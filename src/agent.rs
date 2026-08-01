use anyhow::Result;
use reqwest::Client;
use serde_json::Value;
use std::sync::{Arc, OnceLock};

use crate::llm::{build_tools_schema, chat, chat_stream, Delta, LlmConfig, Message, ToolCall};
use crate::memory::long_term::LongTermMemory;
use crate::memory::short_term::History;
use crate::tools::ToolMap;

pub const MAX_TURNS: usize = 64;

/// 记忆抽取的 system prompt。
///
/// 目标：从一轮对话（用户输入 + 助手答复）里抽出值得**长期**记住的信息，
/// 跳过寒暄、一次性指令、琐碎细节。输出 JSON 数组（每条是一句可独立理解的陈述），
/// 无值得记的内容时输出 `[]`。
///
/// 选 JSON 数组而非自由文本：便于 `serde_json::from_str` 解析，解析失败兜底跳过
/// （绝不写入垃圾到记忆库）。
const EXTRACT_PROMPT: &str = "\
你是一个记忆抽取器。分析下面这轮对话（用户输入 + 助手答复），提取值得长期记住的信息。\n\
\n\
值得记的：\n\
- 用户偏好、习惯、个人背景（如「用户偏好 Rust」「用户在用 applepi 项目」）\n\
- 重要事实、约定、决策结论（如「项目部署在 example.com」）\n\
- 任务的关键结果（如「已把搜索换成 FTS5」）\n\
\n\
不要记的：寒暄、一次性指令、过程性琐碎细节、常识、本次对话中已显而易见的临时信息。\n\
\n\
输出格式：严格的 JSON 字符串数组，每条是一个可独立理解、带主语的陈述句。\
不要输出数组以外的任何文字（不要 markdown 围栏、不要解释）。\n\
示例：[\"用户喜欢用 Rust 写后端\", \"用户的项目 applepi 部署在 example.com\"]\n\
无值得记的内容时输出：[]";

/// 判断一条 agent 回复是否标记为静默（不应推送给用户）。
/// agent 在回复开头或结尾加 `[SILENT]` 即触发，调用方据此跳过发送。
pub fn is_silent(s: &str) -> bool {
    let t = s.trim();
    t.starts_with("[SILENT]") || t.ends_with("[SILENT]")
}

/// 解析记忆抽取 LLM 的输出，返回有效记忆条目列表。
///
/// 期望格式：JSON 字符串数组，如 `["用户偏好 Rust", "项目部署在 example.com"]`。
/// 容错：模型偶尔会在数组外加 markdown 围栏（```json ... ```）或前后噪声文字，
/// 这里取**首个 `[` 到末个 `]`** 的子串再解析，兼容这种情况。
///
/// 解析失败、非数组、含非字符串元素、或全部为空白条目时返回空 Vec
/// （调用方据此跳过写入，绝不污染记忆库）。
pub fn parse_extract_output(raw: &str) -> Vec<String> {
    let trimmed = raw.trim();
    // 提取首个 [ 到末个 ] 的子串，兼容前后噪声 / 围栏
    let slice = match (trimmed.find('['), trimmed.rfind(']')) {
        (Some(start), Some(end)) if start < end => &trimmed[start..=end],
        _ => return Vec::new(),
    };
    let parsed: serde_json::Value = match serde_json::from_str(slice) {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };
    let arr = match parsed.as_array() {
        Some(a) => a,
        None => return Vec::new(),
    };
    arr.iter()
        .filter_map(|v| v.as_str().map(|s| s.trim().to_string()))
        .filter(|s| !s.is_empty())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::parse_extract_output;

    #[test]
    fn parse_plain_array() {
        let out = parse_extract_output(r#"["用户偏好 Rust", "项目部署在 example.com"]"#);
        assert_eq!(out, vec!["用户偏好 Rust", "项目部署在 example.com"]);
    }

    #[test]
    fn parse_empty_array() {
        let out = parse_extract_output("[]");
        assert!(out.is_empty());
    }

    #[test]
    fn parse_with_markdown_fence() {
        let raw = "```json\n[\"条目一\", \"条目二\"]\n```";
        let out = parse_extract_output(raw);
        assert_eq!(out, vec!["条目一", "条目二"]);
    }

    #[test]
    fn parse_with_leading_trailing_noise() {
        // 模型偶尔会在数组前后加解释文字
        let raw = "好的，这是抽取结果：[\"a\",\"b\"]\n以上。";
        let out = parse_extract_output(raw);
        assert_eq!(out, vec!["a", "b"]);
    }

    #[test]
    fn parse_invalid_returns_empty() {
        // 非 JSON / 不是数组 / 缺括号 —— 全部返回空，绝不 panic
        assert!(parse_extract_output("not json at all").is_empty());
        assert!(parse_extract_output("{\"key\": \"value\"}").is_empty());
        assert!(parse_extract_output("[unclosed").is_empty());
    }

    #[test]
    fn parse_skips_blank_and_non_string() {
        // 混合类型 / 空白条目：只保留有效非空字符串
        let raw = r#"["有效", "", 123, "  ", "也有效"]"#;
        let out = parse_extract_output(raw);
        assert_eq!(out, vec!["有效", "也有效"]);
    }
}

/// Agent 主入口：持有配置、工具、记忆。
pub struct Agent {
    pub(crate) cfg: LlmConfig,
    pub(crate) tools: ToolMap,
    pub(crate) http: Client,
    pub(crate) history: History,
    pub(crate) long_term: Option<Arc<LongTermMemory>>,
    pub(crate) top_k: usize,
    /// 最近一次用户输入，供续跑成功后写入长期记忆用。
    pub(crate) last_input: Option<String>,
    /// tools schema 序列化缓存：工具集在 Agent 生命周期内不变，
    /// 首轮构建一次后复用，避免每轮 LLM 请求都重算。
    pub(crate) tools_schema_cache: OnceLock<Vec<Value>>,
}

/// 一次对话产出的事件流。调用方据此渲染 UI（终端逐字、Telegram 增量编辑）。
#[derive(Debug, Clone)]
pub enum AgentEvent {
    /// 增量文本（流式）
    Text(String),
    /// 触发了工具（调试/可见性用）
    ToolCall { name: String, args: String },
    /// 一轮的最终文本（聚合后）
    Final(String),
    /// 工具调用出错（已捕获，作为结果喂回模型继续）
    ToolError(String),
    /// 达到本轮上限（MAX_TURNS）仍未给出最终答复，需询问用户是否继续。
    /// 携带字符串 = 给用户看的提示语。同意续跑则用 continue_stream 重置计数继续。
    ContinuePrompt(String),
}

/// 把所有 `mcp__` 前缀的工具汇总成一段「可用工具」清单，连同通用使用引导
/// 一起追加到 system message。新增/移除 MCP server 自动反映，无需改文档。
/// 没有 MCP 工具时返回空串（不污染 system message）。
fn build_mcp_summary(tools: &ToolMap) -> String {
    let mut names: Vec<&String> = tools.keys().filter(|n| n.starts_with("mcp__")).collect();
    if names.is_empty() {
        return String::new();
    }
    names.sort();
    let listing = names
        .iter()
        .map(|n| {
            let desc = tools
                .get(*n)
                .map(|t| t.description())
                .unwrap_or("")
                .trim();
            if desc.is_empty() {
                format!("- `{n}`")
            } else {
                // description 可能多行，压成一行避免破坏清单结构
                let one_line = desc.split_whitespace().collect::<Vec<_>>().join(" ");
                format!("- `{n}`：{one_line}")
            }
        })
        .collect::<Vec<_>>()
        .join("\n");

    format!(
        "\n\n## MCP 工具（可用清单 + 使用准则）\n\
以下是以 `mcp__` 开头的外部能力工具（联网、浏览器、数据库等），按各自描述提供能力：\n\n\
{listing}\n\n\
**通用准则**：当用户的请求落在某个工具的能力范围内时，主动调用它——不要凭自身记忆回答\
需要实时/外部数据的问题（如最新资讯、网页内容、外部系统数据），也不要反问「要不要用某工具」。\
按工具描述判断是否匹配即可。"
    )
}

impl Agent {
    pub fn new(
        cfg: LlmConfig,
        persona: String,
        tools: ToolMap,
        long_term: Option<Arc<LongTermMemory>>,
        top_k: usize,
        http: Client,
    ) -> Self {
        let mut persona = persona;
        persona.push_str(&build_mcp_summary(&tools));
        Self {
            cfg,
            tools,
            http,
            history: History::new(persona),
            long_term,
            top_k,
            last_input: None,
            tools_schema_cache: OnceLock::new(),
        }
    }

    /// 流式对话。返回事件流。
    pub async fn chat_stream(&mut self, input: &str) -> Result<Vec<AgentEvent>> {
        self.last_input = Some(input.to_string());
        // 长期记忆拼到当前 user 消息尾部（而非注入 system），保持 system 消息稳定、
        // 最大化 prompt 前缀缓存命中。
        let memory_block = self.recall(input).await;
        let user_msg = match memory_block {
            Some(m) => format!("{input}\n\n{m}"),
            None => input.to_string(),
        };
        self.history.add(Message::user(user_msg));
        self.turn_loop().await
    }

    /// 续跑：在达到本轮上限后，经用户同意则调用本方法，重置计数从 0 重新开始。
    /// 复用现有 history（末尾是 tool_result），不重新注入用户输入。
    pub async fn continue_stream(&mut self) -> Result<Vec<AgentEvent>> {
        self.turn_loop().await
    }

    /// 一轮上限内的循环：最多 MAX_TURNS 次模型往返。
    /// - 收到纯文本答复 → 入历史、存记忆、发 Final、结束。
    /// - 全部耗尽仍未收尾 → 发 ContinuePrompt、结束（由调用方决定是否 continue_stream 续跑）。
    async fn turn_loop(&mut self) -> Result<Vec<AgentEvent>> {
        let mut events = Vec::new();
        // tools schema 首轮构建一次，后续轮次复用（工具集在 Agent 生命周期内不变）。
        if self.tools_schema_cache.get().is_none() {
            let _ = self
                .tools_schema_cache
                .set(build_tools_schema(&self.tools));
        }
        let tools_schema = self
            .tools_schema_cache
            .get()
            .expect("tools_schema_cache 已初始化")
            .clone();
        for _ in 0..MAX_TURNS {
            let mut rx = chat_stream(
                self.cfg.clone(),
                self.history.all().to_vec(),
                tools_schema.clone(),
                self.http.clone(),
            );

            let mut text_buf = String::new();
            let mut tool_calls: Option<Vec<ToolCall>> = None;
            let mut truncated = false;

            while let Some(item) = rx.recv().await {
                match item? {
                    Delta::Text(t) => {
                        events.push(AgentEvent::Text(t.clone()));
                        text_buf.push_str(&t);
                    }
                    Delta::ToolCalls(calls) => {
                        tool_calls = Some(calls);
                    }
                    Delta::Final => {}
                    Delta::Truncated => {
                        truncated = true;
                    }
                }
            }

            if let Some(calls) = tool_calls {
                // 把 assistant 消息（含 tool_calls）入历史
                self.history.add(Message {
                    role: "assistant".into(),
                    content: if text_buf.is_empty() {
                        None
                    } else {
                        Some(text_buf.clone())
                    },
                    tool_calls: Some(calls.clone()),
                    ..Default::default()
                });
                for call in &calls {
                    events.push(AgentEvent::ToolCall {
                        name: call.function.name.clone(),
                        args: call.function.arguments.clone(),
                    });
                    let result = match self.dispatch(call).await {
                        Ok(s) => s,
                        Err(e) => {
                            let msg = format!("[工具错误] {e}");
                            events.push(AgentEvent::ToolError(msg.clone()));
                            msg
                        }
                    };
                    self.history.add(Message::tool_result(
                        &call.id,
                        &call.function.name,
                        result,
                    ));
                }
                // 继续下一轮
                continue;
            }

            // 被 max_tokens 截断：把已有的半截文本入历史，让模型在下一轮接续写完，
            // 而不是把半句话当成最终答复推送出去（典型表现如"输出被截断了，让我
            // 提取完整的 message ID 列表进行对比"被原样发出去）。
            if truncated {
                self.history.add(Message {
                    role: "assistant".into(),
                    content: if text_buf.is_empty() {
                        None
                    } else {
                        Some(text_buf.clone())
                    },
                    ..Default::default()
                });
                continue;
            }

            // 纯文本答复结束。
            // 注意：模型有时在多步任务末尾返回空 content（既不调工具也不输出
            // 文字，视为已完成），这里照常产出 Final("")，由调用方各自决定
            // 如何呈现（bot 显示兜底文案；cron 静默跳过）。
            self.history.add(Message {
                role: "assistant".into(),
                content: if text_buf.is_empty() {
                    None
                } else {
                    Some(text_buf.clone())
                },
                ..Default::default()
            });
            events.push(AgentEvent::Final(text_buf.clone()));
            // 异步抽取长期记忆（spawn 后立即返回，不阻塞）
            if let Some(input) = self.last_input.as_deref() {
                self.maybe_remember(input, &text_buf);
            }
            return Ok(events);
        }
        // 全部轮次耗尽仍未给出最终答复：询问用户是否继续
        events.push(AgentEvent::ContinuePrompt(format!(
            "（已达到最大轮次 {MAX_TURNS}，是否继续？回复「继续」即可）"
        )));
        Ok(events)
    }

    /// 非流式便捷封装（CLI 用）
    pub async fn chat(&mut self, input: &str) -> Result<String> {
        let events = self.chat_stream(input).await?;
        let mut out = String::new();
        for e in events {
            match e {
                AgentEvent::Text(t) | AgentEvent::Final(t) => out.push_str(&t),
                _ => {}
            }
        }
        Ok(out)
    }

    /// 检索长期记忆，命中则返回格式化文本块，供调用方拼入当前 user 消息。
    /// 不再写入 system 消息，以保持 system 稳定、提高 prompt 前缀缓存命中率。
    ///
    /// 短输入跳过：太短（<4 字符）或纯斜杠命令（`/xxx`）的输入不检索，
    /// 省一次本地推理且这类输入通常也不需要记忆上下文。
    async fn recall(&self, input: &str) -> Option<String> {
        let trimmed = input.trim();
        if trimmed.len() < 4 || trimmed.starts_with('/') {
            return None;
        }
        let mem = self.long_term.as_ref()?;
        match mem.recall(input, self.top_k).await {
            Ok(hits) if !hits.is_empty() => {
                let block = hits
                    .iter()
                    .map(|h| format!("- {h}"))
                    .collect::<Vec<_>>()
                    .join("\n");
                Some(format!("[长期记忆]\n{block}"))
            }
            _ => None,
        }
    }

    /// 异步抽取长期记忆：spawn 一个独立任务，从本轮对话（input + reply）
    /// 让 LLM 抽取值得长期记住的信息，批量写入记忆库。
    ///
    /// 设计要点：
    /// - **异步**：spawn 后立即返回，不阻塞对话主流程。代价是进程崩溃时这一轮
    ///   记忆会丢（可接受，下一轮还会触发，且 write_memory 工具可补）。
    /// - **轻量过滤**：spawn 前同步跳过明显不值得抽取的对话（过短 / 斜杠命令 /
    ///   空答复），省一次 LLM 调用。
    /// - **失败静默**：抽取 / 解析 / 写入任一步失败都只 `tracing::warn`，
    ///   绝不影响对话；解析失败宁可跳过也不写脏数据。
    /// - **共享实例**：所有 move 进任务的字段都是 Clone 便宜的（cfg / http /
    ///   Arc<LongTermMemory>），任务独立运行不借用 &mut self。
    fn maybe_remember(&self, input: &str, reply: &str) {
        let mem = match self.long_term.clone() {
            Some(m) => m,
            None => return, // 记忆未启用
        };

        // 轻量过滤：不值得抽取的对话直接跳过，省一次 LLM 调用
        let input_t = input.trim();
        if input_t.len() < 8 || input_t.starts_with('/') || reply.trim().is_empty() {
            return;
        }

        let cfg = self.cfg.clone();
        let http = self.http.clone();
        let input = input.to_string();
        let reply = reply.to_string();
        tokio::spawn(async move {
            let user_msg = format!(
                "用户输入：\n{input}\n\n助手答复：\n{reply}\n\n请抽取值得长期记住的信息。"
            );
            let messages = vec![
                Message::system(EXTRACT_PROMPT),
                Message::user(user_msg),
            ];
            let raw = match chat(cfg, messages, http).await {
                Ok(s) => s,
                Err(e) => {
                    tracing::warn!("记忆抽取 LLM 调用失败（已跳过）: {e}");
                    return;
                }
            };
            let items = parse_extract_output(&raw);
            if items.is_empty() {
                tracing::debug!("记忆抽取：本轮无可记内容");
                return;
            }
            match mem.remember_batch(&items).await {
                Ok(_) => tracing::info!("记忆抽取：写入 {} 条", items.len()),
                Err(e) => tracing::warn!("记忆抽取写入失败（已跳过）: {e}"),
            }
        });
    }

    async fn dispatch(&self, call: &ToolCall) -> Result<String> {
        let tool = self
            .tools
            .get(&call.function.name)
            .ok_or_else(|| anyhow::anyhow!("未知工具: {}", call.function.name))?;
        let args = if call.function.arguments.trim().is_empty() {
            serde_json::Value::Object(Default::default())
        } else {
            serde_json::from_str(&call.function.arguments)?
        };
        tool.run(args).await
    }
}
