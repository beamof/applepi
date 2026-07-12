use anyhow::Result;
use reqwest::Client;

use crate::llm::{chat_stream, Delta, LlmConfig, Message, ToolCall};
use crate::memory::long_term::LongTermMemory;
use crate::memory::short_term::History;
use crate::tools::ToolMap;

pub const MAX_TURNS: usize = 64;

/// Agent 主入口：持有配置、工具、记忆。
pub struct Agent {
    pub(crate) cfg: LlmConfig,
    pub(crate) tools: ToolMap,
    pub(crate) http: Client,
    pub(crate) history: History,
    pub(crate) long_term: Option<LongTermMemory>,
    pub(crate) top_k: usize,
    /// 最近一次用户输入，供续跑成功后写入长期记忆用。
    pub(crate) last_input: Option<String>,
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
        long_term: Option<LongTermMemory>,
        top_k: usize,
    ) -> Self {
        let mut persona = persona;
        persona.push_str(&build_mcp_summary(&tools));
        Self {
            cfg,
            tools,
            http: Client::new(),
            history: History::new(persona),
            long_term,
            top_k,
            last_input: None,
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
        for _ in 0..MAX_TURNS {
            let mut rx = chat_stream(
                self.cfg.clone(),
                self.history.all().to_vec(),
                self.tools.clone(),
                self.http.clone(),
            );

            let mut text_buf = String::new();
            let mut tool_calls: Option<Vec<ToolCall>> = None;

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
            events.push(AgentEvent::Final(text_buf));
            // 异步存记忆（不阻塞返回）
            if let Some(input) = self.last_input.as_deref() {
                self.maybe_remember(input).await;
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
    async fn recall(&self, input: &str) -> Option<String> {
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

    async fn maybe_remember(&self, input: &str) {
        if let Some(mem) = self.long_term.as_ref() {
            // 简单策略：直接存用户原话。生产中可让 LLM 抽取要点。
            let _ = mem.remember(input).await;
        }
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
