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

impl Agent {
    pub fn new(
        cfg: LlmConfig,
        persona: String,
        tools: ToolMap,
        long_term: Option<LongTermMemory>,
        top_k: usize,
    ) -> Self {
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
        self.recall_and_inject(input).await;
        self.history.add(Message::user(input));
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

            // 纯文本答复结束
            self.history.add(Message {
                role: "assistant".into(),
                content: Some(text_buf.clone()),
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

    async fn recall_and_inject(&mut self, input: &str) {
        let Some(mem) = self.long_term.as_ref() else {
            return;
        };
        match mem.recall(input, self.top_k).await {
            Ok(hits) if !hits.is_empty() => {
                let block = hits
                    .iter()
                    .map(|h| format!("- {h}"))
                    .collect::<Vec<_>>()
                    .join("\n");
                let note = format!("\n\n[长期记忆]\n{block}");
                if let Some(sys) = self.history.system_mut() {
                    // 末尾追加，避免覆盖 persona
                    sys.content = Some(format!(
                        "{}{note}",
                        sys.content.clone().unwrap_or_default()
                    ));
                }
            }
            _ => {}
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
