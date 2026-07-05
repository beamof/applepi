use anyhow::Result;
use reqwest::Client;

use crate::llm::{chat_stream, Delta, LlmConfig, Message, ToolCall};
use crate::memory::long_term::LongTermMemory;
use crate::memory::short_term::History;
use crate::tools::ToolMap;

const MAX_TURNS: usize = 6;

/// Agent 主入口：持有配置、工具、记忆。
pub struct Agent {
    pub(crate) cfg: LlmConfig,
    pub(crate) tools: ToolMap,
    pub(crate) http: Client,
    pub(crate) history: History,
    pub(crate) long_term: Option<LongTermMemory>,
    pub(crate) top_k: usize,
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
        }
    }

    /// 流式对话。返回事件流。
    pub async fn chat_stream(&mut self, input: &str) -> Result<Vec<AgentEvent>> {
        self.recall_and_inject(input).await;
        self.history.add(Message::user(input));

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
            self.maybe_remember(input).await;
            return Ok(events);
        }
        events.push(AgentEvent::Final(
            "（达到最大轮次，仍未给出最终答复）".into(),
        ));
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
