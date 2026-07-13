use anyhow::{anyhow, Result};
use futures::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::sync::mpsc;

use crate::tools::ToolMap;

#[derive(Clone, Default, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "tool_call_id")]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl Message {
    pub fn system(s: impl Into<String>) -> Self {
        Self { role: "system".into(), content: Some(s.into()), ..Default::default() }
    }
    pub fn user(s: impl Into<String>) -> Self {
        Self { role: "user".into(), content: Some(s.into()), ..Default::default() }
    }
    pub fn tool_result(id: &str, name: &str, content: String) -> Self {
        Self {
            role: "tool".into(),
            content: Some(content),
            tool_call_id: Some(id.into()),
            name: Some(name.into()),
            ..Default::default()
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub r#type: String,
    pub function: FunctionCall,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Clone)]
pub struct LlmConfig {
    pub api_base: String,
    pub api_key: String,
    pub model: String,
}

/// 一次非流式调用，用于在流式循环里完成 tool_calls（流式拼接 arguments 复杂，
/// 多轮工具调用走非流式更稳）。
pub async fn chat(
    cfg: &LlmConfig,
    messages: &[Message],
    tools: &ToolMap,
    client: &Client,
) -> Result<LlmResponse> {
    let body = build_body(cfg, messages, tools, false);
    let resp = send(cfg, body, client).await?;
    parse_choice(resp)
}

pub struct LlmResponse {
    pub content: Option<String>,
    pub tool_calls: Option<Vec<ToolCall>>,
    /// finish_reason：stop（正常结束）/ length（达 max_tokens 被截断）/ tool_calls 等。
    /// 调用方据此区分"模型说完了"和"被截断了"。
    pub finish_reason: Option<String>,
}

impl LlmResponse {
    pub fn into_message(self) -> Message {
        Message {
            role: "assistant".into(),
            content: self.content,
            tool_calls: self.tool_calls,
            ..Default::default()
        }
    }
}

// ---------- 流式 ----------

#[derive(Debug, Clone)]
pub enum Delta {
    /// 增量文本
    Text(String),
    /// 这一轮结束；若 Some 则表示需要调工具
    ToolCalls(Vec<ToolCall>),
    /// 本轮已给出最终文本答复（content 字段结束）
    Final,
    /// 本轮因 max_tokens 被截断（finish_reason=length），调用方应继续下一轮接续
    Truncated,
}

/// 流式 chat。tool_calls 走非流式（在内部完成）；纯文本走 SSE 增量推送。
/// 通过 `Delta` 统一对外，调用方无需关心差异。
pub fn chat_stream(
    cfg: LlmConfig,
    messages: Vec<Message>,
    tools: ToolMap,
    client: Client,
) -> mpsc::Receiver<Result<Delta>> {
    let (tx, rx) = mpsc::channel::<Result<Delta>>(32);
    tokio::spawn(async move {
        // 先发非流式请求探测：是否要调工具
        let probe = match chat(&cfg, &messages, &tools, &client).await {
            Ok(p) => p,
            Err(e) => {
                let _ = tx.send(Err(e)).await;
                return;
            }
        };
        if let Some(calls) = probe.tool_calls {
            let _ = tx.send(Ok(Delta::ToolCalls(calls))).await;
            let _ = tx.send(Ok(Delta::Final)).await;
            return;
        }
        // 无 tool_calls 但被 max_tokens 截断：发 Truncated，让 agent 续轮接续，
        // 而不是把半截文字当成最终答复。
        if probe.finish_reason.as_deref() == Some("length") {
            let _ = tx.send(Ok(Delta::Truncated)).await;
            // 把已有的半截文本也透传给 agent，由 agent 入历史后继续
            if let Some(t) = probe.content {
                if !t.is_empty() {
                    let _ = tx.send(Ok(Delta::Text(t))).await;
                }
            }
            let _ = tx.send(Ok(Delta::Final)).await;
            return;
        }

        // 纯文本路径：SSE 流
        let body = build_body(&cfg, &messages, &tools, true);
        let resp = match client
            .post(format!("{}/chat/completions", cfg.api_base))
            .bearer_auth(&cfg.api_key)
            .json(&body)
            .send()
            .await
        {
            Ok(r) => r,
            Err(e) => {
                let _ = tx.send(Err(anyhow!("请求失败: {e}"))).await;
                return;
            }
        };
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            let _ = tx.send(Err(anyhow!("LLM 流式请求失败 [{status}]: {text}"))).await;
            return;
        }

        let mut byte_stream = resp.bytes_stream();
        let mut buf = String::new();
        let mut finished = false;
        while let Some(chunk) = byte_stream.next().await {
            let chunk = match chunk {
                Ok(c) => c,
                Err(e) => {
                    let _ = tx.send(Err(anyhow!("stream error: {e}"))).await;
                    return;
                }
            };
            buf.push_str(std::str::from_utf8(&chunk).unwrap_or(""));
            while let Some(idx) = buf.find('\n') {
                let line: String = buf.drain(..=idx).collect();
                let line = line.trim();
                if line.is_empty() || line.starts_with(':') {
                    continue;
                }
                let Some(data) = line.strip_prefix("data: ") else {
                    continue;
                };
                if data == "[DONE]" {
                    finished = true;
                    break;
                }
                let v: Value = match serde_json::from_str(data) {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                let delta = v
                    .get("choices")
                    .and_then(|c| c.get(0))
                    .and_then(|c| c.get("delta"))
                    .and_then(|d| d.get("content"))
                    .and_then(|c| c.as_str());
                if let Some(text) = delta {
                    if !text.is_empty() {
                        if tx.send(Ok(Delta::Text(text.to_string()))).await.is_err() {
                            return; // 接收端提前结束
                        }
                    }
                }
            }
            if finished {
                break;
            }
        }
        let _ = tx.send(Ok(Delta::Final)).await;
    });
    rx
}

// ---------- 内部 ----------

fn build_body(
    cfg: &LlmConfig,
    messages: &[Message],
    tools: &ToolMap,
    stream: bool,
) -> Value {
    let tools_schema: Vec<Value> = tools
        .values()
        .map(|t| {
            json!({
                "type": "function",
                "function": {
                    "name": t.name(),
                    "description": t.description(),
                    "parameters": t.parameters(),
                }
            })
        })
        .collect();

    let mut body = json!({
        "model": cfg.model,
        "messages": messages,
        "tools": tools_schema,
        "stream": stream,
    });
    if stream {
        body["stream_options"] = json!({ "include_usage": false });
    }
    body
}

async fn send(cfg: &LlmConfig, body: Value, client: &Client) -> Result<Value> {
    let resp = client
        .post(format!("{}/chat/completions", cfg.api_base))
        .bearer_auth(&cfg.api_key)
        .json(&body)
        .send()
        .await?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        return Err(anyhow!("LLM 请求失败 [{status}]: {text}"));
    }
    Ok(resp.json().await?)
}

fn parse_choice(json: Value) -> Result<LlmResponse> {
    let msg = json
        .get("choices")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("message"))
        .ok_or_else(|| anyhow!("响应缺少 choices[0].message"))?;

    let content = msg.get("content").and_then(|v| v.as_str()).map(String::from);
    let tool_calls = msg
        .get("tool_calls")
        .and_then(|v| serde_json::from_value(v.clone()).ok());
    // finish_reason 在 choices[0] 上，而非 message 上
    let finish_reason = json
        .get("choices")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("finish_reason"))
        .and_then(|v| v.as_str())
        .map(String::from);

    Ok(LlmResponse { content, tool_calls, finish_reason })
}
