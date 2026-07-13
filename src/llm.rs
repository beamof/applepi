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

/// 流式 chat，单次请求完成。SSE 流内同时处理三种增量：
/// - 文本（delta.content）→ `Delta::Text` 增量透传
/// - 工具调用（delta.tool_calls）→ 按 index 聚合（首片带 id/type/name，
///   arguments 跨多片拼接）
/// - finish_reason → 区分 stop（正常）/ tool_calls / length（截断）
///
/// 收尾时按结果发 `ToolCalls` / `Truncated` / 都不发，最后发 `Final`。
pub fn chat_stream(
    cfg: LlmConfig,
    messages: Vec<Message>,
    tools: ToolMap,
    client: Client,
) -> mpsc::Receiver<Result<Delta>> {
    let (tx, rx) = mpsc::channel::<Result<Delta>>(32);
    tokio::spawn(async move {
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

        // SSE 内 tool_calls 按 index 分片到达，需跨片聚合。
        let mut tool_calls: Vec<ToolCall> = Vec::new();
        let mut finish_reason: Option<String> = None;

        let mut byte_stream = resp.bytes_stream();
        let mut buf = String::new();
        let mut done = false;
        while !done {
            let chunk = match byte_stream.next().await {
                Some(Ok(c)) => c,
                Some(Err(e)) => {
                    let _ = tx.send(Err(anyhow!("stream error: {e}"))).await;
                    return;
                }
                None => break,
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
                    done = true;
                    break;
                }
                let v: Value = match serde_json::from_str(data) {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                let choice = match v.get("choices").and_then(|c| c.get(0)) {
                    Some(c) => c,
                    None => continue,
                };
                let delta = choice.get("delta");
                // 增量文本
                if let Some(text) = delta
                    .and_then(|d| d.get("content"))
                    .and_then(|c| c.as_str())
                {
                    if !text.is_empty()
                        && tx.send(Ok(Delta::Text(text.to_string()))).await.is_err()
                    {
                        return; // 接收端提前结束
                    }
                }
                // 工具调用分片聚合
                if let Some(tc_arr) = delta.and_then(|d| d.get("tool_calls")) {
                    merge_tool_call_deltas(&mut tool_calls, tc_arr);
                }
                // finish_reason（通常在最后一片给出）
                if let Some(fr) = choice.get("finish_reason").and_then(|f| f.as_str()) {
                    finish_reason = Some(fr.to_string());
                }
            }
        }

        // 收尾：按 finish_reason 和聚合结果决定语义
        if !tool_calls.is_empty() {
            let _ = tx.send(Ok(Delta::ToolCalls(tool_calls))).await;
        } else if finish_reason.as_deref() == Some("length") {
            // 被 max_tokens 截断：文本已增量透传，只发标志让 agent 续轮
            let _ = tx.send(Ok(Delta::Truncated)).await;
        }
        let _ = tx.send(Ok(Delta::Final)).await;
    });
    rx
}

/// 聚合 SSE 流里分片到达的 tool_calls。
/// 每片形如 `{"index":N,"id":..,"type":..,"function":{"name":..,"arguments":..}}`：
/// 首片带 id/type/function.name，后续片只增量追加 function.arguments。
fn merge_tool_call_deltas(out: &mut Vec<ToolCall>, arr: &Value) {
    let Some(arr) = arr.as_array() else { return };
    for d in arr {
        let idx = d.get("index").and_then(|i| i.as_u64()).unwrap_or(0) as usize;
        while out.len() <= idx {
            out.push(ToolCall {
                id: String::new(),
                r#type: "function".into(),
                function: FunctionCall {
                    name: String::new(),
                    arguments: String::new(),
                },
            });
        }
        let tc = &mut out[idx];
        if let Some(id) = d.get("id").and_then(|v| v.as_str()) {
            tc.id = id.to_string();
        }
        if let Some(ty) = d.get("type").and_then(|v| v.as_str()) {
            tc.r#type = ty.to_string();
        }
        if let Some(func) = d.get("function") {
            if let Some(name) = func.get("name").and_then(|v| v.as_str()) {
                tc.function.name = name.to_string();
            }
            if let Some(args) = func.get("arguments").and_then(|v| v.as_str()) {
                tc.function.arguments.push_str(args);
            }
        }
    }
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
