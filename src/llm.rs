use anyhow::{anyhow, Result};
use futures::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio::sync::RwLock;

use crate::tools::ToolMap;

/// 接口协议类型。决定请求路径、body 结构与 SSE 解析方式。
/// - `Responses`：较新的 `/v1/responses`（OpenAI 原生、部分新模型仅支持此接口），**默认优先尝试**
/// - `ChatCompletions`：经典 `/v1/chat/completions`（OpenAI 兼容生态通用），Responses 不可用时回退到此
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ApiType {
    ChatCompletions,
    Responses,
}

/// 进程级协议探测结果缓存。
/// - `None`：尚未探测，先按默认（Responses）尝试
/// - `Some(ChatCompletions)`：已确认端点不支持 Responses，后续所有请求直接走 Chat Completions
///
/// 用 `Arc<RwLock>` 共享：首个失败请求一次性回退并记住结论，避免每条消息都先失败一次。
/// 探测成功（Responses 可用）时不写入，保持 `None`，让每次请求继续走 Responses。
pub(crate) type Fallback = Arc<RwLock<Option<ApiType>>>;

/// 构造一个初始的协议探测状态（`None` = 先试 Responses）。
pub(crate) fn new_fallback() -> Fallback {
    Arc::new(RwLock::new(None))
}

/// 判断一个 HTTP 状态码是否表示「Responses 接口不可用，应回退到 Chat Completions」。
/// 覆盖：404（路径不存在）、400/404/405（模型不支持 / 方法不允许）。
/// 不含 401/429/5xx——那些是鉴权 / 限流 / 服务端问题，回退也救不了。
fn should_fallback(status: reqwest::StatusCode) -> bool {
    use reqwest::StatusCode;
    matches!(status, StatusCode::NOT_FOUND | StatusCode::BAD_REQUEST | StatusCode::METHOD_NOT_ALLOWED)
}

/// 统一构建共享 HTTP 客户端：池化连接、TCP keepalive、合理超时。
/// 整个进程复用同一个 Client（clone 便宜，连接池共享），避免每个 Agent / 每次触发
/// 都新建 Client 导致首次 TLS 握手成本落到首条用户消息上。
pub fn build_http_client() -> Result<Client> {
    Ok(Client::builder()
        .pool_max_idle_per_host(32)
        .pool_idle_timeout(Duration::from_secs(90))
        .tcp_keepalive(Duration::from_secs(60))
        .connect_timeout(Duration::from_secs(15))
        .timeout(Duration::from_secs(300))
        .build()?)
}

/// 拉取接口支持的模型列表（OpenAI 兼容的 GET /models）。
///
/// 返回去重、按字典序排序后的模型 id，用于 /model 命令向用户展示可切换的模型。
/// 端点不支持或请求失败时返回 Err（由调用方反馈给用户，不影响会话）。
pub async fn list_models(cfg: &LlmConfig, client: &Client) -> Result<Vec<String>> {
    let resp = client
        .get(format!("{}/models", cfg.api_base))
        .bearer_auth(&cfg.api_key)
        .send()
        .await?;
    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        return Err(anyhow!("获取模型列表失败 [{status}]: {text}"));
    }
    let v: Value = resp.json().await?;
    let mut ids: Vec<String> = v
        .get("data")
        .and_then(|d| d.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|m| m.get("id").and_then(|i| i.as_str()).map(String::from))
                .collect()
        })
        .unwrap_or_default();
    ids.sort();
    ids.dedup();
    Ok(ids)
}

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
    /// 是否给 system 消息追加 `cache_control: {type: "ephemeral"}`，
    /// 显式声明 prompt 前缀缓存（Anthropic / 部分 OpenAI 兼容端点支持）。
    /// DeepSeek / GLM 等自动缓存的端点保持 false 即可，避免无效字段干扰。
    pub prompt_cache_control: bool,
    /// 协议探测缓存：默认先尝试 Responses，端点不支持时回退到 Chat Completions 并记住。
    /// 同一进程内共享一份，首个失败请求触发回退，后续请求直接走已确认可用的协议。
    pub(crate) fallback: Fallback,
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

/// 流式 chat 入口：默认优先 Responses，端点不支持时自动回退到 Chat Completions。
///
/// 探测策略（避免每条消息都先失败一次）：
/// 1. 先读 `cfg.fallback`：已回退则直接走 Chat Completions；否则按 Responses 发请求。
/// 2. Responses 返回「不支持」类状态（404/400/405）→ 记入 `fallback`，本条用 Chat Completions 重试。
/// 3. Responses 成功 / 其它失败（401/429/5xx 等）→ 原样处理，不回退。
///
/// 单次请求完成，返回 `Delta` 事件流（统一抽象，调用方无需关心底层协议）。
pub fn chat_stream(
    cfg: LlmConfig,
    messages: Vec<Message>,
    tools_schema: Vec<Value>,
    client: Client,
) -> mpsc::Receiver<Result<Delta>> {
    let (tx, rx) = mpsc::channel::<Result<Delta>>(32);
    tokio::spawn(async move {
        // 已回退过 → 直接 Chat Completions
        if cfg.fallback.read().await.is_some() {
            relay_channel(&tx, chat_completions_stream(cfg, messages, tools_schema, client)).await;
            return;
        }

        // 先按 Responses 发请求；先读状态码再决定是否进入流式。
        let body = build_responses_body(&cfg, &messages, &tools_schema, true);
        let resp = match client
            .post(format!("{}/responses", cfg.api_base))
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

        if resp.status().is_success() {
            // Responses 可用：直接流入流式解析。
            relay_channel(&tx, responses_consume(resp, cfg.fallback.clone())).await;
            return;
        }

        // Responses 返回「不支持」类状态 → 记入 fallback 并用 Chat Completions 重试。
        if should_fallback(resp.status()) {
            let status = resp.status();
            // 消费掉 body 释放连接
            let _ = resp.text().await;
            tracing::info!("Responses 接口不可用（{status}），回退到 Chat Completions");
            *cfg.fallback.write().await = Some(ApiType::ChatCompletions);
            relay_channel(&tx, chat_completions_stream(cfg, messages, tools_schema, client)).await;
            return;
        }

        // 其它失败：透传错误，不回退
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        let _ = tx
            .send(Err(anyhow!("LLM 流式请求失败 [{status}]: {text}")))
            .await;
    });
    rx
}

/// 把一个内部实现的 `Receiver<Result<Delta>>` 透传到外层 `tx`。
/// 调用方拿到的 channel 行为与直接 spawn 一致。
async fn relay_channel(tx: &mpsc::Sender<Result<Delta>>, mut rx: mpsc::Receiver<Result<Delta>>) {
    while let Some(item) = rx.recv().await {
        if tx.send(item).await.is_err() {
            return; // 外层接收端已关闭
        }
    }
}

/// 经典 Chat Completions 协议的流式实现。
///
/// SSE 流内同时处理三种增量：
/// - 文本（delta.content）→ `Delta::Text` 增量透传
/// - 工具调用（delta.tool_calls）→ 按 index 聚合（首片带 id/type/name，
///   arguments 跨多片拼接）
/// - finish_reason → 区分 stop（正常）/ tool_calls / length（截断）
///
/// 收尾时按结果发 `ToolCalls` / `Truncated` / 都不发，最后发 `Final`。
fn chat_completions_stream(
    cfg: LlmConfig,
    messages: Vec<Message>,
    tools_schema: Vec<Value>,
    client: Client,
) -> mpsc::Receiver<Result<Delta>> {
    let (tx, rx) = mpsc::channel::<Result<Delta>>(32);
    tokio::spawn(async move {
        let body = build_chat_completions_body(&cfg, &messages, &tools_schema, true);
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

// ---------- 非流式 ----------

/// 非流式 chat 入口：默认优先 Responses，端点不支持时自动回退到 Chat Completions。
///
/// 用于不需要流式 / 不需要工具的后台任务（如长期记忆抽取）：
/// 发一个请求，等响应回来把模型输出的文本拼成 String。失败返回 Err。
///
/// 探测策略与 [`chat_stream`] 一致：先查 `cfg.fallback`，未回退则试 Responses；
/// Responses 返回「不支持」类状态（404/400/405）→ 记入 fallback 并用 Chat Completions 重试。
pub async fn chat(
    cfg: LlmConfig,
    messages: Vec<Message>,
    client: Client,
) -> Result<String> {
    // 已回退过 → 直接 Chat Completions
    if cfg.fallback.read().await.is_some() {
        return chat_completions(cfg, messages, client).await;
    }

    let body = build_responses_body(&cfg, &messages, &[], false);
    let resp = client
        .post(format!("{}/responses", cfg.api_base))
        .bearer_auth(&cfg.api_key)
        .json(&body)
        .send()
        .await?;

    if resp.status().is_success() {
        let v: Value = resp.json().await?;
        return Ok(extract_responses_text(&v));
    }

    // Responses 返回「不支持」类状态 → 记入 fallback 并用 Chat Completions 重试。
    if should_fallback(resp.status()) {
        let status = resp.status();
        // 消费掉 body 释放连接
        let _ = resp.text().await;
        tracing::info!("Responses 接口不可用（{status}），回退到 Chat Completions");
        *cfg.fallback.write().await = Some(ApiType::ChatCompletions);
        return chat_completions(cfg, messages, client).await;
    }

    let status = resp.status();
    let text = resp.text().await.unwrap_or_default();
    Err(anyhow!("LLM 非流式请求失败 [{status}]: {text}"))
}

/// 经典 Chat Completions 协议的非流式实现。
/// 发一个 `stream:false` 请求，等响应回来把 `choices[0].message.content` 拼成 String。
async fn chat_completions(
    cfg: LlmConfig,
    messages: Vec<Message>,
    client: Client,
) -> Result<String> {
    let body = build_chat_completions_body(&cfg, &messages, &[], false);
    let resp = client
        .post(format!("{}/chat/completions", cfg.api_base))
        .bearer_auth(&cfg.api_key)
        .json(&body)
        .send()
        .await?;
    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        return Err(anyhow!("LLM 非流式请求失败 [{status}]: {text}"));
    }
    let v: Value = resp.json().await?;
    let content = v
        .get("choices")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("message"))
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_str())
        .unwrap_or("")
        .to_string();
    Ok(content)
}

// ---------- 内部 ----------

/// 一次性把 ToolMap 序列化成 OpenAI tools schema 数组。
/// 调用方缓存结果（如 Agent::tools_schema_cache），避免每轮 LLM 请求重复构建。
pub fn build_tools_schema(tools: &ToolMap) -> Vec<Value> {
    tools
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
        .collect()
}

fn build_chat_completions_body(
    cfg: &LlmConfig,
    messages: &[Message],
    tools_schema: &[Value],
    stream: bool,
) -> Value {
    // 可选：给首条 system 消息追加 cache_control，显式声明 prompt 前缀缓存。
    // 仅当配置打开且首条是 system 时生效；其余情况直接序列化 messages。
    let messages_value: Value = if cfg.prompt_cache_control {
        annotate_system_cache_control(messages)
    } else {
        serde_json::to_value(messages).unwrap_or(Value::Array(vec![]))
    };

    let mut body = json!({
        "model": cfg.model,
        "messages": messages_value,
        "tools": tools_schema,
        "stream": stream,
    });
    if stream {
        body["stream_options"] = json!({ "include_usage": false });
    }
    body
}

/// 把 messages 序列化为 JSON 数组，并为首条 system 消息追加 `cache_control`。
/// 失败则回退为不带 annotation 的纯序列化（保证请求至少能发出）。
fn annotate_system_cache_control(messages: &[Message]) -> Value {
    let mut arr: Vec<Value> = match serde_json::to_value(messages) {
        Ok(Value::Array(a)) => a,
        other => return other.unwrap_or(Value::Array(vec![])),
    };
    if let Some(first) = arr.first_mut() {
        let is_system = first
            .get("role")
            .and_then(|r| r.as_str())
            .map(|s| s == "system")
            .unwrap_or(false);
        if is_system {
            if let Some(obj) = first.as_object_mut() {
                obj.insert(
                    "cache_control".into(),
                    json!({ "type": "ephemeral" }),
                );
            }
        }
    }
    Value::Array(arr)
}

// ========================================================================
// Responses API（/v1/responses）实现
//
// 与 Chat Completions 的差异：
// - 请求体用 `input`（数组）替代 `messages`；system 用 role=system 的 message item。
// - 工具结果用独立的 `function_call_output` item 回传（不再用 role=tool 消息），
//   call_id 关联到对应的 function_call。
// - 历史里的 assistant tool_calls 要还原成 `function_call` item（而非 message），
//   否则模型无法接上工具上下文。
// - SSE 是命名事件（`event: <type>\ndata: {...}`），按事件类型分派，
//   终止信号是 `response.completed` / `response.failed` / `response.incomplete`，
//   而非 `data: [DONE]`。
// ========================================================================

/// Responses API 的流式解析（消费一个已发送且状态码为成功的 `Response`）。
///
/// 增量语义（映射到统一的 `Delta`）：
/// - `response.output_text.delta` → `Delta::Text` 逐 token 透传
/// - `response.function_call_arguments.delta` → 按 item_id 累积 arguments
/// - `response.output_item.added`（type=function_call）→ 记下 call_id/name 占位
/// - `response.completed` → 收尾：有工具调用发 `ToolCalls`，否则按 status 判断截断
/// - `response.incomplete` → 视作截断（reason 通常是 max_output_tokens / length）
/// - `response.failed` → 把错误透传为 Err
fn responses_consume(resp: reqwest::Response, _fallback: Fallback) -> mpsc::Receiver<Result<Delta>> {
    let (tx, rx) = mpsc::channel::<Result<Delta>>(32);
    tokio::spawn(async move {
        // 按 item_id 聚合进行中的 function_call（arguments 跨多片 delta 拼接）。
        // 用 Vec 按 output_index 排序，保证返回顺序与模型产出一致。
        let mut pending_calls: Vec<ToolCall> = Vec::new();
        let mut index_to_pos: std::collections::HashMap<u64, usize> =
            std::collections::HashMap::new();
        let mut status: Option<String> = None;
        let mut stream_ended = false;

        let mut byte_stream = resp.bytes_stream();
        let mut buf = String::new();
        while !stream_ended {
            let chunk = match byte_stream.next().await {
                Some(Ok(c)) => c,
                Some(Err(e)) => {
                    let _ = tx.send(Err(anyhow!("stream error: {e}"))).await;
                    return;
                }
                None => break,
            };
            buf.push_str(std::str::from_utf8(&chunk).unwrap_or(""));
            // SSE：事件间以空行分隔，每个事件由 `event:` 行 + `data:` 行组成。
            // 按空行切块处理完整事件，残余留在 buf 等下次补齐。
            while let Some(idx) = buf.find("\n\n") {
                let event_block: String = buf.drain(..idx + 2).collect();
                let (event_type, data) = parse_sse_event(&event_block);
                if event_type.is_empty() || data.is_empty() {
                    continue;
                }
                let v: Value = match serde_json::from_str(&data) {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                match event_type.as_str() {
                    "response.output_text.delta" => {
                        if let Some(text) = v.get("delta").and_then(|d| d.as_str()) {
                            if !text.is_empty()
                                && tx.send(Ok(Delta::Text(text.to_string()))).await.is_err()
                            {
                                return; // 接收端提前结束
                            }
                        }
                    }
                    "response.output_item.added" => {
                        // function_call 项在这里出现：记下 call_id / name 占位，
                        // arguments 随后由 function_call_arguments.delta 流式补全。
                        if let Some(item) = v.get("item") {
                            if item.get("type").and_then(|t| t.as_str()) == Some("function_call") {
                                let idx = v
                                    .get("output_index")
                                    .and_then(|i| i.as_u64())
                                    .unwrap_or(0);
                                let call_id = item
                                    .get("call_id")
                                    .and_then(|c| c.as_str())
                                    .unwrap_or("")
                                    .to_string();
                                let name = item
                                    .get("name")
                                    .and_then(|n| n.as_str())
                                    .unwrap_or("")
                                    .to_string();
                                let pos = pending_calls.len();
                                index_to_pos.insert(idx, pos);
                                pending_calls.push(ToolCall {
                                    id: call_id,
                                    r#type: "function".into(),
                                    function: FunctionCall {
                                        name,
                                        arguments: String::new(),
                                    },
                                });
                            }
                        }
                    }
                    "response.function_call_arguments.delta" => {
                        let delta = v.get("delta").and_then(|d| d.as_str()).unwrap_or("");
                        if !delta.is_empty() {
                            append_call_args(&mut pending_calls, &index_to_pos, &v, delta);
                        }
                    }
                    "response.function_call_arguments.done" => {
                        // done 里带完整 arguments，作为 delta 拼接的权威兜底：
                        // 若累积结果与 done 不一致（极端分片丢失），以 done 为准。
                        if let Some(args) = v.get("arguments").and_then(|a| a.as_str()) {
                            if !args.is_empty() {
                                append_call_args(&mut pending_calls, &index_to_pos, &v, args);
                            }
                        }
                    }
                    "response.completed" => {
                        if let Some(s) = v
                            .get("response")
                            .and_then(|r| r.get("status"))
                            .and_then(|s| s.as_str())
                        {
                            status = Some(s.to_string());
                        }
                        stream_ended = true;
                    }
                    "response.incomplete" => {
                        // 未完成（典型 max_output_tokens 截断）：视为 Truncated。
                        status = Some("incomplete".into());
                        stream_ended = true;
                    }
                    "response.failed" => {
                        let msg = v
                            .get("response")
                            .and_then(|r| r.get("error"))
                            .map(|e| e.to_string())
                            .unwrap_or_else(|| "response failed".into());
                        let _ = tx.send(Err(anyhow!("LLM response 失败: {msg}"))).await;
                        return;
                    }
                    "error" => {
                        let msg = v
                            .get("message")
                            .and_then(|m| m.as_str())
                            .unwrap_or("unknown error");
                        let _ = tx.send(Err(anyhow!("LLM error 事件: {msg}"))).await;
                        return;
                    }
                    _ => {} // 其余事件（reasoning / content_part / usage 等）忽略
                }
            }
        }

        // 收尾：工具调用优先；否则按 status 判断是否截断。
        if !pending_calls.is_empty() {
            let _ = tx.send(Ok(Delta::ToolCalls(pending_calls))).await;
        } else if status.as_deref() == Some("incomplete") {
            let _ = tx.send(Ok(Delta::Truncated)).await;
        }
        let _ = tx.send(Ok(Delta::Final)).await;
    });
    rx
}

/// 从 Responses API 的非流式响应里抽取全部文本。
/// 遍历 `response.output`：message 项里 content[].text 拼接；其余项（function_call 等）跳过。
pub fn extract_responses_text(v: &Value) -> String {
    let mut out = String::new();
    if let Some(items) = v.get("output").and_then(|o| o.as_array()) {
        for item in items {
            if item.get("type").and_then(|t| t.as_str()) != Some("message") {
                continue;
            }
            if let Some(contents) = item.get("content").and_then(|c| c.as_array()) {
                for part in contents {
                    if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
                        out.push_str(text);
                    }
                }
            }
        }
    }
    out
}

/// 构造 Responses API 请求体。
///
/// 关键转换：把内部统一的 `Message` 列表映射成 Responses 的 input items：
/// - system/user/assistant 纯文本 → `{type:"message", role, content:[{type:...text, text}]}`
/// - assistant 带 tool_calls → 拆成多个 `{type:"function_call", call_id, name, arguments}`
/// - role=tool（工具结果）→ `{type:"function_call_output", call_id, output}`
fn build_responses_body(
    cfg: &LlmConfig,
    messages: &[Message],
    tools_schema: &[Value],
    stream: bool,
) -> Value {
    let mut input: Vec<Value> = Vec::with_capacity(messages.len());
    for m in messages {
        match m.role.as_str() {
            "system" | "developer" => {
                input.push(json!({
                    "type": "message",
                    "role": m.role,
                    "content": message_content_parts(m),
                }));
            }
            "user" => {
                input.push(json!({
                    "type": "message",
                    "role": "user",
                    "content": message_content_parts(m),
                }));
            }
            "assistant" => {
                // 若带 tool_calls：先输出可能的文本 message，再逐个 function_call item。
                // 实际上 Chat 风格里 assistant 同时有 content + tool_calls 的情况，
                // Responses 把文本和 function_call 视为并列的独立 output item。
                if let Some(content) = &m.content {
                    if !content.is_empty() {
                        input.push(json!({
                            "type": "message",
                            "role": "assistant",
                            "content": vec![json!({ "type": "output_text", "text": content })],
                        }));
                    }
                }
                if let Some(calls) = &m.tool_calls {
                    for c in calls {
                        input.push(json!({
                            "type": "function_call",
                            "call_id": c.id,
                            "name": c.function.name,
                            "arguments": c.function.arguments,
                        }));
                    }
                }
            }
            "tool" => {
                // 工具结果：用 tool_call_id（即 Responses 的 call_id）关联。
                let call_id = m.tool_call_id.clone().unwrap_or_default();
                let output = m.content.clone().unwrap_or_default();
                input.push(json!({
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": output,
                }));
            }
            other => {
                // 未知角色兜底按 message 处理，避免丢消息
                input.push(json!({
                    "type": "message",
                    "role": other,
                    "content": message_content_parts(m),
                }));
            }
        }
    }

    // Responses 的 tools 字段结构与 Chat Completions 一致（{type:function, function:{...}}），
    // 可直接复用 tools_schema。空数组时省略，避免部分端点拒收空 tools。
    let mut body = json!({
        "model": cfg.model,
        "input": input,
        "stream": stream,
    });
    if !tools_schema.is_empty() {
        body["tools"] = json!(tools_schema);
    }
    body
}

/// 把 Message 的 content 转成 Responses 风格的 content parts 数组。
/// - user/system 的文本 → `input_text`
/// - assistant 的文本 → `output_text`
fn message_content_parts(m: &Message) -> Vec<Value> {
    let text = m.content.clone().unwrap_or_default();
    let part_type = if m.role == "assistant" {
        "output_text"
    } else {
        "input_text"
    };
    vec![json!({ "type": part_type, "text": text })]
}

/// 把 function_call_arguments 的 delta / done 片段追加到对应的 pending call。
fn append_call_args(
    pending: &mut [ToolCall],
    index_to_pos: &std::collections::HashMap<u64, usize>,
    v: &Value,
    fragment: &str,
) {
    // 优先用 output_index 定位；缺省回退到 item_id（极少数端点不带 output_index）。
    let pos = v
        .get("output_index")
        .and_then(|i| i.as_u64())
        .and_then(|idx| index_to_pos.get(&idx).copied());
    let pos = match pos {
        Some(p) => p,
        None => {
            // 按 item_id 找已登记的 call（output_item.added 时记的位置）
            let item_id = v.get("item_id").and_then(|i| i.as_str());
            pending
                .iter()
                .position(|c| item_id.map(|id| !id.is_empty() && c.id.ends_with(id)).unwrap_or(false))
                .unwrap_or(0)
        }
    };
    if pos < pending.len() {
        pending[pos].function.arguments.push_str(fragment);
    }
}

/// 解析一个 SSE 事件块（由 `event:` 行 + `data:` 行组成），返回 (type, data)。
/// 缺 event 行时（部分端点只发 data），type 返回空串，由调用方按 data 内容兜底。
fn parse_sse_event(block: &str) -> (String, String) {
    let mut event_type = String::new();
    let mut data_parts: Vec<&str> = Vec::new();
    for line in block.lines() {
        let line = line.trim_end_matches(['\r', '\n']);
        if let Some(rest) = line.strip_prefix("event:") {
            event_type = rest.trim().to_string();
        } else if let Some(rest) = line.strip_prefix("data:") {
            data_parts.push(rest.trim_start_matches(' '));
        }
    }
    (event_type, data_parts.join("\n"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> LlmConfig {
        LlmConfig {
            api_base: "https://example.invalid/v1".into(),
            api_key: "k".into(),
            model: "m".into(),
            prompt_cache_control: false,
            fallback: Arc::new(RwLock::new(None)),
        }
    }

    // ---- should_fallback：状态码分类（决定是否回退到 Chat Completions） ----

    #[test]
    fn fallback_status_triggers_on_not_found_bad_request_method_not_allowed() {
        // 这些状态码表示「Responses 接口/模型不存在或不支持」→ 应回退
        assert!(should_fallback(reqwest::StatusCode::NOT_FOUND)); // 404
        assert!(should_fallback(reqwest::StatusCode::BAD_REQUEST)); // 400
        assert!(should_fallback(reqwest::StatusCode::METHOD_NOT_ALLOWED)); // 405
    }

    #[test]
    fn fallback_status_does_not_trigger_on_auth_rate_limit_or_server_errors() {
        // 这些是鉴权 / 限流 / 服务端问题，回退到 Chat Completions 也救不了
        assert!(!should_fallback(reqwest::StatusCode::UNAUTHORIZED)); // 401
        assert!(!should_fallback(reqwest::StatusCode::FORBIDDEN)); // 403
        assert!(!should_fallback(reqwest::StatusCode::TOO_MANY_REQUESTS)); // 429
        assert!(!should_fallback(reqwest::StatusCode::INTERNAL_SERVER_ERROR)); // 500
        assert!(!should_fallback(reqwest::StatusCode::BAD_GATEWAY)); // 502
        assert!(!should_fallback(reqwest::StatusCode::SERVICE_UNAVAILABLE)); // 503
    }

    #[test]
    fn fallback_starts_none() {
        // 初始探测状态为 None（先试 Responses）
        let fb = new_fallback();
        assert!(fb.try_read().map(|g| g.is_none()).unwrap_or(false));
    }

    // ---- parse_sse_event ----

    #[test]
    fn sse_event_parses_type_and_data() {
        let block = "event: response.output_text.delta\ndata: {\"delta\":\"hi\"}\n\n";
        let (ty, data) = parse_sse_event(block);
        assert_eq!(ty, "response.output_text.delta");
        assert_eq!(data, "{\"delta\":\"hi\"}");
    }

    #[test]
    fn sse_event_strips_optional_space_after_colon() {
        // OpenAI 规范是 `data: `（带空格），这里验证空格被正确去除
        let block = "event:response.completed\ndata: {\"a\":1}\n\n";
        let (ty, data) = parse_sse_event(block);
        assert_eq!(ty, "response.completed");
        assert_eq!(data, "{\"a\":1}");
    }

    #[test]
    fn sse_event_multiline_data_joined() {
        let block = "event: x\ndata: line1\ndata:line2\n\n";
        let (_, data) = parse_sse_event(block);
        assert_eq!(data, "line1\nline2");
    }

    #[test]
    fn sse_event_missing_event_returns_empty_type() {
        let (_, data) = parse_sse_event("data: {}\n\n");
        assert_eq!(data, "{}");
    }

    // ---- extract_responses_text ----

    #[test]
    fn extract_text_single_message() {
        let v = json!({
            "output": [{
                "type": "message",
                "content": [{"type":"output_text","text":"你好"}]
            }]
        });
        assert_eq!(extract_responses_text(&v), "你好");
    }

    #[test]
    fn extract_text_skips_function_call_items() {
        let v = json!({
            "output": [
                {"type":"function_call","call_id":"c1","name":"f","arguments":"{}"},
                {"type":"message","content":[{"type":"output_text","text":"done"}]}
            ]
        });
        assert_eq!(extract_responses_text(&v), "done");
    }

    #[test]
    fn extract_text_empty_output() {
        assert_eq!(extract_responses_text(&json!({})), "");
        assert_eq!(extract_responses_text(&json!({"output": []})), "");
    }

    // ---- build_responses_body：核心是把统一 Message 历史翻译成 Responses input ----

    #[test]
    fn responses_body_system_user_become_messages() {
        let body = build_responses_body(
            &cfg(),
            &[
                Message::system("你是助手"),
                Message::user("你好"),
            ],
            &[],
            false,
        );
        let input = body.get("input").and_then(|i| i.as_array()).unwrap();
        assert_eq!(input.len(), 2);
        assert_eq!(input[0]["type"], "message");
        assert_eq!(input[0]["role"], "system");
        assert_eq!(input[0]["content"][0]["type"], "input_text");
        assert_eq!(input[0]["content"][0]["text"], "你是助手");
        assert_eq!(input[1]["role"], "user");
    }

    #[test]
    fn responses_body_assistant_tool_calls_become_function_call_items() {
        let assistant = Message {
            role: "assistant".into(),
            content: None,
            tool_calls: Some(vec![ToolCall {
                id: "call_1".into(),
                r#type: "function".into(),
                function: FunctionCall {
                    name: "get_weather".into(),
                    arguments: r#"{"city":"北京"}"#.into(),
                },
            }]),
            ..Default::default()
        };
        let body = build_responses_body(&cfg(), &[assistant], &[], false);
        let input = body.get("input").and_then(|i| i.as_array()).unwrap();
        assert_eq!(input.len(), 1);
        assert_eq!(input[0]["type"], "function_call");
        assert_eq!(input[0]["call_id"], "call_1");
        assert_eq!(input[0]["name"], "get_weather");
        assert_eq!(input[0]["arguments"], r#"{"city":"北京"}"#);
    }

    #[test]
    fn responses_body_tool_result_becomes_function_call_output() {
        let result = Message::tool_result("call_1", "get_weather", "晴".into());
        let body = build_responses_body(&cfg(), &[result], &[], false);
        let input = body.get("input").and_then(|i| i.as_array()).unwrap();
        assert_eq!(input.len(), 1);
        assert_eq!(input[0]["type"], "function_call_output");
        assert_eq!(input[0]["call_id"], "call_1");
        assert_eq!(input[0]["output"], "晴");
    }

    #[test]
    fn responses_body_assistant_text_uses_output_text_part() {
        let m = Message {
            role: "assistant".into(),
            content: Some("回复内容".into()),
            ..Default::default()
        };
        let body = build_responses_body(&cfg(), &[m], &[], false);
        let part_type = &body["input"][0]["content"][0]["type"];
        assert_eq!(part_type, "output_text");
    }

    #[test]
    fn responses_body_omits_empty_tools_and_sets_stream() {
        let body = build_responses_body(
            &cfg(),
            &[Message::user("hi")],
            &[],
            true,
        );
        assert!(body.get("tools").is_none());
        assert_eq!(body["stream"], true);
    }

    #[test]
    fn responses_body_includes_tools_when_non_empty() {
        let tool = json!({"type":"function","function":{"name":"f","parameters":{}}});
        let body = build_responses_body(
            &cfg(),
            &[Message::user("hi")],
            &[tool],
            false,
        );
        assert_eq!(body["tools"][0]["function"]["name"], "f");
    }
}
