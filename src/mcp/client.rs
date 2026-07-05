//! MCP Streamable HTTP 客户端。
//!
//! 协议（2025-06-18 规范）要点：
//! - 单 endpoint，客户端 POST JSON-RPC 请求，带 `Accept: application/json, text/event-stream`。
//! - 服务器响应两种 Content-Type：
//!   - `application/json`  → 响应体即单个 JSON-RPC 消息
//!   - `text/event-stream` → SSE，每个 `data:` 行是一条 JSON-RPC 消息
//! - 响应头可能返回 `Mcp-Session-Id`，后续请求需带回。
//! - 握手：initialize → notifications/initialized → 才能调其他方法。
//! - tools/list 返回 `{tools:[{name, description, inputSchema}]}`。
//! - tools/call 入参 `{name, arguments}`，返回 `{content:[{type:"text",text:"..."}], isError}`。

use anyhow::{anyhow, Result};
use reqwest::Client;
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

/// MCP 协议版本
const PROTOCOL_VERSION: &str = "2025-06-18";

/// 远端工具的描述（来自 tools/list）。
#[derive(Debug, Clone)]
pub struct RemoteTool {
    pub name: String,
    pub description: String,
    /// 工具参数的 JSON Schema，原样作为本地 Tool::parameters()。
    pub input_schema: Value,
}

/// 单个 MCP 服务器（HTTP 传输）的客户端。
pub struct McpClient {
    url: String,
    headers: HashMap<String, String>,
    http: Client,
    /// 来自服务器响应头 `Mcp-Session-Id`，后续请求需带回。
    session_id: Option<String>,
    next_id: AtomicU64,
}

impl McpClient {
    pub fn new(url: String, headers: HashMap<String, String>) -> Result<Self> {
        Ok(Self {
            url,
            headers,
            http: Client::builder().build()?,
            session_id: None,
            next_id: AtomicU64::new(1),
        })
    }

    /// 发一个 JSON-RPC request（带 id，等响应）；解析其 `result` 字段。
    async fn request<R: for<'de> Deserialize<'de>>(
        &mut self,
        method: &str,
        params: Value,
    ) -> Result<R> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let body = json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        });

        let mut req = self
            .http
            .post(&self.url)
            .header("Accept", "application/json, text/event-stream")
            .header("Content-Type", "application/json")
            .json(&body);
        for (k, v) in &self.headers {
            req = req.header(k, v);
        }
        if let Some(sid) = &self.session_id {
            req = req.header("Mcp-Session-Id", sid);
        }

        let resp = req.send().await?;
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(anyhow!("MCP 请求失败 [{status}]: {text}"));
        }
        // 记录 session id（幂等，可能为 None）
        if let Some(sid) = resp.headers().get("mcp-session-id").and_then(|v| v.to_str().ok()) {
            self.session_id = Some(sid.to_string());
        }

        let ct = resp
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_lowercase();

        // 两种响应形态，统一抽取出第一条带 id 的 JSON-RPC 响应（跳过 notification）。
        let msg: Value = if ct.contains("text/event-stream") {
            self.read_sse_until_id(resp, id).await?
        } else {
            resp.json::<Value>().await?
        };

        if let Some(err) = msg.get("error") {
            return Err(anyhow!("MCP error: {err}"));
        }
        let result = msg
            .get("result")
            .ok_or_else(|| anyhow!("MCP 响应缺少 result 字段: {msg}"))?;
        Ok(serde_json::from_value(result.clone())?)
    }

    /// 发一个 JSON-RPC notification（无 id，不等响应）。
    async fn notify(&mut self, method: &str, params: Value) -> Result<()> {
        let body = json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        });
        let mut req = self
            .http
            .post(&self.url)
            .header("Accept", "application/json, text/event-stream")
            .header("Content-Type", "application/json")
            .json(&body);
        for (k, v) in &self.headers {
            req = req.header(k, v);
        }
        if let Some(sid) = &self.session_id {
            req = req.header("Mcp-Session-Id", sid);
        }
        let resp = req.send().await?;
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(anyhow!("MCP notify 失败 [{status}]: {text}"));
        }
        // notification 通常无响应体（或返回 202）；忽略 body。
        Ok(())
    }

    /// 从 SSE 流中读出第一条 `id` 与目标相符的 JSON-RPC 响应（跳过 notification）。
    async fn read_sse_until_id(&self, resp: reqwest::Response, target: u64) -> Result<Value> {
        use futures::StreamExt;
        let mut stream = resp.bytes_stream();
        let mut buf = String::new();
        let mut data_line = String::new();

        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            buf.push_str(std::str::from_utf8(&chunk).unwrap_or(""));
            // SSE 事件以空行分隔；逐行处理。
            while let Some(idx) = buf.find('\n') {
                let line: String = buf.drain(..=idx).collect();
                let line = line.trim_end_matches(['\r', '\n']);
                if line.is_empty() {
                    // 事件边界：尝试解析已累积的 data 行。
                    if !data_line.is_empty() {
                        if let Ok(v) = serde_json::from_str::<Value>(&data_line) {
                            // 只认带 id 的 response（跳过 notification）。
                            if v.get("id").and_then(|i| i.as_u64()) == Some(target) {
                                return Ok(v);
                            }
                        }
                        data_line.clear();
                    }
                    continue;
                }
                if let Some(data) = line.strip_prefix("data:") {
                    let data = data.strip_prefix(' ').unwrap_or(data);
                    if data_line.is_empty() {
                        data_line.push_str(data);
                    } else {
                        data_line.push('\n');
                        data_line.push_str(data);
                    }
                }
            }
        }
        // 流末尾再检查一次
        if !data_line.is_empty() {
            if let Ok(v) = serde_json::from_str::<Value>(&data_line) {
                if v.get("id").and_then(|i| i.as_u64()) == Some(target) {
                    return Ok(v);
                }
            }
        }
        Err(anyhow!("SSE 流结束，未找到 id={target} 的响应"))
    }

    /// 完成握手：initialize → notifications/initialized。
    pub async fn initialize(&mut self) -> Result<()> {
        let _result: Value = self
            .request(
                "initialize",
                json!({
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": { "name": "applepi", "version": "0.1.0" },
                }),
            )
            .await?;
        self.notify("notifications/initialized", json!({})).await?;
        Ok(())
    }

    /// 拉取服务器工具列表。
    pub async fn list_tools(&mut self) -> Result<Vec<RemoteTool>> {
        #[derive(Deserialize)]
        struct ListResult {
            #[serde(default)]
            tools: Vec<RawTool>,
        }
        #[derive(Deserialize)]
        struct RawTool {
            name: String,
            #[serde(default)]
            description: String,
            #[serde(default = "default_object")]
            #[serde(rename = "inputSchema")]
            input_schema: Value,
        }
        fn default_object() -> Value {
            json!({})
        }

        let res: ListResult = self.request("tools/list", json!({})).await?;
        Ok(res
            .tools
            .into_iter()
            .map(|t| RemoteTool {
                name: t.name,
                description: t.description,
                input_schema: t.input_schema,
            })
            .collect())
    }

    /// 调用一个远端工具，返回拼接后的文本结果。
    pub async fn call_tool(&mut self, name: &str, args: Value) -> Result<String> {
        #[derive(Deserialize)]
        struct CallResult {
            #[serde(default)]
            content: Vec<Content>,
            #[serde(default)]
            #[serde(rename = "isError")]
            is_error: bool,
        }
        #[derive(Deserialize)]
        struct Content {
            #[serde(rename = "type")]
            kind: String,
            #[serde(default)]
            text: String,
        }

        let res: CallResult = self
            .request("tools/call", json!({ "name": name, "arguments": args }))
            .await?;

        let mut parts: Vec<String> = Vec::new();
        for c in res.content {
            if c.kind == "text" {
                parts.push(c.text);
            }
        }
        let out = parts.join("\n");
        if res.is_error {
            Err(anyhow!("MCP 工具返回错误: {out}"))
        } else {
            Ok(out)
        }
    }

    /// 一体化连接：initialize + list_tools，返回 self 与工具列表。
    pub async fn connect_and_list(mut self) -> Result<(Self, Vec<RemoteTool>)> {
        self.initialize().await?;
        let tools = self.list_tools().await?;
        Ok((self, tools))
    }
}
