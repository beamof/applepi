//! MCP（Model Context Protocol）接入：Streamable HTTP 传输。
//!
//! 启动时遍历配置里的 MCP 服务器，握手 + 拉取工具列表，
//! 把每个远端工具包装成本地 `Tool` 注入 Agent。
//! 单个 server 连接失败只警告，不阻断启动。

pub mod client;
pub mod tool;

use anyhow::Result;
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::config::McpServerConfig;
use crate::mcp::client::McpClient;
use crate::mcp::tool::McpTool;
use crate::tools::{Tool, ToolMap};

/// 加载所有已启用的 MCP 服务器工具，合并进 ToolMap。
///
/// - 单个 server 连接/握手/拉取失败：打印 `[MCP]` 警告并跳过。
/// - 同名工具后者覆盖前者（HashMap 语义），覆盖时打印警告。
pub async fn load_mcp_tools(servers: &[McpServerConfig]) -> Result<ToolMap> {
    let mut map: ToolMap = ToolMap::new();
    if servers.is_empty() {
        return Ok(map);
    }

    for s in servers.iter().filter(|s| s.enabled) {
        match connect_one(s).await {
            Ok((server_name, client, tools)) => {
                let shared = Arc::new(Mutex::new(client));
                eprintln!("[MCP] {server_name}: 发现 {} 个工具", tools.len());
                for t in tools {
                    let tool = McpTool::new(
                        t.name.clone(),
                        t.description.clone(),
                        t.input_schema.clone(),
                        shared.clone(),
                    );
                    let name = t.name.clone();
                    let arc: Arc<dyn Tool> = Arc::new(tool);
                    if map.contains_key(&name) {
                        eprintln!("[MCP] 警告: 工具 {name} 已存在，将被覆盖");
                    }
                    map.insert(name, arc);
                }
            }
            Err(e) => {
                eprintln!("[MCP] 警告: 连接服务器 {} 失败，已跳过: {e}", s.name);
            }
        }
    }
    Ok(map)
}

/// 握手 + 拉取工具列表。失败时返回 anyhow 错误，由调用方决定如何处理。
async fn connect_one(
    s: &McpServerConfig,
) -> Result<(String, McpClient, Vec<client::RemoteTool>)> {
    let client = McpClient::new(s.url.clone(), s.headers.clone())?;
    let (client, tools) = client.connect_and_list().await?;
    Ok((s.name.clone(), client, tools))
}
