//! MCP 工具适配器：把远端工具包装成本地 Tool trait。

use anyhow::Result;
use async_trait::async_trait;
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::mcp::client::McpClient;
use crate::tools::Tool;

/// 一个远端 MCP 工具的本地包装。
///
/// 同一 server 的多个 McpTool 共享同一个 `Arc<Mutex<McpClient>>`，
/// 避免重复握手；调用时加锁串行化（与 agent 主循环的串行 dispatch 语义一致）。
pub struct McpTool {
    name: String,
    description: String,
    schema: Value,
    client: Arc<Mutex<McpClient>>,
}

impl McpTool {
    pub fn new(
        name: String,
        description: String,
        schema: Value,
        client: Arc<Mutex<McpClient>>,
    ) -> Self {
        Self {
            name,
            description,
            schema,
            client,
        }
    }
}

#[async_trait]
impl Tool for McpTool {
    fn name(&self) -> &str {
        &self.name
    }
    fn description(&self) -> &str {
        &self.description
    }
    /// 裸 JSON Schema（build_body 会包 function 外壳）。
    fn parameters(&self) -> Value {
        self.schema.clone()
    }
    async fn run(&self, args: Value) -> Result<String> {
        let mut client = self.client.lock().await;
        client.call_tool(&self.name, args).await
    }
}
