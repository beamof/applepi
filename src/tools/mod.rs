pub mod echo;
pub mod fs;
// pub mod search; // 联网搜索工具示例：取消注释并在 default_tools 中注册

use anyhow::Result;
use async_trait::async_trait;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;

/// 工具接口：实现它即获得被 Agent 调用的能力。
#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    /// 参数的 JSON Schema（描述 parameters 对象）。
    fn parameters(&self) -> Value;
    async fn run(&self, args: Value) -> Result<String>;
}

pub type ToolMap = HashMap<String, Arc<dyn Tool>>;

/// 默认工具集。新增工具：实现 Tool + 在这里加一行。
pub fn default_tools() -> ToolMap {
    let tools: Vec<Arc<dyn Tool>> = vec![
        Arc::new(echo::Echo),
        Arc::new(fs::ReadFile),
        // Arc::new(search::Search::default()),
    ];
    tools
        .into_iter()
        .map(|t| (t.name().to_string(), t))
        .collect()
}
