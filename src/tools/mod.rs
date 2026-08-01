pub mod cron;
pub mod echo;
pub mod fs;
pub mod memory;
pub mod shell;
pub mod skill;
// pub mod search; // 联网搜索工具示例：取消注释并在 default_tools 中注册

use anyhow::Result;
use async_trait::async_trait;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;

use crate::memory::long_term::LongTermMemory;

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
        Arc::new(skill::SkillCreateTool),
        Arc::new(skill::SkillUseTool),
        // Arc::new(search::Search::default()),
    ];
    tools
        .into_iter()
        .map(|t| (t.name().to_string(), t))
        .collect()
}

/// 写入长期记忆的工具：需要共享的长期记忆库实例。
/// `long_term` 为 None（记忆未启用）时不注入，避免 agent 调用空写。
pub fn write_memory_tool(long_term: Arc<LongTermMemory>) -> Arc<dyn Tool> {
    Arc::new(memory::WriteMemoryTool::new(long_term))
}
