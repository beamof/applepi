//! 记忆工具：写入长期记忆。
//!
//! 长期记忆库（`LongTermMemory`）基于 SQLite FTS5 全文检索，跨会话保留。
//! 本工具让 agent 可以**主动**把需要长期记住的信息写入记忆库——例如用户
//! 偏好、重要事实、约定、决策结论等。检索是自动的（每轮按当前输入召回），
//! 这里只暴露写入能力。
//!
//! 设计：
//! - 持有 `Arc<LongTermMemory>`（与 Agent 共享同一实例），写入即对所有
//!   actor / cron job 立刻可见。
//! - 记忆库未启用（`memory.enabled = false`）时该工具不注入，避免 agent
//!   调用一个写不进任何地方的空操作。

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use serde_json::{json, Value};
use std::sync::Arc;

use crate::memory::long_term::LongTermMemory;
use crate::tools::Tool;

/// 写入长期记忆。
pub struct WriteMemoryTool {
    mem: Arc<LongTermMemory>,
}

impl WriteMemoryTool {
    pub fn new(mem: Arc<LongTermMemory>) -> Self {
        Self { mem }
    }
}

#[async_trait]
impl Tool for WriteMemoryTool {
    fn name(&self) -> &str {
        "write_memory"
    }

    fn description(&self) -> &str {
        "把一条信息写入长期记忆库（跨会话保留，按相关性自动召回）。\
         用于记住用户的偏好、重要事实、约定、任务结论等需要长期记住的内容。\
         不要写入琐碎或一次性的对话信息。\
         内容建议是一句完整、可独立理解的话（检索时按字面召回）。"
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "要记住的内容（一句完整、可独立理解的话）"
                }
            },
            "required": ["content"]
        })
    }

    async fn run(&self, args: Value) -> Result<String> {
        let content = args
            .get("content")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("缺少 content 参数"))?;
        let content = content.trim();
        if content.is_empty() {
            return Err(anyhow!("content 不能为空"));
        }
        // 软上限：避免 agent 写入超大段落污染记忆库与检索质量。
        const MAX_CHARS: usize = 2000;
        let content: String = content.chars().take(MAX_CHARS).collect();

        self.mem.remember(&content).await?;
        Ok(format!("已写入长期记忆（{} 字符）", content.chars().count()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 端到端：写记忆工具 → 召回，验证工具与共享 Arc 协作正常。
    /// 共享语义：同一 Arc 给工具和 recall，写入后立即对持有同份 Arc 的调用方可见。
    #[tokio::test]
    async fn write_memory_then_recall() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!(
            "applepi_mem_test_{}.db",
            std::process::id()
        ));
        // 预清理（前次失败可能残留）
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(format!("{}-wal", path.display()));
        let _ = std::fs::remove_file(format!("{}-shm", path.display()));

        let mem = Arc::new(LongTermMemory::open(path.to_str().unwrap()).unwrap());
        let tool = WriteMemoryTool::new(mem.clone());

        // 写一条记忆
        let result = tool
            .run(json!({ "content": "用户喜欢用 Rust 写后端服务" }))
            .await
            .unwrap();
        assert!(result.contains("已写入长期记忆"));

        // 通过同一 Arc 召回（模拟 Agent 持有共享实例做检索）
        let hits = mem.recall("Rust 后端", 5).await.unwrap();
        assert!(hits.iter().any(|h| h.contains("Rust")));

        // 校验：空 content 报错
        let err = tool.run(json!({ "content": "   " })).await;
        assert!(err.is_err());

        // 清理
        drop(tool);
        drop(mem);
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(format!("{}-wal", path.display()));
        let _ = std::fs::remove_file(format!("{}-shm", path.display()));
    }
}
