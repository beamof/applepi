//! Cron 管理工具：让 agent 在对话中直接创建/列出/暂停/删除定时任务。
//!
//! 复用 CronStore（SQLite）+ watch 热重载机制。agent 调用本工具改动 DB 后，
//! 立即通知 scheduler 重载任务，无需重启进程。

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use serde_json::{json, Value};
use std::sync::Arc;
use tokio::sync::watch;

use crate::cron::store::CronStore;
use crate::tools::Tool;

/// 定时任务管理工具（stateful：持有 DB 句柄与重载信号）。
pub struct CronTool {
    store: Arc<CronStore>,
    reload: watch::Sender<()>,
}

impl CronTool {
    pub fn new(store: Arc<CronStore>, reload: watch::Sender<()>) -> Self {
        Self { store, reload }
    }
}

#[async_trait]
impl Tool for CronTool {
    fn name(&self) -> &str {
        "cron"
    }

    fn description(&self) -> &str {
        "管理定时任务（cron）。到点会自动触发并把结果推送到指定 Telegram chat_id。\
         支持动作：add（新增）/ list（列出）/ pause（暂停）/ resume（恢复）/ delete（删除）。\
         schedule 为北京时间 cron 表达式（分 时 日 月 周，如 \"0 9 * * *\" = 每天 9:00）。"
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["add", "list", "pause", "resume", "delete"],
                    "description": "要执行的动作"
                },
                "name": {
                    "type": "string",
                    "description": "任务名称（add 时必填）"
                },
                "schedule": {
                    "type": "string",
                    "description": "北京时间 cron 表达式，5 字段：分 时 日 月 周。如 \"0 9 * * *\"。add 时必填"
                },
                "chat_id": {
                    "type": "integer",
                    "description": "推送目标的 Telegram chat_id。add 时必填"
                },
                "prompt": {
                    "type": "string",
                    "description": "触发时发给 assistant 的提示词。add 时必填"
                },
                "id": {
                    "type": "integer",
                    "description": "任务 id（pause/resume/delete 时必填）"
                }
            },
            "required": ["action"]
        })
    }

    async fn run(&self, args: Value) -> Result<String> {
        let action = args
            .get("action")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("缺少 action 参数"))?;

        let result = match action {
            "add" => self.action_add(&args).await,
            "list" => self.action_list().await,
            "pause" => self.action_set(&args, false).await,
            "resume" => self.action_set(&args, true).await,
            "delete" => self.action_delete(&args).await,
            other => Err(anyhow!("未知 action: {other}")),
        };

        // 任何写操作都通知 scheduler 重载（失败也无妨，重载幂等）
        if matches!(action, "add" | "pause" | "resume" | "delete") {
            let _ = self.reload.send(());
        }

        result
    }
}

impl CronTool {
    async fn action_add(&self, args: &Value) -> Result<String> {
        let name = args
            .get("name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("缺少 name 参数"))?;
        let schedule = args
            .get("schedule")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("缺少 schedule 参数"))?;
        let chat_id = args
            .get("chat_id")
            .and_then(|v| v.as_i64())
            .ok_or_else(|| anyhow!("缺少 chat_id 参数"))?;
        let prompt = args
            .get("prompt")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("缺少 prompt 参数"))?;

        // 校验 cron 表达式
        if let Err(e) = croner::Cron::new(schedule).parse() {
            return Err(anyhow!("cron 表达式 '{schedule}' 非法: {e}"));
        }

        let id = self
            .store
            .insert(name, schedule, prompt, chat_id, true)?;
        Ok(format!(
            "已创建定时任务 [{id}] {name}：schedule={schedule}（北京时间） chat_id={chat_id}\n已立即生效。"
        ))
    }

    async fn action_list(&self) -> Result<String> {
        let jobs = self.store.list()?;
        if jobs.is_empty() {
            return Ok("当前没有任何定时任务。".into());
        }
        let mut out = String::from("定时任务列表：\n");
        for j in jobs {
            let status = if j.enabled { "启用" } else { "暂停" };
            out.push_str(&format!(
                "[{}] {}（{}）\n  schedule: {}\n  chat_id: {}\n  prompt: {}\n",
                j.id, j.name, status, j.schedule, j.chat_id, j.prompt
            ));
        }
        Ok(out)
    }

    async fn action_set(&self, args: &Value, enabled: bool) -> Result<String> {
        let id = args
            .get("id")
            .and_then(|v| v.as_i64())
            .ok_or_else(|| anyhow!("缺少 id 参数"))?;
        let job = self
            .store
            .get(id)?
            .ok_or_else(|| anyhow!("任务 {id} 不存在"))?;
        self.store.set_enabled(id, enabled)?;
        Ok(format!(
            "已{}任务 [{id}] {}",
            if enabled { "恢复" } else { "暂停" },
            job.name
        ))
    }

    async fn action_delete(&self, args: &Value) -> Result<String> {
        let id = args
            .get("id")
            .and_then(|v| v.as_i64())
            .ok_or_else(|| anyhow!("缺少 id 参数"))?;
        let job = self
            .store
            .get(id)?
            .ok_or_else(|| anyhow!("任务 {id} 不存在"))?;
        self.store.delete(id)?;
        Ok(format!("已删除任务 [{id}] {}", job.name))
    }
}
