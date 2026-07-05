//! Cron 任务的 SQLite 持久化层。
//!
//! Schema：cron_jobs(id, name, schedule, prompt, chat_id, enabled, created_at)
//! 不存 last_run（错过不补执行，无需记录）。
//! 与长期记忆库分库，避免互相干扰。

use anyhow::Result;
use rusqlite::{params, Connection};
use std::sync::Mutex;

/// DB 中的任务记录（带自增 id）。
#[derive(Debug, Clone)]
pub struct JobRecord {
    pub id: i64,
    pub name: String,
    pub schedule: String,
    pub prompt: String,
    pub chat_id: i64,
    pub enabled: bool,
}

pub struct CronStore {
    conn: Mutex<Connection>,
}

impl CronStore {
    pub fn open(db_path: &str) -> Result<Self> {
        if let Some(parent) = std::path::Path::new(db_path).parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).ok();
            }
        }
        let conn = Connection::open(db_path)?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS cron_jobs (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                name       TEXT NOT NULL,
                schedule   TEXT NOT NULL,
                prompt     TEXT NOT NULL,
                chat_id    INTEGER NOT NULL,
                enabled    INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );",
        )?;
        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    /// 列出所有任务。
    pub fn list(&self) -> Result<Vec<JobRecord>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt =
            conn.prepare("SELECT id, name, schedule, prompt, chat_id, enabled FROM cron_jobs")?;
        let rows = stmt.query_map([], |row| {
            Ok(JobRecord {
                id: row.get(0)?,
                name: row.get(1)?,
                schedule: row.get(2)?,
                prompt: row.get(3)?,
                chat_id: row.get(4)?,
                enabled: row.get::<_, i64>(5)? != 0,
            })
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    /// 仅列出启用的任务（scheduler 用）。
    pub fn list_enabled(&self) -> Result<Vec<JobRecord>> {
        Ok(self.list()?.into_iter().filter(|j| j.enabled).collect())
    }

    /// 插入任务，返回新 id。
    pub fn insert(
        &self,
        name: &str,
        schedule: &str,
        prompt: &str,
        chat_id: i64,
        enabled: bool,
    ) -> Result<i64> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO cron_jobs (name, schedule, prompt, chat_id, enabled) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![name, schedule, prompt, chat_id, enabled as i64],
        )?;
        Ok(conn.last_insert_rowid())
    }

    /// 按 name 查询（种子去重用）。
    pub fn find_by_name(&self, name: &str) -> Result<Option<JobRecord>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, name, schedule, prompt, chat_id, enabled FROM cron_jobs WHERE name = ?1",
        )?;
        let mut rows = stmt.query_map(params![name], |row| {
            Ok(JobRecord {
                id: row.get(0)?,
                name: row.get(1)?,
                schedule: row.get(2)?,
                prompt: row.get(3)?,
                chat_id: row.get(4)?,
                enabled: row.get::<_, i64>(5)? != 0,
            })
        })?;
        match rows.next() {
            Some(r) => Ok(Some(r?)),
            None => Ok(None),
        }
    }

    pub fn delete(&self, id: i64) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute("DELETE FROM cron_jobs WHERE id = ?1", params![id])?;
        Ok(())
    }

    pub fn set_enabled(&self, id: i64, enabled: bool) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "UPDATE cron_jobs SET enabled = ?1 WHERE id = ?2",
            params![enabled as i64, id],
        )?;
        Ok(())
    }

    /// 按 id 查询单条（命令回显用）。
    pub fn get(&self, id: i64) -> Result<Option<JobRecord>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, name, schedule, prompt, chat_id, enabled FROM cron_jobs WHERE id = ?1",
        )?;
        let mut rows = stmt.query_map(params![id], |row| {
            Ok(JobRecord {
                id: row.get(0)?,
                name: row.get(1)?,
                schedule: row.get(2)?,
                prompt: row.get(3)?,
                chat_id: row.get(4)?,
                enabled: row.get::<_, i64>(5)? != 0,
            })
        })?;
        match rows.next() {
            Some(r) => Ok(Some(r?)),
            None => Ok(None),
        }
    }
}
