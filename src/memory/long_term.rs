use anyhow::Result;
use rusqlite::{params, Connection};
use serde_json;
use std::sync::Mutex;

use super::embed::LocalEmbedder;

/// Schema 版本号：写在 SQLite 的 `PRAGMA user_version`。
/// - 0: 旧版（远程云端 embedding，1536 维）
/// - 2: 本地 embedding（bge-small-zh 等，维度由模型决定）
///
/// 升级到 2 时会清空 memories 表（远程向量维度与本地不兼容）。
const SCHEMA_VERSION: u32 = 2;

/// 长期记忆：用 SQLite 存文本+向量，cosine 相似度检索。
/// 设计权衡：避免引入额外向量库依赖，记忆条数在几千以内足够用。
///
/// 向量由本地 `LocalEmbedder` 生成（candle 纯 Rust 推理），无网络往返。
pub struct LongTermMemory {
    conn: Mutex<Connection>,
}

impl LongTermMemory {
    pub fn open(db_path: &str) -> Result<Self> {
        if let Some(parent) = std::path::Path::new(db_path).parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).ok();
            }
        }
        let conn = Connection::open(db_path)?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                embedding TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );",
        )?;
        migrate(&conn)?;
        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    /// 存一条记忆。embedding 来自进程级单例 `LocalEmbedder`。
    pub async fn remember(&self, text: &str) -> Result<()> {
        let emb = LocalEmbedder::global("", None)?.embed(text.to_string()).await?;
        let emb_json = serde_json::to_string(&emb)?;
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO memories (text, embedding) VALUES (?1, ?2)",
            params![text, emb_json],
        )?;
        Ok(())
    }

    /// 检索 Top-K 相关记忆。query embedding 来自本地推理，无网络往返。
    pub async fn recall(&self, query: &str, top_k: usize) -> Result<Vec<String>> {
        let q_emb = LocalEmbedder::global("", None)?.embed(query.to_string()).await?;
        let conn = self.conn.lock().unwrap();
        let mut stmt =
            conn.prepare("SELECT text, embedding FROM memories ORDER BY id DESC LIMIT 2000")?;
        let rows = stmt
            .query_map([], |row| {
                let text: String = row.get(0)?;
                let emb_json: String = row.get(1)?;
                let emb: Vec<f32> =
                    serde_json::from_str(&emb_json).unwrap_or_default();
                Ok((text, emb))
            })?
            .filter_map(|r| r.ok())
            .collect::<Vec<_>>();

        let mut scored: Vec<(f32, String)> = rows
            .into_iter()
            .map(|(text, emb)| (cosine(&q_emb, &emb), text))
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        Ok(scored.into_iter().take(top_k).map(|(_, t)| t).collect())
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

/// Schema 迁移：根据 `PRAGMA user_version` 决定动作。
///
/// v0 → v2：检测旧版云端 embedding（1536 维，与新本地模型维度不兼容），
/// 清空 memories 表后写入新版本号。
fn migrate(conn: &Connection) -> Result<()> {
    let current: u32 = conn.query_row("PRAGMA user_version", [], |row| row.get(0))?;
    if current >= SCHEMA_VERSION {
        return Ok(());
    }
    // 仅在当前为 v0 且表里有数据时清空。
    let count: u64 = conn.query_row("SELECT COUNT(*) FROM memories", [], |row| row.get(0))?;
    if count > 0 {
        tracing::warn!(
            "检测到旧版远程 embedding（{} 条），与新本地模型维度不兼容；清空 memories 表重建。",
            count
        );
        conn.execute("DELETE FROM memories", [])?;
    }
    conn.execute_batch(&format!("PRAGMA user_version = {SCHEMA_VERSION}"))?;
    Ok(())
}
