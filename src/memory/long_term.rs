use anyhow::{anyhow, Result};
use reqwest::Client;
use rusqlite::{params, Connection};
use serde::Deserialize;
use serde_json::json;
use std::sync::Mutex;

/// 长期记忆：用 SQLite 存文本+向量，cosine 相似度检索。
/// 设计权衡：避免引入额外向量库依赖，记忆条数在几千以内足够用。
pub struct LongTermMemory {
    conn: Mutex<Connection>,
    embed: EmbedConfig,
    http: Client,
}

#[derive(Clone)]
pub struct EmbedConfig {
    pub api_base: String,
    pub api_key: String,
    pub model: String,
}

#[derive(Deserialize)]
struct EmbedResp {
    data: Vec<EmbedItem>,
}

#[derive(Deserialize)]
struct EmbedItem {
    embedding: Vec<f32>,
}

impl LongTermMemory {
    pub fn open(db_path: &str, embed: EmbedConfig) -> Result<Self> {
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
        Ok(Self { conn: Mutex::new(conn), embed, http: Client::new() })
    }

    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let body = json!({
            "model": self.embed.model,
            "input": text,
        });
        let resp = self
            .http
            .post(format!("{}/embeddings", self.embed.api_base))
            .bearer_auth(&self.embed.api_key)
            .json(&body)
            .send()
            .await?;
        if !resp.status().is_success() {
            let status = resp.status();
            let t = resp.text().await.unwrap_or_default();
            return Err(anyhow!("embedding 请求失败 [{status}]: {t}"));
        }
        let parsed: EmbedResp = resp.json().await?;
        parsed
            .data
            .into_iter()
            .next()
            .map(|i| i.embedding)
            .ok_or_else(|| anyhow!("embedding 响应为空"))
    }

    /// 存一条记忆
    pub async fn remember(&self, text: &str) -> Result<()> {
        let emb = self.embed(text).await?;
        let emb_json = serde_json::to_string(&emb)?;
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO memories (text, embedding) VALUES (?1, ?2)",
            params![text, emb_json],
        )?;
        Ok(())
    }

    /// 检索 Top-K 相关记忆
    pub async fn recall(&self, query: &str, top_k: usize) -> Result<Vec<String>> {
        let q_emb = self.embed(query).await?;
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
