//! 长期记忆：基于 SQLite FTS5 全文检索（BM25 排序）。
//!
//! 设计权衡：
//! - 不用 embedding 模型（规避 ort 静态链接坑、模型下载、API 依赖）。
//! - 直接用 SQLite 内置的 FTS5 虚拟表 + BM25 评分，零外部依赖。
//! - rusqlite 的 `bundled` feature 默认启用 `SQLITE_ENABLE_FTS5`，无需额外编译。
//!
//! 中文分词：FTS5 内置 `unicode61` 分词器**不会**按字切分中文——它把连续汉字
//! 当作一个 token，导致无法按字检索。解决：写入和查询前都把字符串按字符切分
//! （中间插空格），这样 unicode61 就会把每个字符当作独立 token 建索引。
//! 验证："苹果派很好吃" 存为 "苹 果 派 很 好 吃"，query "苹果" 切为 "苹 果"，
//! FTS5 隐式 AND 命中所有含这两字的记录，BM25 评分排序。
//!
//! Schema：FTS5 虚拟表含两列：
//! - `text_indexed`：按字切分后的文本，用于索引/检索
//! - `text_raw`（UNINDEXED）：原始文本，查询时返回给 LLM，避免看到带空格的怪文本
//!
//! Query 安全：用户输入按字切分后已经是合法 FTS5 token（每个字独立、无特殊语义），
//! 不需要额外转义。

use anyhow::Result;
use rusqlite::{params, Connection};
use std::sync::Mutex;

/// Schema 版本号：写在 SQLite 的 `PRAGMA user_version`。
/// - 0: 旧版（远程云端 embedding，1536 维向量）
/// - 2: 本地 embedding（fastembed/ONNX 向量）
/// - 3: FTS5 全文检索（当前版本，无向量）
///
/// 升级到 3 时会丢弃旧向量数据，只保留 text 内容迁移到 FTS 表。
const SCHEMA_VERSION: u32 = 3;

/// 长期记忆库：FTS5 虚拟表存文本，BM25 评分检索。
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
        // 提升 SQLite 吞吐：WAL + 更大的 cache。FTS5 写入是批量场景，值得调。
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "synchronous", "NORMAL")?;
        conn.execute_batch(
            // FTS5 虚拟表：
            // - text_indexed：按字切分后的文本，参与索引（检索依据）
            // - text_raw UNINDEXED：原始文本，仅存储不索引（返回给调用方）
            // - created_at UNINDEXED：时间戳，仅存储
            // tokenize='unicode61 remove_diacritics 2'：默认分词器。
            "CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                text_indexed,
                text_raw UNINDEXED,
                created_at UNINDEXED,
                tokenize = 'unicode61 remove_diacritics 2'
            );",
        )?;
        migrate(&conn)?;
        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    /// 存一条记忆。text 会按字切分后建索引，原始文本另存供返回。
    pub async fn remember(&self, text: &str) -> Result<()> {
        let raw = text.to_string();
        let indexed = char_split(text);
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO memories_fts (text_indexed, text_raw, created_at)
             VALUES (?1, ?2, datetime('now'))",
            params![indexed, raw],
        )?;
        Ok(())
    }

    /// 检索 Top-K 相关记忆（BM25 评分升序，分数越低越相关）。返回原始文本。
    pub async fn recall(&self, query: &str, top_k: usize) -> Result<Vec<String>> {
        let fts_query = char_split(query);
        let conn = self.conn.lock().unwrap();
        if fts_query.trim().is_empty() {
            // 退化：无有效 token 时返回最近的 top_k 条（按 rowid 倒序）。
            let mut stmt = conn.prepare(
                "SELECT text_raw FROM memories_fts ORDER BY rowid DESC LIMIT ?1",
            )?;
            let rows: Vec<String> = stmt
                .query_map(params![top_k as i64], |row| row.get::<_, String>(0))?
                .filter_map(|r| r.ok())
                .collect();
            return Ok(rows);
        }

        // bm25() 返回相关性分数（越负越相关，FTS5 约定），所以 ASC 排序。
        let mut stmt = conn.prepare(
            "SELECT text_raw FROM memories_fts
             WHERE memories_fts MATCH ?1
             ORDER BY bm25(memories_fts) ASC
             LIMIT ?2",
        )?;
        let rows: Vec<String> = stmt
            .query_map(params![fts_query, top_k as i64], |row| {
                row.get::<_, String>(0)
            })?
            .filter_map(|r| r.ok())
            .collect();
        Ok(rows)
    }
}

/// 按字切分字符串：每个字符间插空格，让 unicode61 分词器把每个字符当独立 token。
///
/// 同时过滤掉 FTS5 会解析为语法字符的符号（`"`、`*`、`(`、`)` 等）和空白/控制符，
/// 这些字符没有检索意义且会破坏 MATCH 表达式。
fn char_split(s: &str) -> String {
    s.chars()
        .filter(|c| !c.is_whitespace() && !c.is_control())
        .filter(|c| !matches!(c, '"' | '*' | '(' | ')' | ':' | '^' | '-' | '+' | '|' ))
        .map(|c| c.to_string())
        .collect::<Vec<_>>()
        .join(" ")
}

/// Schema 迁移：根据 `PRAGMA user_version` 决定动作。
///
/// - v0/v2（向量时代）→ v3：旧 memories 表里的 text 字段迁移到 FTS 表，
///   向量字段丢弃。embedding 字段无法转 FTS，但文本内容保留。
fn migrate(conn: &Connection) -> Result<()> {
    let current: u32 = conn.query_row("PRAGMA user_version", [], |row| row.get(0))?;
    if current >= SCHEMA_VERSION {
        return Ok(());
    }

    // 旧版有 memories 表（向量 schema）。检查是否存在。
    let has_old_table: bool = conn
        .query_row(
            "SELECT COUNT(*) > 0 FROM sqlite_master
             WHERE type='table' AND name='memories'",
            [],
            |row| row.get::<_, i64>(0),
        )
        .map(|v| v != 0)
        .unwrap_or(false);

    if has_old_table {
        let count: u64 = conn.query_row("SELECT COUNT(*) FROM memories", [], |row| row.get(0))?;
        if count > 0 {
            tracing::info!(
                "迁移：从旧向量 schema (v{current}) 导入 {count} 条记忆到 FTS5（向量字段丢弃）"
            );
            // 把旧表的 text 字段按字切分后导入 FTS 索引列，原始文本存到 text_raw。
            // 用一个 SELECT + INSERT，char_split 在 Rust 侧做不了，所以先读出来再批量插。
            let mut stmt = conn.prepare("SELECT text, created_at FROM memories")?;
            let rows: Vec<(String, String)> = stmt
                .query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)))?
                .filter_map(|r| r.ok())
                .collect();
            drop(stmt);
            for (text, created_at) in rows {
                let indexed = char_split(&text);
                conn.execute(
                    "INSERT INTO memories_fts (text_indexed, text_raw, created_at)
                     VALUES (?1, ?2, ?3)",
                    params![indexed, text, created_at],
                )?;
            }
        }
        // 丢弃旧表（含 embedding 列，体积大）。
        conn.execute("DROP TABLE memories", [])?;
    }

    conn.execute_batch(&format!("PRAGMA user_version = {SCHEMA_VERSION}"))?;
    Ok(())
}

