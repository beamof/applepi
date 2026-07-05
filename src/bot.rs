//! Telegram Bot：纯 reqwest 实现，长轮询 getUpdates。
//! 一个用户一条 Agent 会话（按 chat_id 隔离）。
use anyhow::Result;
use reqwest::Client;
use serde::Deserialize;
use serde_json::json;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::agent::{Agent, AgentEvent};
use crate::config::Config;

#[derive(Deserialize)]
struct Update {
    update_id: i64,
    message: Option<Message>,
}

#[derive(Deserialize)]
struct Message {
    chat: Chat,
    text: Option<String>,
}

#[derive(Deserialize)]
struct Chat {
    id: i64,
}

const TG_API: &str = "https://api.telegram.org";

pub async fn run(cfg: Config, api_key: String) -> Result<()> {
    let token = if !cfg.telegram.bot_token.is_empty() {
        cfg.telegram.bot_token.clone()
    } else {
        std::env::var("TELEGRAM_BOT_TOKEN")?
    };
    if token.is_empty() {
        anyhow::bail!("未配置 TELEGRAM_BOT_TOKEN");
    }

    let http = Client::new();
    let base = format!("{TG_API}/bot{token}");

    // 每个 chat_id 一个独立 Agent（独立记忆上下文）
    let agents: Arc<Mutex<HashMap<i64, Agent>>> = Arc::new(Mutex::new(HashMap::new()));

    let llm_cfg = cfg.llm_config(api_key.clone());
    let embed_cfg = cfg.embeddings_config(api_key.clone());
    let persona = cfg.agent.persona.clone();
    // 合并默认工具 + MCP 远端工具（启动时一次性握手拉取，每个 chat 共享同一份）
    let mut tools = crate::tools::default_tools();
    tools.extend(crate::mcp::load_mcp_tools(&cfg.mcp_servers).await?);
    let top_k = cfg.memory.top_k_or(3);

    // 预热 long_term（共享 embedding 客户端即可，每个 agent 各持一份简化处理）
    let long_term = if cfg.memory.enabled {
        Some(crate::memory::long_term::LongTermMemory::open(
            &cfg.memory.db_path,
            embed_cfg.clone(),
        )?)
    } else {
        None
    };
    // 注意：long_term 内含 Mutex<Connection>，跨 agent 共享需 Arc。
    // 这里为简化，每个 agent 重新 open 一份（SQLite 多连接没问题）。
    drop(long_term);

    let mut offset: Option<i64> = None;
    tracing::info!("Telegram bot 已启动，开始长轮询...");

    loop {
        let mut params = json!({ "timeout": 30 });
        if let Some(o) = offset {
            params["offset"] = json!(o);
        }
        let resp = http
            .post(format!("{base}/getUpdates"))
            .json(&params)
            .send()
            .await?;

        if !resp.status().is_success() {
            tracing::warn!("getUpdates 状态码 {}", resp.status());
            tokio::time::sleep(std::time::Duration::from_secs(2)).await;
            continue;
        }

        let updates: Vec<Update> = resp
            .json::<serde_json::Value>()
            .await?
            .get("result")
            .cloned()
            .map(serde_json::from_value)
            .transpose()?
            .unwrap_or_default();

        for u in updates {
            offset = Some(u.update_id + 1);
            let Some(msg) = u.message else { continue };
            let Some(text) = msg.text else { continue };
            let chat_id = msg.chat.id;
            if text.trim().is_empty() {
                continue;
            }

            // 取/建该 chat 的 Agent
            {
                let mut map = agents.lock().await;
                map.entry(chat_id).or_insert_with(|| {
                    let lt = if cfg.memory.enabled {
                        crate::memory::long_term::LongTermMemory::open(
                            &cfg.memory.db_path,
                            embed_cfg.clone(),
                        )
                        .ok()
                    } else {
                        None
                    };
                    Agent::new(
                        llm_cfg.clone(),
                        persona.clone(),
                        tools.clone(),
                        lt,
                        top_k,
                    )
                });
            }

            // 先发一条占位消息，后续逐段编辑更新（流式体验）
            let placeholder = http
                .post(format!("{base}/sendMessage"))
                .json(&json!({ "chat_id": chat_id, "text": "…" }))
                .send()
                .await?
                .json::<serde_json::Value>()
                .await?;
            let msg_id = placeholder
                .get("result")
                .and_then(|r| r.get("message_id"))
                .and_then(|v| v.as_i64());

            let result: Result<String> = async {
                let mut map = agents.lock().await;
                let agent = map.get_mut(&chat_id).unwrap();
                let events = agent.chat_stream(&text).await?;
                let mut buf = String::new();
                let mut last_len = 0;
                for ev in events {
                    match ev {
                        AgentEvent::Text(t) => {
                            buf.push_str(&t);
                            // 节流：每攒够 ~120 字符或完成时编辑一次
                            if buf.len().saturating_sub(last_len) >= 120 {
                                last_len = buf.len();
                                edit_text(&http, &base, chat_id, msg_id, &buf).await;
                            }
                        }
                        AgentEvent::Final(t) => {
                            buf = t;
                        }
                        _ => {}
                    }
                }
                Ok(buf)
            }
            .await;

            match result {
                Ok(reply) => {
                    edit_text(&http, &base, chat_id, msg_id, &reply).await;
                }
                Err(e) => {
                    let _ = edit_text(&http, &base, chat_id, msg_id, &format!("[错误] {e}")).await;
                }
            }
        }
    }
}

async fn edit_text(http: &Client, base: &str, chat_id: i64, msg_id: Option<i64>, text: &str) {
    let Some(msg_id) = msg_id else { return };
    // Telegram 文本上限 4096，截断保护
    let safe: String = text.chars().take(4000).collect();
    let _ = http
        .post(format!("{base}/editMessageText"))
        .json(&json!({ "chat_id": chat_id, "message_id": msg_id, "text": safe }))
        .send()
        .await;
}
