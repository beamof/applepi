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
use crate::tools::Tool;

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
    // chat_id → 正在等待用户确认是否续跑的占位消息 id。下一条消息为肯定词则续跑，
    // 否则视作新输入。
    let pending_continue: Arc<Mutex<HashMap<i64, i64>>> = Arc::new(Mutex::new(HashMap::new()));

    let llm_cfg = cfg.llm_config(api_key.clone());
    let embed_cfg = cfg.embeddings_config(api_key.clone());
    let mut persona = crate::config::load_persona("AGENTS.md")?;
    persona.push_str(&crate::config::load_skills_summary("skills"));

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

    // Cron scheduler：仅当启用时启动。watch 通道用于 agent 通过 cron 工具改动后通知重载。
    // 必须在 tools 构建前完成，以便把 store 注入 CronTool。
    let (reload_tx, reload_rx) = tokio::sync::watch::channel(());
    let cron_store: Option<Arc<crate::cron::store::CronStore>> = if cfg.cron.enabled {
        match crate::cron::store::CronStore::open(&cfg.cron.db_path) {
            Ok(s) => {
                let store = Arc::new(s);
                let n = cfg.cron.jobs.len();
                tracing::info!("Cron 已启用（{} 个种子任务待写入）", n);
                tokio::spawn(crate::cron::run(
                    cfg.clone(),
                    api_key.clone(),
                    http.clone(),
                    base.clone(),
                    reload_rx,
                ));
                Some(store)
            }
            Err(e) => {
                tracing::error!("Cron DB 打开失败，scheduler 未启动: {e}");
                None
            }
        }
    } else {
        None
    };

    // 合并默认工具 + MCP 远端工具 + Cron 管理工具（启动时一次性构建，每个 chat 共享同一份）
    let mut tools = crate::tools::default_tools();
    tools.extend(crate::mcp::load_mcp_tools(&cfg.mcp_servers).await?);
    if let Some(store) = &cron_store {
        let t = Arc::new(crate::tools::cron::CronTool::new(store.clone(), reload_tx.clone()));
        tools.insert(t.name().to_string(), t);
    }
    // Shell 工具（可选，受白名单/黑名单约束）
    if cfg.shell.enabled {
        let t = Arc::new(crate::tools::shell::ShellTool::new(&cfg.shell));
        tools.insert(t.name().to_string(), t);
    }
    let top_k = cfg.memory.top_k_or(3);

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

            // 先取走待续跑确认状态（若上一轮耗尽）
            let pending_msg_id = pending_continue.lock().await.remove(&chat_id);

            let result: Result<()> = async {
                let mut map = agents.lock().await;
                let agent = map.get_mut(&chat_id).unwrap();

                // 续跑确认分支：上一轮耗尽，用户回复肯定词则继续（沿用原占位消息）
                if let Some(msg_id) = pending_msg_id {
                    if is_affirmative(&text) {
                        let events = agent.continue_stream().await?;
                        let (buf, pending) =
                            render_events(&http, &base, chat_id, Some(msg_id), events).await;
                        if pending {
                            pending_continue.lock().await.insert(chat_id, msg_id);
                        } else {
                            edit_text(&http, &base, chat_id, Some(msg_id), &buf).await;
                        }
                        return Ok(());
                    }
                    // 非肯定词 → 放弃续跑，按新输入处理
                }

                // 正常新输入：先发一条占位消息，后续逐段编辑更新（流式体验）
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

                // 把当前 chat_id 注入上下文，供 cron 等工具使用（用户无需手动提供）
                let input = format!("[chat_id: {chat_id}]\n\n{text}");
                let events = agent.chat_stream(&input).await?;
                let (buf, pending) = render_events(&http, &base, chat_id, msg_id, events).await;
                if pending {
                    // 耗尽：挂起等待用户确认，不做最终编辑（消息已显示提问）
                    if let Some(id) = msg_id {
                        pending_continue.lock().await.insert(chat_id, id);
                    }
                } else {
                    edit_text(&http, &base, chat_id, msg_id, &buf).await;
                }
                Ok(())
            }
            .await;

            if let Err(e) = result {
                // 优先复用已知 msg_id 编辑，否则新发一条错误消息
                if let Some(msg_id) = pending_msg_id {
                    let _ =
                        edit_text(&http, &base, chat_id, Some(msg_id), &format!("[错误] {e}")).await;
                } else {
                    let _ = http
                        .post(format!("{base}/sendMessage"))
                        .json(&json!({ "chat_id": chat_id, "text": format!("[错误] {e}") }))
                        .send()
                        .await;
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

/// 把事件流渲染到指定占位消息（沿用节流编辑策略）。
/// 返回 `(最终文本, 是否以 ContinuePrompt 结束)`。
/// - 节流：每攒够 ~120 字符编辑一次。
/// - `Final` 覆盖 buf；`ContinuePrompt` 追加提示语并立即编辑一次，返回 `pending=true`。
async fn render_events(
    http: &Client,
    base: &str,
    chat_id: i64,
    msg_id: Option<i64>,
    events: Vec<AgentEvent>,
) -> (String, bool) {
    let mut buf = String::new();
    let mut last_len = 0;
    let mut pending = false;
    for ev in events {
        match ev {
            AgentEvent::Text(t) => {
                buf.push_str(&t);
                if buf.len().saturating_sub(last_len) >= 120 {
                    last_len = buf.len();
                    edit_text(http, base, chat_id, msg_id, &buf).await;
                }
            }
            AgentEvent::Final(t) => {
                buf = t;
            }
            AgentEvent::ContinuePrompt(note) => {
                if !buf.is_empty() {
                    buf.push_str("\n\n");
                }
                buf.push_str(&note);
                edit_text(http, base, chat_id, msg_id, &buf).await;
                pending = true;
            }
            _ => {}
        }
    }
    (buf, pending)
}

/// 判断用户回复是否为肯定（继续/继续吧/continue/yes/y/是/好的/1，忽略大小写与空白）。
fn is_affirmative(s: &str) -> bool {
    matches!(
        s.trim().to_ascii_lowercase().as_str(),
        "继续" | "继续吧" | "是" | "好" | "好的" | "ok" | "continue" | "yes" | "y" | "1"
    )
}
