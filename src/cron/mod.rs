//! Cron 定时任务调度（仅 bot 模式）。
//!
//! 设计：
//! - job 存 SQLite（`data/cron.db`），重启不丢失。
//! - 每个 enabled job 一个独立常驻任务，构造专用 Agent，不碰 bot 的 agents 锁。
//! - 重载机制：agent 通过 `cron` 工具改动 DB 后，经 watch 通道发信号；
//!   scheduler 收到后 abort 全部任务、重新拉起所有当前 enabled job。
//! - 时区：cron 表达式按北京时间（UTC+8）解释。用 FixedOffset::east(8*3600)
//!   作为 croner 的参考时区，find_next_occurrence 返回北京时间的触发时刻，
//!   转成 UTC 绝对时长后 sleep。

pub mod store;

use anyhow::Result;
use chrono::{DateTime, FixedOffset};
use croner::Cron;
use reqwest::Client;
use serde_json::json;
use std::sync::Arc;
use tokio::sync::watch;

use crate::agent::{Agent, AgentEvent};
use crate::config::Config;
use crate::memory::long_term::LongTermMemory;
use crate::tools::{Tool, ToolMap};

/// 北京时间（UTC+8）。
fn beijing_tz() -> FixedOffset {
    FixedOffset::east_opt(8 * 3600).unwrap()
}

/// 启动 scheduler。
///
/// - `reload_tx`：agent 通过 cron 工具改动 DB 后发信号触发重载（与 bot 侧共享同一通道）。
/// - `reload_rx`：收到信号就重载 DB（增删改 job 后 bot 侧会发信号）。
///
/// 该函数会一直运行（直到进程退出），应在 `tokio::spawn` 中调用。
pub async fn run(
    cfg: Config,
    api_key: String,
    http: Client,
    base: String,
    reload_tx: watch::Sender<()>,
    mut reload_rx: watch::Receiver<()>,
) -> Result<()> {
    if !cfg.cron.enabled {
        return Ok(());
    }

    let store = Arc::new(store::CronStore::open(&cfg.cron.db_path)?);

    // 启动时种子写入（按 name 去重，已存在不重复插入）
    seed_jobs(&store, &cfg).await;

    // 预构造 agent 所需的共享件
    let llm_cfg = cfg.llm_config(api_key.clone());
    let embed_cfg = cfg.embeddings_config(api_key.clone());
    let persona = crate::config::load_persona("AGENTS.md")?;
    // 工具集与 bot.rs 对齐：default + MCP + Cron 管理 + Shell（按开关）
    let mut tools = crate::tools::default_tools();
    tools.extend(crate::mcp::load_mcp_tools(&cfg.mcp_servers).await?);
    let t = Arc::new(crate::tools::cron::CronTool::new(store.clone(), reload_tx.clone()));
    tools.insert(t.name().to_string(), t);
    if cfg.shell.enabled {
        let t = Arc::new(crate::tools::shell::ShellTool::new(&cfg.shell));
        tools.insert(t.name().to_string(), t);
    }
    let tools = Arc::new(tools);
    let top_k = cfg.memory.top_k_or(3);
    let db_path = cfg.memory.db_path.clone();
    let memory_enabled = cfg.memory.enabled;

    tracing::info!("Cron scheduler 已启动");

    let mut handles: Vec<tokio::task::JoinHandle<()>> = Vec::new();

    loop {
        // 拉取当前 enabled job，全量重建任务
        let jobs = match store.list_enabled() {
            Ok(j) => j,
            Err(e) => {
                tracing::error!("[cron] 读取 DB 失败: {e}");
                // 等下一次重载再试
                let _ = reload_rx.changed().await;
                continue;
            }
        };

        // abort 全部旧任务
        for h in handles.drain(..) {
            h.abort();
        }

        tracing::info!("[cron] 加载 {} 个任务", jobs.len());
        for job in jobs {
            let ctx = Arc::new(JobCtx {
                llm_cfg: llm_cfg.clone(),
                embed_cfg: embed_cfg.clone(),
                persona: persona.clone(),
                tools: tools.clone(),
                top_k,
                db_path: db_path.clone(),
                memory_enabled,
                http: http.clone(),
                base: base.clone(),
            });
            let mut rx = reload_rx.clone();
            let handle = tokio::spawn(async move {
                run_job(job, ctx, &mut rx).await;
            });
            handles.push(handle);
        }

        // 等下一次重载信号
        if reload_rx.changed().await.is_err() {
            // 发送端关闭，退出
            break;
        }
    }

    // 清理
    for h in handles {
        h.abort();
    }
    Ok(())
}

/// 单个 job 的运行上下文（共享、只读）。
struct JobCtx {
    llm_cfg: crate::llm::LlmConfig,
    embed_cfg: crate::memory::long_term::EmbedConfig,
    persona: String,
    tools: Arc<ToolMap>,
    top_k: usize,
    db_path: String,
    memory_enabled: bool,
    http: Client,
    base: String,
}

/// 启动时种子写入：cfg.cron.jobs 中 name 不存在于 DB 的才插入。
async fn seed_jobs(store: &Arc<store::CronStore>, cfg: &Config) {
    for j in &cfg.cron.jobs {
        match store.find_by_name(&j.name) {
            Ok(Some(_)) => {} // 已存在，跳过
            Ok(None) => match store.insert(&j.name, &j.schedule, &j.prompt, j.chat_id, j.enabled) {
                Ok(id) => tracing::info!("[cron] 种子任务 '{}' 已写入 (id={id})", j.name),
                Err(e) => tracing::error!("[cron] 种子任务 '{}' 写入失败: {e}", j.name),
            },
            Err(e) => tracing::error!("[cron] 查询种子任务 '{}' 失败: {e}", j.name),
        }
    }
}

/// 单个 job 的常驻循环。
async fn run_job(job: store::JobRecord, ctx: Arc<JobCtx>, reload_rx: &mut watch::Receiver<()>) {
    let cron: Cron = match Cron::new(&job.schedule).parse() {
        Ok(c) => c,
        Err(e) => {
            tracing::error!("[cron:{}] 表达式 '{}' 非法: {e}", job.name, job.schedule);
            return;
        }
    };
    let tz = beijing_tz();

    loop {
        // 用北京时间的"现在"算下次触发（返回值也是北京时间）
        let now_bj = chrono::Local::now().with_timezone(&tz);
        let next_bj: DateTime<FixedOffset> = match cron.find_next_occurrence(&now_bj, false) {
            Ok(t) => t,
            Err(e) => {
                tracing::error!("[cron:{}] 计算下次触发失败: {e}", job.name);
                break;
            }
        };
        // 转成 UTC 绝对等待时长
        let next_utc = next_bj.with_timezone(&chrono::Utc);
        let sleep_dur = match (next_utc - chrono::Utc::now()).to_std() {
            Ok(d) => d,
            Err(_) => {
                // 已过（理论不该发生，inclusive=false），立即触发
                std::time::Duration::ZERO
            }
        };

        tracing::info!(
            "[cron:{}] 下次触发: {}（北京时间），等待 {:.0}s",
            job.name,
            next_bj.format("%Y-%m-%d %H:%M:%S"),
            sleep_dur.as_secs_f64()
        );

        // 等到触发点；若收到重载信号则退出（主循环会重新拉起）
        tokio::select! {
            _ = tokio::time::sleep(sleep_dur) => {}
            _ = reload_rx.changed() => {
                tracing::info!("[cron:{}] 收到重载信号，退出当前循环", job.name);
                return;
            }
        }

        // 触发：构造专用 Agent 执行 prompt 并推送
        if let Err(e) = trigger_job(&job, &ctx).await {
            tracing::error!("[cron:{}] 触发失败: {e}", job.name);
        }
    }
}

/// 触发一次 job：构造 Agent → chat_stream → 拼接最终文本 → 推送到 Telegram。
async fn trigger_job(job: &store::JobRecord, ctx: &JobCtx) -> Result<()> {
    let long_term = if ctx.memory_enabled {
        Some(LongTermMemory::open(&ctx.db_path, ctx.embed_cfg.clone())?)
    } else {
        None
    };
    let mut agent = Agent::new(
        ctx.llm_cfg.clone(),
        ctx.persona.clone(),
        ctx.tools.as_ref().clone(),
        long_term,
        ctx.top_k,
    );

    let events = agent.chat_stream(&job.prompt).await?;
    let mut reply = String::new();
    let mut exhausted = false;
    for ev in events {
        match ev {
            AgentEvent::Final(t) => {
                reply = t;
                break;
            }
            AgentEvent::ContinuePrompt(_) => {
                // 定时任务无人值守，无法询问用户，记为耗尽自动停止
                exhausted = true;
            }
            _ => {}
        }
    }
    if reply.is_empty() {
        reply = if exhausted {
            "（已达最大轮次上限，定时任务无人值守，自动停止）".into()
        } else {
            "（无输出）".into()
        };
    }
    // Telegram 文本上限 4096，截断保护
    let safe: String = reply.chars().take(4000).collect();
    let resp = ctx
        .http
        .post(format!("{}/sendMessage", ctx.base))
        .json(&json!({ "chat_id": job.chat_id, "text": safe }))
        .send()
        .await?;
    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        return Err(anyhow::anyhow!("Telegram 推送失败 [{status}]: {text}"));
    }
    tracing::info!("[cron:{}] 已推送到 chat_id={}", job.name, job.chat_id);
    Ok(())
}
