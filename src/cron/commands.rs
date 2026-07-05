//! `/cron` Telegram 命令解析与处理。
//!
//! 命令格式：
//!   /cron list
//!   /cron add <name> "<cron>" <chat_id> <prompt...>
//!   /cron del <id>
//!   /cron pause <id>
//!   /cron resume <id>
//!   /cron help
//!
//! cron 表达式按北京时间解释。<cron> 用双引号包裹（含空格/星号）。
//! 处理完成后通过 reload_tx 发送重载信号，scheduler 会热更新。

use std::sync::Arc;
use tokio::sync::watch;

use super::store::CronStore;

const HELP: &str = "\
/cron 命令用法：
  /cron list
      列出所有定时任务
  /cron add <name> \"<cron 表达式>\" <chat_id> <prompt...>
      新增任务（cron 按北京时间，例如 \"0 9 * * *\" = 每天 9:00）
      示例: /cron add 早报 \"0 9 * * *\" 123456 给出今日待办
  /cron del <id>
      删除任务
  /cron pause <id>
      暂停任务
  /cron resume <id>
      恢复任务
  /cron help
      显示本帮助";

/// 处理 `/cron ...` 命令，返回要回复给用户的文本。
pub async fn handle(
    text: &str,
    store: &Arc<CronStore>,
    reload_tx: &watch::Sender<()>,
) -> String {
    // 去掉 "/cron" 前缀及多余空白
    let rest = text.strip_prefix("/cron").unwrap_or(text).trim();
    let reply = dispatch(rest, store).await;
    // 任何命令都通知重载（即使失败也无妨，重载是幂等的）
    let _ = reload_tx.send(());
    reply
}

async fn dispatch(rest: &str, store: &Arc<CronStore>) -> String {
    if rest.is_empty() {
        return HELP.into();
    }
    let (cmd, args) = match rest.split_once(char::is_whitespace) {
        Some((c, a)) => (c.trim(), a.trim()),
        None => (rest.trim(), ""),
    };
    match cmd {
        "list" => cmd_list(store).await,
        "add" => cmd_add(args, store).await,
        "del" => cmd_del(args, store).await,
        "pause" => cmd_set_enabled(args, store, false).await,
        "resume" => cmd_set_enabled(args, store, true).await,
        "help" => HELP.into(),
        other => format!("未知子命令 '{other}'。\n\n{HELP}"),
    }
}

async fn cmd_list(store: &Arc<CronStore>) -> String {
    match store.list() {
        Ok(jobs) if jobs.is_empty() => "当前没有任何定时任务。".into(),
        Ok(jobs) => {
            let mut out = String::from("定时任务列表：\n");
            for j in jobs {
                let status = if j.enabled { "✅" } else { "⏸️" };
                out.push_str(&format!(
                    "{} [{}] {}\n    schedule: {}\n    chat_id: {}\n    prompt: {}\n",
                    status, j.id, j.name, j.schedule, j.chat_id, j.prompt
                ));
            }
            out
        }
        Err(e) => format!("读取失败: {e}"),
    }
}

/// 解析 `add` 的参数：<name> "<cron>" <chat_id> <prompt...>
async fn cmd_add(args: &str, store: &Arc<CronStore>) -> String {
    // name（第一个 token）
    let Some((name, rest)) = args.split_once(char::is_whitespace) else {
        return "用法: /cron add <name> \"<cron>\" <chat_id> <prompt...>".into();
    };
    let name = name.trim();
    let rest = rest.trim();

    // cron（双引号包裹）
    let Some(rest) = rest.strip_prefix('"') else {
        return "cron 表达式需用双引号包裹，例如 \"0 9 * * *\"".into();
    };
    let Some((cron, rest)) = rest.split_once('"') else {
        return "cron 表达式的右引号缺失".into();
    };
    let rest = rest.trim_start();

    // chat_id（下一个 token）
    let Some((chat_str, prompt)) = rest.split_once(char::is_whitespace) else {
        return "缺少 chat_id 或 prompt".into();
    };
    let chat_id: i64 = match chat_str.trim().parse() {
        Ok(v) => v,
        Err(_) => return format!("chat_id 非法: {}", chat_str.trim()),
    };
    let prompt = prompt.trim();
    if prompt.is_empty() {
        return "prompt 不能为空".into();
    }

    // 校验 cron 表达式
    if let Err(e) = croner::Cron::new(cron).parse() {
        return format!("cron 表达式 '{}' 非法: {e}", cron);
    }

    match store.insert(name, cron, prompt, chat_id, true) {
        Ok(id) => format!("✅ 已添加任务 [{id}] {name}（下次重载后生效）"),
        Err(e) => format!("添加失败: {e}"),
    }
}

async fn cmd_del(args: &str, store: &Arc<CronStore>) -> String {
    let id: i64 = match args.trim().parse() {
        Ok(v) => v,
        Err(_) => return "用法: /cron del <id>".into(),
    };
    match store.get(id) {
        Ok(Some(j)) => match store.delete(id) {
            Ok(()) => format!("✅ 已删除任务 [{id}] {}", j.name),
            Err(e) => format!("删除失败: {e}"),
        },
        Ok(None) => format!("任务 {id} 不存在"),
        Err(e) => format!("查询失败: {e}"),
    }
}

async fn cmd_set_enabled(args: &str, store: &Arc<CronStore>, enabled: bool) -> String {
    let id: i64 = match args.trim().parse() {
        Ok(v) => v,
        Err(_) => {
            return format!(
                "用法: /cron {} <id>",
                if enabled { "resume" } else { "pause" }
            )
        }
    };
    match store.get(id) {
        Ok(Some(j)) => match store.set_enabled(id, enabled) {
            Ok(()) => format!(
                "{} 任务 [{id}] {}",
                if enabled { "▶️ 已恢复" } else { "⏸️ 已暂停" },
                j.name
            ),
            Err(e) => format!("操作失败: {e}"),
        },
        Ok(None) => format!("任务 {id} 不存在"),
        Err(e) => format!("查询失败: {e}"),
    }
}
