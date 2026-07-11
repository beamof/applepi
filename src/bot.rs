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
/// 单条消息处理超时：超时后向用户报错，避免占位消息无限停在「…」。
const CHAT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(180);

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
    // 这里为简化，每个 actor 重新 open 一份（SQLite 多连接没问题）。
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

    // 合并默认工具 + MCP 远端工具 + Cron 管理工具（启动时一次性构建，每个 actor 共享同一份）
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

    // 构建新 actor 所需的共享上下文（Clone 便宜：全是 Arc）
    let ctx = Arc::new(ChatCtx {
        llm_cfg,
        embed_cfg,
        persona,
        tools,
        cfg: cfg.clone(),
        top_k,
        http: http.clone(),
        base: base.clone(),
    });

    // 每个 chat_id 一个独立 actor task（拥有自己的 Agent，互不阻塞）。
    // 主循环只负责 getUpdates + 派发，绝不 await LLM 调用。
    let actors: Arc<Mutex<HashMap<i64, ChatHandle>>> =
        Arc::new(Mutex::new(HashMap::new()));

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

            // 斜杠命令在主循环直接处理（轻量、不涉及 LLM），/new 会重建 actor
            if let Some(rest) = text.trim().strip_prefix('/') {
                let cmd_name = rest.split_whitespace().next().unwrap_or("");
                match cmd_name {
                    "new" | "clear" => {
                        // 中止旧 actor（若有）并移除，下一条消息自然触发新建
                        let had = actors.lock().await.remove(&chat_id).is_some();
                        let reply = if had {
                            "✅ 已开启新会话，上下文已清空。"
                        } else {
                            "（当前本就是新会话，无需清空。）"
                        };
                        let _ = http
                            .post(format!("{base}/sendMessage"))
                            .json(&json!({ "chat_id": chat_id, "text": reply, "parse_mode": "HTML" }))
                            .send()
                            .await;
                    }
                    "help" | "start" => {
                        let help = "<b>命令</b>\n\
/new · /clear — 开启新会话，清空当前上下文\n\
/help — 显示本帮助\n\n\
直接发消息即可对话。长期记忆跨会话保留。";
                        let _ = http
                            .post(format!("{base}/sendMessage"))
                            .json(&json!({ "chat_id": chat_id, "text": help, "parse_mode": "HTML" }))
                            .send()
                            .await;
                    }
                    _ => {
                        let reply = format!("未知命令 /{cmd_name}。发送 /help 查看可用命令。");
                        let _ = http
                            .post(format!("{base}/sendMessage"))
                            .json(&json!({ "chat_id": chat_id, "text": reply }))
                            .send()
                            .await;
                    }
                }
                continue;
            }

            // 取/建该 chat 的 actor，把消息送进去（非阻塞：channel send 即返回）
            let handle = {
                let mut map = actors.lock().await;
                map.entry(chat_id)
                    .or_insert_with(|| ChatActor::spawn(chat_id, ctx.clone()))
                    .clone()
            };
            // channel 满则丢弃消息（actor 正忙；Telegram 侧用户会看到无响应，但不会冻结其他 chat）
            let _ = handle.tx.try_send(text);
        }
    }
}

// ---------- per-chat actor ----------

/// 构建新 actor 所需的共享上下文。所有字段 Clone 便宜（Arc 或已 Clone 的配置）。
struct ChatCtx {
    llm_cfg: crate::llm::LlmConfig,
    embed_cfg: crate::memory::long_term::EmbedConfig,
    persona: String,
    tools: crate::tools::ToolMap,
    cfg: Config,
    top_k: usize,
    http: Client,
    base: String,
}

/// 主循环持有的 actor 句柄。Clone = 共享同一个 actor。
#[derive(Clone)]
struct ChatHandle {
    tx: tokio::sync::mpsc::Sender<String>,
    _task: Arc<tokio::task::JoinHandle<()>>,
}

/// actor 内部状态：一个 Agent + 续跑/超时交互的挂起状态。
struct ChatActor {
    chat_id: i64,
    agent: Agent,
    http: Client,
    base: String,
    /// 续跑确认：上一轮耗尽，等用户肯定词继续
    pending_continue: Option<i64>,
    /// 超时询问：等用户回复 "1"(继续) / "2"(终止)
    pending_timeout: Option<i64>,
}

impl ChatActor {
    fn spawn(chat_id: i64, ctx: Arc<ChatCtx>) -> ChatHandle {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<String>(16);
        let lt = if ctx.cfg.memory.enabled {
            crate::memory::long_term::LongTermMemory::open(
                &ctx.cfg.memory.db_path,
                ctx.embed_cfg.clone(),
            )
            .ok()
        } else {
            None
        };
        let agent = Agent::new(
            ctx.llm_cfg.clone(),
            ctx.persona.clone(),
            ctx.tools.clone(),
            lt,
            ctx.top_k,
        );
        let mut actor = ChatActor {
            chat_id,
            agent,
            http: ctx.http.clone(),
            base: ctx.base.clone(),
            pending_continue: None,
            pending_timeout: None,
        };
        let task = tokio::spawn(async move {
            while let Some(text) = rx.recv().await {
                actor.handle_message(text).await;
            }
        });
        ChatHandle {
            tx,
            _task: Arc::new(task),
        }
    }

    /// 处理一条用户消息。串行调用（actor 单线程），所有交互状态都在 self 上。
    async fn handle_message(&mut self, text: String) {
        // 1) 超时询问待回应
        if let Some(msg_id) = self.pending_timeout.take() {
            match text.trim() {
                "1" => {
                    // 继续等待：continue_stream 接着已有历史跑，仍受超时保护
                    self.process(Some(msg_id), None).await;
                    return;
                }
                "2" => {
                    edit_text(&self.http, &self.base, self.chat_id, Some(msg_id), "🛑 已终止本次处理。").await;
                    return;
                }
                _ => {
                    // 非数字：放弃超时上下文，按新输入正常处理
                }
            }
        }

        // 2) 续跑确认待回应
        if let Some(msg_id) = self.pending_continue.take() {
            if is_affirmative(&text) {
                self.process(Some(msg_id), None).await;
                return;
            }
            // 非肯定词 → 放弃续跑，按新输入处理
        }

        // 3) 正常新输入：发占位消息后处理
        let msg_id = send_placeholder(&self.http, &self.base, self.chat_id).await;
        self.process(msg_id, Some(&text)).await;
    }

    /// 执行一次 agent 处理，带超时保护。
    /// - `new_input = Some(text)`：正常新对话（构造 `[chat_id:..]\n\n{text}` 喂给 agent）
    /// - `new_input = None`：续跑（continue_stream，沿用已有历史，用于超时选"继续"和续跑确认）
    /// 超时时编辑占位消息为询问，并设置 `pending_timeout` 状态。
    async fn process(&mut self, msg_id: Option<i64>, new_input: Option<&str>) {
        let chat_id = self.chat_id;
        let started = std::time::Instant::now();
        tracing::info!("chat {} 开始处理, msg_id={:?}", chat_id, msg_id);
        let result = tokio::time::timeout(
            CHAT_TIMEOUT,
            self.process_inner(msg_id, chat_id, new_input),
        )
        .await;
        tracing::info!("chat {} 处理结束, 耗时 {:?}", chat_id, started.elapsed());

        match result {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                report_error(&self.http, &self.base, chat_id, msg_id, &e.to_string()).await;
            }
            Err(_elapsed) => {
                // 超时：询问用户继续/终止，设置挂起状态
                tracing::warn!("chat {} 处理超时，msg_id={:?}", chat_id, msg_id);
                self.pending_timeout = msg_id;
                let ask = "⏰ 处理已超过 3 分钟仍未完成。\n\n回复数字选择：\n\
1️⃣ 继续等待\n\
2️⃣ 终止本次处理";
                // msg_id 为 None（占位消息没发成功）时 edit_text 会静默返回；
                // 改用 sendMessage 保底，确保超时提示一定发得出。
                if let Some(id) = msg_id {
                    edit_text(&self.http, &self.base, chat_id, Some(id), ask).await;
                } else {
                    let _ = self
                        .http
                        .post(format!("{}/sendMessage", self.base))
                        .json(&json!({ "chat_id": chat_id, "text": ask }))
                        .send()
                        .await;
                }
            }
        }
    }

    /// 实际的 agent 调用 + 渲染。被 `process` 包裹超时。
    async fn process_inner(
        &mut self,
        msg_id: Option<i64>,
        chat_id: i64,
        new_input: Option<&str>,
    ) -> Result<(), anyhow::Error> {
        let events = if let Some(text) = new_input {
            let input = format!("[chat_id: {chat_id}]\n\n{text}");
            self.agent.chat_stream(&input).await?
        } else {
            self.agent.continue_stream().await?
        };
        let (buf, pending) = render_events(&self.http, &self.base, chat_id, msg_id, events).await;
        if pending {
            self.pending_continue = msg_id;
        } else {
            edit_text(&self.http, &self.base, chat_id, msg_id, &buf).await;
        }
        Ok(())
    }
}

/// 发送占位消息，返回 message_id。
async fn send_placeholder(http: &Client, base: &str, chat_id: i64) -> Option<i64> {
    let resp = http
        .post(format!("{base}/sendMessage"))
        .json(&json!({ "chat_id": chat_id, "text": "…", "parse_mode": "HTML" }))
        .send()
        .await
        .ok()?;
    let v: serde_json::Value = resp.json().await.ok()?;
    v.get("result")
        .and_then(|r| r.get("message_id"))
        .and_then(|m| m.as_i64())
}

/// 把常见 markdown 转成 Telegram 支持的 HTML 子集。
///
/// 覆盖的语法：```代码块```、`行内代码`、**粗体**、*斜体*、~~删除线~~、
/// # 标题、-/* 列表项、[文本](url) 链接。其余原样保留。
///
/// 设计保守：先对整段做 HTML 转义（& < >），再处理 markdown 标记，
/// 替换进去的标签不会被二次转义。代码块内容只整体转义一次，不解析行内标记。
/// 无法识别的语法原样输出文本，宁可少渲染也不产生非法 HTML 导致发送失败。
fn md_to_tg_html(md: &str) -> String {
    // 1. 整体 HTML 转义
    let escaped = md
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;");

    // 2. 按围栏代码块分段处理（``` 或 ~~~）
    let mut out = String::with_capacity(escaped.len());
    let mut in_code = false;
    let mut fence = String::new();
    let mut code_buf = String::new();

    for line in escaped.split_inclusive('\n') {
        let trimmed = line.trim_end_matches(['\n', '\r']);
        let opens = trimmed.starts_with("```") || trimmed.starts_with("~~~");
        let closes = in_code && !fence.is_empty() && trimmed.starts_with(fence.as_str());

        if !in_code {
            if opens {
                in_code = true;
                fence = trimmed.chars().take(3).collect();
                code_buf.clear();
                // 末尾换行属于代码块外，不复制
            } else {
                out.push_str(&convert_inline_line(trimmed));
                if line.ends_with('\n') {
                    out.push('\n');
                }
            }
        } else if closes {
            // 闭合代码块
            in_code = false;
            fence.clear();
            out.push_str("<pre><code>");
            out.push_str(code_buf.trim_end_matches(['\n', '\r']));
            out.push_str("</code></pre>");
            if line.ends_with('\n') {
                out.push('\n');
            }
            code_buf.clear();
        } else {
            // 代码块内部：累计（已转义过，原样）
            code_buf.push_str(line);
        }
    }
    // 未闭合的代码块兜底：原样包起来输出，避免吞内容
    if in_code {
        out.push_str("<pre><code>");
        out.push_str(code_buf.trim_end_matches(['\n', '\r']));
        out.push_str("</code></pre>");
    }
    out
}

/// 转换一行非代码内容：标题/列表标记 + 行内标记。
fn convert_inline_line(line: &str) -> String {
    // 行首结构性标记（仅处理前缀，不碰行内可能出现的 * 等）
    let mut s = line.to_string();

    // ATX 标题：# ~ ###### 开头 → <b>（Telegram 无 h1..h6，统一粗体）
    if let Some(rest) = strip_atx_header(&s) {
        s = format!("<b>{rest}</b>");
    } else if s.starts_with("- ") || s.starts_with("* ") {
        // 无序列表项：保留 `- ` 前缀，内容转行内
        let (mark, rest) = s.split_at(2);
        s = format!("{mark}{}", convert_inline(rest));
    } else if let Some(rest) = strip_ordered_item(&s) {
        // 有序列表项 "1. " → 保留前缀
        let dot = s.len() - rest.len();
        let mark = &s[..dot];
        s = format!("{mark}{}", convert_inline(rest));
    } else {
        s = convert_inline(&s);
    }
    s
}

/// 去掉行首的 `#`..`######` 与空格，返回剩余内容；不匹配返回 None。
fn strip_atx_header(s: &str) -> Option<String> {
    let hashes = s.chars().take_while(|&c| c == '#').count();
    if hashes == 0 || hashes > 6 {
        return None;
    }
    let rest = &s[hashes..];
    if rest.is_empty() {
        return Some(String::new());
    }
    // # 后必须紧跟空格才算标题
    if rest.starts_with(' ') {
        Some(rest.trim_start().to_string())
    } else {
        None
    }
}

/// 识别有序列表项前缀 "数字. "，返回前缀之后的剩余内容；不匹配返回 None。
fn strip_ordered_item(s: &str) -> Option<&str> {
    let digits_end = s.chars().take_while(|c| c.is_ascii_digit()).count();
    if digits_end == 0 {
        return None;
    }
    let rest = &s[digits_end..];
    if let Some(after) = rest.strip_prefix(". ") {
        Some(after)
    } else if rest == "." {
        Some("")
    } else {
        None
    }
}

/// 处理行内标记：`code`、**bold**、*italic*/_italic_、~~del~~、[text](url)。
/// 输入已是 HTML 转义后的文本。
fn convert_inline(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let chars: Vec<char> = s.chars().collect();
    let n = chars.len();
    let mut i = 0;

    let push_run = |out: &mut String, token: &str, content: &str, close: &str| {
        out.push_str(token);
        out.push_str(content);
        out.push_str(close);
    };

    while i < n {
        let c = chars[i];

        // 行内代码 `...`：到下一个反引号为止，不做任何解析
        if c == '`' {
            if let Some(end) = chars[i + 1..].iter().position(|&ch| ch == '`') {
                let content: String = chars[i + 1..i + 1 + end].iter().collect();
                push_run(&mut out, "<code>", &content, "</code>");
                i += end + 2;
                continue;
            }
        }

        // 粗体 **...** 或 __...__
        if (c == '*' && i + 1 < n && chars[i + 1] == '*')
            || (c == '_' && i + 1 < n && chars[i + 1] == '_')
        {
            let pair: String = std::iter::repeat(c).take(2).collect();
            if let Some(end) = find_marker(&chars, i + 2, &pair) {
                let content: String = chars[i + 2..end].iter().collect();
                push_run(&mut out, "<b>", &content, "</b>");
                i = end + 2;
                continue;
            }
        }

        // 斜体 *...* 或 _..._（单个标记，需配对且标记两侧不都是空白）
        if (c == '*' || c == '_') && (i + 1 < n && chars[i + 1] != c) {
            // 左侧不能是字母数字（避免匹配 a*b 中的 *），右侧首个不能是空白
            let left_ok = i == 0 || !chars[i - 1].is_alphanumeric();
            let right_ok = i + 1 < n && !chars[i + 1].is_whitespace();
            if left_ok && right_ok {
                if let Some(end) = find_single(&chars, i + 1, c) {
                    let content: String = chars[i + 1..end].iter().collect();
                    push_run(&mut out, "<i>", &content, "</i>");
                    i = end + 1;
                    continue;
                }
            }
        }

        // 删除线 ~~...~~
        if c == '~' && i + 1 < n && chars[i + 1] == '~' {
            if let Some(end) = find_marker(&chars, i + 2, "~~") {
                let content: String = chars[i + 2..end].iter().collect();
                push_run(&mut out, "<s>", &content, "</s>");
                i = end + 2;
                continue;
            }
        }

        // 链接 [text](url) —— url 中的 & 已被转义成 &amp;，原样保留即可
        if c == '[' {
            if let Some(close_bracket) = chars[i + 1..].iter().position(|&ch| ch == ']') {
                let after = i + 1 + close_bracket + 1;
                if after < n && chars[after] == '(' {
                    if let Some(close_paren) = chars[after + 1..].iter().position(|&ch| ch == ')') {
                        let text: String = chars[i + 1..i + 1 + close_bracket].iter().collect();
                        let url: String = chars[after + 1..after + 1 + close_paren].iter().collect();
                        let inner = convert_inline(&text);
                        out.push_str(&format!("<a href=\"{url}\">{inner}</a>"));
                        i = after + 1 + close_paren + 1;
                        continue;
                    }
                }
            }
        }

        out.push(c);
        i += 1;
    }
    out
}

/// 从 `from` 开始查找连续的 `marker`（如 "**"），返回其起始索引；找不到返回 None。
fn find_marker(chars: &[char], from: usize, marker: &str) -> Option<usize> {
    let m: Vec<char> = marker.chars().collect();
    let ml = m.len();
    if ml == 0 || from + ml > chars.len() {
        return None;
    }
    let mut i = from;
    while i + ml <= chars.len() {
        if chars[i..i + ml] == m[..] {
            return Some(i);
        }
        i += 1;
    }
    None
}

/// 从 `from` 开始查找单个 `ch`，要求其后一个字符不是同一个（避免与 ** 混淆），
/// 且标记本身不是单词内部的一部分。
fn find_single(chars: &[char], from: usize, ch: char) -> Option<usize> {
    let mut i = from;
    while i < chars.len() {
        if chars[i] == ch {
            // 不能紧接另一个相同字符（那应该是 ** 的情况）
            let next_is_same = i + 1 < chars.len() && chars[i + 1] == ch;
            // 内容不能为空，且结束标记前一个字符不能是空白
            let prev_not_ws = i > 0 && !chars[i - 1].is_whitespace();
            if !next_is_same && prev_not_ws {
                return Some(i);
            }
            // 跳过这对
            i += 2;
            continue;
        }
        i += 1;
    }
    None
}

/// 统一的错误反馈：优先编辑已知占位消息，否则新发一条。
async fn report_error(
    http: &Client,
    base: &str,
    chat_id: i64,
    msg_id: Option<i64>,
    msg: &str,
) {
    if let Some(id) = msg_id {
        let _ = edit_text(http, base, chat_id, Some(id), &format!("[错误] {msg}")).await;
    } else {
        let _ = http
            .post(format!("{base}/sendMessage"))
            .json(&json!({ "chat_id": chat_id, "text": format!("[错误] {msg}"), "parse_mode": "HTML" }))
            .send()
            .await;
    }
}

async fn edit_text(http: &Client, base: &str, chat_id: i64, msg_id: Option<i64>, text: &str) {
    let Some(msg_id) = msg_id else { return };
    // Telegram 文本上限 4096，截断保护
    let safe: String = text.chars().take(4000).collect();
    let html = md_to_tg_html(&safe);
    // 先尝试 HTML 渲染；若被拒（解析失败）则回退纯文本，保证消息至少能发出
    let resp = http
        .post(format!("{base}/editMessageText"))
        .json(&json!({ "chat_id": chat_id, "message_id": msg_id, "text": html, "parse_mode": "HTML" }))
        .send()
        .await;
    if let Ok(r) = resp {
        let ok = r
            .json::<serde_json::Value>()
            .await
            .ok()
            .and_then(|v| v.get("ok").and_then(|o| o.as_bool()))
            .unwrap_or(false);
        if !ok {
            let _ = http
                .post(format!("{base}/editMessageText"))
                .json(&json!({ "chat_id": chat_id, "message_id": msg_id, "text": safe }))
                .send()
                .await;
        }
    }
}

/// 工具名简化：`mcp__server__tool` → `server__tool`；内置工具无前缀，原样返回。
fn short_tool_name(name: &str) -> String {
    name.strip_prefix("mcp__").unwrap_or(name).to_string()
}

/// 从工具调用的 JSON 参数里提取一个人类可读的摘要。
/// 按字段优先级查找常见字段；找不到则回退到截断的原始 JSON。
fn summarize_args(args: &str) -> String {
    let trimmed = args.trim();
    if trimmed.is_empty() {
        return "(无参数)".to_string();
    }
    let Ok(v) = serde_json::from_str::<serde_json::Value>(trimmed) else {
        return truncate_chars(trimmed, 80);
    };
    let keys = [
        "url", "command", "query", "text", "prompt", "message", "name", "file_path", "selector",
        "path", "skill",
    ];
    if let Some(obj) = v.as_object() {
        for key in keys {
            if let Some(val) = obj.get(key).and_then(|x| x.as_str()) {
                if !val.is_empty() {
                    return truncate_chars(val, 80);
                }
            }
        }
    }
    // 没匹配到已知字段：显示截断后的紧凑 JSON
    let compact = serde_json::to_string(&v).unwrap_or_else(|_| trimmed.to_string());
    truncate_chars(&compact, 60)
}

/// 按字符数（非字节）截断，避免切断多字节 UTF-8。
fn truncate_chars(s: &str, max: usize) -> String {
    s.chars().take(max).collect()
}

/// 把正文与操作日志拼成待显示的文本。操作日志为空时只返回正文。
fn compose_display(buf: &str, tool_log: &str) -> String {
    if tool_log.is_empty() {
        return buf.to_string();
    }
    let sep = if buf.is_empty() { "" } else { "\n\n" };
    format!("{buf}{sep}🔧 操作中\n{tool_log}")
}

/// 把事件流渲染到指定占位消息（沿用节流编辑策略）。
/// 返回 `(最终文本, 是否以 ContinuePrompt 结束)`。
/// - 正文 `Text` 增量：节流编辑（攒够 ~120 字符）。
/// - `ToolCall`/`ToolError`：追加到操作日志并立即编辑（工具调用值得即时可见）。
/// - `Final`：用最终答复覆盖 buf，并清空操作日志（过程信息退场）。
/// - `ContinuePrompt`：追加提示语并立即编辑一次，返回 `pending=true`。
async fn render_events(
    http: &Client,
    base: &str,
    chat_id: i64,
    msg_id: Option<i64>,
    events: Vec<AgentEvent>,
) -> (String, bool) {
    let mut buf = String::new();
    let mut last_len = 0;
    let mut tool_log = String::new();
    let mut pending = false;
    for ev in events {
        match ev {
            AgentEvent::Text(t) => {
                buf.push_str(&t);
                if buf.len().saturating_sub(last_len) >= 120 {
                    last_len = buf.len();
                    let display = compose_display(&buf, &tool_log);
                    edit_text(http, base, chat_id, msg_id, &display).await;
                }
            }
            AgentEvent::ToolCall { name, args } => {
                if !tool_log.is_empty() {
                    tool_log.push('\n');
                }
                tool_log.push_str(&format!(
                    "• {} → {}",
                    short_tool_name(&name),
                    summarize_args(&args)
                ));
                let display = compose_display(&buf, &tool_log);
                edit_text(http, base, chat_id, msg_id, &display).await;
            }
            AgentEvent::ToolError(msg) => {
                if !tool_log.is_empty() {
                    tool_log.push('\n');
                }
                tool_log.push_str(&format!("• ⚠️ {}", truncate_chars(&msg, 80)));
                let display = compose_display(&buf, &tool_log);
                edit_text(http, base, chat_id, msg_id, &display).await;
            }
            AgentEvent::Final(t) => {
                buf = t;
                tool_log.clear();
            }
            AgentEvent::ContinuePrompt(note) => {
                if !buf.is_empty() {
                    buf.push_str("\n\n");
                }
                buf.push_str(&note);
                let display = compose_display(&buf, &tool_log);
                edit_text(http, base, chat_id, msg_id, &display).await;
                pending = true;
            }
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
