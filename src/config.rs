use crate::llm::LlmConfig;
use anyhow::Result;
use serde::Deserialize;
use std::collections::HashMap;

#[derive(Deserialize, Clone)]
pub struct Config {
    pub llm: LlmSection,
    #[serde(default)]
    pub embeddings: EmbeddingsSection,
    #[serde(default)]
    pub memory: MemorySection,
    #[serde(default)]
    pub telegram: TelegramSection,
    /// MCP 服务器列表（Streamable HTTP 传输）。默认空。
    #[serde(default)]
    pub mcp_servers: Vec<McpServerConfig>,
    /// Cron 定时任务（仅 bot 模式生效）。默认禁用。
    #[serde(default)]
    pub cron: CronSection,
    /// Shell 工具配置。默认禁用。
    #[serde(default)]
    pub shell: ShellSection,
}

#[derive(Deserialize, Clone)]
pub struct LlmSection {
    pub api_base: String,
    pub model: String,
    /// 留空则回退到环境变量 OPENAI_API_KEY（或 API_KEY）
    #[serde(default)]
    pub api_key: String,
}

#[derive(Deserialize, Clone, Default)]
pub struct EmbeddingsSection {
    pub model: String,
    /// 留空则复用 llm.api_base
    pub api_base: Option<String>,
}

#[derive(Deserialize, Clone, Default)]
pub struct MemorySection {
    pub enabled: bool,
    pub db_path: String,
    pub top_k: usize,
}

impl MemorySection {
    pub fn top_k_or(&self, default: usize) -> usize {
        if self.top_k == 0 { default } else { self.top_k }
    }
}

#[derive(Deserialize, Clone, Default)]
pub struct TelegramSection {
    #[serde(default)]
    pub bot_token: String,
}

/// 单个 MCP 服务器配置（HTTP/SSE 传输）。
#[derive(Deserialize, Clone)]
pub struct McpServerConfig {
    /// 名称，仅用于日志标识。
    pub name: String,
    /// MCP endpoint URL。
    pub url: String,
    /// 额外请求头（如 Authorization）。
    #[serde(default)]
    pub headers: HashMap<String, String>,
    /// 是否启用，默认 true。
    #[serde(default = "default_true")]
    pub enabled: bool,
}

fn default_true() -> bool {
    true
}

fn default_cron_db() -> String {
    "data/cron.db".into()
}

/// Cron 定时任务配置（仅 bot 模式生效）。
#[derive(Deserialize, Clone, Default)]
pub struct CronSection {
    /// 总开关，默认 false。
    #[serde(default)]
    pub enabled: bool,
    /// SQLite 路径，默认 data/cron.db（与长期记忆库分库）。
    #[serde(default = "default_cron_db")]
    pub db_path: String,
    /// 启动时种子 job：首次启动按 name 去重写入 DB；已存在不重复插入。
    /// 之后所有增删改通过 /cron 命令操作 DB，这里改动不再生效。
    #[serde(default)]
    pub jobs: Vec<CronJob>,
}

/// 单个 Cron 定时任务（配置种子形态，无 id 字段）。
#[derive(Deserialize, Clone)]
pub struct CronJob {
    /// 名称，用于日志标识与种子去重。
    pub name: String,
    /// 标准 cron 表达式（北京时间），如 "0 9 * * *"。支持 5 字段（分 时 日 月 周）。
    pub schedule: String,
    /// 触发时发给 agent 的 prompt。
    pub prompt: String,
    /// 推送目标 Telegram chat_id。
    pub chat_id: i64,
    /// 是否启用，默认 true。
    #[serde(default = "default_true")]
    pub enabled: bool,
}

fn default_timeout() -> u64 {
    30
}

/// 默认危险命令黑名单（即使配置里不写 deny 也有这层保护）。
fn default_deny() -> Vec<String> {
    [
        "rm -rf", "sudo ", " >/", " >>/", "mkfs", "dd if=", ":(){",
        "chmod 777", "reboot", "shutdown", "halt", ":(){:|:&};:",
    ]
    .iter()
    .map(|s| (*s).to_string())
    .collect()
}

/// Shell 工具配置。
#[derive(Deserialize, Clone)]
pub struct ShellSection {
    /// 是否启用 shell 工具，默认 false（安全默认）。
    #[serde(default)]
    pub enabled: bool,
    /// 允许的命令前缀白名单。非空时只允许命令以其中任一前缀开头；为空则不限制（仅靠黑名单）。
    #[serde(default)]
    pub allow: Vec<String>,
    /// 禁止的子串黑名单。命令含其中任一子串即拒绝。为空时使用内置默认黑名单。
    #[serde(default)]
    pub deny: Vec<String>,
    /// 执行超时（秒），默认 30。
    #[serde(default = "default_timeout")]
    pub timeout: u64,
    /// 工作目录，留空 = 当前目录。
    #[serde(default)]
    pub workdir: Option<String>,
}

impl Default for ShellSection {
    fn default() -> Self {
        Self {
            enabled: false,
            allow: Vec::new(),
            deny: Vec::new(),
            timeout: default_timeout(),
            workdir: None,
        }
    }
}

impl ShellSection {
    /// 取生效的黑名单：配置非空用配置，否则用默认。
    pub fn effective_deny(&self) -> Vec<String> {
        if self.deny.is_empty() {
            default_deny()
        } else {
            self.deny.clone()
        }
    }
}

pub fn load(path: &str) -> Result<Config> {
    let raw = std::fs::read_to_string(path)?;
    Ok(serde_yaml::from_str(&raw)?)
}

/// 从 AGENTS.md 读取人设（启动时读一次）。
/// 文件不存在时报错并提示创建。
pub fn load_persona(path: &str) -> Result<String> {
    let raw = std::fs::read_to_string(path).map_err(|e| {
        anyhow::anyhow!(
            "读取人设文件 {path} 失败: {e}\n\
             请在项目根目录创建 AGENTS.md 描述助手人设。"
        )
    })?;
    Ok(raw.trim().to_string())
}

/// 扫描 skills/ 目录，返回所有技能的 name + description 摘要（用于注入人设）。
///
/// 遍历 `skills/<name>/SKILL.md`，解析 YAML frontmatter 提取 name + description。
/// 无技能目录、单文件解析失败均不阻断，返回空串或部分结果。
pub fn load_skills_summary(skills_dir: &str) -> String {
    let entries = match std::fs::read_dir(skills_dir) {
        Ok(e) => e,
        Err(_) => return String::new(), // 目录不存在 = 无技能，静默
    };

    let mut items: Vec<(String, String)> = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path().join("SKILL.md");
        let Ok(content) = std::fs::read_to_string(&path) else {
            continue; // 该子目录无 SKILL.md，跳过
        };
        if let Some((name, desc)) = parse_frontmatter(&content) {
            // name 缺失时用目录名兜底
            let name = if name.is_empty() {
                entry.file_name().to_string_lossy().into_owned()
            } else {
                name
            };
            items.push((name, desc));
        }
    }

    if items.is_empty() {
        return String::new();
    }

    let mut out = String::from("\n\n## 可用技能\n");
    for (name, desc) in &items {
        out.push_str(&format!("- {name}: {desc}\n"));
    }
    out.push_str("用 skill_use(\"技能名\") 运行技能；用 skill_create 创建新技能。");
    out
}

/// 从 SKILL.md 内容解析 frontmatter 的 name 与 description。
/// 格式：首行 `---`，随后若干 `key: value`，再 `---` 结束。
fn parse_frontmatter(content: &str) -> Option<(String, String)> {
    let content = content.trim_start();
    let after_opening = content.strip_prefix("---")?;
    // 找闭合 ---
    let end = after_opening.find("\n---")?;
    let fm = &after_opening[..end];

    let mut name = String::new();
    let mut desc = String::new();
    for line in fm.lines() {
        let line = line.trim();
        if let Some(v) = line.strip_prefix("name:") {
            name = v.trim().trim_matches('"').to_string();
        } else if let Some(v) = line.strip_prefix("description:") {
            desc = v.trim().trim_matches('"').to_string();
        }
    }
    Some((name, desc))
}

impl Config {
    /// 统一取 API key：config 优先，环境变量兜底。
    pub fn resolve_api_key(&self) -> Result<String> {
        if !self.llm.api_key.trim().is_empty() {
            Ok(self.llm.api_key.trim().to_string())
        } else {
            std::env::var("OPENAI_API_KEY")
                .or_else(|_| std::env::var("API_KEY"))
                .map_err(|_| {
                    anyhow::anyhow!(
                        "未找到 API key：请在 config.yaml 的 llm.api_key 填写，\
                         或设置环境变量 OPENAI_API_KEY / API_KEY"
                    )
                })
        }
    }

    pub fn llm_config(&self, api_key: String) -> LlmConfig {
        LlmConfig {
            api_base: self.llm.api_base.clone(),
            api_key,
            model: self.llm.model.clone(),
        }
    }

    pub fn embeddings_config(&self, api_key: String) -> crate::memory::long_term::EmbedConfig {
        crate::memory::long_term::EmbedConfig {
            api_base: self
                .embeddings
                .api_base
                .clone()
                .unwrap_or_else(|| self.llm.api_base.clone()),
            api_key,
            model: self.embeddings.model.clone(),
        }
    }
}
