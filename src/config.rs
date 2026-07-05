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
