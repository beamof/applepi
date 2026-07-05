//! Shell 工具：让 agent 执行 shell 命令。
//!
//! 安全策略：白名单（命令前缀）+ 黑名单（子串）。两者均为字符串匹配，
//! **不是真正的沙箱**——无法防御利用 shell 特性构造的绕过（管道、变量拼接等）。
//! 仅适合可信环境、可信输入。生产/多用户场景请用容器隔离或禁用本工具。

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use serde_json::{json, Value};
use std::path::PathBuf;
use std::time::Duration;
use tokio::process::Command;

use crate::config::ShellSection;
use crate::tools::Tool;

/// 输出最大字符数（防止 token 爆炸）。
const MAX_OUTPUT_CHARS: usize = 5000;

pub struct ShellTool {
    allow: Vec<String>,
    deny: Vec<String>,
    timeout: Duration,
    workdir: Option<PathBuf>,
}

impl ShellTool {
    pub fn new(cfg: &ShellSection) -> Self {
        Self {
            allow: cfg.allow.clone(),
            deny: cfg.effective_deny(),
            timeout: Duration::from_secs(cfg.timeout),
            workdir: cfg.workdir.as_ref().map(PathBuf::from),
        }
    }

    /// 策略检查：返回 Ok(()) 通过，Err(原因) 拒绝。
    fn check_policy(&self, command: &str) -> Result<()> {
        // 先过黑名单（子串匹配）
        for pat in &self.deny {
            if command.contains(pat.as_str()) {
                return Err(anyhow!(
                    "命令被黑名单拦截（命中 '{pat}'）。如需执行，请调整命令或联系管理员修改 shell.deny 配置。"
                ));
            }
        }
        // 再过白名单（前缀匹配）；白名单为空则不限制
        if !self.allow.is_empty() {
            let allowed = self
                .allow
                .iter()
                .any(|prefix| command.trim_start().starts_with(prefix.as_str()));
            if !allowed {
                return Err(anyhow!(
                    "命令不在白名单内（允许的前缀：{}）。请改用允许的命令，或联系管理员扩展 shell.allow 配置。",
                    self.allow.join(", ")
                ));
            }
        }
        Ok(())
    }
}

#[async_trait]
impl Tool for ShellTool {
    fn name(&self) -> &str {
        "shell"
    }

    fn description(&self) -> &str {
        "执行 shell 命令并返回输出（合并 stdout 与 stderr，含退出码）。\
         受白名单/黑名单约束：命中黑名单子串或不在白名单前缀内会被拒绝。\
         有超时限制（默认 30 秒）。请只执行只读或安全命令，避免破坏性操作。"
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "要执行的 shell 命令"
                }
            },
            "required": ["command"]
        })
    }

    async fn run(&self, args: Value) -> Result<String> {
        let command = args
            .get("command")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("缺少 command 参数"))?;

        // 策略检查
        self.check_policy(command)?;

        // 构造 shell 命令（跨平台）
        let mut cmd = shell_command(command);
        if let Some(dir) = &self.workdir {
            cmd.current_dir(dir);
        }
        // 合并 stderr 到 stdout
        cmd.stdout(std::process::Stdio::piped());
        cmd.stderr(std::process::Stdio::piped());

        // 带超时执行
        let output = match tokio::time::timeout(self.timeout, cmd.output()).await {
            Ok(Ok(o)) => o,
            Ok(Err(e)) => return Err(anyhow!("执行失败: {e}")),
            Err(_) => {
                return Err(anyhow!(
                    "命令执行超时（{} 秒）",
                    self.timeout.as_secs()
                ))
            }
        };

        // 合并输出
        let mut combined = String::new();
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        if !stdout.is_empty() {
            combined.push_str(&stdout);
        }
        if !stderr.is_empty() {
            if !combined.is_empty() {
                combined.push_str("\n[stderr]\n");
            }
            combined.push_str(&stderr);
        }

        // 截断保护
        let combined = truncate(&combined, MAX_OUTPUT_CHARS);

        let code = output.status.code().unwrap_or(-1);
        if output.status.success() {
            if combined.is_empty() {
                Ok(format!("（命令执行成功，无输出。退出码 {code}）"))
            } else {
                Ok(format!("{combined}\n[退出码 {code}]"))
            }
        } else {
            // 非零退出码：作为错误喂回模型，agent 能看到输出并调整
            Err(anyhow!(
                "命令退出码非零（{code}）。输出：\n{combined}"
            ))
        }
    }
}

/// 按平台构造 shell 命令。
#[cfg(unix)]
fn shell_command(command: &str) -> Command {
    let mut cmd = Command::new("sh");
    cmd.arg("-c").arg(command);
    cmd
}

#[cfg(windows)]
fn shell_command(command: &str) -> Command {
    let mut cmd = Command::new("cmd");
    cmd.arg("/C").arg(command);
    cmd
}

#[cfg(not(any(unix, windows)))]
fn shell_command(_command: &str) -> Command {
    compile_error!("shell 工具仅支持 unix/windows 平台")
}

/// 截断字符串：超长时保留头部 + 省略提示 + 尾部。
fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let head: String = s.chars().take(max * 4 / 5).collect();
    let tail: String = s.chars().skip(s.chars().count().saturating_sub(max / 5)).collect();
    format!("{head}\n…[输出过长，已截断 {max} 字符]…\n{tail}")
}
