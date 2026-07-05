//! 技能（Skill）工具：创建与运行 Claude Skills 风格的 SKILL.md。
//!
//! 技能 = 持久化的工作流/指令模板，存 `skills/<name>/SKILL.md`。
//! - skill_create：创建/更新技能文件（YAML frontmatter + Markdown 正文）
//! - skill_use：加载技能正文为本次任务上下文，agent 据此执行后续工作
//!
//! 「运行」语义忠于 Claude Skills：不启动子进程，而是把指令文本交给 agent。

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use serde_json::{json, Value};
use tokio::fs;

use crate::tools::Tool;

/// 技能根目录（相对项目工作目录）。
const SKILLS_DIR: &str = "skills";
/// 单个技能文件名。
const SKILL_FILE: &str = "SKILL.md";

/// 创建/更新技能。
pub struct SkillCreateTool;

#[async_trait]
impl Tool for SkillCreateTool {
    fn name(&self) -> &str {
        "skill_create"
    }

    fn description(&self) -> &str {
        "创建或更新一个技能（持久化的工作流/指令模板，存为 SKILL.md 文件）。\
         适合把重复性任务（代码评审、周报生成、代码风格检查等）沉淀为可复用技能。\
         技能创建后需重启程序才会在可用列表中显示，但可立即用 skill_use 运行。"
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "技能标识（用作目录名，建议小写连字符，如 code-review）"
                },
                "description": {
                    "type": "string",
                    "description": "技能的一句话描述（agent 据此判断何时使用该技能）"
                },
                "content": {
                    "type": "string",
                    "description": "技能正文（Markdown），即具体的工作流/指令"
                }
            },
            "required": ["name", "description", "content"]
        })
    }

    async fn run(&self, args: Value) -> Result<String> {
        let name = args
            .get("name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("缺少 name 参数"))?;
        let description = args
            .get("description")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("缺少 description 参数"))?;
        let content = args
            .get("content")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("缺少 content 参数"))?;

        // 简单校验 name（避免路径穿越）
        if name.is_empty()
            || name.contains('/')
            || name.contains('\\')
            || name.contains("..")
        {
            return Err(anyhow!("name 非法（含路径分隔符或为空）"));
        }

        let dir = format!("{SKILLS_DIR}/{name}");
        fs::create_dir_all(&dir)
            .await
            .map_err(|e| anyhow!("创建目录 {dir} 失败: {e}"))?;

        // frontmatter 单行 description（YAML 要求），正文原样
        let mut file_content = String::new();
        file_content.push_str("---\n");
        file_content.push_str(&format!("name: {name}\n"));
        // description 单行；含冒号时用引号包裹
        let desc_field = if description.contains(':') {
            format!("description: \"{}\"", description.replace('"', "\\\""))
        } else {
            format!("description: {description}")
        };
        file_content.push_str(&desc_field);
        file_content.push('\n');
        file_content.push_str("---\n\n");
        file_content.push_str(content);

        let path = format!("{dir}/{SKILL_FILE}");
        fs::write(&path, &file_content)
            .await
            .map_err(|e| anyhow!("写入 {path} 失败: {e}"))?;

        Ok(format!(
            "已创建技能 '{name}'，写入 {path}。\n\
             可立即用 skill_use(\"{name}\") 运行；下次启动后该技能会出现在可用列表中。"
        ))
    }
}

/// 运行技能：加载 SKILL.md 正文为上下文。
pub struct SkillUseTool;

#[async_trait]
impl Tool for SkillUseTool {
    fn name(&self) -> &str {
        "skill_use"
    }

    fn description(&self) -> &str {
        "运行一个技能：加载其 SKILL.md 内容作为本次任务的指令上下文，\
         然后你按这些指令完成用户请求。技能可定义工作流、检查清单、输出格式等。"
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "要运行的技能名（与 skill_create 的 name 一致）"
                }
            },
            "required": ["name"]
        })
    }

    async fn run(&self, args: Value) -> Result<String> {
        let name = args
            .get("name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("缺少 name 参数"))?;

        if name.contains('/') || name.contains('\\') || name.contains("..") {
            return Err(anyhow!("name 非法"));
        }

        let path = format!("{SKILLS_DIR}/{name}/{SKILL_FILE}");
        let content = fs::read_to_string(&path)
            .await
            .map_err(|e| anyhow!("读取技能 '{name}' 失败（{path}）: {e}"))?;

        Ok(format!("（已加载技能 {name} 的指令，请据此执行）\n\n{content}"))
    }
}
