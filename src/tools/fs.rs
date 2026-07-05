use crate::tools::Tool;
use anyhow::{anyhow, Result};
use async_trait::async_trait;
use serde_json::{json, Value};
use tokio::fs;

/// 读取本地 UTF-8 文本文件，最多返回前 5000 字符。
pub struct ReadFile;

#[async_trait]
impl Tool for ReadFile {
    fn name(&self) -> &str {
        "read_file"
    }
    fn description(&self) -> &str {
        "读取本地 UTF-8 文本文件内容，最多返回前 5000 字符。"
    }
    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "文件绝对或相对路径" }
            },
            "required": ["path"]
        })
    }
    async fn run(&self, args: Value) -> Result<String> {
        let path = args
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("缺少 path 参数"))?;
        let mut content =
            fs::read_to_string(path).await.map_err(|e| anyhow!("读取失败: {e}"))?;
        if content.chars().count() > 5000 {
            content = content.chars().take(5000).collect::<String>() + "...(截断)";
        }
        Ok(content)
    }
}
