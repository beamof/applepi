use crate::tools::Tool;
use anyhow::Result;
use async_trait::async_trait;
use serde_json::{json, Value};

/// 示例工具：原样回显，用来验证 tool-calling 链路。
pub struct Echo;

#[async_trait]
impl Tool for Echo {
    fn name(&self) -> &str {
        "echo"
    }
    fn description(&self) -> &str {
        "原样返回输入文本。用于测试。"
    }
    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "text": { "type": "string", "description": "要回显的内容" }
            },
            "required": ["text"]
        })
    }
    async fn run(&self, args: Value) -> Result<String> {
        let text = args.get("text").and_then(|v| v.as_str()).unwrap_or("");
        Ok(text.to_string())
    }
}
