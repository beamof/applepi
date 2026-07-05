mod lib_decl {
    // 主 bin 复用 lib
}

use applepi::agent::{Agent, AgentEvent};
use applepi::config;
use applepi::mcp;
use applepi::memory::long_term::LongTermMemory;
use applepi::tools::{default_tools, Tool};
use std::io::{self, BufRead, Write};
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();
    tracing_subscriber::fmt::init();

    let cfg = config::load("config.yaml")?;
    let api_key = cfg.resolve_api_key()?;
    let mut persona = config::load_persona("AGENTS.md")?;
    persona.push_str(&config::load_skills_summary("skills"));

    let long_term = if cfg.memory.enabled {
        Some(LongTermMemory::open(&cfg.memory.db_path, cfg.embeddings_config(api_key.clone()))?)
    } else {
        None
    };

    // 合并默认工具 + MCP 远端工具
    let mut tools = default_tools();
    tools.extend(mcp::load_mcp_tools(&cfg.mcp_servers).await?);
    // Shell 工具（可选，受白名单/黑名单约束）
    if cfg.shell.enabled {
        let t = Arc::new(applepi::tools::shell::ShellTool::new(&cfg.shell));
        tools.insert(t.name().to_string(), t);
    }

    let mut agent = Agent::new(
        cfg.llm_config(api_key),
        persona,
        tools,
        long_term,
        cfg.memory.top_k_or(3),
    );

    println!("applepi 已就绪（CLI 模式）。输入 /quit 退出。\n");
    let stdin = io::stdin();
    let mut lines = stdin.lock().lines();
    loop {
        print!("你: ");
        io::stdout().flush()?;
        let Some(line) = lines.next() else { break };
        let line = line?;
        if line.trim() == "/quit" {
            break;
        }
        if line.trim().is_empty() {
            continue;
        }
        print!("\napplepi: ");
        io::stdout().flush()?;
        match agent.chat_stream(&line).await {
            Ok(events) => {
                for ev in events {
                    match ev {
                        AgentEvent::Text(t) => {
                            print!("{t}");
                            io::stdout().flush()?;
                        }
                        AgentEvent::Final(_) => {} // 已增量打印
                        AgentEvent::ToolCall { name, args } => {
                            eprintln!("\n  [工具 {name}: {args}]");
                        }
                        AgentEvent::ToolError(e) => {
                            eprintln!("\n  [{e}]");
                        }
                    }
                }
                println!("\n");
            }
            Err(e) => eprintln!("\n[错误] {e}\n"),
        }
    }
    Ok(())
}
