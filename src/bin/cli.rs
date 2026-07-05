mod lib_decl {
    // 主 bin 复用 lib
}

use applepi::agent::{Agent, AgentEvent};
use applepi::config;
use applepi::memory::long_term::LongTermMemory;
use applepi::tools::default_tools;
use std::io::{self, BufRead, Write};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();
    tracing_subscriber::fmt::init();

    let cfg = config::load("config.yaml")?;
    let api_key = cfg.resolve_api_key()?;

    let long_term = if cfg.memory.enabled {
        Some(LongTermMemory::open(&cfg.memory.db_path, cfg.embeddings_config(api_key.clone()))?)
    } else {
        None
    };
    let mut agent = Agent::new(
        cfg.llm_config(api_key),
        cfg.agent.persona,
        default_tools(),
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
