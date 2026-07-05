use applepi::{bot, config};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();
    tracing_subscriber::fmt::init();

    let cfg = config::load("config.yaml")?;
    let api_key = cfg.resolve_api_key()?;

    bot::run(cfg, api_key).await
}
