use ferrous_waves::cache::Cache;
use ferrous_waves::mcp::server::FerrousWavesMcp;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging to stderr (stdout is used for MCP communication)
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("ferrous_waves=info".parse()?)
                .add_directive(tracing::Level::DEBUG.into()),
        )
        .with_writer(std::io::stderr)
        .with_ansi(false)
        .init();

    tracing::info!("Starting Ferrous Waves MCP server...");

    // Create cache
    let cache = Cache::new();

    // Create and start MCP server
    let server = FerrousWavesMcp::with_cache(cache);

    if let Err(e) = server.start().await {
        tracing::error!("Failed to start MCP server: {:?}", e);
        return Err(e.into());
    }

    Ok(())
}
