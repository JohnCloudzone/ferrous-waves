use clap::Parser;
use ferrous_waves::cli::commands;
use ferrous_waves::cli::{Cli, Commands};
use ferrous_waves::Result;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    let filter = match std::env::var("RUST_LOG") {
        Ok(level) => level,
        Err(_) => {
            let cli = Cli::parse();
            match cli.verbose {
                0 => "error".to_string(),
                1 => "warn".to_string(),
                2 => "info".to_string(),
                3 => "debug".to_string(),
                _ => "trace".to_string(),
            }
        }
    };

    tracing_subscriber::fmt().with_env_filter(filter).init();

    let cli = Cli::parse();

    // Execute command
    match cli.command {
        Commands::Serve { cache } => {
            commands::run_serve(cache).await?;
        }
        Commands::Analyze {
            file,
            output,
            format,
            fft_size,
            hop_size,
            no_cache,
        } => {
            commands::run_analyze(file, output, format, fft_size, hop_size, no_cache).await?;
        }
        Commands::Compare {
            file_a,
            file_b,
            format,
        } => {
            commands::run_compare(file_a, file_b, format).await?;
        }
        Commands::Tempo { file, show_beats } => {
            commands::run_tempo(file, show_beats).await?;
        }
        Commands::Onsets { file, format } => {
            commands::run_onsets(file, format).await?;
        }
        Commands::Batch {
            directory,
            pattern,
            output,
            parallel,
        } => {
            commands::run_batch(directory, pattern, output, parallel).await?;
        }
        Commands::ClearCache { confirm } => {
            commands::run_clear_cache(confirm)?;
        }
        Commands::CacheStats => {
            commands::run_cache_stats()?;
        }
    }

    Ok(())
}
