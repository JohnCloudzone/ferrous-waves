pub mod commands;

use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "ferrous-waves")]
#[command(author = "Brandon Williams")]
#[command(version)]
#[command(about = "High-fidelity audio analysis bridge for development workflows", long_about = None)]
pub struct Cli {
    /// Sets the verbosity level
    #[arg(short, long, action = clap::ArgAction::Count)]
    pub verbose: u8,

    /// Path to configuration file
    #[arg(short, long)]
    pub config: Option<PathBuf>,

    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Start MCP server
    Serve {
        /// Port to listen on
        #[arg(short, long, default_value = "3030")]
        port: u16,

        /// Host to bind to
        #[arg(short = 'H', long, default_value = "127.0.0.1")]
        host: String,

        /// Enable cache
        #[arg(long, default_value = "true")]
        cache: bool,
    },

    /// Analyze a single audio file
    Analyze {
        /// Path to audio file
        file: PathBuf,

        /// Output directory for results
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Output format (json, text, visual)
        #[arg(short, long, default_value = "json")]
        format: String,

        /// FFT size for analysis
        #[arg(long, default_value = "2048")]
        fft_size: usize,

        /// Hop size for STFT
        #[arg(long, default_value = "512")]
        hop_size: usize,

        /// Disable cache
        #[arg(long)]
        no_cache: bool,
    },

    /// Compare two audio files
    Compare {
        /// First audio file
        file_a: PathBuf,

        /// Second audio file
        file_b: PathBuf,

        /// Output format
        #[arg(short, long, default_value = "json")]
        format: String,
    },

    /// Extract tempo from audio file
    Tempo {
        /// Path to audio file
        file: PathBuf,

        /// Show beat positions
        #[arg(long)]
        show_beats: bool,
    },

    /// Detect onsets in audio file
    Onsets {
        /// Path to audio file
        file: PathBuf,

        /// Output format (json, csv, text)
        #[arg(short, long, default_value = "text")]
        format: String,
    },

    /// Batch analyze multiple audio files
    Batch {
        /// Directory containing audio files
        directory: PathBuf,

        /// File pattern to match (e.g., "*.wav")
        #[arg(short, long, default_value = "*")]
        pattern: String,

        /// Output directory for results
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Number of parallel jobs
        #[arg(short = 'j', long, default_value = "4")]
        parallel: usize,
    },

    /// Clear the cache
    ClearCache {
        /// Confirm cache clearing
        #[arg(long)]
        confirm: bool,
    },

    /// Show cache statistics
    CacheStats,
}
