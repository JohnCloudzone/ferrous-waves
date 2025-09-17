//! Ferrous Waves - High-fidelity audio analysis library

pub mod audio;
pub mod analysis;
pub mod cache;
pub mod cli;
pub mod mcp;
pub mod utils;
pub mod visualization;

pub use crate::audio::AudioFile;
pub use crate::analysis::engine::AnalysisEngine;
pub use crate::utils::error::{FerrousError, Result};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
