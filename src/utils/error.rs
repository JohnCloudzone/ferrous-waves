use thiserror::Error;

#[derive(Error, Debug)]
pub enum FerrousError {
    #[error("Audio decode error: {0}")]
    AudioDecode(String),

    #[error("Analysis error: {0}")]
    Analysis(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Cache error: {0}")]
    Cache(String),

    #[error("MCP error: {0}")]
    Mcp(String),

    #[error("Visualization error: {0}")]
    Visualization(String),

    #[error("Configuration error: {0}")]
    Config(#[from] config::ConfigError),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, FerrousError>;