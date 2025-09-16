use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub analysis: AnalysisConfig,
    pub cache: CacheConfig,
    pub mcp: McpConfig,
    pub output: OutputConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisConfig {
    pub fft_size: usize,
    pub hop_size: usize,
    pub sample_rate: Option<u32>,
    pub window_type: WindowType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WindowType {
    Hann,
    Hamming,
    Blackman,
    Kaiser,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    pub enabled: bool,
    pub directory: PathBuf,
    pub max_size_gb: f32,
    pub ttl_hours: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpConfig {
    pub enabled: bool,
    pub port: u16,
    pub host: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputConfig {
    pub directory: PathBuf,
    pub format: OutputFormat,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OutputFormat {
    Json,
    Bundle,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            analysis: AnalysisConfig {
                fft_size: 2048,
                hop_size: 512,
                sample_rate: None,
                window_type: WindowType::Hann,
            },
            cache: CacheConfig {
                enabled: true,
                directory: PathBuf::from("~/.ferrous-waves/cache"),
                max_size_gb: 10.0,
                ttl_hours: 24,
            },
            mcp: McpConfig {
                enabled: false,
                port: 3030,
                host: "127.0.0.1".to_string(),
            },
            output: OutputConfig {
                directory: PathBuf::from("./output"),
                format: OutputFormat::Bundle,
            },
        }
    }
}