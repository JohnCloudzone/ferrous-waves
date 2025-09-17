use blake3::Hasher;
use serde::Serialize;
use std::path::Path;

/// Generate a cache key based on file path and analysis parameters
pub fn generate_cache_key(file_path: &Path, params: &impl Serialize) -> String {
    let mut hasher = Hasher::new();

    // Hash the file path
    if let Some(path_str) = file_path.to_str() {
        hasher.update(path_str.as_bytes());
    }

    // Hash the file metadata if available
    if let Ok(metadata) = std::fs::metadata(file_path) {
        if let Ok(modified) = metadata.modified() {
            if let Ok(duration) = modified.duration_since(std::time::UNIX_EPOCH) {
                hasher.update(&duration.as_secs().to_le_bytes());
            }
        }
        hasher.update(&metadata.len().to_le_bytes());
    }

    // Hash the analysis parameters
    if let Ok(params_bytes) = bincode::serialize(params) {
        hasher.update(&params_bytes);
    }

    format!("{}", hasher.finalize())
}

/// Generate a simple cache key from a string
pub fn generate_simple_key(content: &str) -> String {
    let mut hasher = Hasher::new();
    hasher.update(content.as_bytes());
    format!("{}", hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_key_generation() {
        let key1 = generate_simple_key("test_content");
        let key2 = generate_simple_key("test_content");
        let key3 = generate_simple_key("different_content");

        assert_eq!(key1, key2);
        assert_ne!(key1, key3);
    }

    #[test]
    fn test_cache_key_deterministic() {
        let key1 = generate_simple_key("deterministic_test");
        let key2 = generate_simple_key("deterministic_test");

        assert_eq!(key1, key2);
    }
}