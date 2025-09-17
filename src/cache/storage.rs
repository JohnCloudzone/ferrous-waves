use dashmap::DashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{SystemTime, Duration};
use serde::{Serialize, Deserialize};
use blake3::Hasher;
use crate::utils::error::{Result, FerrousError};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry {
    pub key: String,
    pub data: Vec<u8>,
    pub created_at: SystemTime,
    pub accessed_at: SystemTime,
    pub size_bytes: usize,
}

#[derive(Clone)]
pub struct Cache {
    entries: Arc<DashMap<String, CacheEntry>>,
    directory: PathBuf,
    max_size_bytes: usize,
    ttl: Duration,
}

impl Cache {
    pub fn new() -> Self {
        Self::with_config(
            PathBuf::from(".ferrous-waves-cache"),
            10 * 1024 * 1024 * 1024,  // 10GB
            Duration::from_secs(24 * 3600),  // 24 hours
        )
    }

    pub fn with_config(directory: PathBuf, max_size_bytes: usize, ttl: Duration) -> Self {
        std::fs::create_dir_all(&directory).ok();

        Self {
            entries: Arc::new(DashMap::new()),
            directory,
            max_size_bytes,
            ttl,
        }
    }

    pub fn generate_key(file_path: &str, params: &impl Serialize) -> String {
        let mut hasher = Hasher::new();
        hasher.update(file_path.as_bytes());
        hasher.update(&bincode::serialize(params).unwrap_or_default());
        format!("{}", hasher.finalize())
    }

    pub fn get(&self, key: &str) -> Option<Vec<u8>> {
        if let Some(mut entry) = self.entries.get_mut(key) {
            // Check TTL
            if entry.created_at.elapsed().ok()? > self.ttl {
                drop(entry);
                self.entries.remove(key);
                return None;
            }

            entry.accessed_at = SystemTime::now();
            Some(entry.data.clone())
        } else {
            // Try to load from disk
            self.load_from_disk(key)
        }
    }

    pub fn put(&self, key: String, data: Vec<u8>) -> Result<()> {
        let size = data.len();

        // Check if we need to evict entries
        if self.total_size() + size > self.max_size_bytes {
            self.evict_lru(size)?;
        }

        let entry = CacheEntry {
            key: key.clone(),
            data: data.clone(),
            created_at: SystemTime::now(),
            accessed_at: SystemTime::now(),
            size_bytes: size,
        };

        // Save to disk
        self.save_to_disk(&key, &data)?;

        // Store in memory
        self.entries.insert(key, entry);

        Ok(())
    }

    fn save_to_disk(&self, key: &str, data: &[u8]) -> Result<()> {
        // Ensure directory exists
        std::fs::create_dir_all(&self.directory)?;
        let path = self.directory.join(format!("{}.cache", key));
        std::fs::write(path, data)?;
        Ok(())
    }

    fn load_from_disk(&self, key: &str) -> Option<Vec<u8>> {
        let path = self.directory.join(format!("{}.cache", key));
        std::fs::read(path).ok()
    }

    fn total_size(&self) -> usize {
        self.entries.iter().map(|entry| entry.size_bytes).sum()
    }

    fn evict_lru(&self, needed_space: usize) -> Result<()> {
        let mut entries: Vec<_> = self.entries
            .iter()
            .map(|entry| (entry.key().clone(), entry.accessed_at))
            .collect();

        entries.sort_by_key(|(_, accessed)| *accessed);

        let mut freed_space = 0;
        for (key, _) in entries {
            if freed_space >= needed_space {
                break;
            }

            if let Some((_, entry)) = self.entries.remove(&key) {
                freed_space += entry.size_bytes;
                let path = self.directory.join(format!("{}.cache", key));
                std::fs::remove_file(path).ok();
            }
        }

        if freed_space < needed_space {
            return Err(FerrousError::Cache(
                "Unable to free enough space in cache".to_string()
            ));
        }

        Ok(())
    }

    pub fn clear(&self) -> Result<()> {
        self.entries.clear();

        // Remove all cache files from disk
        if let Ok(entries) = std::fs::read_dir(&self.directory) {
            for entry in entries {
                if let Ok(entry) = entry {
                    if entry.path().extension().and_then(|s| s.to_str()) == Some("cache") {
                        std::fs::remove_file(entry.path()).ok();
                    }
                }
            }
        }

        Ok(())
    }

    pub fn stats(&self) -> CacheStats {
        let total_entries = self.entries.len();
        let total_size = self.total_size();

        CacheStats {
            total_entries,
            total_size_bytes: total_size,
            max_size_bytes: self.max_size_bytes,
            directory: self.directory.clone(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    pub total_entries: usize,
    pub total_size_bytes: usize,
    pub max_size_bytes: usize,
    pub directory: PathBuf,
}

impl Default for Cache {
    fn default() -> Self {
        Self::new()
    }
}