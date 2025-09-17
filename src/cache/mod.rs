pub mod storage;
pub mod key;

pub use storage::{Cache, CacheEntry, CacheStats};
pub use key::{generate_cache_key, generate_simple_key};