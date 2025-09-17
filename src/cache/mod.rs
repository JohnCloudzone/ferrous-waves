pub mod key;
pub mod storage;

pub use key::{generate_cache_key, generate_simple_key};
pub use storage::{Cache, CacheEntry, CacheStats};
