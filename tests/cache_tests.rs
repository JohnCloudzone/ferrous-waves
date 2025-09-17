#[cfg(test)]
mod cache_tests {
    use ferrous_waves::cache::{Cache, generate_simple_key};
    use std::time::Duration;
    use tempfile::tempdir;

    #[test]
    fn test_cache_creation() {
        let cache = Cache::new();
        let stats = cache.stats();
        assert_eq!(stats.total_entries, 0);
        assert_eq!(stats.total_size_bytes, 0);
    }

    #[test]
    fn test_cache_with_config() {
        let dir = tempdir().unwrap();
        let cache = Cache::with_config(
            dir.path().to_path_buf(),
            1024 * 1024,  // 1MB
            Duration::from_secs(3600),  // 1 hour
        );
        let stats = cache.stats();
        assert_eq!(stats.max_size_bytes, 1024 * 1024);
    }

    #[test]
    fn test_cache_put_and_get() {
        let cache = Cache::new();
        let key = "test_key".to_string();
        let data = vec![1, 2, 3, 4, 5];

        cache.put(key.clone(), data.clone()).unwrap();

        let retrieved = cache.get(&key);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap(), data);
    }

    #[test]
    fn test_cache_key_generation() {
        let key1 = generate_simple_key("test_content");
        let key2 = generate_simple_key("test_content");
        let key3 = generate_simple_key("different_content");

        assert_eq!(key1, key2);
        assert_ne!(key1, key3);
    }

    #[test]
    fn test_cache_disk_persistence() {
        let dir = tempdir().unwrap();
        let cache = Cache::with_config(
            dir.path().to_path_buf(),
            1024 * 1024,
            Duration::from_secs(3600),
        );

        let key = "persist_key".to_string();
        let data = vec![10, 20, 30];

        cache.put(key.clone(), data.clone()).unwrap();

        // Create new cache instance with same directory
        let cache2 = Cache::with_config(
            dir.path().to_path_buf(),
            1024 * 1024,
            Duration::from_secs(3600),
        );

        // Should load from disk
        let retrieved = cache2.get(&key);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap(), data);
    }

    #[test]
    fn test_cache_clear() {
        let cache = Cache::new();

        for i in 0..5 {
            let key = format!("key_{}", i);
            let data = vec![i as u8; 100];
            cache.put(key, data).unwrap();
        }

        let stats = cache.stats();
        assert_eq!(stats.total_entries, 5);

        cache.clear().unwrap();

        let stats = cache.stats();
        assert_eq!(stats.total_entries, 0);
        assert_eq!(stats.total_size_bytes, 0);
    }

    #[test]
    fn test_cache_lru_eviction() {
        let dir = tempdir().unwrap();
        let cache = Cache::with_config(
            dir.path().to_path_buf(),
            1000,  // Very small cache
            Duration::from_secs(3600),
        );

        // Add entries that will exceed cache size
        for i in 0..10 {
            let key = format!("key_{}", i);
            let data = vec![i as u8; 200];  // Each entry is 200 bytes
            cache.put(key, data).ok();
        }

        // Cache should have evicted older entries
        let stats = cache.stats();
        assert!(stats.total_size_bytes <= 1000);
    }

    #[test]
    fn test_cache_ttl() {
        use std::thread;

        let dir = tempdir().unwrap();
        let cache = Cache::with_config(
            dir.path().to_path_buf(),
            1024 * 1024,
            Duration::from_millis(100),  // Very short TTL
        );

        let key = "ttl_key".to_string();
        let data = vec![1, 2, 3];

        cache.put(key.clone(), data.clone()).unwrap();

        // Data should be available immediately
        assert!(cache.get(&key).is_some());

        // Wait for TTL to expire
        thread::sleep(Duration::from_millis(200));

        // Data should be expired
        assert!(cache.get(&key).is_none());
    }

    #[test]
    fn test_cache_stats() {
        let cache = Cache::new();

        let key1 = "key1".to_string();
        let data1 = vec![1; 100];
        cache.put(key1, data1).unwrap();

        let key2 = "key2".to_string();
        let data2 = vec![2; 200];
        cache.put(key2, data2).unwrap();

        let stats = cache.stats();
        assert_eq!(stats.total_entries, 2);
        assert_eq!(stats.total_size_bytes, 300);
    }
}