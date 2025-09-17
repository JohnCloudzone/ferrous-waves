use ferrous_waves::cache::Cache;
use ferrous_waves::{AnalysisEngine, AudioFile, Result};
use std::time::Instant;

#[tokio::main]
async fn main() -> Result<()> {
    // Create cache with custom settings
    let cache = Cache::with_config(
        std::path::PathBuf::from("./analysis_cache"),
        50 * 1024 * 1024,                     // 50MB max size
        std::time::Duration::from_secs(3600), // 1 hour TTL
    );

    // Create analysis engine with cache
    let engine = AnalysisEngine::new().with_cache(cache.clone());

    // Load audio file
    let audio = AudioFile::load("samples/large_file.wav")?;

    // First analysis (will be computed)
    println!("First analysis (computing)...");
    let start = Instant::now();
    let _result1 = engine.analyze(&audio).await?;
    let first_time = start.elapsed();
    println!("  Time: {:.3}s", first_time.as_secs_f32());

    // Second analysis (from cache)
    println!("Second analysis (from cache)...");
    let start = Instant::now();
    let _result2 = engine.analyze(&audio).await?;
    let second_time = start.elapsed();
    println!("  Time: {:.3}s", second_time.as_secs_f32());

    // Print cache statistics
    let stats = cache.stats();
    println!("\nCache Statistics:");
    println!("  Entries: {}", stats.total_entries);
    println!(
        "  Size: {:.2}MB",
        stats.total_size_bytes as f64 / 1_048_576.0
    );
    println!(
        "  Speedup: {:.1}x",
        first_time.as_secs_f32() / second_time.as_secs_f32()
    );

    Ok(())
}
