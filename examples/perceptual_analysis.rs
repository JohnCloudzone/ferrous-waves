//! Example demonstrating perceptual metrics analysis including LUFS loudness measurement

use ferrous_waves::{AnalysisEngine, AudioFile};
use std::error::Error;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    println!("Perceptual Metrics Analysis Example");
    println!("====================================\n");

    // Generate sample files if needed
    // Run: cargo run --example generate_samples
    // to create the sample files

    // Analyze different audio files to show perceptual metrics
    let files = vec!["samples/test.wav", "samples/music.wav", "samples/drums.wav"];

    let engine = AnalysisEngine::new();

    for file_path in files {
        println!("Analyzing: {}", file_path);
        println!("----------");

        let audio = AudioFile::load(file_path)?;
        let result = engine.analyze(&audio).await?;

        // Display perceptual metrics
        println!("  Loudness: {:.1} LUFS", result.perceptual.loudness_lufs);
        println!("  True Peak: {:.1} dBFS", result.perceptual.true_peak_dbfs);
        println!("  Dynamic Range: {:.1} dB", result.perceptual.dynamic_range);
        println!(
            "  Loudness Range: {:.1} LU",
            result.perceptual.loudness_range
        );
        println!("  Crest Factor: {:.2}", result.perceptual.crest_factor);
        println!(
            "  Energy Level: {:.0}%",
            result.perceptual.energy_level * 100.0
        );

        // Show relevant insights
        println!("\n  Insights:");
        for insight in result
            .insights
            .iter()
            .filter(|i| i.contains("LUFS") || i.contains("peak") || i.contains("range"))
        {
            println!("    - {}", insight);
        }

        // Show recommendations if any
        if !result.recommendations.is_empty() {
            println!("\n  Recommendations:");
            for rec in &result.recommendations {
                println!("    - {}", rec);
            }
        }

        println!();
    }

    println!("Perceptual Metrics Reference:");
    println!("------------------------------");
    println!("LUFS (Loudness Units Full Scale):");
    println!("  - Streaming platforms: -14 to -16 LUFS");
    println!("  - Broadcast (EBU R 128): -23 LUFS");
    println!("  - Cinema: -27 to -31 LUFS");
    println!("\nTrue Peak:");
    println!("  - Recommended maximum: -1 dBFS (to prevent inter-sample peaks)");
    println!("  - 0 dBFS or above indicates clipping");
    println!("\nLoudness Range (LRA):");
    println!("  - < 5 LU: Low dynamics (compressed)");
    println!("  - 5-20 LU: Normal dynamics");
    println!("  - > 20 LU: High dynamics");

    Ok(())
}
