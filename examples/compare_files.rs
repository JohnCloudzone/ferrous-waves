use ferrous_waves::{AnalysisEngine, AudioFile, Result};

#[tokio::main]
async fn main() -> Result<()> {
    // Load two audio files to compare
    let audio_a = AudioFile::load("samples/original.wav")?;
    let audio_b = AudioFile::load("samples/processed.wav")?;

    // Create analysis engine
    let engine = AnalysisEngine::new();

    // Compare the files
    let comparison = engine.compare(&audio_a, &audio_b).await;

    // Print comparison results
    println!("Audio Comparison:");
    println!("  File A: {}", comparison.file_a.path);
    println!("  File B: {}", comparison.file_b.path);
    println!();
    println!("Differences:");
    println!("  Duration: {:.3}s", comparison.comparison.duration_difference);
    println!("  Sample Rate Match: {}", comparison.comparison.sample_rate_match);

    if let Some(tempo_diff) = comparison.comparison.tempo_difference {
        println!("  Tempo Difference: {:.1} BPM", tempo_diff);
    }

    if let Some(spectral_sim) = comparison.comparison.spectral_similarity {
        println!("  Spectral Similarity: {:.2}%", spectral_sim * 100.0);
    }

    Ok(())
}