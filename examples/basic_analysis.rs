use ferrous_waves::{AnalysisEngine, AudioFile, Result};

#[tokio::main]
async fn main() -> Result<()> {
    // Load an audio file
    let audio = AudioFile::load("samples/test.wav")?;

    // Create analysis engine with default settings
    let engine = AnalysisEngine::new();

    // Perform analysis
    let result = engine.analyze(&audio).await?;

    // Print summary
    println!("Audio Analysis Summary:");
    println!("  Duration: {:.2}s", result.summary.duration);
    println!("  Sample Rate: {}Hz", result.summary.sample_rate);
    println!("  Channels: {}", result.summary.channels);
    println!("  Peak Amplitude: {:.3}", result.summary.peak_amplitude);
    println!("  RMS Level: {:.3}", result.summary.rms_level);

    if let Some(tempo) = result.temporal.tempo {
        println!("  Estimated Tempo: {:.1} BPM", tempo);
    }

    println!("  Onsets Detected: {}", result.temporal.onsets.len());

    Ok(())
}