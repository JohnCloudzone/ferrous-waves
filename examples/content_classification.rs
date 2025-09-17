//! Example demonstrating content classification (speech/music/silence detection)

use ferrous_waves::{AnalysisEngine, AudioFile};
use std::error::Error;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    println!("Content Classification Example");
    println!("==============================\n");

    // Analyze different audio files to show classification
    let files = vec![
        ("samples/test.wav", "Pure tone"),
        ("samples/music.wav", "C major chord"),
        ("samples/drums.wav", "Drum pattern"),
    ];

    let engine = AnalysisEngine::new();

    for (file_path, description) in files {
        println!("Analyzing: {} ({})", file_path, description);
        println!("----------------------------------------");

        let audio = AudioFile::load(file_path)?;
        let result = engine.analyze(&audio).await?;

        // Display classification results
        let classification = &result.classification;

        println!("  Primary Type: {:?}", classification.primary_type);
        println!("  Confidence: {:.1}%", classification.confidence * 100.0);
        println!("\n  Detailed Scores:");
        println!("    Speech: {:.1}%", classification.scores.speech * 100.0);
        println!("    Music:  {:.1}%", classification.scores.music * 100.0);
        println!("    Silence: {:.1}%", classification.scores.silence * 100.0);

        // Show key features used for classification
        println!("\n  Classification Features:");
        println!(
            "    Zero Crossing Rate: {:.4} (σ={:.4})",
            classification.features.zcr_mean, classification.features.zcr_std
        );
        println!(
            "    Spectral Rolloff: {:.0} Hz",
            classification.features.spectral_rolloff_mean
        );
        println!(
            "    Energy Level: {:.4} (σ={:.4})",
            classification.features.energy_mean, classification.features.energy_std
        );
        println!(
            "    Low Energy Rate: {:.1}%",
            classification.features.low_energy_rate * 100.0
        );
        println!(
            "    Harmonic-to-Noise Ratio: {:.3}",
            classification.features.hnr
        );

        // Show temporal segments if available
        if !classification.segments.is_empty() {
            println!("\n  Temporal Segments:");
            for (i, segment) in classification.segments.iter().enumerate().take(5) {
                println!(
                    "    Segment {}: {:.1}s-{:.1}s - {:?} ({:.0}%)",
                    i + 1,
                    segment.start_time,
                    segment.end_time,
                    segment.content_type,
                    segment.confidence * 100.0
                );
            }
            if classification.segments.len() > 5 {
                println!(
                    "    ... and {} more segments",
                    classification.segments.len() - 5
                );
            }
        }

        println!();
    }

    println!("\nClassification Guidelines:");
    println!("==========================");
    println!("Speech Characteristics:");
    println!("  - Higher zero crossing rate (consonants)");
    println!("  - More variable energy (pauses between words)");
    println!("  - Lower spectral rolloff (energy in speech frequencies)");
    println!("  - Higher harmonic-to-noise ratio (vocal harmonics)");
    println!("\nMusic Characteristics:");
    println!("  - More stable energy levels");
    println!("  - Higher spectral flux (timbral variation)");
    println!("  - Wider frequency range");
    println!("  - Presence of rhythm/beats");
    println!("\nMixed Content:");
    println!("  - Songs with vocals");
    println!("  - Podcasts with background music");
    println!("  - Content with alternating speech and music");

    Ok(())
}
