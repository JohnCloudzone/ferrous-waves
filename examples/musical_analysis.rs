//! Example demonstrating musical analysis with enhanced key detection

use ferrous_waves::{AnalysisEngine, AudioFile};
use std::error::Error;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    println!("Musical Analysis Example");
    println!("========================\n");

    // Analyze different audio files to show musical features
    let files = vec![
        ("samples/music.wav", "C major chord"),
        ("samples/test.wav", "440Hz sine wave"),
        ("samples/drums.wav", "Drum pattern"),
    ];

    let engine = AnalysisEngine::new();

    for (file_path, description) in files {
        println!("Analyzing: {} ({})", file_path, description);
        println!("----------------------------------------");

        let audio = AudioFile::load(file_path)?;
        let result = engine.analyze(&audio).await?;

        // Display musical key detection
        let musical = &result.musical;

        println!("  Key Detection:");
        println!("    Detected Key: {}", musical.key.key);
        println!("    Root Note: {}", musical.key.root);
        println!("    Mode: {:?}", musical.key.mode);
        println!("    Confidence: {:.1}%", musical.key.confidence * 100.0);

        // Show alternative key interpretations
        if !musical.key.alternatives.is_empty() {
            println!("\n  Alternative Keys:");
            for (i, alt) in musical.key.alternatives.iter().take(3).enumerate() {
                println!("    {}. {} (score: {:.2})", i + 1, alt.key, alt.score);
            }
        }

        // Show chroma vector (pitch class distribution)
        println!("\n  Pitch Class Distribution (Chroma):");
        let notes = [
            "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
        ];
        for (note, &value) in notes.iter().zip(&musical.chroma_vector.values) {
            let bar_length = (value * 40.0) as usize;
            let bar = "█".repeat(bar_length);
            println!("    {:2}: {:>4.1}% {}", note, value * 100.0, bar);
        }

        // Musical characteristics
        println!("\n  Musical Characteristics:");
        println!("    Tonality Strength: {:.1}%", musical.tonality * 100.0);
        println!("    Mode Clarity: {:.1}%", musical.mode_clarity * 100.0);
        println!(
            "    Harmonic Complexity: {:.1}%",
            musical.harmonic_complexity * 100.0
        );

        // Chord progression if detected
        if let Some(ref progression) = musical.chord_progression {
            if !progression.chords.is_empty() {
                println!("\n  Detected Chords:");
                for (i, chord) in progression.chords.iter().take(5).enumerate() {
                    println!(
                        "    {:>4.1}s: {} (confidence: {:.0}%)",
                        chord.start_time,
                        chord.chord,
                        chord.confidence * 100.0
                    );
                    if i == 4 && progression.chords.len() > 5 {
                        println!("    ... and {} more", progression.chords.len() - 5);
                    }
                }
            }
        }

        // Time signature
        if let Some(ref time_sig) = musical.time_signature {
            println!(
                "\n  Time Signature: {}/{} (confidence: {:.0}%)",
                time_sig.numerator,
                time_sig.denominator,
                time_sig.confidence * 100.0
            );
        }

        println!();
    }

    println!("\nMusical Analysis Reference:");
    println!("===========================");
    println!("Key Detection:");
    println!("  - Uses Krumhansl-Kessler key profiles");
    println!("  - Analyzes pitch class distribution (chroma)");
    println!("  - Provides confidence scores and alternatives");
    println!("\nTonality Strength:");
    println!("  - How well the audio fits a tonal profile");
    println!("  - High = clear tonal center, Low = atonal/noise");
    println!("\nMode Clarity:");
    println!("  - How clearly major vs minor is defined");
    println!("  - Based on third and fifth degree analysis");
    println!("\nHarmonic Complexity:");
    println!("  - Entropy of pitch class distribution");
    println!("  - High = many different notes, Low = few notes");

    Ok(())
}
