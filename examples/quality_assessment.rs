//! Example demonstrating audio quality assessment

use ferrous_waves::{AnalysisEngine, AudioFile};
use std::error::Error;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    println!("Audio Quality Assessment Example");
    println!("================================\n");

    // Analyze different audio files to demonstrate quality detection
    let files = vec![
        ("samples/test.wav", "Clean sine wave"),
        ("samples/drums.wav", "Drum pattern"),
        ("samples/music.wav", "Musical content"),
    ];

    let engine = AnalysisEngine::new();

    for (file_path, description) in files {
        println!("Analyzing: {} ({})", file_path, description);
        println!("----------------------------------------");

        let audio = AudioFile::load(file_path)?;
        let result = engine.analyze(&audio).await?;

        // Display quality assessment
        let quality = &result.quality;

        println!(
            "  Overall Quality Score: {:.0}%",
            quality.overall_score * 100.0
        );
        println!(
            "  Assessment Confidence: {:.0}%",
            quality.confidence * 100.0
        );

        // Display key metrics
        println!("\n  Key Metrics:");
        println!(
            "    Signal-to-Noise Ratio: {:.1} dB",
            quality.metrics.snr_db
        );
        println!(
            "    Total Harmonic Distortion: {:.2}%",
            quality.metrics.thd_percent
        );
        println!("    DC Offset: {:.4}", quality.metrics.dc_offset);
        println!(
            "    Clipping: {:.3}%",
            quality.metrics.clipping_ratio * 100.0
        );
        println!("    Noise Floor: {:.1} dB", quality.metrics.noise_floor_db);
        println!(
            "    Dynamic Range: {:.1} dB",
            quality.metrics.dynamic_range_db
        );
        println!(
            "    Frequency Response Score: {:.0}%",
            quality.metrics.frequency_response_score * 100.0
        );
        println!(
            "    Phase Coherence: {:.0}%",
            quality.metrics.phase_coherence * 100.0
        );
        println!(
            "    Aliasing Score: {:.1}%",
            quality.metrics.aliasing_score * 100.0
        );
        println!("    Dropouts Detected: {}", quality.metrics.dropout_count);

        // Sample rate and bit depth quality
        println!("\n  Technical Quality:");
        println!(
            "    Sample Rate Quality: {:?}",
            quality.metrics.sample_rate_quality
        );
        println!(
            "    Bit Depth Quality: {:?}",
            quality.metrics.bit_depth_quality
        );

        // Display detected issues
        if !quality.issues.is_empty() {
            println!("\n  Detected Issues:");
            for issue in &quality.issues {
                println!("    [{:?}] {}", issue.severity, issue.description);
                if !issue.impact.is_empty() {
                    println!("      Impact: {}", issue.impact);
                }
                if !issue.time_ranges.is_empty() && issue.time_ranges.len() <= 3 {
                    for (start, end) in &issue.time_ranges {
                        println!("      Occurs at: {:.2}s - {:.2}s", start, end);
                    }
                }
            }
        } else {
            println!("\n  No quality issues detected!");
        }

        // Display recommendations
        if !quality.recommendations.is_empty() {
            println!("\n  Recommendations:");
            for rec in &quality.recommendations {
                println!("    • {}", rec);
            }
        }

        println!();
    }

    // Demonstrate quality impact on analysis
    println!("\nQuality Impact on Analysis:");
    println!("===========================");
    println!("High Quality (>80%):");
    println!("  - All analysis features reliable");
    println!("  - Accurate pitch and key detection");
    println!("  - Precise onset and beat tracking");
    println!("\nMedium Quality (50-80%):");
    println!("  - Most features reliable");
    println!("  - Some spectral features may be affected");
    println!("  - Temporal analysis still accurate");
    println!("\nLow Quality (<50%):");
    println!("  - Analysis results should be verified");
    println!("  - Spectral features may be unreliable");
    println!("  - Consider preprocessing or re-recording");

    // Quality issues reference
    println!("\nCommon Quality Issues:");
    println!("======================");
    println!("Clipping:");
    println!("  - Causes: Input gain too high");
    println!("  - Impact: Distortion, affects all frequency analysis");
    println!("  - Fix: Reduce input gain before recording");
    println!("\nHigh Noise Floor:");
    println!("  - Causes: Poor recording environment, low-quality equipment");
    println!("  - Impact: Reduces pitch detection accuracy");
    println!("  - Fix: Use noise reduction or better recording setup");
    println!("\nDC Offset:");
    println!("  - Causes: Hardware issues, poor ADC");
    println!("  - Impact: Reduces headroom, may affect some algorithms");
    println!("  - Fix: Apply high-pass filter at 20Hz");
    println!("\nDropouts:");
    println!("  - Causes: Buffer underruns, corrupted data");
    println!("  - Impact: Discontinuities affect temporal analysis");
    println!("  - Fix: Check audio interface settings, increase buffer size");

    Ok(())
}
