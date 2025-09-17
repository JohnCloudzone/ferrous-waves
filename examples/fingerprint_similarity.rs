//! Example demonstrating audio fingerprinting and similarity detection

use ferrous_waves::analysis::fingerprint::{
    FingerprintDatabase, FingerprintGenerator, FingerprintMatcher,
};
use ferrous_waves::{AnalysisEngine, AudioFile};
use std::error::Error;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    println!("Audio Fingerprinting and Similarity Detection Example");
    println!("====================================================\n");

    // Analyze files and generate fingerprints
    let files = vec![
        ("samples/drums.wav", "Drum pattern"),
        ("samples/music.wav", "Musical content"),
        ("samples/test.wav", "Test signal"),
    ];

    let engine = AnalysisEngine::new();
    let generator = FingerprintGenerator::new(44100.0);
    let mut database = FingerprintDatabase::new();

    println!("Generating fingerprints...");
    println!("-------------------------");

    for (file_path, description) in &files {
        println!("\nAnalyzing: {} ({})", file_path, description);

        let audio = AudioFile::load(file_path)?;
        let result = engine.analyze(&audio).await?;

        // Display fingerprint info
        let fingerprint = &result.fingerprint;

        println!("  Fingerprint Details:");
        println!("    Perceptual hash: {:016x}", fingerprint.perceptual_hash);
        println!("    Spectral hashes: {}", fingerprint.spectral_hashes.len());
        println!("    Landmarks: {}", fingerprint.landmarks.len());
        println!(
            "    Sub-fingerprints: {}",
            fingerprint.sub_fingerprints.len()
        );
        println!(
            "    Compact size: {} bytes",
            fingerprint.fingerprint.len() * 8
        );

        // Show dominant frequencies
        if !fingerprint.metadata.dominant_frequencies.is_empty() {
            println!("    Dominant frequencies:");
            for freq in fingerprint.metadata.dominant_frequencies.iter().take(3) {
                println!("      {:.1} Hz", freq);
            }
        }

        // Show landmark types
        let spectral_peaks = fingerprint
            .landmarks
            .iter()
            .filter(|l| {
                matches!(
                    l.landmark_type,
                    ferrous_waves::analysis::fingerprint::LandmarkType::SpectralPeak
                )
            })
            .count();
        let onsets = fingerprint
            .landmarks
            .iter()
            .filter(|l| {
                matches!(
                    l.landmark_type,
                    ferrous_waves::analysis::fingerprint::LandmarkType::OnsetEvent
                )
            })
            .count();

        println!("    Landmark breakdown:");
        println!("      Spectral peaks: {}", spectral_peaks);
        println!("      Onset events: {}", onsets);

        // Add to database
        database.insert(file_path.to_string(), fingerprint.clone());
    }

    // Compare fingerprints
    println!("\n\nSimilarity Comparison:");
    println!("======================");

    let matcher = FingerprintMatcher::new();

    for i in 0..files.len() {
        for j in i + 1..files.len() {
            let (file_a, desc_a) = files[i];
            let (file_b, desc_b) = files[j];

            let audio_a = AudioFile::load(file_a)?;
            let audio_b = AudioFile::load(file_b)?;

            let fp_a = generator.generate(&audio_a.buffer.to_mono())?;
            let fp_b = generator.generate(&audio_b.buffer.to_mono())?;

            let match_result = matcher.compare(&fp_a, &fp_b);

            println!("\n{} vs {}", desc_a, desc_b);
            println!(
                "  Overall similarity: {:.1}%",
                match_result.similarity * 100.0
            );
            println!("  Match type: {:?}", match_result.match_type);
            println!("  Confidence: {:.1}%", match_result.confidence * 100.0);

            println!("  Detailed scores:");
            println!("    Spectral: {:.1}%", match_result.scores.spectral * 100.0);
            println!("    Temporal: {:.1}%", match_result.scores.temporal * 100.0);
            println!("    Energy: {:.1}%", match_result.scores.energy * 100.0);
            println!("    Landmark: {:.1}%", match_result.scores.landmark * 100.0);
            println!(
                "    Perceptual: {:.1}%",
                match_result.scores.perceptual * 100.0
            );

            if !match_result.matched_segments.is_empty() {
                println!(
                    "  Matched segments: {}",
                    match_result.matched_segments.len()
                );
                for (idx, segment) in match_result.matched_segments.iter().enumerate().take(3) {
                    println!(
                        "    {}. [{:.1}s] ↔ [{:.1}s] (quality: {:.1}%)",
                        idx + 1,
                        segment.time_a,
                        segment.time_b,
                        segment.quality * 100.0
                    );
                }
            }

            if let Some(offset) = match_result.time_offset {
                println!("  Time offset detected: {:.2}s", offset);
            }
        }
    }

    // Database search demonstration
    println!("\n\nDatabase Search:");
    println!("================");

    // Search with the first file
    if let Some((query_file, query_desc)) = files.first() {
        let audio = AudioFile::load(query_file)?;
        let query_fp = generator.generate(&audio.buffer.to_mono())?;

        println!("Searching for: {} in database", query_desc);

        let results = database.search(&query_fp, 0.3);

        println!("Found {} matches:", results.len());
        for (id, match_result) in results {
            println!(
                "  - {} (similarity: {:.1}%, type: {:?})",
                id,
                match_result.similarity * 100.0,
                match_result.match_type
            );
        }
    }

    // Self-similarity test
    println!("\n\nSelf-Similarity Test:");
    println!("=====================");

    let test_audio = AudioFile::load("samples/test.wav")?;
    let fp1 = generator.generate(&test_audio.buffer.to_mono())?;
    let fp2 = generator.generate(&test_audio.buffer.to_mono())?;

    let self_match = matcher.compare(&fp1, &fp2);

    println!("Same audio compared to itself:");
    println!("  Similarity: {:.1}%", self_match.similarity * 100.0);
    println!("  Match type: {:?}", self_match.match_type);
    println!("  Expected: >99% similarity for identical audio");

    // Reference information
    println!("\n\nFingerprinting Reference:");
    println!("========================");
    println!("Match Types:");
    println!("  Identical: >95% similarity");
    println!("  Very Similar: 85-95% similarity");
    println!("  Similar: 70-85% similarity");
    println!("  Partially Similar: 50-70% similarity");
    println!("  Different: <50% similarity");
    println!("\nUse Cases:");
    println!("  - Duplicate detection in music libraries");
    println!("  - Copyright and content identification");
    println!("  - Version tracking (remixes, covers)");
    println!("  - Audio synchronization and alignment");
    println!("  - Partial matching for samples and loops");
    println!("\nFingerprint Components:");
    println!("  - Spectral hashes: Frequency pattern encoding");
    println!("  - Landmarks: Significant acoustic events");
    println!("  - Perceptual hash: Overall audio signature");
    println!("  - Sub-fingerprints: Partial matching support");

    Ok(())
}
