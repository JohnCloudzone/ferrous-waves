//! Example demonstrating segment-based temporal analysis

use ferrous_waves::{AnalysisEngine, AudioFile};
use std::error::Error;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    println!("Segment-Based Temporal Analysis Example");
    println!("=======================================\n");

    // Analyze different audio files to show temporal structure
    let files = vec![
        ("samples/music.wav", "Musical content"),
        ("samples/drums.wav", "Drum pattern"),
        ("samples/test.wav", "Test signal"),
    ];

    let engine = AnalysisEngine::new();

    for (file_path, description) in files {
        println!("Analyzing: {} ({})", file_path, description);
        println!("----------------------------------------");

        let audio = AudioFile::load(file_path)?;
        let result = engine.analyze(&audio).await?;

        // Display segment analysis
        let segments = &result.segments;

        println!("  Temporal Overview:");
        println!("    Total segments: {}", segments.segments.len());
        println!("    Structural sections: {}", segments.structure.len());
        println!(
            "    Temporal complexity: {:.1}%",
            segments.temporal_complexity * 100.0
        );
        println!(
            "    Segment coherence: {:.1}%",
            segments.coherence_score * 100.0
        );

        // Display segments
        println!("\n  Segment Timeline:");
        for (i, segment) in segments.segments.iter().enumerate().take(5) {
            println!(
                "    [{:>5.2}s - {:>5.2}s] {:?} (energy: {:.2}, content: {:?})",
                segment.start_time,
                segment.start_time + segment.duration,
                segment.label,
                segment.energy,
                segment.content_type
            );
            if i == 4 && segments.segments.len() > 5 {
                println!("    ... and {} more segments", segments.segments.len() - 5);
            }
        }

        // Display structural sections
        if !segments.structure.is_empty() {
            println!("\n  Structural Sections:");
            for section in &segments.structure {
                println!(
                    "    {:?}: {:.2}s - {:.2}s (avg energy: {:.2})",
                    section.section_type, section.start_time, section.end_time, section.avg_energy
                );
            }
        }

        // Display temporal patterns
        println!("\n  Temporal Patterns:");
        println!(
            "    Energy profile shape: {:?}",
            segments.patterns.energy_profile.shape
        );

        if !segments.patterns.energy_profile.peaks.is_empty() {
            println!("    Energy peaks:");
            for (time, energy) in segments.patterns.energy_profile.peaks.iter().take(3) {
                println!("      {:.2}s: {:.2}", time, energy);
            }
        }

        if !segments.patterns.repetitions.is_empty() {
            println!(
                "    Repetition patterns: {}",
                segments.patterns.repetitions.len()
            );
            for pattern in segments.patterns.repetitions.iter().take(2) {
                println!(
                    "      Pattern length: {} segments, similarity: {:.1}%",
                    pattern.length,
                    pattern.similarity * 100.0
                );
            }
        }

        if !segments.patterns.periodic_events.is_empty() {
            println!("    Periodic events:");
            for event in &segments.patterns.periodic_events {
                println!(
                    "      {}: period = {:.2}s, strength = {:.1}%",
                    event.event_type,
                    event.period,
                    event.strength * 100.0
                );
            }
        }

        // Display transitions
        if !segments.transitions.is_empty() {
            println!("\n  Transitions:");
            for (i, transition) in segments.transitions.iter().enumerate().take(3) {
                println!(
                    "    {:.2}s: {:?} (strength: {:.1}%)",
                    transition.time,
                    transition.transition_type,
                    transition.strength * 100.0
                );
                if i == 2 && segments.transitions.len() > 3 {
                    println!(
                        "    ... and {} more transitions",
                        segments.transitions.len() - 3
                    );
                }
            }
        }

        // Display tension profile
        if !segments.patterns.tension_profile.is_empty() {
            println!("\n  Tension Profile:");
            let buildups = segments
                .patterns
                .tension_profile
                .iter()
                .filter(|t| {
                    matches!(
                        t.change_type,
                        ferrous_waves::analysis::segments::TensionChange::BuildUp
                    )
                })
                .count();
            let releases = segments
                .patterns
                .tension_profile
                .iter()
                .filter(|t| {
                    matches!(
                        t.change_type,
                        ferrous_waves::analysis::segments::TensionChange::Release
                    )
                })
                .count();
            println!("    Build-ups: {}", buildups);
            println!("    Releases: {}", releases);
        }

        println!();
    }

    // Reference information
    println!("\nSegment Analysis Reference:");
    println!("==========================");
    println!("Temporal Complexity:");
    println!("  0-30%: Simple, repetitive structure");
    println!("  30-70%: Moderate variation and development");
    println!("  70-100%: Complex, highly varied structure");
    println!("\nSegment Coherence:");
    println!("  0-50%: Abrupt changes, contrasting sections");
    println!("  50-80%: Balanced transitions");
    println!("  80-100%: Smooth, gradual transitions");
    println!("\nEnergy Shapes:");
    println!("  Flat: Consistent energy throughout");
    println!("  Increasing: Build-up or crescendo");
    println!("  Decreasing: Fade-out or diminuendo");
    println!("  Oscillating: Regular energy variations");
    println!("\nSegment Labels:");
    println!("  Musical: Intro, Verse, Chorus, Bridge, Outro");
    println!("  Electronic: BuildUp, Drop, Break");
    println!("  General: Transition, Silence, Speech, Ambient");

    Ok(())
}
