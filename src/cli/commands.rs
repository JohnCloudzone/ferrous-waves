use crate::analysis::spectral::WindowFunction;
use crate::cache::Cache;
use crate::mcp::server::FerrousWavesMcp;
use crate::utils::error::Result;
use crate::{AnalysisEngine, AudioFile};
use futures::stream::{self, StreamExt};
use glob::glob;
use std::fs;
use std::path::PathBuf;

pub async fn run_serve(port: u16, host: String, cache_enabled: bool) -> Result<()> {
    println!("Starting Ferrous Waves MCP server on {}:{}...", host, port);

    let server = if cache_enabled {
        FerrousWavesMcp::with_cache(Cache::new())
    } else {
        FerrousWavesMcp::new()
    };

    server.start().await?;
    Ok(())
}

pub async fn run_analyze(
    file: PathBuf,
    output: Option<PathBuf>,
    format: String,
    fft_size: usize,
    hop_size: usize,
    no_cache: bool,
) -> Result<()> {
    println!("Analyzing {}...", file.display());

    let audio = AudioFile::load(&file)?;

    let engine = if no_cache {
        AnalysisEngine::with_config(fft_size, hop_size, WindowFunction::Hann).without_cache()
    } else {
        AnalysisEngine::with_config(fft_size, hop_size, WindowFunction::Hann)
    };

    let result = engine.analyze(&audio).await?;

    // Output results based on format
    match format.as_str() {
        "json" => {
            let json = serde_json::to_string_pretty(&result)?;
            if let Some(output_path) = output {
                fs::create_dir_all(&output_path)?;
                let json_path = output_path.join("analysis.json");
                fs::write(json_path, json)?;
                println!("Analysis saved to {}", output_path.display());
            } else {
                println!("{}", json);
            }
        }
        "text" => {
            println!("\n=== Audio Analysis Results ===\n");
            println!("Duration: {:.2} seconds", result.summary.duration);
            println!("Sample Rate: {} Hz", result.summary.sample_rate);
            println!("Channels: {}", result.summary.channels);
            println!("Format: {}", result.summary.format);
            println!("Peak Amplitude: {:.3}", result.summary.peak_amplitude);
            println!("RMS Level: {:.3}", result.summary.rms_level);
            println!("Dynamic Range: {:.1} dB", result.summary.dynamic_range);

            if let Some(tempo) = result.temporal.tempo {
                println!("\nTempo: {:.1} BPM", tempo);
                println!(
                    "Tempo Stability: {:.1}%",
                    result.temporal.tempo_stability * 100.0
                );
            }

            println!("Onsets Detected: {}", result.temporal.onsets.len());
            println!(
                "Rhythmic Complexity: {:.2}",
                result.temporal.rhythmic_complexity
            );

            if !result.insights.is_empty() {
                println!("\nInsights:");
                for insight in &result.insights {
                    println!("  • {}", insight);
                }
            }

            if !result.recommendations.is_empty() {
                println!("\nRecommendations:");
                for rec in &result.recommendations {
                    println!("  • {}", rec);
                }
            }
        }
        "visual" => {
            if let Some(output_path) = output {
                fs::create_dir_all(&output_path)?;

                // Save visualizations from base64
                if let Some(waveform) = result.visuals.waveform {
                    use base64::Engine;
                    let data = base64::engine::general_purpose::STANDARD
                        .decode(waveform)
                        .map_err(|e| crate::utils::error::FerrousError::Analysis(e.to_string()))?;
                    fs::write(output_path.join("waveform.png"), data)?;
                }

                if let Some(spectrogram) = result.visuals.spectrogram {
                    use base64::Engine;
                    let data = base64::engine::general_purpose::STANDARD
                        .decode(spectrogram)
                        .map_err(|e| crate::utils::error::FerrousError::Analysis(e.to_string()))?;
                    fs::write(output_path.join("spectrogram.png"), data)?;
                }

                if let Some(power_curve) = result.visuals.power_curve {
                    use base64::Engine;
                    let data = base64::engine::general_purpose::STANDARD
                        .decode(power_curve)
                        .map_err(|e| crate::utils::error::FerrousError::Analysis(e.to_string()))?;
                    fs::write(output_path.join("power_curve.png"), data)?;
                }

                println!("Visualizations saved to {}", output_path.display());
            } else {
                println!("Please specify an output directory for visualizations");
            }
        }
        _ => {
            println!("Unknown format: {}. Using json.", format);
            let json = serde_json::to_string_pretty(&result)?;
            println!("{}", json);
        }
    }

    Ok(())
}

pub async fn run_compare(file_a: PathBuf, file_b: PathBuf, format: String) -> Result<()> {
    println!("Comparing {} and {}...", file_a.display(), file_b.display());

    let audio_a = AudioFile::load(&file_a)?;
    let audio_b = AudioFile::load(&file_b)?;

    let engine = AnalysisEngine::new();
    let comparison = engine.compare(&audio_a, &audio_b).await;

    match format.as_str() {
        "json" => {
            let json = serde_json::to_string_pretty(&comparison)?;
            println!("{}", json);
        }
        "text" => {
            println!("\n=== Audio Comparison Results ===\n");
            println!("File A: {}", comparison.file_a.path);
            println!("  Duration: {:.2}s", comparison.file_a.duration);
            println!("  Sample Rate: {} Hz", comparison.file_a.sample_rate);
            println!("  Channels: {}", comparison.file_a.channels);
            if let Some(tempo) = comparison.file_a.tempo {
                println!("  Tempo: {:.1} BPM", tempo);
            }

            println!("\nFile B: {}", comparison.file_b.path);
            println!("  Duration: {:.2}s", comparison.file_b.duration);
            println!("  Sample Rate: {} Hz", comparison.file_b.sample_rate);
            println!("  Channels: {}", comparison.file_b.channels);
            if let Some(tempo) = comparison.file_b.tempo {
                println!("  Tempo: {:.1} BPM", tempo);
            }

            println!("\nComparison:");
            println!(
                "  Duration Difference: {:.2}s",
                comparison.comparison.duration_difference
            );
            println!(
                "  Sample Rate Match: {}",
                comparison.comparison.sample_rate_match
            );
            if let Some(tempo_diff) = comparison.comparison.tempo_difference {
                println!("  Tempo Difference: {:.1} BPM", tempo_diff);
            }
        }
        _ => {
            let json = serde_json::to_string_pretty(&comparison)?;
            println!("{}", json);
        }
    }

    Ok(())
}

pub async fn run_tempo(file: PathBuf, show_beats: bool) -> Result<()> {
    println!("Detecting tempo in {}...", file.display());

    let audio = AudioFile::load(&file)?;
    let engine = AnalysisEngine::new();
    let result = engine.analyze(&audio).await?;

    if let Some(tempo) = result.temporal.tempo {
        println!("Tempo: {:.1} BPM", tempo);
        println!(
            "Tempo Stability: {:.1}%",
            result.temporal.tempo_stability * 100.0
        );

        if show_beats {
            println!("\nBeat positions (seconds):");
            for (i, beat) in result.temporal.beats.iter().enumerate() {
                println!("  Beat {}: {:.3}s", i + 1, beat);
                if i >= 19 {
                    // Show first 20 beats
                    println!("  ... ({} more beats)", result.temporal.beats.len() - 20);
                    break;
                }
            }
        }
    } else {
        println!("Could not detect tempo in this file");
    }

    Ok(())
}

pub async fn run_onsets(file: PathBuf, format: String) -> Result<()> {
    println!("Detecting onsets in {}...", file.display());

    let audio = AudioFile::load(&file)?;
    let engine = AnalysisEngine::new();
    let result = engine.analyze(&audio).await?;

    let onsets = &result.temporal.onsets;

    match format.as_str() {
        "json" => {
            let json = serde_json::to_string_pretty(&onsets)?;
            println!("{}", json);
        }
        "csv" => {
            println!("onset_time");
            for onset in onsets {
                println!("{:.6}", onset);
            }
        }
        _ => {
            println!("Found {} onsets:", onsets.len());
            for (i, onset) in onsets.iter().enumerate() {
                println!("  Onset {}: {:.3}s", i + 1, onset);
                if i >= 49 {
                    // Show first 50 onsets
                    println!("  ... ({} more onsets)", onsets.len() - 50);
                    break;
                }
            }
        }
    }

    Ok(())
}

pub async fn run_batch(
    directory: PathBuf,
    pattern: String,
    output: Option<PathBuf>,
    parallel: usize,
) -> Result<()> {
    println!("Batch analyzing files in {}...", directory.display());

    let search_pattern = directory.join(&pattern);
    let paths: Vec<PathBuf> = glob(search_pattern.to_str().unwrap())
        .map_err(|e| {
            crate::utils::error::FerrousError::Io(std::io::Error::other(e))
        })?
        .filter_map(|r| r.ok())
        .collect();

    println!("Found {} files to analyze", paths.len());

    let engine = AnalysisEngine::new();
    let output_dir = output.unwrap_or_else(|| directory.join("analysis_results"));
    fs::create_dir_all(&output_dir)?;

    stream::iter(paths)
        .map(|path| {
            let engine = engine.clone();
            let output_dir = output_dir.clone();
            async move {
                match AudioFile::load(&path) {
                    Ok(audio) => match engine.analyze(&audio).await {
                        Ok(result) => {
                            let file_stem = path
                                .file_stem()
                                .and_then(|s| s.to_str())
                                .unwrap_or("unknown");
                            let json_path = output_dir.join(format!("{}.json", file_stem));
                            let json = serde_json::to_string_pretty(&result)?;
                            fs::write(json_path, json)?;
                            println!("✓ Analyzed: {}", path.display());
                            Ok::<_, crate::utils::error::FerrousError>(())
                        }
                        Err(e) => {
                            println!("✗ Failed to analyze {}: {}", path.display(), e);
                            Ok(())
                        }
                    },
                    Err(e) => {
                        println!("✗ Failed to load {}: {}", path.display(), e);
                        Ok(())
                    }
                }
            }
        })
        .buffer_unordered(parallel)
        .collect::<Vec<_>>()
        .await;

    println!(
        "\nBatch analysis complete. Results saved to {}",
        output_dir.display()
    );

    Ok(())
}

pub fn run_clear_cache(confirm: bool) -> Result<()> {
    if !confirm {
        println!("Please confirm cache clearing with --confirm flag");
        return Ok(());
    }

    let cache = Cache::new();
    cache.clear()?;
    println!("Cache cleared successfully");

    Ok(())
}

pub fn run_cache_stats() -> Result<()> {
    let cache = Cache::new();
    let stats = cache.stats();

    println!("=== Cache Statistics ===");
    println!("Total Entries: {}", stats.total_entries);
    println!(
        "Total Size: {:.2} MB",
        stats.total_size_bytes as f64 / 1_048_576.0
    );
    println!(
        "Max Size: {:.2} GB",
        stats.max_size_bytes as f64 / 1_073_741_824.0
    );
    println!("Cache Directory: {}", stats.directory.display());

    let usage_percent = (stats.total_size_bytes as f64 / stats.max_size_bytes as f64) * 100.0;
    println!("Usage: {:.1}%", usage_percent);

    Ok(())
}
