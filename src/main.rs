use clap::{Parser, Subcommand};
use ferrous_waves::mcp::server::FerrousWavesMcp;
use ferrous_waves::audio::AudioFile;
use ferrous_waves::analysis::spectral::{FftProcessor, StftProcessor, WindowFunction};
use ferrous_waves::analysis::temporal::{OnsetDetector, BeatTracker};
use ferrous_waves::utils::error::Result;
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(
    name = "ferrous-waves",
    version,
    about = "High-fidelity audio analysis bridge for development workflows",
    long_about = None
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Enable verbose output
    #[arg(short, long, global = true)]
    verbose: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the MCP server for AI integration
    Serve {
        /// Port to listen on (stdio by default)
        #[arg(short, long)]
        port: Option<u16>,
    },

    /// Analyze an audio file
    Analyze {
        /// Path to the audio file
        file: String,

        /// Output format (json, text, summary)
        #[arg(short = 'f', long, default_value = "json")]
        format: String,

        /// Output file path (stdout by default)
        #[arg(short, long)]
        output: Option<String>,
    },

    /// Compare two audio files
    Compare {
        /// First audio file
        file_a: String,

        /// Second audio file
        file_b: String,

        /// Comparison metrics to calculate
        #[arg(short, long)]
        metrics: Vec<String>,
    },

    /// Extract tempo from an audio file
    Tempo {
        /// Path to the audio file
        file: String,
    },

    /// Detect onsets in an audio file
    Onsets {
        /// Path to the audio file
        file: String,

        /// Output format (json, text, csv)
        #[arg(short = 'f', long, default_value = "text")]
        format: String,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Set up logging
    let filter = if cli.verbose {
        EnvFilter::new("ferrous_waves=debug,rmcp=debug")
    } else {
        EnvFilter::from_default_env()
            .add_directive("ferrous_waves=info".parse().unwrap())
            .add_directive("rmcp=info".parse().unwrap())
    };

    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .init();

    match cli.command {
        Commands::Serve { port } => {
            if port.is_some() {
                tracing::warn!("Port option not yet implemented, using stdio");
            }
            tracing::info!("Starting Ferrous Waves MCP server...");
            let server = FerrousWavesMcp::new();
            server.start().await?;
        }

        Commands::Analyze { file, format, output } => {
            tracing::info!("Analyzing audio file: {}", file);

            let audio = AudioFile::load(&file)?;
            let mono = audio.buffer.to_mono();

            // Calculate basic metrics
            let peak = mono.iter()
                .map(|s| s.abs())
                .fold(0.0f32, |a, b| a.max(b));

            let rms = (mono.iter()
                .map(|s| s * s)
                .sum::<f32>() / mono.len() as f32)
                .sqrt();

            // Spectral analysis
            let fft = FftProcessor::new(2048);
            let stft = StftProcessor::new(2048, 512, WindowFunction::Hann);
            let stft_frames = stft.process(&mono);

            // Temporal analysis
            let onset_detector = OnsetDetector::new();
            let mut spectral_flux = Vec::new();
            let num_frames = stft_frames.shape()[0];

            for i in 1..num_frames {
                let curr_frame = stft_frames.row(i);
                let prev_frame = stft_frames.row(i - 1);

                let curr_mag = fft.magnitude_spectrum(curr_frame.as_slice().unwrap());
                let prev_mag = fft.magnitude_spectrum(prev_frame.as_slice().unwrap());

                let flux: f32 = curr_mag.iter()
                    .zip(prev_mag.iter())
                    .map(|(&c, &p)| (c - p).max(0.0).powi(2))
                    .sum();
                spectral_flux.push(flux.sqrt());
            }

            let onsets = onset_detector.detect_onsets(&spectral_flux, 512, audio.buffer.sample_rate);

            // Beat tracking
            let beat_tracker = BeatTracker::new();
            let tempo = beat_tracker.estimate_tempo(&onsets);

            let result = match format.as_str() {
                "json" => {
                    serde_json::json!({
                        "file": file,
                        "duration": audio.buffer.duration_seconds,
                        "sample_rate": audio.buffer.sample_rate,
                        "channels": audio.buffer.channels,
                        "format": format!("{:?}", audio.format),
                        "peak_amplitude": peak,
                        "rms_level": rms,
                        "dynamic_range": if rms > 0.0 { 20.0 * (peak / rms).log10() } else { 0.0 },
                        "tempo": tempo,
                        "onset_count": onsets.len(),
                    }).to_string()
                }
                "summary" => {
                    format!(
                        "File: {}\nDuration: {:.2}s\nTempo: {}\nOnsets: {}\nPeak: {:.3}\nRMS: {:.3}",
                        file,
                        audio.buffer.duration_seconds,
                        tempo.map_or("Unknown".to_string(), |t| format!("{:.1} BPM", t)),
                        onsets.len(),
                        peak,
                        rms
                    )
                }
                _ => {
                    format!(
                        "Audio Analysis Results\n{}\n\nFile: {}\nDuration: {:.2} seconds\nSample Rate: {} Hz\nChannels: {}\nFormat: {:?}\n\nAmplitude:\n  Peak: {:.3}\n  RMS: {:.3}\n  Dynamic Range: {:.1} dB\n\nTemporal:\n  Tempo: {}\n  Onsets Detected: {}",
                        "=".repeat(50),
                        file,
                        audio.buffer.duration_seconds,
                        audio.buffer.sample_rate,
                        audio.buffer.channels,
                        audio.format,
                        peak,
                        rms,
                        if rms > 0.0 { 20.0 * (peak / rms).log10() } else { 0.0 },
                        tempo.map_or("Unknown".to_string(), |t| format!("{:.1} BPM", t)),
                        onsets.len()
                    )
                }
            };

            if let Some(output_path) = output {
                std::fs::write(output_path, result)?;
            } else {
                println!("{}", result);
            }
        }

        Commands::Compare { file_a, file_b, metrics } => {
            tracing::info!("Comparing {} and {}", file_a, file_b);

            let audio_a = AudioFile::load(&file_a)?;
            let audio_b = AudioFile::load(&file_b)?;

            println!("Comparison Results");
            println!("{}", "=".repeat(50));
            println!("\nFile A: {}", file_a);
            println!("  Duration: {:.2}s", audio_a.buffer.duration_seconds);
            println!("  Sample Rate: {} Hz", audio_a.buffer.sample_rate);
            println!("  Channels: {}", audio_a.buffer.channels);

            println!("\nFile B: {}", file_b);
            println!("  Duration: {:.2}s", audio_b.buffer.duration_seconds);
            println!("  Sample Rate: {} Hz", audio_b.buffer.sample_rate);
            println!("  Channels: {}", audio_b.buffer.channels);

            println!("\nDifferences:");
            println!("  Duration: {:.2}s", audio_a.buffer.duration_seconds - audio_b.buffer.duration_seconds);
            println!("  Sample Rate Match: {}", audio_a.buffer.sample_rate == audio_b.buffer.sample_rate);

            if !metrics.is_empty() {
                println!("\nRequested metrics: {:?}", metrics);
                println!("(Advanced metrics not yet implemented)");
            }
        }

        Commands::Tempo { file } => {
            tracing::info!("Extracting tempo from: {}", file);

            let audio = AudioFile::load(&file)?;
            let mono = audio.buffer.to_mono();

            // Spectral flux for onset detection
            let fft = FftProcessor::new(2048);
            let stft = StftProcessor::new(2048, 512, WindowFunction::Hann);
            let stft_frames = stft.process(&mono);

            let mut spectral_flux = Vec::new();
            let num_frames = stft_frames.shape()[0];

            for i in 1..num_frames {
                let curr_frame = stft_frames.row(i);
                let prev_frame = stft_frames.row(i - 1);

                let curr_mag = fft.magnitude_spectrum(curr_frame.as_slice().unwrap());
                let prev_mag = fft.magnitude_spectrum(prev_frame.as_slice().unwrap());

                let flux: f32 = curr_mag.iter()
                    .zip(prev_mag.iter())
                    .map(|(&c, &p)| (c - p).max(0.0).powi(2))
                    .sum();
                spectral_flux.push(flux.sqrt());
            }

            let onset_detector = OnsetDetector::new();
            let onsets = onset_detector.detect_onsets(&spectral_flux, 512, audio.buffer.sample_rate);

            let beat_tracker = BeatTracker::new();
            if let Some(tempo) = beat_tracker.estimate_tempo(&onsets) {
                println!("{:.1} BPM", tempo);
            } else {
                println!("Unable to detect tempo");
            }
        }

        Commands::Onsets { file, format } => {
            tracing::info!("Detecting onsets in: {}", file);

            let audio = AudioFile::load(&file)?;
            let mono = audio.buffer.to_mono();

            // Spectral flux for onset detection
            let fft = FftProcessor::new(2048);
            let stft = StftProcessor::new(2048, 512, WindowFunction::Hann);
            let stft_frames = stft.process(&mono);

            let mut spectral_flux = Vec::new();
            let num_frames = stft_frames.shape()[0];

            for i in 1..num_frames {
                let curr_frame = stft_frames.row(i);
                let prev_frame = stft_frames.row(i - 1);

                let curr_mag = fft.magnitude_spectrum(curr_frame.as_slice().unwrap());
                let prev_mag = fft.magnitude_spectrum(prev_frame.as_slice().unwrap());

                let flux: f32 = curr_mag.iter()
                    .zip(prev_mag.iter())
                    .map(|(&c, &p)| (c - p).max(0.0).powi(2))
                    .sum();
                spectral_flux.push(flux.sqrt());
            }

            let onset_detector = OnsetDetector::new();
            let onsets = onset_detector.detect_onsets(&spectral_flux, 512, audio.buffer.sample_rate);

            match format.as_str() {
                "json" => {
                    let json = serde_json::json!({
                        "file": file,
                        "onset_count": onsets.len(),
                        "onset_times": onsets,
                    });
                    println!("{}", serde_json::to_string_pretty(&json)?);
                }
                "csv" => {
                    println!("time_seconds");
                    for onset in onsets {
                        println!("{:.4}", onset);
                    }
                }
                _ => {
                    println!("Onset Detection Results");
                    println!("{}", "=".repeat(50));
                    println!("File: {}", file);
                    println!("Total onsets detected: {}", onsets.len());
                    println!("\nOnset times (seconds):");
                    for (i, onset) in onsets.iter().enumerate() {
                        print!("{:8.3}", onset);
                        if (i + 1) % 8 == 0 {
                            println!();
                        }
                    }
                    println!();
                }
            }
        }
    }

    Ok(())
}