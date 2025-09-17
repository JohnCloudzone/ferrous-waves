# Ferrous Waves

High-fidelity audio analysis bridge for development workflows. Analyze audio files, extract metrics, and integrate with AI tools through MCP.

## Features

- **Multi-format Support**: WAV, MP3, FLAC, OGG, M4A
- **Spectral Analysis**: FFT/STFT, mel-scale spectrograms, spectral features
- **Temporal Analysis**: Tempo detection, beat tracking, onset detection
- **Perceptual Metrics**: LUFS loudness measurement (EBU R 128), true peak detection, dynamic range analysis
- **Content Classification**: Speech/music/silence detection with confidence scores
- **Musical Analysis**: Key detection with confidence, chord progression, harmonic complexity
- **Quality Assessment**: SNR, THD, clipping detection, noise floor, and reliability scoring
- **Segment Analysis**: Temporal structure detection, pattern recognition, coherence analysis
- **Audio Fingerprinting**: Similarity detection, duplicate finding, content identification
- **Visualization**: Waveforms, spectrograms, power curves (base64 encoded)
- **MCP Integration**: Direct integration with AI assistants via Model Context Protocol
- **Content-based Caching**: Fast re-analysis with BLAKE3 hashing
- **CLI Interface**: Command-line tools for batch processing and automation

## Installation

```bash
# Clone the repository
git clone https://github.com/willibrandon/ferrous-waves
cd ferrous-waves

# Build the project
cargo build --release

# Run tests
cargo test
```

## Usage

### CLI Commands

```bash
# Analyze a single audio file
ferrous-waves analyze audio.mp3 --format json

# Extract tempo
ferrous-waves tempo song.wav --show-beats

# Detect onsets
ferrous-waves onsets track.flac --format csv

# Compare two audio files
ferrous-waves compare file1.mp3 file2.mp3

# Batch process multiple files
ferrous-waves batch ./music --pattern "*.mp3" --parallel 4

# Start MCP server
ferrous-waves serve
```

### Library Usage

```rust
use ferrous_waves::{AudioFile, AnalysisEngine};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load audio file
    let audio = AudioFile::load("song.mp3")?;

    // Create analysis engine
    let engine = AnalysisEngine::new();

    // Run analysis
    let result = engine.analyze(&audio).await?;

    // Access results
    println!("Tempo: {:?} BPM", result.temporal.tempo);
    println!("Peak amplitude: {}", result.summary.peak_amplitude);
    println!("Loudness: {:.1} LUFS", result.perceptual.loudness_lufs);
    println!("True peak: {:.1} dBFS", result.perceptual.true_peak_dbfs);
    println!("Content type: {:?}", result.classification.primary_type);
    println!("Audio quality score: {:.1}%", result.quality.overall_score * 100.0);
    println!("Fingerprint hash: {:016x}", result.fingerprint.perceptual_hash);

    Ok(())
}
```

### MCP Integration

Start the MCP server:
```bash
ferrous-waves serve
```

The server exposes three tools:

#### `analyze_audio` - Analyze audio files with configurable depth
Parameters:
- `file_path` (required): Path to audio file
- `analysis_profile`: Choose analysis depth (default: "standard")
  - `"quick"`: Basic metrics only (fastest)
  - `"standard"`: Core audio features
  - `"detailed"`: All analysis modules
  - `"fingerprint"`: Focus on similarity detection
  - `"mastering"`: Focus on loudness and quality metrics
- `return_format`: "summary" (default), "full", or "visual_only"
- `include_visuals`: Include base64-encoded images (default: false, WARNING: very large)
- `include_spectral`: Include spectral data arrays (default: false)
- `include_temporal`: Include temporal data arrays (default: false)
- `max_data_points`: Limit array sizes for pagination (default: 1000)
- `cursor`: Continue from previous response's next_cursor

Response format includes:
- Audio properties (format, duration, sample rate, channels)
- Content type and confidence scores
- Quality metrics (loudness LUFS, true peak, dynamic range)
- Issues categorized by severity (critical, warnings, info)
- Fingerprint data for similarity matching

#### `compare_audio` - Compare two audio files
Parameters:
- `file_a`, `file_b` (required): Paths to audio files
- `metrics`: Optional comparison metrics to calculate

Returns:
- Similarity scores: overall, fingerprint, spectral, perceptual, structural (0.0-1.0 scale)
- Differences: loudness delta, tempo delta, quality delta, key change
- Match type: Identical, Very Similar, Similar, Different
- Use case suggestions: "Same recording", "Different master", "Cover version", etc.

#### `get_job_status` - Check analysis job status
Parameters:
- `job_id` (required): Job ID from previous analysis

## Architecture

```
src/
├── audio/          # Audio file loading and buffer management
├── analysis/       # Core analysis engine
│   ├── spectral/   # FFT, STFT, mel-scale processing
│   ├── temporal/   # Beat tracking, onset detection
│   ├── perceptual.rs # LUFS, dynamic range, psychoacoustic metrics
│   ├── classification.rs # Speech/music/silence detection
│   ├── musical.rs  # Key detection, chord progression, harmonic analysis
│   ├── quality.rs  # Audio quality assessment and issue detection
│   ├── segments.rs # Segment-based temporal structure analysis
│   └── fingerprint.rs # Audio fingerprinting and similarity detection
├── visualization/  # Waveform and spectrogram generation
├── cache/          # Content-based caching system
├── mcp/           # MCP server implementation
└── cli/           # Command-line interface
```

## Configuration

Cache settings can be adjusted when creating an `AnalysisEngine`:

```rust
use ferrous_waves::{AnalysisEngine, Cache};
use std::time::Duration;

let cache = Cache::with_config(
    PathBuf::from(".cache"),
    1024 * 1024 * 100,  // 100MB max size
    Duration::from_secs(3600),  // 1 hour TTL
);

let engine = AnalysisEngine::new().with_cache(cache);
```

## Performance

- SIMD-optimized FFT operations via rustfft
- Parallel processing for batch operations
- Content-based caching reduces re-analysis time
- Async I/O for non-blocking operations

## Examples

The `examples/` directory contains runnable demonstrations:

```bash
# Generate sample audio files
cargo run --example generate_samples

# Basic analysis
cargo run --example basic_analysis

# Spectral analysis (FFT/STFT)
cargo run --example spectral_analysis

# Onset detection
cargo run --example onset_detection

# Perceptual metrics (LUFS, dynamic range)
cargo run --example perceptual_analysis

# Content classification (speech/music/silence)
cargo run --example content_classification

# Musical analysis (key detection, chords)
cargo run --example musical_analysis

# Audio quality assessment
cargo run --example quality_assessment

# Segment-based temporal analysis
cargo run --example segment_analysis

# Audio fingerprinting and similarity detection
cargo run --example fingerprint_similarity

# Compare two audio files
cargo run --example compare_files

# Cached analysis demonstration
cargo run --example cached_analysis

# Batch processing
cargo run --example batch_processing

# Envelope visualization (creates PNG)
cargo run --example envelope_visualization
```

See [examples/README.md](examples/README.md) for more details.

### Quick Start Example
```rust
use ferrous_waves::{AudioFile, AnalysisEngine};

let audio = AudioFile::load("song.mp3")?;
let engine = AnalysisEngine::new();
let result = engine.analyze(&audio).await?;

if let Some(tempo) = result.temporal.tempo {
    println!("BPM: {:.1}", tempo);
}
```

## Contributing

Pull requests welcome. Please ensure all tests pass and add tests for new features.

```bash
cargo test
cargo fmt -- --check
cargo clippy --all-targets --all-features -- -D warnings
```

## License

This project is licensed under the [MIT License](LICENSE).