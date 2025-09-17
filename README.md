# Ferrous Waves

High-fidelity audio analysis bridge for development workflows. Analyze audio files, extract metrics, and integrate with AI tools through MCP.

## Features

- **Multi-format Support**: WAV, MP3, FLAC, OGG, M4A
- **Spectral Analysis**: FFT/STFT, mel-scale spectrograms, spectral features
- **Temporal Analysis**: Tempo detection, beat tracking, onset detection
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

    Ok(())
}
```

### MCP Integration

Start the MCP server:
```bash
ferrous-waves serve
```

The server exposes three tools:
- `analyze_audio` - Full audio analysis with spectral and temporal features
- `compare_audio` - Compare two audio files
- `get_job_status` - Check analysis job status

## Architecture

```
src/
├── audio/          # Audio file loading and buffer management
├── analysis/       # Core analysis engine
│   ├── spectral/   # FFT, STFT, mel-scale processing
│   └── temporal/   # Beat tracking, onset detection
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

### Tempo Detection
```rust
let audio = AudioFile::load("dance_track.mp3")?;
let engine = AnalysisEngine::new();
let result = engine.analyze(&audio).await?;

if let Some(tempo) = result.temporal.tempo {
    println!("BPM: {:.1}", tempo);
    println!("Beats at: {:?}", result.temporal.beats);
}
```

### Spectral Analysis
```rust
let result = engine.analyze(&audio).await?;
let centroids = &result.spectral.spectral_centroid;
let flux = &result.spectral.spectral_flux;
```

## Contributing

Pull requests welcome. Please ensure all tests pass and add tests for new features.

```bash
cargo test
cargo fmt -- --check
cargo clippy
```

## License

This project is licensed under the [MIT License](LICENSE).