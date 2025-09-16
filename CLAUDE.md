# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Ferrous Waves is a Rust-based audio analysis pipeline that transforms audio files into comprehensive visual and analytical data packages. It includes native Model Context Protocol (MCP) server integration for seamless interaction with AI-assisted development workflows.

## Build and Development Commands

```bash
# Build the project
cargo build
cargo build --release

# Run tests
cargo test
cargo test --all-features
cargo test --test <specific_test_name>

# Run benchmarks
cargo bench
cargo bench --bench <specific_benchmark>

# Check code quality
cargo fmt -- --check
cargo clippy -- -D warnings
cargo clippy --all-features -- -D warnings

# Run the main binary
cargo run -- analyze <audio_file>
cargo run -- mcp start

# Run with verbose logging
RUST_LOG=debug cargo run -- analyze test.wav

# Run specific examples
cargo run --example <example_name> -- <args>

# Generate documentation
cargo doc --open
cargo doc --no-deps

# Package for distribution
cargo package --allow-dirty
cargo publish --dry-run
```

## Architecture Overview

### Core Module Structure

The project follows a modular architecture with clear separation of concerns:

- **`src/audio/`**: Audio file handling and decoding
  - `decoder.rs`: Symphonia-based multi-format decoder
  - `buffer.rs`: Audio buffer management with channel operations
  - `formats.rs`: Format detection and validation

- **`src/analysis/`**: Core analysis algorithms
  - `spectral/`: FFT, STFT, mel-scale spectrograms
  - `temporal/`: Onset detection, beat tracking, tempo estimation
  - `features/`: Audio feature extraction (MFCC, spectral features)
  - `engine.rs`: Main analysis orchestrator

- **`src/mcp/`**: Model Context Protocol integration
  - `server.rs`: MCP server implementation with tool handlers
  - `tools.rs`: Tool definitions and result structures
  - `handlers.rs`: Async handlers for batch and watch operations

- **`src/visualization/`**: Audio data visualization
  - `renderer.rs`: Plotters-based visualization engine
  - Generates waveforms, spectrograms, power curves

- **`src/cache/`**: Performance optimization
  - Content-based caching with Blake3 hashing
  - LRU eviction strategy
  - Persistent disk cache

### Key Implementation Details

1. **Audio Processing Pipeline**:
   - Uses Symphonia for format-agnostic decoding
   - Processes audio in floating-point for precision
   - Supports WAV, MP3, FLAC, OGG, M4A formats

2. **FFT/Spectral Analysis**:
   - FFT size: 2048 (default), hop size: 512
   - Window functions: Hann, Hamming, Blackman, Nuttall
   - Mel-scale filterbank for perceptual analysis

3. **MCP Server Integration**:
   - Tools: `analyze_audio`, `compare_audio`, `watch_audio`, `get_analysis_status`
   - Async processing with job tracking
   - Configurable return formats (full, summary, visual_only)

4. **Performance Considerations**:
   - Parallel processing with Rayon
   - Content-based caching to avoid redundant analysis
   - Streaming analysis for large files
   - Zero-copy operations where possible

## MCP Server Configuration

The MCP server can be configured for Claude integration:

```toml
# ferrous-waves-mcp.toml
[server]
name = "ferrous-waves"
version = "1.0.0"

[analysis]
cache_dir = "~/.ferrous-waves/cache"
max_cache_size_gb = 10
default_output_format = "full"

[performance]
thread_pool_size = 0  # 0 = number of CPU cores
max_file_size_mb = 500
```

## Implementation Status

The project is designed following the comprehensive guide in `docs/guide.md` which breaks down implementation into 13 phases:

1. Project Foundation (Cargo.toml, error handling, config)
2. Audio Decoding Pipeline (Symphonia integration)
3. FFT and Spectral Analysis (rustfft, STFT)
4. Temporal Analysis (onset/beat detection)
5. MCP Server Implementation
6. Visualization Generation (Plotters)
7. Cache System
8. Analysis Engine Integration
9. Feature Extraction
10. CLI Interface
11. Testing Suite
12. Documentation
13. Final Integration

## Testing Strategy

- Unit tests for individual components
- Integration tests for full pipeline (`tests/integration/`)
- Benchmarks for performance-critical paths (`benches/`)
- Test with synthetic audio (sine waves, noise) and real files

## Key Dependencies

- `symphonia`: Audio decoding (all formats)
- `rustfft`: Fast Fourier Transform
- `plotters`: Visualization rendering
- `mcp-rs`/`rmcp`: Model Context Protocol SDK
- `tokio`: Async runtime
- `clap`: CLI argument parsing
- `dashmap`: Concurrent cache storage
- `blake3`: Content hashing for cache keys