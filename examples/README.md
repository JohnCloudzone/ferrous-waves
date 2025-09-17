# Ferrous Waves Examples

This directory contains examples demonstrating different ways to use Ferrous Waves.

## Setup

First, generate the sample audio files:

```bash
cargo run --example generate_samples
```

This creates a `samples/` directory with test WAV files including sine waves, chords, and drum patterns.

## Library Examples

- **basic_analysis.rs** - Simple audio file analysis using the high-level API
- **compare_files.rs** - Compare two audio files and show differences
- **spectral_analysis.rs** - Direct use of FFT and STFT processors
- **onset_detection.rs** - Detect note onsets and transients in audio
- **perceptual_analysis.rs** - LUFS loudness measurement and perceptual metrics
- **content_classification.rs** - Speech/music/silence detection with confidence scores
- **cached_analysis.rs** - Using the cache system for faster repeated analysis
- **batch_processing.rs** - Process multiple files in parallel
- **envelope_visualization.rs** - Generate waveform visualization with peak and RMS envelopes
- **generate_samples.rs** - Generate test WAV files for the examples

## Running Examples

After generating samples, run any example:

```bash
cargo run --example basic_analysis
cargo run --example compare_files
cargo run --example spectral_analysis
cargo run --example onset_detection
cargo run --example perceptual_analysis
cargo run --example content_classification
cargo run --example cached_analysis
cargo run --example batch_processing
cargo run --example envelope_visualization
```

The `envelope_visualization` example creates a PNG image showing waveform with peak and RMS envelopes.

## MCP Server

The `mcp_client.rs` example shows the JSON format for MCP tool calls. To use the actual MCP server:

```bash
# Start the MCP server
cargo run --bin mcp_server

# The server communicates via stdio and can be integrated with
# AI assistants that support the Model Context Protocol
```

## CLI Usage

The main CLI provides these commands:

```bash
# Analyze a single file
ferrous-waves analyze audio.wav

# Compare two files
ferrous-waves compare original.wav processed.wav

# Extract tempo
ferrous-waves tempo music.wav --show-beats

# Detect onsets
ferrous-waves onsets drums.wav

# Batch process files
ferrous-waves batch ./music "*.wav" --output results/

# Start HTTP server
ferrous-waves serve --port 8080
```