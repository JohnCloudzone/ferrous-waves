use ferrous_waves::{AudioFile, AnalysisEngine, Result};
use ferrous_waves::analysis::spectral::WindowFunction;
use ferrous_waves::cache::Cache;
use tempfile::TempDir;

#[tokio::test]
async fn test_analysis_engine_basic() -> Result<()> {
    // Create a test signal
    let sample_rate = 44100;
    let duration = 1.0;
    let samples_per_channel = (sample_rate as f32 * duration) as usize;
    let channels = 2;

    // Generate a simple sine wave (interleaved stereo)
    let mut data = vec![0.0f32; samples_per_channel * channels];
    for i in 0..samples_per_channel {
        let sample = (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate as f32).sin() * 0.5;
        data[i * channels] = sample;      // Left channel
        data[i * channels + 1] = sample;  // Right channel
    }

    // Create audio buffer
    let buffer = ferrous_waves::audio::AudioBuffer::new(data, sample_rate, channels);

    // Create audio file
    let audio = AudioFile {
        path: "test.wav".to_string(),
        format: ferrous_waves::audio::AudioFormat::Wav,
        buffer,
    };

    // Run analysis
    let engine = AnalysisEngine::new();
    let result = engine.analyze(&audio).await?;

    // Verify results
    assert_eq!(result.summary.sample_rate, sample_rate);
    assert_eq!(result.summary.channels, 2);
    assert!((result.summary.duration - duration).abs() < 0.01);
    assert!(result.summary.peak_amplitude > 0.4 && result.summary.peak_amplitude < 0.6);

    Ok(())
}

#[tokio::test]
async fn test_analysis_engine_with_cache() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let cache = Cache::with_config(
        temp_dir.path().to_path_buf(),
        10 * 1024 * 1024, // 10MB max size
        std::time::Duration::from_secs(3600), // 1 hour TTL
    );

    // Create a test signal
    let sample_rate = 44100;
    let duration = 0.5;
    let samples = (sample_rate as f32 * duration) as usize;

    let mut data = vec![0.0f32; samples];
    for i in 0..samples {
        data[i] = (2.0 * std::f32::consts::PI * 220.0 * i as f32 / sample_rate as f32).sin() * 0.3;
    }

    let buffer = ferrous_waves::audio::AudioBuffer::new(data, sample_rate, 1);

    let audio = AudioFile {
        path: "test_cached.wav".to_string(),
        format: ferrous_waves::audio::AudioFormat::Wav,
        buffer,
    };

    let engine = AnalysisEngine::new().with_cache(cache.clone());

    // First analysis - should compute
    let result1 = engine.analyze(&audio).await?;

    // Second analysis - should use cache
    let result2 = engine.analyze(&audio).await?;

    // Results should be identical
    assert_eq!(result1.summary.peak_amplitude, result2.summary.peak_amplitude);
    assert_eq!(result1.summary.rms_level, result2.summary.rms_level);

    // Check cache stats
    let stats = cache.stats();
    assert!(stats.total_entries > 0);

    Ok(())
}

#[tokio::test]
async fn test_comparison() -> Result<()> {
    let sample_rate = 44100;

    // Create two different test signals
    let audio_a = {
        let samples = (sample_rate as f32 * 0.5) as usize;
        let mut data = vec![0.0f32; samples];
        for i in 0..samples {
            data[i] = (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate as f32).sin() * 0.5;
        }

        AudioFile {
            path: "a.wav".to_string(),
            format: ferrous_waves::audio::AudioFormat::Wav,
            buffer: ferrous_waves::audio::AudioBuffer::new(data, sample_rate, 1),
        }
    };

    let audio_b = {
        let samples = (sample_rate as f32 * 1.0) as usize;
        let mut data = vec![0.0f32; samples];
        for i in 0..samples {
            data[i] = (2.0 * std::f32::consts::PI * 880.0 * i as f32 / sample_rate as f32).sin() * 0.3;
        }

        AudioFile {
            path: "b.wav".to_string(),
            format: ferrous_waves::audio::AudioFormat::Wav,
            buffer: ferrous_waves::audio::AudioBuffer::new(data, sample_rate, 1),
        }
    };

    let engine = AnalysisEngine::new();
    let comparison = engine.compare(&audio_a, &audio_b).await;

    assert_eq!(comparison.file_a.path, "a.wav");
    assert_eq!(comparison.file_b.path, "b.wav");
    assert_eq!(comparison.comparison.duration_difference, -0.5);
    assert!(comparison.comparison.sample_rate_match);

    Ok(())
}

#[test]
fn test_window_function_config() {
    // Test that we can create an engine with custom config
    // The actual values are private, but we can test that it works
    let _engine = AnalysisEngine::with_config(4096, 1024, WindowFunction::Hamming);
    // If it compiles and doesn't panic, the config is working
}

#[tokio::test]
async fn test_tempo_detection() -> Result<()> {
    // Create a simple rhythmic pattern
    let sample_rate = 44100;
    let bpm = 120.0;
    let duration = 4.0;
    let samples = (sample_rate as f32 * duration) as usize;

    let mut data = vec![0.0f32; samples];
    let beat_interval = 60.0 / bpm;
    let samples_per_beat = (sample_rate as f32 * beat_interval) as usize;

    // Add click on each beat
    for beat in 0..((duration / beat_interval) as usize) {
        let pos = beat * samples_per_beat;
        if pos < samples {
            // Add a short click
            for i in 0..100.min(samples - pos) {
                data[pos + i] = (0.9 - i as f32 / 100.0) * if i < 50 { 1.0 } else { -1.0 };
            }
        }
    }

    let buffer = ferrous_waves::audio::AudioBuffer::new(data, sample_rate, 1);

    let audio = AudioFile {
        path: "tempo_test.wav".to_string(),
        format: ferrous_waves::audio::AudioFormat::Wav,
        buffer,
    };

    let engine = AnalysisEngine::new();
    let result = engine.analyze(&audio).await?;

    // Check if tempo detection is within reasonable range
    if let Some(detected_tempo) = result.temporal.tempo {
        // Allow for some variance in tempo detection
        assert!(detected_tempo > 60.0 && detected_tempo < 240.0);
    }

    Ok(())
}