use ferrous_waves::{
    audio::{AudioBuffer, AudioFile},
    analysis::spectral::{FftProcessor, StftProcessor, MelFilterBank, WindowFunction},
    analysis::temporal::{OnsetDetector, BeatTracker},
};
use std::f32::consts::PI;

/// Helper function to generate test audio
fn generate_test_audio(duration_seconds: f32, sample_rate: u32) -> AudioBuffer {
    let num_samples = (duration_seconds * sample_rate as f32) as usize;
    let mut samples = Vec::with_capacity(num_samples * 2); // Stereo

    // Generate a complex test signal with multiple components
    for i in 0..num_samples {
        let t = i as f32 / sample_rate as f32;

        // Left channel: 440Hz sine + 880Hz harmonics
        let left = (2.0 * PI * 440.0 * t).sin() + 0.3 * (2.0 * PI * 880.0 * t).sin();

        // Right channel: 440Hz with slight phase shift + noise
        let right = (2.0 * PI * 440.0 * t + 0.1).sin() + 0.1 * ((i * 12345) as f32 / 65536.0).sin();

        samples.push(left * 0.5);
        samples.push(right * 0.5);
    }

    AudioBuffer::new(samples, sample_rate, 2)
}

#[test]
fn test_full_analysis_pipeline() {
    // Generate test audio
    let audio_buffer = generate_test_audio(2.0, 44100);

    // Convert to mono for analysis
    let mono = audio_buffer.to_mono();

    // FFT Analysis
    let fft = FftProcessor::new(2048);
    let window_size = 2048;
    let hop_size = 512;

    // Take a window of audio
    let window: Vec<f32> = mono.iter().take(window_size).copied().collect();
    let magnitude = fft.magnitude_spectrum(&window);

    assert_eq!(magnitude.len(), 1025); // 2048/2 + 1

    // STFT Analysis
    let stft = StftProcessor::new(window_size, hop_size, WindowFunction::Hann);
    let spectrogram = stft.process(&mono);

    assert!(spectrogram.shape()[0] > 0);
    assert!(spectrogram.shape()[1] > 0);

    // Mel Spectrogram
    let mel_filter = MelFilterBank::new(40, 44100, window_size);
    let mel_spectrogram = mel_filter.apply(&spectrogram);

    assert_eq!(mel_spectrogram.shape()[0], 40);
    assert_eq!(mel_spectrogram.shape()[1], spectrogram.shape()[1]);

    // Onset Detection
    let onset_detector = OnsetDetector::new();

    // Calculate spectral flux for onset detection
    let num_frames = spectrogram.shape()[1];
    let mut spectral_flux = vec![0.0; num_frames];

    for i in 1..num_frames {
        let mut flux = 0.0;
        for bin in 0..spectrogram.shape()[0] {
            let diff = spectrogram[[bin, i]] - spectrogram[[bin, i - 1]];
            if diff > 0.0 {
                flux += diff * diff;
            }
        }
        spectral_flux[i] = flux.sqrt();
    }

    let onsets = onset_detector.detect_onsets(&spectral_flux, hop_size, 44100);

    // Beat Tracking
    let beat_tracker = BeatTracker::new();
    let tempo = beat_tracker.estimate_tempo(&onsets);

    // The generated audio doesn't have a clear beat, so tempo might be None
    // But the pipeline should complete without errors
}

#[test]
fn test_stereo_to_mono_pipeline() {
    let stereo_buffer = generate_test_audio(1.0, 48000);

    // Test channel extraction
    let left = stereo_buffer.get_channel(0).unwrap();
    let right = stereo_buffer.get_channel(1).unwrap();

    assert_eq!(left.len(), 48000);
    assert_eq!(right.len(), 48000);

    // Test mono conversion
    let mono = stereo_buffer.to_mono();
    assert_eq!(mono.len(), 48000);

    // Process mono through FFT
    let fft = FftProcessor::new(1024);
    let window: Vec<f32> = mono.iter().take(1024).copied().collect();
    let spectrum = fft.magnitude_spectrum(&window);

    assert_eq!(spectrum.len(), 513);
}

#[test]
fn test_different_sample_rates() {
    let sample_rates = vec![22050, 44100, 48000];

    for sr in sample_rates {
        let buffer = generate_test_audio(0.5, sr);
        let mono = buffer.to_mono();

        let stft = StftProcessor::new(1024, 512, WindowFunction::Hann);
        let spectrogram = stft.process(&mono);

        // Verify frequency bins are correct for each sample rate
        let freq_bins = stft.frequency_bins(sr);
        assert_eq!(freq_bins.len(), 513);
        assert_eq!(freq_bins[0], 0.0); // DC
        assert!((freq_bins.last().unwrap() - sr as f32 / 2.0).abs() < 1.0); // Nyquist
    }
}

#[test]
fn test_window_hop_combinations() {
    let audio_buffer = generate_test_audio(1.0, 44100);
    let mono = audio_buffer.to_mono();

    let combinations = vec![
        (256, 128),
        (512, 256),
        (1024, 512),
        (2048, 1024),
        (4096, 2048),
    ];

    for (window_size, hop_size) in combinations {
        let stft = StftProcessor::new(window_size, hop_size, WindowFunction::Hann);
        let spectrogram = stft.process(&mono);

        let expected_frames = (mono.len() - window_size) / hop_size + 1;
        assert_eq!(spectrogram.shape()[1], expected_frames);
        assert_eq!(spectrogram.shape()[0], window_size / 2 + 1);
    }
}

#[test]
fn test_tempo_detection_synthetic() {
    // Create synthetic onset times at 120 BPM
    let bpm = 120.0;
    let beat_period = 60.0 / bpm;
    let duration = 10.0;

    let mut onset_times = Vec::new();
    let mut time = 0.0;
    while time < duration {
        onset_times.push(time);
        time += beat_period;
    }

    // Add slight jitter to make it more realistic
    for (i, onset) in onset_times.iter_mut().enumerate().skip(1) {
        *onset += (i as f32 * 0.1).sin() * 0.01; // Small variation
    }

    let tracker = BeatTracker::new();
    let detected_tempo = tracker.estimate_tempo(&onset_times);

    assert!(detected_tempo.is_some());
    let tempo = detected_tempo.unwrap();
    assert!((tempo - bpm).abs() < 5.0, "Expected ~{} BPM, got {}", bpm, tempo);

    // Test beat tracking
    let beats = tracker.track_beats(&onset_times, tempo);
    assert!(!beats.is_empty());

    // Verify beat regularity
    if beats.len() > 2 {
        let intervals: Vec<f32> = beats.windows(2).map(|w| w[1] - w[0]).collect();
        let expected_interval = 60.0 / tempo;

        for interval in intervals {
            assert!((interval - expected_interval).abs() < 0.01);
        }
    }
}

#[test]
fn test_mel_scale_pipeline() {
    let audio = generate_test_audio(1.0, 22050);
    let mono = audio.to_mono();

    // Different mel configurations
    let configs = vec![
        (20, 512),
        (40, 1024),
        (80, 2048),
    ];

    for (num_filters, fft_size) in configs {
        let stft = StftProcessor::new(fft_size, fft_size / 2, WindowFunction::Hann);
        let spectrogram = stft.process(&mono);

        let mel = MelFilterBank::new(num_filters, 22050, fft_size);
        let mel_spec = mel.apply(&spectrogram);

        assert_eq!(mel_spec.shape()[0], num_filters);

        // Verify mel spectrogram has reasonable values
        let max_val = mel_spec.iter().fold(0.0f32, |a, &b| a.max(b));
        let min_val = mel_spec.iter().fold(f32::INFINITY, |a, &b| a.min(b));

        assert!(max_val > 0.0, "Mel spectrogram should have positive values");
        assert!(min_val >= 0.0, "Mel spectrogram should be non-negative");
    }
}