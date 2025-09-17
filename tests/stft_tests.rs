use ferrous_waves::analysis::spectral::{StftProcessor, WindowFunction};
use std::f32::consts::PI;

#[test]
fn test_stft_processor_creation() {
    StftProcessor::new(1024, 512, WindowFunction::Hann);
    // Should create without panic
}

#[test]
fn test_stft_frame_count() {
    let processor = StftProcessor::new(1024, 512, WindowFunction::Hann);
    let signal = vec![0.0; 4096];

    let spectrogram = processor.process(&signal);

    // Expected number of frames: (4096 - 1024) / 512 + 1 = 7
    assert_eq!(spectrogram.shape()[1], 7);
    // Number of frequency bins: 1024 / 2 + 1 = 513
    assert_eq!(spectrogram.shape()[0], 513);
}

#[test]
fn test_stft_constant_signal() {
    let processor = StftProcessor::new(512, 256, WindowFunction::Rectangular);
    let signal = vec![1.0; 2048];

    let spectrogram = processor.process(&signal);

    // For a constant signal, energy should be concentrated at DC (bin 0)
    for frame_idx in 0..spectrogram.shape()[1] {
        let dc_value = spectrogram[[0, frame_idx]];
        let other_value = spectrogram[[10, frame_idx]];

        assert!(dc_value > other_value * 10.0, "DC should dominate");
    }
}

#[test]
fn test_stft_sine_sweep() {
    let processor = StftProcessor::new(1024, 512, WindowFunction::Hann);
    let sample_rate = 44100.0;
    let duration = 2.0;
    let num_samples = (sample_rate * duration) as usize;

    // Create a frequency sweep from 100Hz to 1000Hz
    let mut signal = vec![0.0; num_samples];
    for i in 0..num_samples {
        let t = i as f32 / sample_rate;
        let freq = 100.0 + (900.0 * t / duration); // Linear sweep
        signal[i] = (2.0 * PI * freq * t).sin();
    }

    let spectrogram = processor.process(&signal);
    let num_frames = spectrogram.shape()[1];

    // Early frames should have energy at lower frequencies
    // Later frames should have energy at higher frequencies
    let early_frame = 0;
    let late_frame = num_frames - 1;

    // Find peak frequency for early and late frames
    let mut early_peak_bin = 0;
    let mut early_peak_mag = 0.0;
    let mut late_peak_bin = 0;
    let mut late_peak_mag = 0.0;

    for bin in 0..spectrogram.shape()[0] {
        if spectrogram[[bin, early_frame]] > early_peak_mag {
            early_peak_mag = spectrogram[[bin, early_frame]];
            early_peak_bin = bin;
        }
        if spectrogram[[bin, late_frame]] > late_peak_mag {
            late_peak_mag = spectrogram[[bin, late_frame]];
            late_peak_bin = bin;
        }
    }

    // Late frame should have higher frequency peak than early frame
    assert!(late_peak_bin > early_peak_bin, "Frequency should increase over time");
}

#[test]
fn test_stft_to_db() {
    let processor = StftProcessor::new(256, 128, WindowFunction::Hann);
    let signal = vec![1.0; 512];

    let spectrogram = processor.process(&signal);
    let db_spectrogram = processor.to_db(&spectrogram);

    // Check dB conversion
    for frame in 0..spectrogram.shape()[1] {
        for bin in 0..spectrogram.shape()[0] {
            let linear = spectrogram[[bin, frame]];
            let db = db_spectrogram[[bin, frame]];
            let expected_db = 20.0 * (linear + 1e-10).log10();

            assert!((db - expected_db).abs() < 1e-5);
        }
    }
}

#[test]
fn test_stft_frequency_bins() {
    let processor = StftProcessor::new(1024, 512, WindowFunction::Hann);
    let sample_rate = 44100;

    let freq_bins = processor.frequency_bins(sample_rate);

    assert_eq!(freq_bins.len(), 513); // 1024/2 + 1
    assert_eq!(freq_bins[0], 0.0); // DC
    assert!((freq_bins[freq_bins.len() - 1] - sample_rate as f32 / 2.0).abs() < 1.0); // Nyquist

    // Check linear spacing
    let bin_spacing = sample_rate as f32 / 1024.0;
    for i in 1..freq_bins.len() {
        assert!((freq_bins[i] - freq_bins[i - 1] - bin_spacing).abs() < 0.01);
    }
}

#[test]
fn test_stft_time_frames() {
    let processor = StftProcessor::new(1024, 512, WindowFunction::Hann);
    let sample_rate = 44100;
    let num_samples = 10000;

    let time_frames = processor.time_frames(num_samples, sample_rate);

    let expected_frames = (num_samples - 1024) / 512 + 1;
    assert_eq!(time_frames.len(), expected_frames);

    // Check time spacing
    let frame_duration = 512.0 / sample_rate as f32;
    for i in 1..time_frames.len() {
        assert!((time_frames[i] - time_frames[i - 1] - frame_duration).abs() < 1e-6);
    }
}

#[test]
fn test_stft_window_effect() {
    let processor_rect = StftProcessor::new(512, 256, WindowFunction::Rectangular);
    let processor_hann = StftProcessor::new(512, 256, WindowFunction::Hann);

    // Create a signal with a discontinuity
    let mut signal = vec![1.0; 1024];
    for i in 512..1024 {
        signal[i] = -1.0;
    }

    let spec_rect = processor_rect.process(&signal);
    let spec_hann = processor_hann.process(&signal);

    // Hann window should reduce spectral leakage compared to rectangular
    // This is a simplified test - proper validation would be more complex
    assert_ne!(spec_rect[[0, 0]], spec_hann[[0, 0]]);
}