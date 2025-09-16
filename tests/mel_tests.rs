use ferrous_waves::analysis::spectral::MelFilterBank;
use ndarray::Array2;

#[test]
fn test_mel_filterbank_creation() {
    let filterbank = MelFilterBank::new(40, 44100, 2048);
    // Should create without panic
}

#[test]
fn test_mel_scale_conversion() {
    // Test known conversions
    assert_eq!(MelFilterBank::hz_to_mel(0.0), 0.0);

    // 1000 Hz should be approximately 1000 mels
    let mel_1000 = MelFilterBank::hz_to_mel(1000.0);
    assert!((mel_1000 - 1000.0).abs() < 50.0);

    // Test inverse conversion
    let hz = MelFilterBank::mel_to_hz(mel_1000);
    assert!((hz - 1000.0).abs() < 1.0);
}

#[test]
fn test_mel_scale_monotonic() {
    let frequencies = vec![100.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0];
    let mels: Vec<f32> = frequencies.iter().map(|&f| MelFilterBank::hz_to_mel(f)).collect();

    // Mel scale should be monotonically increasing
    for i in 1..mels.len() {
        assert!(mels[i] > mels[i - 1], "Mel scale should be monotonic");
    }
}

#[test]
fn test_mel_to_hz_inverse() {
    let frequencies = vec![100.0, 440.0, 1000.0, 3000.0, 8000.0];

    for &freq in &frequencies {
        let mel = MelFilterBank::hz_to_mel(freq);
        let hz_back = MelFilterBank::mel_to_hz(mel);
        assert!((hz_back - freq).abs() < 0.1, "Conversion should be reversible");
    }
}

#[test]
fn test_filterbank_dimensions() {
    let num_filters = 40;
    let fft_size = 2048;
    let filterbank = MelFilterBank::new(num_filters, 44100, fft_size);

    // Create a dummy spectrogram
    let num_frames = 10;
    let num_bins = fft_size / 2 + 1;
    let spectrogram = Array2::ones((num_bins, num_frames));

    let mel_spec = filterbank.apply(&spectrogram);

    assert_eq!(mel_spec.shape()[0], num_filters);
    assert_eq!(mel_spec.shape()[1], num_frames);
}

#[test]
fn test_filterbank_triangular_filters() {
    let filterbank = MelFilterBank::new(20, 16000, 512);

    // Test a few specific properties of triangular filters
    // Each filter should have a peak value around 1.0
    let filter_bank = filterbank.get_filters();

    for filter_idx in 0..20 {
        let filter = filter_bank.row(filter_idx);
        let max_val = filter.iter().fold(0.0f32, |a, &b| a.max(b));

        // Peak should be close to 1.0 for normalized filters
        assert!(max_val > 0.5 && max_val <= 1.0, "Filter peak should be normalized");

        // Filter should have non-zero values
        let non_zero_count = filter.iter().filter(|&&x| x > 0.0).count();
        assert!(non_zero_count > 0, "Filter should have non-zero values");
    }
}

#[test]
fn test_filterbank_frequency_coverage() {
    let sample_rate = 22050;
    let filterbank = MelFilterBank::new(32, sample_rate, 1024);

    // The filterbank should cover from 0 Hz to Nyquist frequency
    let nyquist = sample_rate / 2;

    // Get the filter bank matrix
    let filters = filterbank.get_filters();

    // Check first filter starts near 0
    let first_filter = filters.row(0);
    let first_nonzero = first_filter.iter().position(|&x| x > 0.0).unwrap_or(0);
    assert!(first_nonzero < 10, "First filter should start near DC");

    // Check last filter extends toward Nyquist
    let last_filter = filters.row(filters.shape()[0] - 1);
    let last_nonzero = last_filter.iter().rposition(|&x| x > 0.0).unwrap_or(0);
    assert!(last_nonzero > filters.shape()[1] / 2, "Last filter should extend toward Nyquist");
}

#[test]
fn test_mel_spectrogram_energy_preservation() {
    let filterbank = MelFilterBank::new(40, 44100, 2048);

    // Create a spectrogram with known energy distribution
    let mut spectrogram = Array2::zeros((1025, 5));

    // Put energy in specific frequency bins
    for frame in 0..5 {
        spectrogram[[100, frame]] = 1.0; // Low frequency
        spectrogram[[500, frame]] = 2.0; // Mid frequency
        spectrogram[[900, frame]] = 0.5; // High frequency
    }

    let mel_spec = filterbank.apply(&spectrogram);

    // Total energy should be preserved (approximately)
    let input_energy: f32 = spectrogram.iter().sum();
    let output_energy: f32 = mel_spec.iter().sum();

    // Some energy loss is expected due to filter overlap
    // but should be within reasonable bounds
    let energy_ratio = output_energy / input_energy;
    assert!(energy_ratio > 0.5 && energy_ratio < 1.5,
            "Energy should be approximately preserved: ratio = {}", energy_ratio);
}

#[test]
fn test_different_num_filters() {
    let configs = vec![
        (10, 16000, 512),
        (20, 22050, 1024),
        (40, 44100, 2048),
        (80, 48000, 4096),
    ];

    for (num_filters, sample_rate, fft_size) in configs {
        let filterbank = MelFilterBank::new(num_filters, sample_rate, fft_size);

        // Create test spectrogram
        let spectrogram = Array2::ones((fft_size / 2 + 1, 10));
        let mel_spec = filterbank.apply(&spectrogram);

        assert_eq!(mel_spec.shape()[0], num_filters,
                   "Should have {} mel bands", num_filters);
        assert_eq!(mel_spec.shape()[1], 10, "Time frames should be preserved");
    }
}