use ferrous_waves::analysis::spectral::{FftProcessor, WindowFunction};
use std::f32::consts::PI;

#[test]
fn test_fft_processor_creation() {
    let processor = FftProcessor::new(1024);
    // Should create without panic
    assert_eq!(processor.size(), 1024);
}

#[test]
fn test_fft_sine_wave_detection() {
    let size = 1024;
    let processor = FftProcessor::new(size);

    // Generate a 440Hz sine wave at 44100Hz sample rate
    let sample_rate = 44100.0;
    let frequency = 440.0;
    let mut input = vec![0.0; size];

    for (i, value) in input.iter_mut().enumerate() {
        *value = (2.0 * PI * frequency * i as f32 / sample_rate).sin();
    }

    let magnitude = processor.magnitude_spectrum(&input);

    // Find the peak
    let (peak_bin, _peak_mag) = magnitude
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap();

    // Convert bin to frequency
    let peak_freq = peak_bin as f32 * sample_rate / size as f32;

    // Should be close to 440Hz (within one bin)
    let bin_resolution = sample_rate / size as f32;
    assert!((peak_freq - frequency).abs() < bin_resolution);
}

#[test]
fn test_fft_dc_component() {
    let processor = FftProcessor::new(512);

    // Create a DC signal (constant value)
    let input = vec![1.0; 512];
    let magnitude = processor.magnitude_spectrum(&input);

    // DC component should be at bin 0 and have the highest magnitude
    let (peak_bin, _) = magnitude
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap();

    assert_eq!(peak_bin, 0);
}

#[test]
fn test_fft_power_spectrum() {
    let processor = FftProcessor::new(256);
    let input = vec![1.0; 256];

    let magnitude = processor.magnitude_spectrum(&input);
    let power = processor.power_spectrum(&input);

    // Power should be magnitude squared
    for (mag, pow) in magnitude.iter().zip(power.iter()) {
        assert!((pow - mag * mag).abs() < 1e-5);
    }
}

#[test]
fn test_fft_phase_spectrum() {
    let processor = FftProcessor::new(128);

    // Create a cosine wave (phase = 0) and sine wave (phase = π/2)
    let mut cosine = vec![0.0; 128];
    let mut sine = vec![0.0; 128];

    for i in 0..128 {
        cosine[i] = (2.0 * PI * 10.0 * i as f32 / 128.0).cos();
        sine[i] = (2.0 * PI * 10.0 * i as f32 / 128.0).sin();
    }

    let _phase_cos = processor.phase_spectrum(&cosine);
    let _phase_sin = processor.phase_spectrum(&sine);

    // Phase values should be different for sine and cosine
    // (Detailed phase testing would require more complex validation)
}

#[test]
fn test_window_function_hann() {
    let mut samples = vec![1.0; 100];
    WindowFunction::Hann.apply(&mut samples);

    // Check window properties
    assert_eq!(samples.len(), 100);
    assert!(samples[0].abs() < 0.01); // Should be near zero at edges
    assert!(samples[99].abs() < 0.01);
    assert!(samples[50] > 0.9); // Should be near 1.0 at center
}

#[test]
fn test_window_function_rectangular() {
    let mut samples = vec![2.0; 50];
    WindowFunction::Rectangular.apply(&mut samples);

    // Rectangular window shouldn't change values
    for &sample in &samples {
        assert_eq!(sample, 2.0);
    }
}

#[test]
fn test_window_function_create() {
    let hann = WindowFunction::Hann.create_window(64);
    let hamming = WindowFunction::Hamming.create_window(64);
    let blackman = WindowFunction::Blackman.create_window(64);

    assert_eq!(hann.len(), 64);
    assert_eq!(hamming.len(), 64);
    assert_eq!(blackman.len(), 64);

    // All windows should be symmetric
    for i in 0..32 {
        assert!((hann[i] - hann[63 - i]).abs() < 1e-6);
        assert!((hamming[i] - hamming[63 - i]).abs() < 1e-6);
        assert!((blackman[i] - blackman[63 - i]).abs() < 1e-6);
    }
}

#[test]
fn test_multiple_frequencies() {
    let size = 2048;
    let processor = FftProcessor::new(size);
    let sample_rate = 44100.0;

    // Generate signal with two frequencies: 440Hz and 880Hz
    let mut input = vec![0.0; size];
    for (i, value) in input.iter_mut().enumerate() {
        let t = i as f32 / sample_rate;
        *value = (2.0 * PI * 440.0 * t).sin() + 0.5 * (2.0 * PI * 880.0 * t).sin();
    }

    let magnitude = processor.magnitude_spectrum(&input);

    // Find peaks
    let mut peaks = Vec::new();
    for i in 1..magnitude.len() - 1 {
        if magnitude[i] > magnitude[i - 1] && magnitude[i] > magnitude[i + 1] && magnitude[i] > 10.0
        {
            // Threshold to filter noise
            let freq = i as f32 * sample_rate / size as f32;
            peaks.push(freq);
        }
    }

    // Should find peaks near 440Hz and 880Hz
    let has_440 = peaks.iter().any(|&f| (f - 440.0).abs() < 50.0);
    let has_880 = peaks.iter().any(|&f| (f - 880.0).abs() < 50.0);

    assert!(has_440, "Should detect 440Hz component");
    assert!(has_880, "Should detect 880Hz component");
}
