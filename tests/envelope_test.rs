use ferrous_waves::visualization::waveform::render_waveform_with_envelope;
use std::fs;
use tempfile::TempDir;

#[test]
fn test_envelope_rendering() {
    // Create test signal with varying amplitude
    let sample_rate = 44100;
    let duration = 2.0;
    let num_samples = (sample_rate as f32 * duration) as usize;
    let mut samples = Vec::with_capacity(num_samples);

    for i in 0..num_samples {
        let t = i as f32 / sample_rate as f32;

        // Create amplitude envelope that changes over time
        let envelope = if t < 0.5 {
            t * 2.0 // Ramp up
        } else if t < 1.5 {
            1.0 // Sustain
        } else {
            (2.0 - t) * 2.0 // Ramp down
        };

        // Modulate a 440Hz sine wave with the envelope
        let sample = (2.0 * std::f32::consts::PI * 440.0 * t).sin() * envelope * 0.8;
        samples.push(sample);
    }

    // Render to temp file
    let temp_dir = TempDir::new().unwrap();
    let output_path = temp_dir.path().join("envelope_test.png");

    let result = render_waveform_with_envelope(&samples, &output_path, 800, 400);
    assert!(result.is_ok());
    assert!(output_path.exists());

    // Verify the file is not empty
    let metadata = fs::metadata(&output_path).unwrap();
    assert!(metadata.len() > 0);
}

#[test]
fn test_envelope_with_silence() {
    // Create signal with silence periods
    let mut samples = vec![0.0; 44100]; // 1 second of silence

    // Add a burst in the middle
    for (i, sample) in samples.iter_mut().enumerate().take(33075).skip(11025) {
        let t = (i - 11025) as f32 / 44100.0;
        *sample = (2.0 * std::f32::consts::PI * 1000.0 * t).sin() * 0.5;
    }

    let temp_dir = TempDir::new().unwrap();
    let output_path = temp_dir.path().join("envelope_silence_test.png");

    let result = render_waveform_with_envelope(&samples, &output_path, 600, 300);
    assert!(result.is_ok());
    assert!(output_path.exists());
}

#[test]
fn test_envelope_with_clipping() {
    // Create signal that clips
    let mut samples = Vec::new();
    for i in 0..22050 {
        let t = i as f32 / 44100.0;
        let sample = (2.0 * std::f32::consts::PI * 200.0 * t).sin() * 2.0; // Amplitude > 1
        samples.push(sample.clamp(-1.0, 1.0)); // Clip to valid range
    }

    let temp_dir = TempDir::new().unwrap();
    let output_path = temp_dir.path().join("envelope_clipping_test.png");

    let result = render_waveform_with_envelope(&samples, &output_path, 600, 300);
    assert!(result.is_ok());
}
