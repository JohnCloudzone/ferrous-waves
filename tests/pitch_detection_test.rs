use ferrous_waves::analysis::pitch::{PitchDetector, PyinDetector, VibratoDetector, YinDetector};
use std::f32::consts::PI;

fn generate_sine_wave(frequency: f32, sample_rate: f32, duration: f32) -> Vec<f32> {
    let num_samples = (sample_rate * duration) as usize;
    (0..num_samples)
        .map(|i| {
            let t = i as f32 / sample_rate;
            (2.0 * PI * frequency * t).sin()
        })
        .collect()
}

fn generate_complex_tone(
    fundamental: f32,
    harmonics: &[(f32, f32)],
    sample_rate: f32,
    duration: f32,
) -> Vec<f32> {
    let num_samples = (sample_rate * duration) as usize;
    (0..num_samples)
        .map(|i| {
            let t = i as f32 / sample_rate;
            let mut sample = (2.0 * PI * fundamental * t).sin();

            for &(harmonic_ratio, amplitude) in harmonics {
                sample += amplitude * (2.0 * PI * fundamental * harmonic_ratio * t).sin();
            }

            sample / (1.0 + harmonics.len() as f32)
        })
        .collect()
}

fn generate_vibrato_signal(
    base_freq: f32,
    vibrato_rate: f32,
    vibrato_depth: f32,
    sample_rate: f32,
    duration: f32,
) -> Vec<f32> {
    let num_samples = (sample_rate * duration) as usize;
    (0..num_samples)
        .map(|i| {
            let t = i as f32 / sample_rate;
            let freq = base_freq * (1.0 + vibrato_depth * (2.0 * PI * vibrato_rate * t).sin());
            (2.0 * PI * freq * t).sin()
        })
        .collect()
}

fn add_noise(samples: &mut [f32], noise_level: f32) {
    use rand::prelude::*;
    let mut rng = thread_rng();

    for sample in samples {
        *sample += (rng.gen::<f32>() - 0.5) * noise_level;
    }
}

#[test]
fn test_yin_detector_accuracy() {
    let sample_rate = 44100.0;
    let test_frequencies = [82.41, 110.0, 220.0, 440.0, 880.0, 1760.0];

    for &frequency in &test_frequencies {
        let samples = generate_sine_wave(frequency, sample_rate, 0.1);
        let detector = YinDetector::new();
        let result = detector.detect_pitch(&samples, sample_rate);

        let error_percent = ((result.frequency - frequency).abs() / frequency) * 100.0;
        assert!(
            error_percent < 1.0,
            "YIN: Frequency {} detected as {} ({}% error)",
            frequency,
            result.frequency,
            error_percent
        );
        assert!(result.confidence > 0.9, "YIN: Low confidence for pure tone");
    }
}

#[test]
fn test_pyin_detector_accuracy() {
    let sample_rate = 44100.0;
    let test_frequencies = [82.41, 110.0, 220.0, 440.0, 880.0, 1760.0];

    for &frequency in &test_frequencies {
        let samples = generate_sine_wave(frequency, sample_rate, 0.1);
        let detector = PyinDetector::new();
        let result = detector.detect_pitch(&samples, sample_rate);

        let error_percent = ((result.frequency - frequency).abs() / frequency) * 100.0;
        assert!(
            error_percent < 3.0,
            "PYIN: Frequency {} detected as {} ({}% error)",
            frequency,
            result.frequency,
            error_percent
        );
        assert!(
            result.confidence > 0.8,
            "PYIN: Low confidence for pure tone"
        );
    }
}

#[test]
fn test_complex_harmonic_detection() {
    let sample_rate = 44100.0;
    let fundamental = 220.0;
    let harmonics = vec![(2.0, 0.7), (3.0, 0.5), (4.0, 0.3), (5.0, 0.2)];

    let samples = generate_complex_tone(fundamental, &harmonics, sample_rate, 0.1);

    let yin_detector = YinDetector::new();
    let yin_result = yin_detector.detect_pitch(&samples, sample_rate);

    let pyin_detector = PyinDetector::new();
    let pyin_result = pyin_detector.detect_pitch(&samples, sample_rate);

    assert!(
        (yin_result.frequency - fundamental).abs() < 10.0,
        "YIN failed on harmonic signal: {} vs {}",
        yin_result.frequency,
        fundamental
    );

    assert!(
        (pyin_result.frequency - fundamental).abs() < 10.0,
        "PYIN failed on harmonic signal: {} vs {}",
        pyin_result.frequency,
        fundamental
    );
}

#[test]
fn test_noisy_signal_detection() {
    let sample_rate = 44100.0;
    let frequency = 440.0;
    let mut samples = generate_sine_wave(frequency, sample_rate, 0.1);

    add_noise(&mut samples, 0.2);

    let yin_detector = YinDetector::new();
    let yin_result = yin_detector.detect_pitch(&samples, sample_rate);

    let pyin_detector = PyinDetector::new();
    let pyin_result = pyin_detector.detect_pitch(&samples, sample_rate);

    assert!(
        (yin_result.frequency - frequency).abs() < 20.0,
        "YIN failed with noise: {} vs {}",
        yin_result.frequency,
        frequency
    );

    assert!(
        (pyin_result.frequency - frequency).abs() < 20.0,
        "PYIN failed with noise: {} vs {}",
        pyin_result.frequency,
        frequency
    );

    assert!(
        yin_result.confidence > 0.3,
        "YIN confidence too low with noise"
    );
    assert!(
        pyin_result.confidence > 0.3,
        "PYIN confidence too low with noise"
    );
}

#[test]
fn test_vibrato_detection() {
    let sample_rate = 44100.0;
    let base_freq = 440.0;
    let vibrato_rate = 5.0;
    let vibrato_depth = 0.03;

    let samples = generate_vibrato_signal(base_freq, vibrato_rate, vibrato_depth, sample_rate, 0.5);

    let detector = PyinDetector::new();
    let hop_size = 512;
    let pitch_track = detector.detect_pitch_track(&samples, sample_rate, hop_size);

    let valid_pitches: Vec<f32> = pitch_track
        .frames
        .iter()
        .filter_map(|f| f.frequency)
        .collect();

    assert!(
        !valid_pitches.is_empty(),
        "No pitches detected in vibrato signal"
    );

    let mean_pitch = valid_pitches.iter().sum::<f32>() / valid_pitches.len() as f32;
    assert!(
        (mean_pitch - base_freq).abs() < 20.0,
        "Mean pitch {} differs from base frequency {}",
        mean_pitch,
        base_freq
    );

    let vibrato_detector = VibratoDetector::new();
    let vibrato_analysis = vibrato_detector.analyze(&valid_pitches, sample_rate, hop_size);

    assert!(vibrato_analysis.is_some(), "Vibrato not detected");

    if let Some(analysis) = vibrato_analysis {
        assert!(
            (analysis.rate - vibrato_rate).abs() < 1.0,
            "Vibrato rate {} differs from expected {}",
            analysis.rate,
            vibrato_rate
        );
        assert!(analysis.presence > 0.5, "Vibrato presence too low");
    }
}

#[test]
fn test_pitch_track_generation() {
    let sample_rate = 44100.0;
    let frequency_sweep = |t: f32| 220.0 * (1.0 + t);
    let duration = 1.0;

    let num_samples = (sample_rate * duration) as usize;
    let samples: Vec<f32> = (0..num_samples)
        .map(|i| {
            let t = i as f32 / sample_rate;
            let freq = frequency_sweep(t);
            (2.0 * PI * freq * t).sin()
        })
        .collect();

    let detector = YinDetector::new();
    let hop_size = 1024;
    let pitch_track = detector.detect_pitch_track(&samples, sample_rate, hop_size);

    assert!(!pitch_track.frames.is_empty());

    for (i, frame) in pitch_track.frames.iter().enumerate() {
        if let Some(detected_freq) = frame.frequency {
            let expected_freq = frequency_sweep(frame.time);
            let error = (detected_freq - expected_freq).abs();

            if error > 50.0 {
                eprintln!(
                    "Frame {}: time={:.3}s, detected={:.1}Hz, expected={:.1}Hz",
                    i, frame.time, detected_freq, expected_freq
                );
            }
        }
    }
}

#[test]
fn test_midi_note_conversion() {
    let sample_rate = 44100.0;

    let test_cases = vec![
        (261.626, 60, "C4"),
        (440.0, 69, "A4"),
        (880.0, 81, "A5"),
        (493.883, 71, "B4"),
    ];

    for (frequency, expected_midi, expected_note) in test_cases {
        let samples = generate_sine_wave(frequency, sample_rate, 0.1);
        let detector = YinDetector::new();
        let result = detector.detect_pitch(&samples, sample_rate);

        assert!(result.midi_note.is_some());
        assert!(result.note_name.is_some());

        if let (Some(midi), Some(note)) = (result.midi_note, &result.note_name) {
            assert!(
                (midi as i32 - expected_midi).abs() <= 1,
                "MIDI note {} differs from expected {}",
                midi,
                expected_midi
            );
            assert_eq!(
                note, expected_note,
                "Note name {} differs from expected {}",
                note, expected_note
            );
        }
    }
}

#[test]
fn test_different_window_sizes() {
    let sample_rate = 44100.0;
    let frequency = 440.0;
    let samples = generate_sine_wave(frequency, sample_rate, 0.1);

    for window_size in [1024, 2048, 4096] {
        let detector = YinDetector::new().with_window_size(window_size);
        let result = detector.detect_pitch(&samples, sample_rate);

        assert!(
            (result.frequency - frequency).abs() < 5.0,
            "Window size {} failed: detected {} vs expected {}",
            window_size,
            result.frequency,
            frequency
        );
    }
}

#[test]
fn test_edge_cases() {
    let sample_rate = 44100.0;

    let silence = vec![0.0; 2048];
    let detector = YinDetector::new();
    let result = detector.detect_pitch(&silence, sample_rate);
    assert_eq!(result.frequency, 0.0);
    assert_eq!(result.confidence, 0.0);

    let very_short = vec![0.5; 10];
    let result = detector.detect_pitch(&very_short, sample_rate);
    assert_eq!(result.frequency, 0.0);

    let empty = Vec::<f32>::new();
    let result = detector.detect_pitch(&empty, sample_rate);
    assert_eq!(result.frequency, 0.0);
}
