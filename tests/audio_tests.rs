use ferrous_waves::audio::{AudioBuffer, AudioFile, AudioFormat};
use std::f32::consts::PI;

#[test]
fn test_audio_format_detection() {
    assert_eq!(AudioFormat::from_path("test.wav"), AudioFormat::Wav);
    assert_eq!(AudioFormat::from_path("test.mp3"), AudioFormat::Mp3);
    assert_eq!(AudioFormat::from_path("test.flac"), AudioFormat::Flac);
    assert_eq!(AudioFormat::from_path("test.ogg"), AudioFormat::Ogg);
    assert_eq!(AudioFormat::from_path("test.m4a"), AudioFormat::M4a);
    assert_eq!(AudioFormat::from_path("test.unknown"), AudioFormat::Unknown);
}

#[test]
fn test_audio_format_case_insensitive() {
    assert_eq!(AudioFormat::from_path("test.WAV"), AudioFormat::Wav);
    assert_eq!(AudioFormat::from_path("test.MP3"), AudioFormat::Mp3);
    assert_eq!(AudioFormat::from_path("test.FLAC"), AudioFormat::Flac);
}

#[test]
fn test_audio_format_is_supported() {
    assert!(AudioFormat::Wav.is_supported());
    assert!(AudioFormat::Mp3.is_supported());
    assert!(AudioFormat::Flac.is_supported());
    assert!(!AudioFormat::Unknown.is_supported());
}

#[test]
fn test_audio_buffer_creation() {
    let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let buffer = AudioBuffer::new(samples.clone(), 44100, 2);

    assert_eq!(buffer.sample_rate, 44100);
    assert_eq!(buffer.channels, 2);
    assert_eq!(buffer.duration_seconds, 3.0 / 44100.0);
}

#[test]
fn test_audio_buffer_to_mono_stereo() {
    let stereo_samples = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let buffer = AudioBuffer::new(stereo_samples, 44100, 2);
    let mono = buffer.to_mono();

    assert_eq!(mono.len(), 3);
    assert_eq!(mono[0], 1.5); // (1.0 + 2.0) / 2
    assert_eq!(mono[1], 3.5); // (3.0 + 4.0) / 2
    assert_eq!(mono[2], 5.5); // (5.0 + 6.0) / 2
}

#[test]
fn test_audio_buffer_to_mono_already_mono() {
    let mono_samples = vec![1.0, 2.0, 3.0];
    let buffer = AudioBuffer::new(mono_samples.clone(), 44100, 1);
    let mono = buffer.to_mono();

    assert_eq!(mono, mono_samples);
}

#[test]
fn test_audio_buffer_get_channel() {
    let samples = vec![
        1.0, 2.0, 3.0,  // Frame 1: L=1.0, R=2.0, C=3.0
        4.0, 5.0, 6.0,  // Frame 2: L=4.0, R=5.0, C=6.0
    ];
    let buffer = AudioBuffer::new(samples, 44100, 3);

    let left = buffer.get_channel(0).unwrap();
    assert_eq!(left, vec![1.0, 4.0]);

    let right = buffer.get_channel(1).unwrap();
    assert_eq!(right, vec![2.0, 5.0]);

    let center = buffer.get_channel(2).unwrap();
    assert_eq!(center, vec![3.0, 6.0]);

    assert!(buffer.get_channel(3).is_none());
}

#[test]
fn test_generate_sine_wave() {
    let sample_rate = 44100.0;
    let frequency = 440.0; // A4 note
    let duration = 1.0;
    let num_samples = (sample_rate * duration) as usize;

    let mut samples = Vec::with_capacity(num_samples);
    for i in 0..num_samples {
        let t = i as f32 / sample_rate;
        samples.push((2.0 * PI * frequency * t).sin());
    }

    let buffer = AudioBuffer::new(samples, sample_rate as u32, 1);

    // Verify buffer properties
    assert_eq!(buffer.sample_rate, 44100);
    assert_eq!(buffer.channels, 1);
    assert!((buffer.duration_seconds - 1.0).abs() < 0.001);

    // Verify the sine wave has expected properties
    let mono = buffer.to_mono();
    let max = mono.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let min = mono.iter().fold(f32::INFINITY, |a, &b| a.min(b));

    assert!((max - 1.0).abs() < 0.01);
    assert!((min + 1.0).abs() < 0.01);
}