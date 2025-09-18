use ferrous_waves::analysis::pitch::{PitchDetector, PyinDetector, VibratoDetector, YinDetector};
use ferrous_waves::audio::decoder::AudioDecoder;
use std::env;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <audio_file>", args[0]);
        std::process::exit(1);
    }

    let file_path = Path::new(&args[1]);
    println!("Analyzing pitch in: {}", file_path.display());

    // Decode audio file
    let mut decoder = AudioDecoder::new(file_path)?;
    let sample_rate = decoder.sample_rate().unwrap_or(44100) as f32;
    let channels = decoder.num_channels().unwrap_or(1);
    let samples = decoder.decode_all()?;

    // Convert to mono for pitch detection
    let mono_samples = if channels > 1 {
        // Average channels to mono
        let samples_per_channel = samples.len() / channels;
        let mut mono = vec![0.0; samples_per_channel];
        for i in 0..samples_per_channel {
            for ch in 0..channels {
                mono[i] += samples[i * channels + ch] / channels as f32;
            }
        }
        mono
    } else {
        samples
    };

    println!("\n=== YIN Algorithm ===");
    analyze_with_yin(&mono_samples, sample_rate);

    println!("\n=== PYIN Algorithm ===");
    analyze_with_pyin(&mono_samples, sample_rate);

    println!("\n=== Pitch Track Analysis ===");
    analyze_pitch_track(&mono_samples, sample_rate);

    Ok(())
}

fn analyze_with_yin(samples: &[f32], sample_rate: f32) {
    let detector = YinDetector::new();

    // Analyze the first few seconds
    let window_size = detector.window_size();
    let hop_size = window_size / 2;

    println!("Analyzing with window size: {} samples", window_size);

    let mut pitch_values = Vec::new();

    for start in (0..samples.len().saturating_sub(window_size))
        .step_by(hop_size)
        .take(20)
    {
        let end = (start + window_size).min(samples.len());
        let window = &samples[start..end];

        let result = detector.detect_pitch(window, sample_rate);

        if result.confidence > 0.5 {
            pitch_values.push(result.frequency);
            println!(
                "Time: {:.2}s - Pitch: {:.1}Hz ({}) - Confidence: {:.2}",
                start as f32 / sample_rate,
                result.frequency,
                result.note_name.as_deref().unwrap_or("N/A"),
                result.confidence
            );
        }
    }

    if !pitch_values.is_empty() {
        let mean_pitch = pitch_values.iter().sum::<f32>() / pitch_values.len() as f32;
        println!("Average pitch: {:.1}Hz", mean_pitch);
    }
}

fn analyze_with_pyin(samples: &[f32], sample_rate: f32) {
    let detector = PyinDetector::new().with_candidates(5);

    let window_size = detector.window_size();
    let hop_size = window_size / 2;

    println!("Analyzing with window size: {} samples", window_size);

    let mut pitch_values = Vec::new();

    for start in (0..samples.len().saturating_sub(window_size))
        .step_by(hop_size)
        .take(20)
    {
        let end = (start + window_size).min(samples.len());
        let window = &samples[start..end];

        let result = detector.detect_pitch(window, sample_rate);

        if result.confidence > 0.5 {
            pitch_values.push(result.frequency);
            println!(
                "Time: {:.2}s - Pitch: {:.1}Hz ({}) - Confidence: {:.2} - Clarity: {:.2}",
                start as f32 / sample_rate,
                result.frequency,
                result.note_name.as_deref().unwrap_or("N/A"),
                result.confidence,
                result.clarity
            );
        }
    }

    if !pitch_values.is_empty() {
        let mean_pitch = pitch_values.iter().sum::<f32>() / pitch_values.len() as f32;
        println!("Average pitch: {:.1}Hz", mean_pitch);
    }
}

fn analyze_pitch_track(samples: &[f32], sample_rate: f32) {
    let detector = PyinDetector::new();
    let hop_size = 512;

    println!("Generating pitch track with hop size: {} samples", hop_size);

    let pitch_track = detector.detect_pitch_track(samples, sample_rate, hop_size);

    // Extract valid pitches for vibrato analysis
    let valid_pitches: Vec<f32> = pitch_track
        .frames
        .iter()
        .filter_map(|f| f.frequency)
        .collect();

    if valid_pitches.is_empty() {
        println!("No valid pitches detected");
        return;
    }

    // Calculate statistics
    let mean_pitch = valid_pitches.iter().sum::<f32>() / valid_pitches.len() as f32;
    let min_pitch = valid_pitches.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max_pitch = valid_pitches
        .iter()
        .fold(f32::NEG_INFINITY, |a, &b| a.max(b));

    println!("Pitch statistics:");
    println!("  Frames analyzed: {}", pitch_track.frames.len());
    println!("  Valid pitches: {}", valid_pitches.len());
    println!("  Mean pitch: {:.1}Hz", mean_pitch);
    println!("  Range: {:.1}Hz - {:.1}Hz", min_pitch, max_pitch);

    // Vibrato detection
    let vibrato_detector = VibratoDetector::new();
    if let Some(vibrato) = vibrato_detector.analyze(&valid_pitches, sample_rate, hop_size) {
        println!("\nVibrato detected:");
        println!("  Rate: {:.2}Hz", vibrato.rate);
        println!("  Depth: {:.1} cents", vibrato.depth_cents);
        println!("  Regularity: {:.2}", vibrato.regularity);
        println!("  Presence: {:.2}", vibrato.presence);

        if let Some(onset) = vibrato.onset_time {
            println!("  Onset: {:.2}s", onset);
        }
    } else {
        println!("\nNo significant vibrato detected");
    }

    // Show pitch contour
    println!("\nPitch contour (first 10 frames with valid pitch):");
    let mut shown = 0;
    for frame in &pitch_track.frames {
        if let Some(freq) = frame.frequency {
            println!("  {:.3}s: {:.1}Hz", frame.time, freq);
            shown += 1;
            if shown >= 10 {
                break;
            }
        }
    }
}
