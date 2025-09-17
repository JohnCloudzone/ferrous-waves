use std::fs::{self, File};
use std::io::Write;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create samples directory
    fs::create_dir_all("samples")?;

    println!("Generating sample audio files...");

    // Generate test.wav - simple 440Hz sine wave
    generate_sine_wave("samples/test.wav", 440.0, 2.0)?;

    // Generate music.wav - chord with multiple frequencies
    generate_chord("samples/music.wav", 3.0)?;

    // Generate drums.wav - rhythmic clicks
    generate_drums("samples/drums.wav", 4.0)?;

    // Generate original.wav and processed.wav for comparison
    generate_sine_wave("samples/original.wav", 440.0, 1.0)?;
    generate_sine_wave("samples/processed.wav", 880.0, 1.5)?;

    // Generate large_file.wav for cache testing
    generate_sine_wave("samples/large_file.wav", 220.0, 10.0)?;

    // Generate track1-3.wav for batch processing
    generate_sine_wave("samples/track1.wav", 330.0, 2.0)?;
    generate_sine_wave("samples/track2.wav", 440.0, 2.5)?;
    generate_sine_wave("samples/track3.wav", 550.0, 3.0)?;

    println!("Sample files generated in samples/ directory");
    Ok(())
}

fn generate_sine_wave(
    path: &str,
    frequency: f32,
    duration: f32,
) -> Result<(), Box<dyn std::error::Error>> {
    let sample_rate = 44100;
    let num_samples = (sample_rate as f32 * duration) as usize;
    let mut data = Vec::with_capacity(num_samples);

    for i in 0..num_samples {
        let t = i as f32 / sample_rate as f32;
        let sample = (2.0 * std::f32::consts::PI * frequency * t).sin() * 0.5;
        data.push(sample);
    }

    write_wav(path, &data, sample_rate)?;
    println!("  Generated {} ({}Hz, {:.1}s)", path, frequency, duration);
    Ok(())
}

fn generate_chord(path: &str, duration: f32) -> Result<(), Box<dyn std::error::Error>> {
    let sample_rate = 44100;
    let num_samples = (sample_rate as f32 * duration) as usize;
    let mut data = Vec::with_capacity(num_samples);

    // C major chord (C4, E4, G4)
    let frequencies = [261.63, 329.63, 392.00];

    for i in 0..num_samples {
        let t = i as f32 / sample_rate as f32;
        let mut sample = 0.0;
        for &freq in &frequencies {
            sample += (2.0 * std::f32::consts::PI * freq * t).sin() * 0.3;
        }
        data.push(sample);
    }

    write_wav(path, &data, sample_rate)?;
    println!("  Generated {} (C major chord, {:.1}s)", path, duration);
    Ok(())
}

fn generate_drums(path: &str, duration: f32) -> Result<(), Box<dyn std::error::Error>> {
    let sample_rate = 44100;
    let num_samples = (sample_rate as f32 * duration) as usize;
    let mut data = vec![0.0; num_samples];

    let bpm = 120.0;
    let beat_interval = 60.0 / bpm;
    let samples_per_beat = (sample_rate as f32 * beat_interval) as usize;

    // Add drum hits
    for beat in 0..((duration / beat_interval) as usize) {
        let pos = beat * samples_per_beat;
        if pos < num_samples {
            // Kick drum on beats 1 and 3
            if beat % 4 == 0 || beat % 4 == 2 {
                add_kick(&mut data, pos, sample_rate);
            }
            // Snare on beats 2 and 4
            if beat % 4 == 1 || beat % 4 == 3 {
                add_snare(&mut data, pos, sample_rate);
            }
            // Hi-hat on every beat
            add_hihat(&mut data, pos, sample_rate);
        }
    }

    write_wav(path, &data, sample_rate)?;
    println!("  Generated {} (120 BPM drums, {:.1}s)", path, duration);
    Ok(())
}

fn add_kick(data: &mut [f32], position: usize, sample_rate: u32) {
    let duration = (sample_rate as f32 * 0.1) as usize;
    for i in 0..duration.min(data.len() - position) {
        let t = i as f32 / sample_rate as f32;
        let envelope = (-t * 20.0).exp();
        let freq = 60.0 * (1.0 + envelope * 2.0); // Pitch sweep
        data[position + i] += (2.0 * std::f32::consts::PI * freq * t).sin() * envelope * 0.8;
    }
}

fn add_snare(data: &mut [f32], position: usize, sample_rate: u32) {
    let duration = (sample_rate as f32 * 0.15) as usize;
    for i in 0..duration.min(data.len() - position) {
        let t = i as f32 / sample_rate as f32;
        let envelope = (-t * 15.0).exp();
        // Mix of tone and noise
        let tone = (2.0 * std::f32::consts::PI * 200.0 * t).sin();
        let noise = (fastrand::f32() * 2.0 - 1.0) * 0.3;
        data[position + i] += (tone + noise) * envelope * 0.6;
    }
}

fn add_hihat(data: &mut [f32], position: usize, sample_rate: u32) {
    let duration = (sample_rate as f32 * 0.05) as usize;
    for i in 0..duration.min(data.len() - position) {
        let t = i as f32 / sample_rate as f32;
        let envelope = (-t * 50.0).exp();
        // High frequency noise
        let noise = fastrand::f32() * 2.0 - 1.0;
        data[position + i] += noise * envelope * 0.3;
    }
}

fn write_wav(path: &str, data: &[f32], sample_rate: u32) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = File::create(path)?;

    // WAV header for mono float32
    let num_samples = data.len() as u32;
    let bytes_per_sample = 4u16; // 32-bit float
    let num_channels = 1u16;
    let byte_rate = sample_rate * num_channels as u32 * bytes_per_sample as u32;
    let block_align = num_channels * bytes_per_sample;
    let data_size = num_samples * bytes_per_sample as u32;
    let file_size = 36 + data_size; // Header is 44 bytes, minus 8 for RIFF header

    // RIFF header
    file.write_all(b"RIFF")?;
    file.write_all(&file_size.to_le_bytes())?;
    file.write_all(b"WAVE")?;

    // fmt chunk
    file.write_all(b"fmt ")?;
    file.write_all(&16u32.to_le_bytes())?; // chunk size
    file.write_all(&3u16.to_le_bytes())?; // format (3 = IEEE float)
    file.write_all(&num_channels.to_le_bytes())?;
    file.write_all(&sample_rate.to_le_bytes())?;
    file.write_all(&byte_rate.to_le_bytes())?;
    file.write_all(&block_align.to_le_bytes())?;
    file.write_all(&(bytes_per_sample * 8).to_le_bytes())?; // bits per sample

    // data chunk
    file.write_all(b"data")?;
    file.write_all(&data_size.to_le_bytes())?;

    // Write samples
    for &sample in data {
        file.write_all(&sample.to_le_bytes())?;
    }

    Ok(())
}
