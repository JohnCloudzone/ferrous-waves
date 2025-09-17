use ferrous_waves::analysis::temporal::onset::OnsetDetector;
use ferrous_waves::analysis::spectral::{StftProcessor, WindowFunction};
use ferrous_waves::{AudioFile, Result};

fn main() -> Result<()> {
    // Load audio file
    let audio = AudioFile::load("samples/drums.wav")?;

    // Extract mono channel
    let mono_samples: Vec<f32> = audio
        .buffer
        .samples
        .chunks(audio.buffer.channels as usize)
        .map(|chunk| chunk[0])
        .collect();

    // Compute spectrogram
    let stft = StftProcessor::new(2048, 512, WindowFunction::Hann);
    let spectrogram = stft.process(&mono_samples);

    // Detect onsets
    let detector = OnsetDetector::new();
    let spectral_flux = detector.spectral_flux(&spectrogram);
    let onset_times = detector.detect_onsets(
        &spectral_flux,
        512, // hop_size
        audio.buffer.sample_rate,
    );

    // Print results
    println!("Detected {} onsets in {}:", onset_times.len(), audio.path);
    for (i, time) in onset_times.iter().enumerate().take(10) {
        println!("  Onset {}: {:.3}s", i + 1, time);
    }
    if onset_times.len() > 10 {
        println!("  ... and {} more", onset_times.len() - 10);
    }

    Ok(())
}