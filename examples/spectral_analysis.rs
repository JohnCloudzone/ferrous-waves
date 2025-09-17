use ferrous_waves::analysis::spectral::{FftProcessor, StftProcessor, WindowFunction};
use ferrous_waves::{AudioFile, Result};

fn main() -> Result<()> {
    // Load audio file
    let audio = AudioFile::load("samples/music.wav")?;

    // Extract mono channel for analysis
    let mono_samples: Vec<f32> = audio
        .buffer
        .samples
        .chunks(audio.buffer.channels)
        .map(|chunk| chunk[0])
        .collect();

    // Perform FFT analysis
    let fft = FftProcessor::new(2048);
    let magnitude_spectrum = fft.magnitude_spectrum(&mono_samples[..2048]);

    // Find peak frequency
    let sample_rate = audio.buffer.sample_rate as f32;
    let (peak_bin, peak_mag) = magnitude_spectrum
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap();

    let peak_freq = peak_bin as f32 * sample_rate / 2048.0;
    println!(
        "Peak frequency: {:.2}Hz (magnitude: {:.3})",
        peak_freq, peak_mag
    );

    // Perform STFT for time-frequency analysis
    let stft = StftProcessor::new(2048, 512, WindowFunction::Hann);
    let spectrogram = stft.process(&mono_samples);

    println!(
        "Spectrogram shape: {} frequency bins x {} time frames",
        spectrogram.shape()[0],
        spectrogram.shape()[1]
    );

    Ok(())
}
