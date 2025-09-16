use crate::analysis::spectral::fft::FftProcessor;
use crate::analysis::spectral::window::WindowFunction;
use ndarray::Array2;

pub struct StftProcessor {
    fft: FftProcessor,
    window: WindowFunction,
    fft_size: usize,
    hop_size: usize,
}

impl StftProcessor {
    pub fn new(fft_size: usize, hop_size: usize, window: WindowFunction) -> Self {
        Self {
            fft: FftProcessor::new(fft_size),
            window,
            fft_size,
            hop_size,
        }
    }

    pub fn process(&self, signal: &[f32]) -> Array2<f32> {
        let num_frames = (signal.len() - self.fft_size) / self.hop_size + 1;
        let num_bins = self.fft_size / 2 + 1;

        let mut spectrogram = Array2::zeros((num_bins, num_frames));
        let window_coeffs = self.window.create_window(self.fft_size);

        for (frame_idx, frame_start) in (0..signal.len())
            .step_by(self.hop_size)
            .enumerate()
            .take(num_frames)
        {
            let frame_end = (frame_start + self.fft_size).min(signal.len());
            let mut frame = vec![0.0; self.fft_size];

            // Copy and pad if necessary
            let copy_len = frame_end - frame_start;
            frame[..copy_len].copy_from_slice(&signal[frame_start..frame_end]);

            // Apply window
            for i in 0..self.fft_size {
                frame[i] *= window_coeffs[i];
            }

            // Compute magnitude spectrum
            let magnitude = self.fft.magnitude_spectrum(&frame);

            // Store in spectrogram
            for (bin_idx, &mag) in magnitude.iter().enumerate() {
                spectrogram[[bin_idx, frame_idx]] = mag;
            }
        }

        spectrogram
    }

    pub fn to_db(&self, spectrogram: &Array2<f32>) -> Array2<f32> {
        spectrogram.mapv(|x| 20.0 * (x + 1e-10).log10())
    }

    pub fn frequency_bins(&self, sample_rate: u32) -> Vec<f32> {
        let num_bins = self.fft_size / 2 + 1;
        (0..num_bins)
            .map(|i| i as f32 * sample_rate as f32 / self.fft_size as f32)
            .collect()
    }

    pub fn time_frames(&self, num_samples: usize, sample_rate: u32) -> Vec<f32> {
        let num_frames = (num_samples - self.fft_size) / self.hop_size + 1;
        (0..num_frames)
            .map(|i| i as f32 * self.hop_size as f32 / sample_rate as f32)
            .collect()
    }
}