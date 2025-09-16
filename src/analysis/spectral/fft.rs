use rustfft::{Fft, FftPlanner};
use num_complex::Complex;
use std::sync::Arc;

pub struct FftProcessor {
    size: usize,
    _planner: FftPlanner<f32>,
    forward_fft: Arc<dyn Fft<f32>>,
}

impl FftProcessor {
    pub fn new(size: usize) -> Self {
        let mut planner = FftPlanner::new();
        let forward_fft = planner.plan_fft_forward(size);

        Self {
            size,
            _planner: planner,
            forward_fft,
        }
    }

    pub fn process(&self, input: &[f32]) -> Vec<Complex<f32>> {
        assert_eq!(input.len(), self.size, "Input size must match FFT size");

        let mut buffer: Vec<Complex<f32>> = input
            .iter()
            .map(|&x| Complex::new(x, 0.0))
            .collect();

        self.forward_fft.process(&mut buffer);
        buffer
    }

    pub fn magnitude_spectrum(&self, input: &[f32]) -> Vec<f32> {
        let complex_output = self.process(input);
        complex_output
            .iter()
            .take(self.size / 2 + 1)  // Only positive frequencies
            .map(|c| c.norm())
            .collect()
    }

    pub fn power_spectrum(&self, input: &[f32]) -> Vec<f32> {
        let complex_output = self.process(input);
        complex_output
            .iter()
            .take(self.size / 2 + 1)
            .map(|c| c.norm_sqr())
            .collect()
    }

    pub fn phase_spectrum(&self, input: &[f32]) -> Vec<f32> {
        let complex_output = self.process(input);
        complex_output
            .iter()
            .take(self.size / 2 + 1)
            .map(|c| c.arg())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    #[test]
    fn test_fft_sine_wave() {
        let size = 1024;
        let processor = FftProcessor::new(size);

        // Generate a 440Hz sine wave at 44100Hz sample rate
        let sample_rate = 44100.0;
        let frequency = 440.0;
        let mut input = vec![0.0; size];

        for i in 0..size {
            input[i] = (2.0 * PI * frequency * i as f32 / sample_rate).sin();
        }

        let magnitude = processor.magnitude_spectrum(&input);

        // Find the peak
        let peak_bin = magnitude
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();

        // Convert bin to frequency
        let peak_freq = peak_bin as f32 * sample_rate / size as f32;

        // Should be close to 440Hz
        assert!((peak_freq - frequency).abs() < sample_rate / size as f32);
    }
}