use super::cpu_features::{log_cpu_features, SimdLevel};
use num_complex::Complex32;
use realfft::RealFftPlanner;
use wide::{f32x4, f32x8};

pub struct SimdFft {
    planner: RealFftPlanner<f32>,
    fft_size: usize,
    simd_level: SimdLevel,
}

impl SimdFft {
    pub fn new(fft_size: usize) -> Self {
        let simd_level = SimdLevel::detect();
        Self {
            planner: RealFftPlanner::new(),
            fft_size,
            simd_level,
        }
    }

    pub fn new_with_logging(fft_size: usize) -> Self {
        log_cpu_features();
        Self::new(fft_size)
    }

    pub fn process(&mut self, input: &[f32]) -> Vec<Complex32> {
        let mut buffer = input.to_vec();
        buffer.resize(self.fft_size, 0.0);

        let fft = self.planner.plan_fft_forward(self.fft_size);
        let mut spectrum = fft.make_output_vec();

        fft.process(&mut buffer, &mut spectrum).unwrap();
        spectrum
    }

    pub fn magnitude_spectrum(&self, spectrum: &[Complex32]) -> Vec<f32> {
        match self.simd_level {
            SimdLevel::None => Self::magnitude_spectrum_scalar(spectrum),
            _ => Self::magnitude_spectrum_simd(spectrum),
        }
    }

    fn magnitude_spectrum_scalar(spectrum: &[Complex32]) -> Vec<f32> {
        spectrum
            .iter()
            .map(|c| (c.re * c.re + c.im * c.im).sqrt())
            .collect()
    }

    #[inline(always)]
    pub fn magnitude_spectrum_simd(spectrum: &[Complex32]) -> Vec<f32> {
        let mut magnitudes = Vec::with_capacity(spectrum.len());

        let chunks = spectrum.chunks_exact(8);
        let remainder = chunks.remainder();

        for chunk in chunks {
            let re0 = f32x8::from([
                chunk[0].re,
                chunk[1].re,
                chunk[2].re,
                chunk[3].re,
                chunk[4].re,
                chunk[5].re,
                chunk[6].re,
                chunk[7].re,
            ]);
            let im0 = f32x8::from([
                chunk[0].im,
                chunk[1].im,
                chunk[2].im,
                chunk[3].im,
                chunk[4].im,
                chunk[5].im,
                chunk[6].im,
                chunk[7].im,
            ]);

            let re_sq = re0 * re0;
            let im_sq = im0 * im0;
            let sum = re_sq + im_sq;
            let mags = sum.sqrt();

            magnitudes.extend_from_slice(&mags.to_array());
        }

        for c in remainder {
            magnitudes.push((c.re * c.re + c.im * c.im).sqrt());
        }

        magnitudes
    }

    #[inline(always)]
    pub fn power_spectrum_simd(spectrum: &[Complex32]) -> Vec<f32> {
        let mut powers = Vec::with_capacity(spectrum.len());

        let chunks = spectrum.chunks_exact(8);
        let remainder = chunks.remainder();

        for chunk in chunks {
            let re0 = f32x8::from([
                chunk[0].re,
                chunk[1].re,
                chunk[2].re,
                chunk[3].re,
                chunk[4].re,
                chunk[5].re,
                chunk[6].re,
                chunk[7].re,
            ]);
            let im0 = f32x8::from([
                chunk[0].im,
                chunk[1].im,
                chunk[2].im,
                chunk[3].im,
                chunk[4].im,
                chunk[5].im,
                chunk[6].im,
                chunk[7].im,
            ]);

            let re_sq = re0 * re0;
            let im_sq = im0 * im0;
            let sum = re_sq + im_sq;

            powers.extend_from_slice(&sum.to_array());
        }

        for c in remainder {
            powers.push(c.re * c.re + c.im * c.im);
        }

        powers
    }

    #[inline(always)]
    pub fn apply_window_simd(samples: &mut [f32], window: &[f32]) {
        assert_eq!(samples.len(), window.len());

        let mut chunks_samples = samples.chunks_exact_mut(8);
        let chunks_window = window.chunks_exact(8);
        let remainder_window = chunks_window.remainder();

        for (sample_chunk, window_chunk) in chunks_samples.by_ref().zip(chunks_window) {
            let samples_vec = f32x8::from(*<&[f32; 8]>::try_from(&sample_chunk[..8]).unwrap());
            let window_vec = f32x8::from(*<&[f32; 8]>::try_from(&window_chunk[..8]).unwrap());
            let result = samples_vec * window_vec;
            sample_chunk.copy_from_slice(&result.to_array());
        }

        let remainder_samples = chunks_samples.into_remainder();
        for (s, w) in remainder_samples.iter_mut().zip(remainder_window) {
            *s *= w;
        }
    }
}

pub struct SimdWindowFunctions;

impl SimdWindowFunctions {
    #[inline(always)]
    pub fn hann_simd(size: usize) -> Vec<f32> {
        let mut window = vec![0.0f32; size];
        let scale = std::f32::consts::TAU / (size - 1) as f32;

        let mut chunks = window.chunks_exact_mut(4);

        let mut i = 0;
        for chunk in chunks.by_ref() {
            let indices = f32x4::from([i as f32, (i + 1) as f32, (i + 2) as f32, (i + 3) as f32]);
            let angles = indices * f32x4::splat(scale);

            for (j, val) in chunk.iter_mut().enumerate().take(4) {
                *val = 0.5 * (1.0 - (angles.as_array_ref()[j]).cos());
            }

            i += 4;
        }

        let remainder = chunks.into_remainder();
        for (j, w) in remainder.iter_mut().enumerate() {
            let angle = (i + j) as f32 * scale;
            *w = 0.5 * (1.0 - angle.cos());
        }

        window
    }

    #[inline(always)]
    pub fn hamming_simd(size: usize) -> Vec<f32> {
        let mut window = vec![0.0f32; size];
        let scale = std::f32::consts::TAU / (size - 1) as f32;

        let mut chunks = window.chunks_exact_mut(4);

        let mut i = 0;
        for chunk in chunks.by_ref() {
            let indices = f32x4::from([i as f32, (i + 1) as f32, (i + 2) as f32, (i + 3) as f32]);
            let angles = indices * f32x4::splat(scale);

            for (j, val) in chunk.iter_mut().enumerate().take(4) {
                *val = 0.54 - 0.46 * angles.as_array_ref()[j].cos();
            }

            i += 4;
        }

        let remainder = chunks.into_remainder();
        for (j, w) in remainder.iter_mut().enumerate() {
            let angle = (i + j) as f32 * scale;
            *w = 0.54 - 0.46 * angle.cos();
        }

        window
    }

    #[inline(always)]
    pub fn blackman_simd(size: usize) -> Vec<f32> {
        let mut window = vec![0.0f32; size];
        let scale = std::f32::consts::TAU / (size - 1) as f32;

        let mut chunks = window.chunks_exact_mut(4);

        let mut i = 0;
        for chunk in chunks.by_ref() {
            let indices = f32x4::from([i as f32, (i + 1) as f32, (i + 2) as f32, (i + 3) as f32]);
            let angles = indices * f32x4::splat(scale);

            for (j, val) in chunk.iter_mut().enumerate().take(4) {
                let angle = angles.as_array_ref()[j];
                *val = 0.42 - 0.5 * angle.cos() + 0.08 * (2.0 * angle).cos();
            }

            i += 4;
        }

        let remainder = chunks.into_remainder();
        for (j, w) in remainder.iter_mut().enumerate() {
            let angle = (i + j) as f32 * scale;
            *w = 0.42 - 0.5 * angle.cos() + 0.08 * (2.0 * angle).cos();
        }

        window
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::spectral::fft::FftProcessor;

    #[test]
    fn test_simd_fft_matches_standard() {
        let size = 2048;
        let input: Vec<f32> = (0..size).map(|i| (i as f32 * 0.1).sin()).collect();

        let mut simd_fft = SimdFft::new(size);
        let simd_result = simd_fft.process(&input);

        let standard_fft = FftProcessor::new(size);
        let standard_result = standard_fft.process(&input);

        for (simd, standard) in simd_result.iter().zip(standard_result.iter()).take(100) {
            assert!((simd.re - standard.re).abs() < 0.001);
            assert!((simd.im - standard.im).abs() < 0.001);
        }
    }

    #[test]
    fn test_simd_magnitude_spectrum() {
        let spectrum: Vec<Complex32> = (0..256)
            .map(|i| Complex32::new(i as f32 * 0.1, i as f32 * 0.05))
            .collect();

        let simd_mags = SimdFft::magnitude_spectrum_simd(&spectrum);

        let standard_mags: Vec<f32> = spectrum
            .iter()
            .map(|c| (c.re * c.re + c.im * c.im).sqrt())
            .collect();

        for (simd, standard) in simd_mags.iter().zip(standard_mags.iter()) {
            assert!((simd - standard).abs() < 0.001);
        }
    }

    #[test]
    fn test_simd_window_functions() {
        let size = 1024;

        let simd_hann = SimdWindowFunctions::hann_simd(size);
        let simd_hamming = SimdWindowFunctions::hamming_simd(size);
        let simd_blackman = SimdWindowFunctions::blackman_simd(size);

        assert_eq!(simd_hann.len(), size);
        assert_eq!(simd_hamming.len(), size);
        assert_eq!(simd_blackman.len(), size);

        assert!((simd_hann[0] - 0.0).abs() < 0.001);
        assert!((simd_hann[size / 2] - 1.0).abs() < 0.01);

        assert!((simd_hamming[0] - 0.08).abs() < 0.01);
        assert!((simd_hamming[size / 2] - 1.0).abs() < 0.01);
    }
}
