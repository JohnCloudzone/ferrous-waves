use super::{normalized_square_difference, PitchDetector, PitchResult};

pub struct YinDetector {
    threshold: f32,
    window_size: usize,
    min_frequency: f32,
    max_frequency: f32,
}

impl YinDetector {
    pub fn new() -> Self {
        Self {
            threshold: 0.1,
            window_size: 2048,
            min_frequency: 50.0,
            max_frequency: 2000.0,
        }
    }

    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold.clamp(0.05, 0.5);
        self
    }

    pub fn with_window_size(mut self, size: usize) -> Self {
        self.window_size = size;
        self
    }

    pub fn with_frequency_range(mut self, min: f32, max: f32) -> Self {
        self.min_frequency = min;
        self.max_frequency = max;
        self
    }

    fn difference_function(&self, samples: &[f32]) -> Vec<f32> {
        let w = samples.len();
        let half_w = w / 2;

        (0..half_w)
            .map(|tau| {
                if tau == 0 {
                    0.0
                } else {
                    normalized_square_difference(samples, tau)
                }
            })
            .collect()
    }

    fn cumulative_mean_normalized_difference(&self, diff: &[f32]) -> Vec<f32> {
        let mut cmnd = vec![0.0; diff.len()];
        cmnd[0] = 1.0;

        let mut running_sum = 0.0;
        for tau in 1..diff.len() {
            running_sum += diff[tau];
            if running_sum > 0.0 {
                cmnd[tau] = diff[tau] * tau as f32 / running_sum;
            } else {
                cmnd[tau] = 1.0;
            }
        }

        cmnd
    }

    fn absolute_threshold(&self, cmnd: &[f32]) -> Option<usize> {
        for tau in 2..cmnd.len() {
            if cmnd[tau] < self.threshold {
                let mut min_tau = tau;
                while min_tau + 1 < cmnd.len() && cmnd[min_tau + 1] < cmnd[min_tau] {
                    min_tau += 1;
                }
                return Some(min_tau);
            }
        }
        None
    }

    fn parabolic_interpolation(&self, cmnd: &[f32], tau_estimate: usize) -> f32 {
        if tau_estimate == 0 || tau_estimate >= cmnd.len() - 1 {
            return tau_estimate as f32;
        }

        let x0 = tau_estimate.saturating_sub(1);
        let x2 = (tau_estimate + 1).min(cmnd.len() - 1);

        let y0 = cmnd[x0];
        let y1 = cmnd[tau_estimate];
        let y2 = cmnd[x2];

        let a = (y2 - 2.0 * y1 + y0) / 2.0;
        let b = (y2 - y0) / 2.0;

        if a.abs() < f32::EPSILON {
            return tau_estimate as f32;
        }

        let x_vertex = tau_estimate as f32 - b / (2.0 * a);
        x_vertex.max(0.0)
    }

    fn calculate_clarity(&self, cmnd: &[f32], best_tau: usize) -> f32 {
        if best_tau >= cmnd.len() {
            return 0.0;
        }

        let min_val = cmnd[best_tau];
        let clarity = 1.0 - min_val;
        clarity.clamp(0.0, 1.0)
    }
}

impl Default for YinDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl PitchDetector for YinDetector {
    fn detect_pitch(&self, samples: &[f32], sample_rate: f32) -> PitchResult {
        if samples.is_empty() {
            return PitchResult::new(0.0, 0.0, 0.0);
        }

        let effective_window = samples.len().min(self.window_size);
        let windowed = &samples[..effective_window];

        let diff = self.difference_function(windowed);
        let cmnd = self.cumulative_mean_normalized_difference(&diff);

        if let Some(tau_estimate) = self.absolute_threshold(&cmnd) {
            let min_tau = (sample_rate / self.max_frequency) as usize;
            let max_tau = (sample_rate / self.min_frequency) as usize;

            if tau_estimate >= min_tau && tau_estimate <= max_tau {
                let refined_tau = self.parabolic_interpolation(&cmnd, tau_estimate);
                let frequency = sample_rate / refined_tau;
                let clarity = self.calculate_clarity(&cmnd, tau_estimate);
                let confidence = clarity * (1.0 - cmnd[tau_estimate]).max(0.0);

                return PitchResult::new(frequency, confidence, clarity);
            }
        }

        PitchResult::new(0.0, 0.0, 0.0)
    }

    fn window_size(&self) -> usize {
        self.window_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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

    #[test]
    fn test_yin_pure_sine() {
        let sample_rate = 44100.0;
        let frequency = 440.0;
        let samples = generate_sine_wave(frequency, sample_rate, 0.1);

        let detector = YinDetector::new();
        let result = detector.detect_pitch(&samples, sample_rate);

        assert!((result.frequency - frequency).abs() < 5.0);
        assert!(result.confidence > 0.8);
        assert!(result.clarity > 0.8);
    }

    #[test]
    fn test_yin_different_frequencies() {
        let sample_rate = 44100.0;
        let test_frequencies = [220.0, 440.0, 880.0, 1760.0];

        for &freq in &test_frequencies {
            let samples = generate_sine_wave(freq, sample_rate, 0.1);
            let detector = YinDetector::new();
            let result = detector.detect_pitch(&samples, sample_rate);

            let error_percent = ((result.frequency - freq).abs() / freq) * 100.0;
            assert!(
                error_percent < 2.0,
                "Frequency {} detected as {} ({}% error)",
                freq,
                result.frequency,
                error_percent
            );
        }
    }

    #[test]
    fn test_yin_with_noise() {
        let sample_rate = 44100.0;
        let frequency = 440.0;
        let mut samples = generate_sine_wave(frequency, sample_rate, 0.1);

        for sample in &mut samples {
            *sample += (rand::random::<f32>() - 0.5) * 0.1;
        }

        let detector = YinDetector::new();
        let result = detector.detect_pitch(&samples, sample_rate);

        assert!((result.frequency - frequency).abs() < 10.0);
        assert!(result.confidence > 0.5);
    }

    #[test]
    fn test_yin_silence() {
        let sample_rate = 44100.0;
        let samples = vec![0.0; 2048];

        let detector = YinDetector::new();
        let result = detector.detect_pitch(&samples, sample_rate);

        assert_eq!(result.frequency, 0.0);
        assert_eq!(result.confidence, 0.0);
    }
}
