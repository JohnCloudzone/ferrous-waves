use serde::{Deserialize, Serialize};
use std::f32::consts::PI;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VibratoAnalysis {
    pub rate: f32,
    pub depth_cents: f32,
    pub regularity: f32,
    pub presence: f32,
    pub onset_time: Option<f32>,
}

pub struct VibratoDetector {
    min_vibrato_rate: f32,
    max_vibrato_rate: f32,
}

impl VibratoDetector {
    pub fn new() -> Self {
        Self {
            min_vibrato_rate: 3.0,
            max_vibrato_rate: 12.0,
        }
    }

    pub fn analyze(
        &self,
        pitch_track: &[f32],
        sample_rate: f32,
        hop_size: usize,
    ) -> Option<VibratoAnalysis> {
        let valid_pitches: Vec<f32> = pitch_track.iter().filter(|&&f| f > 0.0).copied().collect();

        if valid_pitches.len() < 20 {
            return None;
        }

        let mean_pitch = valid_pitches.iter().sum::<f32>() / valid_pitches.len() as f32;

        let cents_deviation: Vec<f32> = valid_pitches
            .iter()
            .map(|&f| 1200.0 * (f / mean_pitch).log2())
            .collect();

        let detrended = self.detrend(&cents_deviation);

        let hop_duration = hop_size as f32 / sample_rate;
        let analysis_rate = 1.0 / hop_duration;

        let vibrato_spectrum = self.compute_spectrum(&detrended, analysis_rate);

        if let Some((rate, strength)) = self.find_vibrato_peak(&vibrato_spectrum, analysis_rate) {
            let depth = self.calculate_depth(&detrended);
            let regularity = self.calculate_regularity(&detrended, rate, analysis_rate);
            let presence = self.calculate_presence(strength, depth, regularity);
            let onset = self.detect_onset(&cents_deviation, rate, analysis_rate);

            Some(VibratoAnalysis {
                rate,
                depth_cents: depth,
                regularity,
                presence,
                onset_time: onset.map(|idx| idx as f32 * hop_duration),
            })
        } else {
            None
        }
    }

    fn detrend(&self, data: &[f32]) -> Vec<f32> {
        let n = data.len() as f32;
        let mean_x = (n - 1.0) / 2.0;
        let mean_y = data.iter().sum::<f32>() / n;

        let mut num = 0.0;
        let mut den = 0.0;
        for (i, &y) in data.iter().enumerate() {
            let x = i as f32;
            num += (x - mean_x) * (y - mean_y);
            den += (x - mean_x) * (x - mean_x);
        }

        let slope = if den > 0.0 { num / den } else { 0.0 };
        let intercept = mean_y - slope * mean_x;

        data.iter()
            .enumerate()
            .map(|(i, &y)| y - (slope * i as f32 + intercept))
            .collect()
    }

    fn compute_spectrum(&self, data: &[f32], sample_rate: f32) -> Vec<(f32, f32)> {
        let n = data.len();
        let mut spectrum = Vec::new();

        let freq_resolution = sample_rate / n as f32;
        let max_bin = ((self.max_vibrato_rate / freq_resolution) as usize).min(n / 2);
        let min_bin = ((self.min_vibrato_rate / freq_resolution) as usize).max(1);

        for bin in min_bin..=max_bin {
            let freq = bin as f32 * freq_resolution;
            let omega = 2.0 * PI * freq / sample_rate;

            let mut real = 0.0;
            let mut imag = 0.0;
            for (i, &sample) in data.iter().enumerate() {
                real += sample * (omega * i as f32).cos();
                imag += sample * (omega * i as f32).sin();
            }

            let magnitude = (real * real + imag * imag).sqrt() / n as f32;
            spectrum.push((freq, magnitude));
        }

        spectrum
    }

    fn find_vibrato_peak(&self, spectrum: &[(f32, f32)], _sample_rate: f32) -> Option<(f32, f32)> {
        let (peak_freq, peak_magnitude) = spectrum
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())?;

        if *peak_magnitude > 0.01 {
            Some((*peak_freq, *peak_magnitude))
        } else {
            None
        }
    }

    fn calculate_depth(&self, detrended: &[f32]) -> f32 {
        let mut max_val = f32::NEG_INFINITY;
        let mut min_val = f32::INFINITY;

        for &val in detrended {
            max_val = max_val.max(val);
            min_val = min_val.min(val);
        }

        (max_val - min_val) / 2.0
    }

    fn calculate_regularity(&self, data: &[f32], rate: f32, sample_rate: f32) -> f32 {
        let period = sample_rate / rate;
        let period_samples = period.round() as usize;

        if period_samples >= data.len() || period_samples == 0 {
            return 0.0;
        }

        let mut correlations = Vec::new();
        let num_periods = data.len() / period_samples;

        for i in 1..num_periods {
            let start1 = 0;
            let end1 = period_samples;
            let start2 = i * period_samples;
            let end2 = ((i + 1) * period_samples).min(data.len());

            if end2 <= start2 || end1 <= start1 {
                continue;
            }

            let segment1 = &data[start1..end1];
            let segment2 = &data[start2..end2];
            let min_len = segment1.len().min(segment2.len());

            let corr = self.correlation(&segment1[..min_len], &segment2[..min_len]);
            correlations.push(corr);
        }

        if correlations.is_empty() {
            return 0.0;
        }

        correlations.iter().sum::<f32>() / correlations.len() as f32
    }

    fn correlation(&self, a: &[f32], b: &[f32]) -> f32 {
        let n = a.len() as f32;
        let mean_a = a.iter().sum::<f32>() / n;
        let mean_b = b.iter().sum::<f32>() / n;

        let mut cov = 0.0;
        let mut var_a = 0.0;
        let mut var_b = 0.0;

        for i in 0..a.len() {
            let da = a[i] - mean_a;
            let db = b[i] - mean_b;
            cov += da * db;
            var_a += da * da;
            var_b += db * db;
        }

        if var_a > 0.0 && var_b > 0.0 {
            cov / (var_a.sqrt() * var_b.sqrt())
        } else {
            0.0
        }
    }

    fn calculate_presence(&self, strength: f32, depth: f32, regularity: f32) -> f32 {
        let depth_factor = (depth / 100.0).min(1.0);
        let strength_factor = (strength * 10.0).min(1.0);
        let regularity_factor = regularity.clamp(0.0, 1.0);

        (depth_factor * 0.4 + strength_factor * 0.3 + regularity_factor * 0.3).min(1.0)
    }

    fn detect_onset(&self, cents_deviation: &[f32], rate: f32, sample_rate: f32) -> Option<usize> {
        let period = sample_rate / rate;
        let window = (period * 2.0).round() as usize;

        if window >= cents_deviation.len() {
            return None;
        }

        let mut max_change = 0.0;
        let mut onset_idx = 0;

        for i in window..cents_deviation.len() - window {
            let before_var = self.variance(&cents_deviation[i - window..i]);
            let after_var = self.variance(&cents_deviation[i..i + window]);

            let change = (after_var - before_var).abs();
            if change > max_change && after_var > before_var * 2.0 {
                max_change = change;
                onset_idx = i;
            }
        }

        if max_change > 10.0 {
            Some(onset_idx)
        } else {
            None
        }
    }

    fn variance(&self, data: &[f32]) -> f32 {
        if data.is_empty() {
            return 0.0;
        }

        let mean = data.iter().sum::<f32>() / data.len() as f32;
        data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32
    }
}

impl Default for VibratoDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_vibrato_pitch_track(
        base_freq: f32,
        vibrato_rate: f32,
        vibrato_depth: f32,
        duration: f32,
        hop_size: usize,
        sample_rate: f32,
    ) -> Vec<f32> {
        let num_frames = ((duration * sample_rate) / hop_size as f32) as usize;
        (0..num_frames)
            .map(|i| {
                let t = (i * hop_size) as f32 / sample_rate;
                base_freq * (1.0 + vibrato_depth * (2.0 * PI * vibrato_rate * t).sin())
            })
            .collect()
    }

    #[test]
    fn test_vibrato_detection() {
        let sample_rate = 44100.0;
        let hop_size = 512;
        let pitch_track =
            generate_vibrato_pitch_track(440.0, 5.0, 0.03, 1.0, hop_size, sample_rate);

        let detector = VibratoDetector::new();
        let analysis = detector.analyze(&pitch_track, sample_rate, hop_size);

        assert!(analysis.is_some());
        let vibrato = analysis.unwrap();
        assert!((vibrato.rate - 5.0).abs() < 0.5);
        assert!(vibrato.depth_cents > 20.0);
        assert!(vibrato.presence > 0.5);
    }

    #[test]
    fn test_no_vibrato() {
        let pitch_track = vec![440.0; 100];
        let detector = VibratoDetector::new();
        let analysis = detector.analyze(&pitch_track, 44100.0, 512);

        assert!(analysis.is_none());
    }
}
