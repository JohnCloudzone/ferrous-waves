use crate::audio::AudioFile;
use crate::analysis::spectral::{StftProcessor, WindowFunction};
use crate::analysis::temporal::{OnsetDetector, BeatTracker};
use crate::mcp::tools::{AnalysisResult, AudioSummary, SpectralAnalysis, TemporalAnalysis, VisualsData};
use crate::utils::error::Result;
use serde::{Serialize, Deserialize};

#[derive(Clone)]
pub struct AnalysisEngine {
    fft_size: usize,
    hop_size: usize,
    window_function: WindowFunction,
}

impl AnalysisEngine {
    pub fn new() -> Self {
        Self {
            fft_size: 2048,
            hop_size: 512,
            window_function: WindowFunction::Hann,
        }
    }

    pub fn with_config(fft_size: usize, hop_size: usize, window_function: WindowFunction) -> Self {
        Self {
            fft_size,
            hop_size,
            window_function,
        }
    }

    pub async fn analyze(&self, audio: &AudioFile) -> Result<AnalysisResult> {
        // Convert to mono for analysis
        let mono = audio.buffer.to_mono();

        // Calculate basic metrics
        let peak_amplitude = mono.iter()
            .map(|s| s.abs())
            .fold(0.0f32, |a, b| a.max(b));

        let rms_level = (mono.iter()
            .map(|s| s * s)
            .sum::<f32>() / mono.len() as f32)
            .sqrt();

        let dynamic_range = if rms_level > 0.0 {
            20.0 * (peak_amplitude / rms_level).log10()
        } else {
            0.0
        };

        // Spectral analysis
        let stft = StftProcessor::new(self.fft_size, self.hop_size, self.window_function);
        let spectrogram = stft.process(&mono);

        // Calculate spectral features
        let mut spectral_centroids = Vec::new();
        let mut spectral_flux = Vec::new();
        let mut dominant_frequencies = Vec::new();

        for frame_idx in 0..spectrogram.shape()[1] {
            let frame = spectrogram.column(frame_idx);
            let frame_slice = frame.as_slice().unwrap();

            // Spectral centroid
            let mut weighted_sum = 0.0;
            let mut magnitude_sum = 0.0;
            let mut peak_freq = 0.0;
            let mut peak_mag = 0.0;

            for (bin, &mag) in frame_slice.iter().enumerate() {
                let freq = bin as f32 * audio.buffer.sample_rate as f32 / self.fft_size as f32;
                weighted_sum += freq * mag;
                magnitude_sum += mag;

                if mag > peak_mag {
                    peak_mag = mag;
                    peak_freq = freq;
                }
            }

            if magnitude_sum > 0.0 {
                spectral_centroids.push(weighted_sum / magnitude_sum);
            }

            if frame_idx < 5 {
                dominant_frequencies.push(peak_freq);
            }

            // Spectral flux
            if frame_idx > 0 {
                let prev_frame = spectrogram.column(frame_idx - 1);
                let flux: f32 = frame_slice.iter()
                    .zip(prev_frame.iter())
                    .map(|(&curr, &prev)| (curr - prev).max(0.0).powi(2))
                    .sum();
                spectral_flux.push(flux.sqrt());
            }
        }

        // Temporal analysis
        let onset_detector = OnsetDetector::new();
        let onsets = onset_detector.detect_onsets(&spectral_flux, self.hop_size, audio.buffer.sample_rate);

        let beat_tracker = BeatTracker::new();
        let tempo = beat_tracker.estimate_tempo(&onsets);
        let beats = tempo.map(|t| beat_tracker.track_beats(&onsets, t))
            .unwrap_or_default();

        // Calculate tempo stability
        let tempo_stability = if beats.len() > 2 {
            let intervals: Vec<f32> = beats.windows(2)
                .map(|w| w[1] - w[0])
                .collect();
            let mean_interval = intervals.iter().sum::<f32>() / intervals.len() as f32;
            let variance = intervals.iter()
                .map(|&i| (i - mean_interval).powi(2))
                .sum::<f32>() / intervals.len() as f32;
            1.0 / (1.0 + variance.sqrt())
        } else {
            0.0
        };

        // Generate insights
        let mut insights = Vec::new();
        let mut recommendations = Vec::new();

        if peak_amplitude > 0.95 {
            insights.push("Audio contains potential clipping".to_string());
            recommendations.push("Consider reducing input gain to avoid distortion".to_string());
        }

        if let Some(t) = tempo {
            insights.push(format!("Detected tempo: {:.1} BPM", t));
            if t < 80.0 {
                insights.push("Slow tempo detected, suitable for ambient or relaxation".to_string());
            } else if t > 140.0 {
                insights.push("Fast tempo detected, suitable for energetic content".to_string());
            }
        }

        if dynamic_range < 6.0 {
            insights.push("Low dynamic range detected".to_string());
            recommendations.push("Consider applying less compression for more dynamic sound".to_string());
        } else if dynamic_range > 20.0 {
            insights.push("High dynamic range preserved".to_string());
        }

        if onsets.len() as f32 / audio.buffer.duration_seconds > 10.0 {
            insights.push("High rhythmic activity detected".to_string());
        }

        Ok(AnalysisResult {
            summary: AudioSummary {
                duration: audio.buffer.duration_seconds,
                sample_rate: audio.buffer.sample_rate,
                channels: audio.buffer.channels,
                format: format!("{:?}", audio.format),
                peak_amplitude,
                rms_level,
                dynamic_range,
            },
            spectral: SpectralAnalysis {
                spectral_centroid: spectral_centroids,
                spectral_rolloff: vec![],
                spectral_flux,
                mfcc: vec![],
                dominant_frequencies,
            },
            temporal: TemporalAnalysis {
                tempo,
                beats,
                onsets: onsets.clone(),
                tempo_stability,
                rhythmic_complexity: onsets.len() as f32 / audio.buffer.duration_seconds,
            },
            visuals: VisualsData {
                waveform: None,
                spectrogram: None,
                mel_spectrogram: None,
                power_curve: None,
            },
            insights,
            recommendations,
        })
    }

    pub async fn compare(&self, audio_a: &AudioFile, audio_b: &AudioFile) -> ComparisonResult {
        let analysis_a = self.analyze(audio_a).await.ok();
        let analysis_b = self.analyze(audio_b).await.ok();

        let tempo_difference = match (&analysis_a, &analysis_b) {
            (Some(a), Some(b)) => {
                match (a.temporal.tempo, b.temporal.tempo) {
                    (Some(ta), Some(tb)) => Some(ta - tb),
                    _ => None,
                }
            }
            _ => None,
        };

        let duration_difference = audio_a.buffer.duration_seconds - audio_b.buffer.duration_seconds;
        let sample_rate_match = audio_a.buffer.sample_rate == audio_b.buffer.sample_rate;

        ComparisonResult {
            file_a: FileInfo {
                path: audio_a.path.clone(),
                duration: audio_a.buffer.duration_seconds,
                sample_rate: audio_a.buffer.sample_rate,
                channels: audio_a.buffer.channels,
                tempo: analysis_a.as_ref().and_then(|a| a.temporal.tempo),
            },
            file_b: FileInfo {
                path: audio_b.path.clone(),
                duration: audio_b.buffer.duration_seconds,
                sample_rate: audio_b.buffer.sample_rate,
                channels: audio_b.buffer.channels,
                tempo: analysis_b.as_ref().and_then(|a| a.temporal.tempo),
            },
            comparison: ComparisonMetrics {
                duration_difference,
                sample_rate_match,
                tempo_difference,
                spectral_similarity: None,
            },
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonResult {
    pub file_a: FileInfo,
    pub file_b: FileInfo,
    pub comparison: ComparisonMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileInfo {
    pub path: String,
    pub duration: f32,
    pub sample_rate: u32,
    pub channels: usize,
    pub tempo: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonMetrics {
    pub duration_difference: f32,
    pub sample_rate_match: bool,
    pub tempo_difference: Option<f32>,
    pub spectral_similarity: Option<f32>,
}

impl Default for AnalysisEngine {
    fn default() -> Self {
        Self::new()
    }
}