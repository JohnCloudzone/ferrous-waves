use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisResult {
    pub summary: AudioSummary,
    pub spectral: SpectralAnalysis,
    pub temporal: TemporalAnalysis,
    pub visuals: VisualsData,
    pub insights: Vec<String>,
    pub recommendations: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioSummary {
    pub duration: f32,
    pub sample_rate: u32,
    pub channels: usize,
    pub format: String,
    pub peak_amplitude: f32,
    pub rms_level: f32,
    pub dynamic_range: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralAnalysis {
    pub spectral_centroid: Vec<f32>,
    pub spectral_rolloff: Vec<f32>,
    pub spectral_flux: Vec<f32>,
    pub mfcc: Vec<Vec<f32>>,
    pub dominant_frequencies: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalAnalysis {
    pub tempo: Option<f32>,
    pub beats: Vec<f32>,
    pub onsets: Vec<f32>,
    pub tempo_stability: f32,
    pub rhythmic_complexity: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualsData {
    pub waveform: Option<String>,  // Base64 encoded PNG
    pub spectrogram: Option<String>,
    pub mel_spectrogram: Option<String>,
    pub power_curve: Option<String>,
}

impl AnalysisResult {
    pub fn get_summary(&self) -> serde_json::Value {
        serde_json::json!({
            "duration": self.summary.duration,
            "tempo": self.temporal.tempo,
            "key": self.estimate_key(),
            "energy_profile": self.calculate_energy_profile(),
            "mood_descriptors": self.generate_mood_descriptors(),
        })
    }

    pub fn get_visuals(&self) -> serde_json::Value {
        serde_json::json!({
            "waveform": self.visuals.waveform,
            "spectrogram": self.visuals.spectrogram,
        })
    }

    fn estimate_key(&self) -> String {
        // Simplified key estimation
        "Am".to_string()
    }

    fn calculate_energy_profile(&self) -> String {
        match self.summary.rms_level {
            x if x < 0.3 => "low",
            x if x < 0.6 => "medium",
            _ => "high",
        }.to_string()
    }

    fn generate_mood_descriptors(&self) -> Vec<String> {
        let mut moods = Vec::new();

        if let Some(tempo) = self.temporal.tempo {
            if tempo < 80.0 {
                moods.push("relaxed".to_string());
            } else if tempo > 140.0 {
                moods.push("energetic".to_string());
            }
        }

        if self.summary.dynamic_range > 20.0 {
            moods.push("dynamic".to_string());
        }

        moods
    }
}