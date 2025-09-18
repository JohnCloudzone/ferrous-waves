use serde::{Deserialize, Serialize};

pub mod pyin;
pub mod vibrato;
pub mod yin;

pub use pyin::PyinDetector;
pub use vibrato::{VibratoAnalysis, VibratoDetector};
pub use yin::YinDetector;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PitchResult {
    pub frequency: f32,
    pub confidence: f32,
    pub clarity: f32,
    pub midi_note: Option<u8>,
    pub note_name: Option<String>,
    pub cents_offset: Option<f32>,
    pub vibrato: Option<VibratoAnalysis>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PitchTrack {
    pub frames: Vec<PitchFrame>,
    pub sample_rate: f32,
    pub hop_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PitchFrame {
    pub time: f32,
    pub frequency: Option<f32>,
    pub confidence: f32,
    pub clarity: f32,
    pub amplitude: f32,
}

pub trait PitchDetector: Send + Sync {
    fn detect_pitch(&self, samples: &[f32], sample_rate: f32) -> PitchResult;

    fn detect_pitch_track(&self, samples: &[f32], sample_rate: f32, hop_size: usize) -> PitchTrack {
        let window_size = self.window_size();
        let mut frames = Vec::new();

        for start in (0..samples.len().saturating_sub(window_size)).step_by(hop_size) {
            let end = (start + window_size).min(samples.len());
            let window = &samples[start..end];

            let result = self.detect_pitch(window, sample_rate);
            let amplitude = window.iter().map(|x| x.abs()).sum::<f32>() / window.len() as f32;

            frames.push(PitchFrame {
                time: start as f32 / sample_rate,
                frequency: if result.confidence > 0.5 {
                    Some(result.frequency)
                } else {
                    None
                },
                confidence: result.confidence,
                clarity: result.clarity,
                amplitude,
            });
        }

        PitchTrack {
            frames,
            sample_rate,
            hop_size,
        }
    }

    fn window_size(&self) -> usize;
}

impl PitchResult {
    pub fn new(frequency: f32, confidence: f32, clarity: f32) -> Self {
        let (midi_note, note_name, cents_offset) = if frequency > 0.0 {
            let midi = frequency_to_midi(frequency);
            let note = midi_to_note_name(midi.round() as u8);
            let cents = (midi - midi.round()) * 100.0;
            (Some(midi.round() as u8), Some(note), Some(cents))
        } else {
            (None, None, None)
        };

        Self {
            frequency,
            confidence,
            clarity,
            midi_note,
            note_name,
            cents_offset,
            vibrato: None,
        }
    }

    pub fn with_vibrato(mut self, vibrato: VibratoAnalysis) -> Self {
        self.vibrato = Some(vibrato);
        self
    }
}

fn frequency_to_midi(frequency: f32) -> f32 {
    69.0 + 12.0 * (frequency / 440.0).log2()
}

fn midi_to_note_name(midi: u8) -> String {
    let notes = [
        "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
    ];
    let octave = (midi / 12) as i32 - 1;
    let note_index = (midi % 12) as usize;
    format!("{}{}", notes[note_index], octave)
}

pub fn autocorrelation(samples: &[f32], lag: usize) -> f32 {
    let n = samples.len();
    if lag >= n {
        return 0.0;
    }

    let mut sum = 0.0;
    for i in 0..n - lag {
        sum += samples[i] * samples[i + lag];
    }
    sum / (n - lag) as f32
}

pub fn normalized_square_difference(samples: &[f32], lag: usize) -> f32 {
    let n = samples.len();
    if lag >= n {
        return f32::MAX;
    }

    let mut diff_sum = 0.0;
    for i in 0..n - lag {
        let diff = samples[i] - samples[i + lag];
        diff_sum += diff * diff;
    }
    diff_sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frequency_to_midi() {
        assert!((frequency_to_midi(440.0) - 69.0).abs() < 0.001);
        assert!((frequency_to_midi(880.0) - 81.0).abs() < 0.001);
        assert!((frequency_to_midi(220.0) - 57.0).abs() < 0.001);
    }

    #[test]
    fn test_note_names() {
        assert_eq!(midi_to_note_name(60), "C4");
        assert_eq!(midi_to_note_name(69), "A4");
        assert_eq!(midi_to_note_name(72), "C5");
    }
}
