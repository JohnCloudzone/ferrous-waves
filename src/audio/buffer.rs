use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct AudioBuffer {
    pub samples: Arc<Vec<f32>>,
    pub sample_rate: u32,
    pub channels: usize,
    pub duration_seconds: f32,
}

impl AudioBuffer {
    pub fn new(samples: Vec<f32>, sample_rate: u32, channels: usize) -> Self {
        let duration_seconds = samples.len() as f32 / (sample_rate as f32 * channels as f32);

        Self {
            samples: Arc::new(samples),
            sample_rate,
            channels,
            duration_seconds,
        }
    }

    pub fn get_channel(&self, channel: usize) -> Option<Vec<f32>> {
        if channel >= self.channels {
            return None;
        }

        let mut channel_samples = Vec::with_capacity(self.samples.len() / self.channels);

        for i in (channel..self.samples.len()).step_by(self.channels) {
            channel_samples.push(self.samples[i]);
        }

        Some(channel_samples)
    }

    pub fn to_mono(&self) -> Vec<f32> {
        if self.channels == 1 {
            return (*self.samples).clone();
        }

        let mut mono = Vec::with_capacity(self.samples.len() / self.channels);

        for chunk in self.samples.chunks(self.channels) {
            let sum: f32 = chunk.iter().sum();
            mono.push(sum / self.channels as f32);
        }

        mono
    }
}