use apodize::{blackman_iter, hamming_iter, hanning_iter, nuttall_iter};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum WindowFunction {
    Hann,
    Hamming,
    Blackman,
    Nuttall,
    Rectangular,
}

impl WindowFunction {
    pub fn apply(&self, samples: &mut [f32]) {
        let len = samples.len();

        match self {
            Self::Hann => {
                for (i, sample) in samples.iter_mut().enumerate() {
                    *sample *= hanning_iter(len).nth(i).unwrap() as f32;
                }
            }
            Self::Hamming => {
                for (i, sample) in samples.iter_mut().enumerate() {
                    *sample *= hamming_iter(len).nth(i).unwrap() as f32;
                }
            }
            Self::Blackman => {
                for (i, sample) in samples.iter_mut().enumerate() {
                    *sample *= blackman_iter(len).nth(i).unwrap() as f32;
                }
            }
            Self::Nuttall => {
                for (i, sample) in samples.iter_mut().enumerate() {
                    *sample *= nuttall_iter(len).nth(i).unwrap() as f32;
                }
            }
            Self::Rectangular => {
                // No windowing needed
            }
        }
    }

    pub fn create_window(&self, size: usize) -> Vec<f32> {
        match self {
            Self::Hann => hanning_iter(size).map(|x| x as f32).collect(),
            Self::Hamming => hamming_iter(size).map(|x| x as f32).collect(),
            Self::Blackman => blackman_iter(size).map(|x| x as f32).collect(),
            Self::Nuttall => nuttall_iter(size).map(|x| x as f32).collect(),
            Self::Rectangular => vec![1.0; size],
        }
    }
}

impl From<crate::utils::config::WindowType> for WindowFunction {
    fn from(wt: crate::utils::config::WindowType) -> Self {
        match wt {
            crate::utils::config::WindowType::Hann => Self::Hann,
            crate::utils::config::WindowType::Hamming => Self::Hamming,
            crate::utils::config::WindowType::Blackman => Self::Blackman,
            crate::utils::config::WindowType::Kaiser => Self::Blackman, // Use Blackman as fallback for Kaiser
        }
    }
}
