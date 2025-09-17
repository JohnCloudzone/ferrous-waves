use crate::visualization::renderer::Renderer;
use crate::utils::error::Result;
use std::path::Path;

pub fn render_spectrogram(spectrogram: &ndarray::Array2<f32>, output_path: &Path, width: u32, height: u32) -> Result<()> {
    let renderer = Renderer::new(width, height);
    renderer.render_spectrogram(spectrogram, output_path, 44100)
}

pub fn render_mel_spectrogram(mel_spec: &ndarray::Array2<f32>, output_path: &Path, width: u32, height: u32) -> Result<()> {
    // Mel spectrogram is similar but with different frequency scaling
    render_spectrogram(mel_spec, output_path, width, height)
}

#[derive(Debug, Clone, Copy)]
pub enum Colormap {
    Viridis,
    Plasma,
    Inferno,
    Grayscale,
}

impl Colormap {
    pub fn apply(&self, value: f32) -> (u8, u8, u8) {
        let v = value.clamp(0.0, 1.0);

        match self {
            Colormap::Viridis => self.viridis(v),
            Colormap::Plasma => self.plasma(v),
            Colormap::Inferno => self.inferno(v),
            Colormap::Grayscale => {
                let g = (v * 255.0) as u8;
                (g, g, g)
            }
        }
    }

    fn viridis(&self, v: f32) -> (u8, u8, u8) {
        let r = if v < 0.5 {
            (68.0 + v * 2.0 * 79.0) as u8
        } else {
            (147.0 + (v - 0.5) * 2.0 * 106.0) as u8
        };

        let g = if v < 0.5 {
            (1.0 + v * 2.0 * 85.0) as u8
        } else {
            (86.0 + (v - 0.5) * 2.0 * 134.0) as u8
        };

        let b = if v < 0.25 {
            (84.0 + v * 4.0 * 30.0) as u8
        } else if v < 0.5 {
            (114.0 - (v - 0.25) * 4.0 * 30.0) as u8
        } else if v < 0.75 {
            (84.0 - (v - 0.5) * 4.0 * 41.0) as u8
        } else {
            (43.0 - (v - 0.75) * 4.0 * 10.0) as u8
        };

        (r, g, b)
    }

    fn plasma(&self, v: f32) -> (u8, u8, u8) {
        let r = (240.0 * v.powf(0.7)) as u8;
        let g = (20.0 + 100.0 * v) as u8;
        let b = (140.0 * (1.0 - v).powf(1.5)) as u8;
        (r, g, b)
    }

    fn inferno(&self, v: f32) -> (u8, u8, u8) {
        let r = ((v * v) * 255.0) as u8;
        let g = (v * 150.0) as u8;
        let b = ((1.0 - v.powf(2.0)) * 200.0) as u8;
        (r, g, b)
    }
}