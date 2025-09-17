use plotters::prelude::*;
use plotters::series::LineSeries;
use image::{ImageBuffer, Rgb};
use std::path::Path;
use crate::utils::error::{Result, FerrousError};
use base64::Engine;

pub struct Renderer {
    width: u32,
    height: u32,
}

impl Renderer {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
        }
    }

    pub fn with_dimensions(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
        }
    }

    pub fn render_waveform(&self, samples: &[f32], output_path: &Path) -> Result<()> {
        let root = BitMapBackend::new(output_path, (self.width, self.height))
            .into_drawing_area();

        root.fill(&WHITE)
            .map_err(|e| FerrousError::Visualization(format!("Failed to fill background: {}", e)))?;

        let max_val = samples.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
        let y_range = if max_val > 0.0 { max_val * 1.1 } else { 1.0 };

        let mut chart = ChartBuilder::on(&root)
            .margin(10)
            .build_cartesian_2d(
                0f32..samples.len() as f32,
                -y_range..y_range,
            )
            .map_err(|e| FerrousError::Visualization(format!("Failed to build chart: {}", e)))?;

        // Downsample if needed for performance
        let step = (samples.len() / (self.width as usize * 2)).max(1);
        let points: Vec<(f32, f32)> = samples
            .iter()
            .step_by(step)
            .enumerate()
            .map(|(i, &s)| (i as f32 * step as f32, s))
            .collect();

        chart.draw_series(LineSeries::new(points, &BLUE))
            .map_err(|e| FerrousError::Visualization(format!("Failed to draw waveform: {}", e)))?;

        // Draw zero line
        chart.draw_series(LineSeries::new(
            vec![(0.0, 0.0), (samples.len() as f32, 0.0)],
            ShapeStyle::from(&BLACK).stroke_width(1),
        ))
        .map_err(|e| FerrousError::Visualization(format!("Failed to draw zero line: {}", e)))?;

        root.present()
            .map_err(|e| FerrousError::Visualization(format!("Failed to present: {}", e)))?;

        Ok(())
    }

    pub fn render_spectrogram(
        &self,
        spectrogram: &ndarray::Array2<f32>,
        output_path: &Path,
        _sample_rate: u32,
    ) -> Result<()> {
        let (num_bins, num_frames) = spectrogram.dim();

        // Create image buffer
        let mut img = ImageBuffer::<Rgb<u8>, Vec<u8>>::new(self.width, self.height);

        // Convert to dB scale for better visualization
        let db_spectrogram: ndarray::Array2<f32> = spectrogram.mapv(|x| 20.0 * (x + 1e-10).log10());

        // Find min and max for normalization
        let min_db = db_spectrogram.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_db = db_spectrogram.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let db_range = max_db - min_db;

        // Calculate scaling factors
        let x_scale = self.width as f32 / num_frames as f32;
        let y_scale = self.height as f32 / num_bins as f32;

        // Render spectrogram with proper scaling
        for y in 0..self.height {
            for x in 0..self.width {
                // Map pixel coordinates to spectrogram indices
                let frame_idx = ((x as f32 / x_scale) as usize).min(num_frames - 1);
                let bin_idx = ((y as f32 / y_scale) as usize).min(num_bins - 1);

                // Flip vertically so low frequencies are at bottom
                let flipped_bin_idx = num_bins - 1 - bin_idx;

                let value = db_spectrogram[[flipped_bin_idx, frame_idx]];
                let normalized = ((value - min_db) / db_range).clamp(0.0, 1.0);

                // Apply viridis colormap
                let (r, g, b) = self.viridis_colormap(normalized);
                img.put_pixel(x, y, Rgb([r, g, b]));
            }
        }

        // Save the image
        img.save(output_path)
            .map_err(|e| FerrousError::Visualization(format!("Failed to save spectrogram: {}", e)))?;

        Ok(())
    }

    pub fn render_power_curve(&self, power: &[f32], output_path: &Path) -> Result<()> {
        let root = BitMapBackend::new(output_path, (self.width, self.height))
            .into_drawing_area();

        root.fill(&WHITE)
            .map_err(|e| FerrousError::Visualization(format!("Failed to fill background: {}", e)))?;

        let max_power = power.iter().fold(0.0f32, |a, &b| a.max(b));
        let min_power = power.iter().fold(f32::INFINITY, |a, &b| a.min(b));

        let mut chart = ChartBuilder::on(&root)
            .margin(10)
            .build_cartesian_2d(
                0f32..power.len() as f32,
                min_power..max_power * 1.1,
            )
            .map_err(|e| FerrousError::Visualization(format!("Failed to build chart: {}", e)))?;

        let points: Vec<(f32, f32)> = power
            .iter()
            .enumerate()
            .map(|(i, &p)| (i as f32, p))
            .collect();

        chart.draw_series(LineSeries::new(points, &RED))
            .map_err(|e| FerrousError::Visualization(format!("Failed to draw series: {}", e)))?;

        root.present()
            .map_err(|e| FerrousError::Visualization(format!("Failed to present: {}", e)))?;

        Ok(())
    }

    pub fn render_to_base64(&self, data: &RenderData) -> Result<String> {
        let temp_path = std::env::temp_dir().join(format!("ferrous_waves_{}.png", uuid::Uuid::new_v4()));

        match data {
            RenderData::Waveform(samples) => {
                self.render_waveform(samples, &temp_path)?;
            }
            RenderData::Spectrogram(spec) => {
                self.render_spectrogram(spec, &temp_path, 44100)?;
            }
            RenderData::PowerCurve(power) => {
                self.render_power_curve(power, &temp_path)?;
            }
        }

        let img_data = std::fs::read(&temp_path)?;
        std::fs::remove_file(&temp_path).ok();
        Ok(base64::engine::general_purpose::STANDARD.encode(img_data))
    }

    pub fn render_to_file<P: AsRef<Path>>(&self, data: &RenderData, output_path: P) -> Result<()> {
        match data {
            RenderData::Waveform(samples) => {
                self.render_waveform(samples, output_path.as_ref())
            }
            RenderData::Spectrogram(spec) => {
                self.render_spectrogram(spec, output_path.as_ref(), 44100)
            }
            RenderData::PowerCurve(power) => {
                self.render_power_curve(power, output_path.as_ref())
            }
        }
    }

    // Viridis colormap implementation
    fn viridis_colormap(&self, value: f32) -> (u8, u8, u8) {
        let v = value.clamp(0.0, 1.0);

        // Simplified viridis colormap
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
}

pub enum RenderData<'a> {
    Waveform(&'a [f32]),
    Spectrogram(&'a ndarray::Array2<f32>),
    PowerCurve(&'a [f32]),
}

impl Default for Renderer {
    fn default() -> Self {
        Self::new(1920, 1080)
    }
}