use crate::visualization::renderer::Renderer;
use crate::utils::error::Result;
use std::path::Path;

pub fn render_waveform(samples: &[f32], output_path: &Path, width: u32, height: u32) -> Result<()> {
    let renderer = Renderer::new(width, height);
    renderer.render_waveform(samples, output_path)
}

pub fn render_waveform_with_envelope(samples: &[f32], output_path: &Path, width: u32, height: u32) -> Result<()> {
    // For now, delegate to regular waveform rendering
    // Envelope rendering can be added as enhancement
    render_waveform(samples, output_path, width, height)
}