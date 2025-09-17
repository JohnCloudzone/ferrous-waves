use crate::utils::error::Result;
use crate::visualization::renderer::Renderer;
use std::path::Path;

pub fn render_waveform(samples: &[f32], output_path: &Path, width: u32, height: u32) -> Result<()> {
    let renderer = Renderer::new(width, height);
    renderer.render_waveform(samples, output_path)
}

pub fn render_waveform_with_envelope(
    samples: &[f32],
    output_path: &Path,
    width: u32,
    height: u32,
) -> Result<()> {
    use plotters::prelude::*;

    // Calculate envelope with a window size
    let window_size = (samples.len() / width as usize).max(1);
    let mut peak_envelope_upper = Vec::new();
    let mut peak_envelope_lower = Vec::new();
    let mut rms_envelope = Vec::new();

    // Process samples in windows
    for chunk in samples.chunks(window_size) {
        let mut max_val = 0.0f32;
        let mut min_val = 0.0f32;
        let mut rms_sum = 0.0f32;

        for &sample in chunk {
            max_val = max_val.max(sample);
            min_val = min_val.min(sample);
            rms_sum += sample * sample;
        }

        peak_envelope_upper.push(max_val);
        peak_envelope_lower.push(min_val);

        let rms = (rms_sum / chunk.len() as f32).sqrt();
        rms_envelope.push(rms);
    }

    // Create the plot
    let root = BitMapBackend::new(output_path, (width, height)).into_drawing_area();

    root.fill(&WHITE).map_err(|e| {
        crate::utils::error::FerrousError::Visualization(format!(
            "Failed to fill background: {}",
            e
        ))
    })?;

    let max_val = samples.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
    let y_range = if max_val > 0.0 { max_val * 1.1 } else { 1.0 };

    let mut chart = ChartBuilder::on(&root)
        .margin(10)
        .build_cartesian_2d(0f32..peak_envelope_upper.len() as f32, -y_range..y_range)
        .map_err(|e| {
            crate::utils::error::FerrousError::Visualization(format!(
                "Failed to build chart: {}",
                e
            ))
        })?;

    // Draw peak envelope as filled area
    let upper_points: Vec<(f32, f32)> = peak_envelope_upper
        .iter()
        .enumerate()
        .map(|(i, &v)| (i as f32, v))
        .collect();

    let lower_points: Vec<(f32, f32)> = peak_envelope_lower
        .iter()
        .enumerate()
        .map(|(i, &v)| (i as f32, v))
        .collect();

    // Draw filled area between envelopes using polygons
    let mut polygon_points = upper_points.clone();
    polygon_points.extend(lower_points.iter().rev());
    polygon_points.push(upper_points[0]); // Close the polygon

    chart
        .draw_series(std::iter::once(Polygon::new(
            polygon_points,
            BLUE.mix(0.2).filled(),
        )))
        .map_err(|e| {
            crate::utils::error::FerrousError::Visualization(format!(
                "Failed to draw envelope fill: {}",
                e
            ))
        })?;

    // Draw RMS envelope
    let rms_points_upper: Vec<(f32, f32)> = rms_envelope
        .iter()
        .enumerate()
        .map(|(i, &v)| (i as f32, v))
        .collect();

    let rms_points_lower: Vec<(f32, f32)> = rms_envelope
        .iter()
        .enumerate()
        .map(|(i, &v)| (i as f32, -v))
        .collect();

    chart
        .draw_series(LineSeries::new(rms_points_upper, &RED))
        .map_err(|e| {
            crate::utils::error::FerrousError::Visualization(format!(
                "Failed to draw RMS upper: {}",
                e
            ))
        })?;

    chart
        .draw_series(LineSeries::new(rms_points_lower, &RED))
        .map_err(|e| {
            crate::utils::error::FerrousError::Visualization(format!(
                "Failed to draw RMS lower: {}",
                e
            ))
        })?;

    // Draw peak envelope outlines
    chart
        .draw_series(LineSeries::new(
            upper_points,
            ShapeStyle::from(&BLUE).stroke_width(2),
        ))
        .map_err(|e| {
            crate::utils::error::FerrousError::Visualization(format!(
                "Failed to draw peak upper: {}",
                e
            ))
        })?;

    chart
        .draw_series(LineSeries::new(
            lower_points,
            ShapeStyle::from(&BLUE).stroke_width(2),
        ))
        .map_err(|e| {
            crate::utils::error::FerrousError::Visualization(format!(
                "Failed to draw peak lower: {}",
                e
            ))
        })?;

    // Draw zero line
    chart
        .draw_series(LineSeries::new(
            vec![(0.0, 0.0), (peak_envelope_upper.len() as f32, 0.0)],
            ShapeStyle::from(&BLACK).stroke_width(1),
        ))
        .map_err(|e| {
            crate::utils::error::FerrousError::Visualization(format!(
                "Failed to draw zero line: {}",
                e
            ))
        })?;

    root.present().map_err(|e| {
        crate::utils::error::FerrousError::Visualization(format!("Failed to present: {}", e))
    })?;

    Ok(())
}
