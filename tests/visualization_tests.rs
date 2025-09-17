#[cfg(test)]
mod visualization_tests {
    use ferrous_waves::visualization::{RenderData, Renderer};
    use ndarray::Array2;
    use tempfile::tempdir;

    #[test]
    fn test_renderer_creation() {
        Renderer::new(1920, 1080);
        Renderer::with_dimensions(800, 600);
        Renderer::default();
    }

    #[test]
    fn test_waveform_rendering() {
        let samples = vec![0.0, 0.5, 1.0, 0.5, 0.0, -0.5, -1.0, -0.5, 0.0];
        let dir = tempdir().unwrap();
        let output_path = dir.path().join("waveform.png");

        let renderer = Renderer::new(800, 600);
        let result = renderer.render_to_file(&RenderData::Waveform(&samples), &output_path);
        assert!(result.is_ok());
        assert!(output_path.exists());
    }

    #[test]
    fn test_waveform_empty_samples() {
        let samples: Vec<f32> = vec![];
        let dir = tempdir().unwrap();
        let output_path = dir.path().join("empty_waveform.png");

        let renderer = Renderer::new(800, 600);
        let result = renderer.render_to_file(&RenderData::Waveform(&samples), &output_path);
        assert!(result.is_ok());
    }

    #[test]
    fn test_spectrogram_rendering() {
        let mut spectrogram = Array2::<f32>::zeros((512, 100));
        for i in 0..100 {
            for j in 0..512 {
                spectrogram[[j, i]] = (i as f32 * j as f32).sin().abs();
            }
        }

        let dir = tempdir().unwrap();
        let output_path = dir.path().join("spectrogram.png");

        let renderer = Renderer::new(800, 600);
        let result = renderer.render_to_file(&RenderData::Spectrogram(&spectrogram), &output_path);
        assert!(result.is_ok());
        assert!(output_path.exists());
    }

    #[test]
    fn test_power_curve_rendering() {
        let power = vec![0.1, 0.3, 0.5, 0.7, 0.9, 0.8, 0.6, 0.4, 0.2, 0.1];

        let dir = tempdir().unwrap();
        let output_path = dir.path().join("power.png");

        let renderer = Renderer::new(800, 600);
        let result = renderer.render_to_file(&RenderData::PowerCurve(&power), &output_path);
        assert!(result.is_ok());
        assert!(output_path.exists());
    }

    #[test]
    fn test_render_to_base64_waveform() {
        let renderer = Renderer::new(400, 300);
        let samples = vec![0.0, 0.25, 0.5, 0.25, 0.0, -0.25, -0.5, -0.25, 0.0];

        let result = renderer.render_to_base64(&RenderData::Waveform(&samples));
        assert!(result.is_ok());

        let base64_str = result.unwrap();
        assert!(!base64_str.is_empty());
        // Base64 should be valid
        assert!(base64_str
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '+' || c == '/' || c == '='));
    }

    #[test]
    fn test_render_to_base64_spectrogram() {
        let renderer = Renderer::new(400, 300);
        let spec = Array2::<f32>::ones((128, 32));

        let result = renderer.render_to_base64(&RenderData::Spectrogram(&spec));
        assert!(result.is_ok());

        let base64_str = result.unwrap();
        assert!(!base64_str.is_empty());
    }

    #[test]
    fn test_render_to_base64_power_curve() {
        let renderer = Renderer::new(400, 300);
        let power = vec![1.0, 2.0, 3.0, 2.0, 1.0];

        let result = renderer.render_to_base64(&RenderData::PowerCurve(&power));
        assert!(result.is_ok());

        let base64_str = result.unwrap();
        assert!(!base64_str.is_empty());
    }

    #[test]
    fn test_spectrogram_db_conversion() {
        let mut spec = Array2::<f32>::zeros((10, 10));
        spec[[5, 5]] = 1.0;
        spec[[3, 3]] = 0.1;
        spec[[7, 7]] = 0.01;

        let dir = tempdir().unwrap();
        let output_path = dir.path().join("db_spectrogram.png");

        let renderer = Renderer::new(100, 100);
        let result = renderer.render_to_file(&RenderData::Spectrogram(&spec), &output_path);
        assert!(result.is_ok());

        // File should exist and be non-empty
        assert!(output_path.exists());
        let metadata = std::fs::metadata(&output_path).unwrap();
        assert!(metadata.len() > 0);
    }

    #[test]
    fn test_waveform_with_extreme_values() {
        let samples = vec![f32::MAX, f32::MIN, 0.0, 1e-10, -1e-10];
        let dir = tempdir().unwrap();
        let output_path = dir.path().join("extreme_waveform.png");

        let renderer = Renderer::new(400, 300);
        let result = renderer.render_to_file(&RenderData::Waveform(&samples), &output_path);
        assert!(result.is_ok());
        assert!(output_path.exists());
    }

    #[test]
    fn test_power_curve_empty() {
        let renderer = Renderer::new(400, 300);
        let power: Vec<f32> = vec![];

        let dir = tempdir().unwrap();
        let output_path = dir.path().join("empty_power.png");

        let result = renderer.render_to_file(&RenderData::PowerCurve(&power), &output_path);
        assert!(result.is_ok());
    }

    #[test]
    fn test_spectrogram_single_frame() {
        let spec = Array2::<f32>::ones((512, 1));

        let dir = tempdir().unwrap();
        let output_path = dir.path().join("single_frame_spec.png");

        let renderer = Renderer::new(400, 300);
        let result = renderer.render_to_file(&RenderData::Spectrogram(&spec), &output_path);
        assert!(result.is_ok());
        assert!(output_path.exists());
    }

    #[test]
    fn test_colormap_viridis() {
        use ferrous_waves::visualization::spectrogram::Colormap;

        let colormap = Colormap::Viridis;

        let (r, g, b) = colormap.apply(0.0);
        assert!(r < 100);
        assert!(g < 50);
        assert!(b > 50);

        let (r, g, b) = colormap.apply(1.0);
        assert!(r > 200);
        assert!(g > 200);
        assert!(b < 100);
    }

    #[test]
    fn test_colormap_plasma() {
        use ferrous_waves::visualization::spectrogram::Colormap;

        let colormap = Colormap::Plasma;

        let (r, g, b) = colormap.apply(0.0);
        assert_eq!(r, 0);
        assert!(g < 50);
        assert!(b > 100);

        let (r, g, b) = colormap.apply(1.0);
        assert!(r > 200);
        assert!(g > 100);
        assert!(b < 50);
    }

    #[test]
    fn test_colormap_inferno() {
        use ferrous_waves::visualization::spectrogram::Colormap;

        let colormap = Colormap::Inferno;

        let (r, g, b) = colormap.apply(0.0);
        assert_eq!(r, 0);
        assert_eq!(g, 0);
        assert!(b > 150);

        let (r, g, b) = colormap.apply(1.0);
        assert_eq!(r, 255);
        assert!(g > 100);
        assert_eq!(b, 0);
    }

    #[test]
    fn test_colormap_grayscale() {
        use ferrous_waves::visualization::spectrogram::Colormap;

        let colormap = Colormap::Grayscale;

        let (r, g, b) = colormap.apply(0.5);
        assert_eq!(r, g);
        assert_eq!(g, b);
        assert_eq!(r, 127);
    }

    #[test]
    fn test_colormap_clamping() {
        use ferrous_waves::visualization::spectrogram::Colormap;

        let colormap = Colormap::Viridis;

        // Test values outside [0, 1] range
        let (r1, g1, b1) = colormap.apply(-0.5);
        let (r2, g2, b2) = colormap.apply(0.0);
        assert_eq!((r1, g1, b1), (r2, g2, b2));

        let (r1, g1, b1) = colormap.apply(1.5);
        let (r2, g2, b2) = colormap.apply(1.0);
        assert_eq!((r1, g1, b1), (r2, g2, b2));
    }
}
