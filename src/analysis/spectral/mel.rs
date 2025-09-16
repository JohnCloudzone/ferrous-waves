use ndarray::Array2;

pub struct MelFilterBank {
    _num_filters: usize,
    _sample_rate: u32,
    _fft_size: usize,
    filter_bank: Array2<f32>,
}

impl MelFilterBank {
    pub fn new(num_filters: usize, sample_rate: u32, fft_size: usize) -> Self {
        let filter_bank = Self::create_filter_bank(num_filters, sample_rate, fft_size);

        Self {
            _num_filters: num_filters,
            _sample_rate: sample_rate,
            _fft_size: fft_size,
            filter_bank,
        }
    }

    fn hz_to_mel(hz: f32) -> f32 {
        2595.0 * (1.0 + hz / 700.0).log10()
    }

    fn mel_to_hz(mel: f32) -> f32 {
        700.0 * (10_f32.powf(mel / 2595.0) - 1.0)
    }

    fn create_filter_bank(num_filters: usize, sample_rate: u32, fft_size: usize) -> Array2<f32> {
        let num_bins = fft_size / 2 + 1;
        let nyquist = sample_rate as f32 / 2.0;

        // Create mel scale points
        let mel_min = Self::hz_to_mel(0.0);
        let mel_max = Self::hz_to_mel(nyquist);
        let mel_points: Vec<f32> = (0..num_filters + 2)
            .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (num_filters + 1) as f32)
            .collect();

        // Convert back to Hz
        let hz_points: Vec<f32> = mel_points.iter().map(|&mel| Self::mel_to_hz(mel)).collect();

        // Convert to FFT bin indices
        let bin_points: Vec<usize> = hz_points
            .iter()
            .map(|&hz| ((fft_size as f32 * hz) / sample_rate as f32).round() as usize)
            .collect();

        // Create triangular filters
        let mut filter_bank = Array2::zeros((num_filters, num_bins));

        for i in 0..num_filters {
            let start = bin_points[i];
            let center = bin_points[i + 1];
            let end = bin_points[i + 2];

            // Rising edge
            for j in start..center {
                filter_bank[[i, j]] = (j - start) as f32 / (center - start) as f32;
            }

            // Falling edge
            for j in center..end.min(num_bins) {
                filter_bank[[i, j]] = 1.0 - (j - center) as f32 / (end - center) as f32;
            }
        }

        filter_bank
    }

    pub fn apply(&self, spectrogram: &Array2<f32>) -> Array2<f32> {
        self.filter_bank.dot(spectrogram)
    }
}