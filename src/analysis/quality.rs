use crate::utils::error::Result;

/// Audio quality assessment results
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct QualityAssessment {
    /// Overall quality score (0.0 to 1.0)
    pub overall_score: f32,

    /// Individual quality metrics
    pub metrics: QualityMetrics,

    /// Detected issues that may affect analysis
    pub issues: Vec<QualityIssue>,

    /// Recommendations for improving quality
    pub recommendations: Vec<String>,

    /// Confidence in the assessment
    pub confidence: f32,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct QualityMetrics {
    /// Signal-to-noise ratio in dB
    pub snr_db: f32,

    /// Total Harmonic Distortion percentage
    pub thd_percent: f32,

    /// DC offset presence
    pub dc_offset: f32,

    /// Clipping detection (0.0 to 1.0, percentage of clipped samples)
    pub clipping_ratio: f32,

    /// Noise floor level in dB
    pub noise_floor_db: f32,

    /// Dynamic range in dB
    pub dynamic_range_db: f32,

    /// Frequency response flatness
    pub frequency_response_score: f32,

    /// Phase coherence
    pub phase_coherence: f32,

    /// Aliasing artifacts detection
    pub aliasing_score: f32,

    /// Dropout detection (brief silence/glitches)
    pub dropout_count: usize,

    /// Sample rate quality indicator
    pub sample_rate_quality: SampleRateQuality,

    /// Bit depth quality
    pub bit_depth_quality: BitDepthQuality,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum SampleRateQuality {
    Low,       // < 22.05 kHz
    Standard,  // 44.1 kHz
    High,      // 48 kHz
    UltraHigh, // >= 96 kHz
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum BitDepthQuality {
    Low,       // 8-bit
    Standard,  // 16-bit
    High,      // 24-bit
    UltraHigh, // 32-bit float
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct QualityIssue {
    /// Issue type
    pub issue_type: IssueType,

    /// Severity level
    pub severity: IssueSeverity,

    /// Description of the issue
    pub description: String,

    /// Time ranges where issue occurs (if applicable)
    pub time_ranges: Vec<(f32, f32)>,

    /// Impact on analysis reliability
    pub impact: String,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum IssueType {
    Clipping,
    HighNoiseFloor,
    DCOffset,
    Distortion,
    Aliasing,
    Dropouts,
    LowDynamicRange,
    PhaseIssues,
    FrequencyImbalance,
    LowSampleRate,
    QuantizationNoise,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum IssueSeverity {
    Critical, // Severely affects analysis
    High,     // Significant impact
    Medium,   // Moderate impact
    Low,      // Minor impact
}

/// Quality analyzer for detecting audio issues
pub struct QualityAnalyzer {
    sample_rate: f32,
    /// Threshold for clipping detection (default: 0.99)
    clipping_threshold: f32,
    /// Minimum SNR for good quality (default: 40 dB)
    min_snr_db: f32,
    /// Maximum acceptable THD (default: 1%)
    max_thd_percent: f32,
}

impl QualityAnalyzer {
    pub fn new(sample_rate: f32) -> Self {
        Self {
            sample_rate,
            clipping_threshold: 0.99,
            min_snr_db: 40.0,
            max_thd_percent: 1.0,
        }
    }

    pub fn analyze(&self, samples: &[f32]) -> Result<QualityAssessment> {
        let metrics = self.calculate_metrics(samples)?;
        let issues = self.detect_issues(&metrics, samples);
        let recommendations = self.generate_recommendations(&metrics, &issues);
        let overall_score = self.calculate_overall_score(&metrics, &issues);
        let confidence = self.calculate_confidence(&metrics, samples.len());

        Ok(QualityAssessment {
            overall_score,
            metrics,
            issues,
            recommendations,
            confidence,
        })
    }

    fn calculate_metrics(&self, samples: &[f32]) -> Result<QualityMetrics> {
        // Calculate SNR
        let snr_db = self.calculate_snr(samples);

        // Calculate THD
        let thd_percent = self.calculate_thd(samples);

        // Detect DC offset
        let dc_offset = self.detect_dc_offset(samples);

        // Detect clipping
        let clipping_ratio = self.detect_clipping(samples);

        // Calculate noise floor
        let noise_floor_db = self.calculate_noise_floor(samples);

        // Calculate dynamic range
        let dynamic_range_db = self.calculate_dynamic_range(samples);

        // Analyze frequency response
        let frequency_response_score = self.analyze_frequency_response(samples);

        // Check phase coherence
        let phase_coherence = self.check_phase_coherence(samples);

        // Detect aliasing
        let aliasing_score = self.detect_aliasing(samples);

        // Detect dropouts
        let dropout_count = self.detect_dropouts(samples);

        // Assess sample rate quality
        let sample_rate_quality = self.assess_sample_rate_quality();

        // Assess bit depth quality (estimated from quantization)
        let bit_depth_quality = self.assess_bit_depth_quality(samples);

        Ok(QualityMetrics {
            snr_db,
            thd_percent,
            dc_offset,
            clipping_ratio,
            noise_floor_db,
            dynamic_range_db,
            frequency_response_score,
            phase_coherence,
            aliasing_score,
            dropout_count,
            sample_rate_quality,
            bit_depth_quality,
        })
    }

    fn calculate_snr(&self, samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }

        // Calculate signal RMS
        let signal_rms = (samples.iter().map(|x| x * x).sum::<f32>() / samples.len() as f32).sqrt();

        if signal_rms < 0.001 {
            return 0.0; // Silent signal
        }

        // Estimate noise floor using spectral analysis
        use rustfft::{num_complex::Complex, FftPlanner};

        let fft_size = 2048.min(samples.len());
        if fft_size < 128 {
            return 40.0; // Default for very short samples
        }

        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(fft_size);

        let mut buffer: Vec<Complex<f32>> = samples
            .iter()
            .take(fft_size)
            .map(|&x| Complex::new(x, 0.0))
            .collect();

        fft.process(&mut buffer);

        // Find noise floor in frequency domain
        let magnitudes: Vec<f32> = buffer[1..fft_size / 2].iter().map(|c| c.norm()).collect();

        if magnitudes.is_empty() {
            return 40.0;
        }

        // Sort to find noise floor (lower percentile of spectrum)
        let mut sorted_mags = magnitudes.clone();
        sorted_mags.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Noise floor is median of lower 30% of spectrum
        let noise_idx = sorted_mags.len() / 3;
        let noise_floor = if noise_idx > 0 {
            sorted_mags[..noise_idx].iter().sum::<f32>() / noise_idx as f32
        } else {
            sorted_mags[0]
        };

        // Peak magnitude (signal)
        let peak_magnitude = sorted_mags[sorted_mags.len() - 1];

        // Calculate SNR
        if noise_floor > 0.0 && peak_magnitude > noise_floor {
            20.0 * (peak_magnitude / noise_floor).log10()
        } else {
            60.0 // High SNR for clean signals
        }
    }

    fn calculate_thd(&self, samples: &[f32]) -> f32 {
        // Simplified THD calculation using FFT
        use rustfft::{num_complex::Complex, FftPlanner};

        let fft_size = 4096.min(samples.len()); // Larger FFT for better frequency resolution
        if fft_size < 256 {
            return 0.0;
        }

        // Apply Hann window to reduce spectral leakage
        let mut windowed_samples: Vec<f32> = Vec::with_capacity(fft_size);
        for (i, &sample) in samples.iter().enumerate().take(fft_size) {
            let window =
                0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (fft_size - 1) as f32).cos());
            windowed_samples.push(sample * window);
        }

        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(fft_size);

        let mut buffer: Vec<Complex<f32>> = windowed_samples
            .iter()
            .map(|&x| Complex::new(x, 0.0))
            .collect();

        fft.process(&mut buffer);

        // Find fundamental frequency (highest peak)
        let mut max_magnitude = 0.0;
        let mut fundamental_bin = 0;

        for (i, complex) in buffer.iter().enumerate().take(fft_size / 2).skip(1) {
            let magnitude = complex.norm();
            if magnitude > max_magnitude {
                max_magnitude = magnitude;
                fundamental_bin = i;
            }
        }

        if fundamental_bin == 0 || max_magnitude == 0.0 {
            return 0.0;
        }

        // Calculate noise floor away from fundamental and harmonics
        let mut noise_samples = Vec::new();
        for (i, complex) in buffer.iter().enumerate().take(fft_size / 2).skip(1) {
            // Skip bins near fundamental and its harmonics
            let mut is_near_harmonic = false;
            for harmonic in 1..=5 {
                let harmonic_bin = fundamental_bin * harmonic;
                if (i as i32 - harmonic_bin as i32).abs() <= 2 {
                    is_near_harmonic = true;
                    break;
                }
            }
            if !is_near_harmonic {
                noise_samples.push(complex.norm());
            }
        }

        let noise_floor = if !noise_samples.is_empty() {
            noise_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
            noise_samples[noise_samples.len() / 2] // Median as noise floor
        } else {
            0.0
        };

        // Sum harmonics (2nd, 3rd, 4th, 5th) above noise floor
        let mut harmonic_power = 0.0;
        for harmonic in 2..=5 {
            let center_bin = fundamental_bin * harmonic;
            if center_bin < fft_size / 2 {
                let mag = buffer[center_bin].norm();
                // Only count if significantly above noise floor (3x noise floor)
                if mag > noise_floor * 3.0 && mag > max_magnitude * 0.001 {
                    harmonic_power += (mag - noise_floor).max(0.0).powi(2);
                }
            }
        }

        let fundamental_power = (max_magnitude - noise_floor).max(0.0).powi(2);

        // THD percentage
        if fundamental_power > 0.0 && harmonic_power > 0.0 {
            (harmonic_power / fundamental_power).sqrt() * 100.0
        } else {
            0.0 // No harmonics detected means no distortion
        }
    }

    fn detect_dc_offset(&self, samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }

        samples.iter().sum::<f32>() / samples.len() as f32
    }

    fn detect_clipping(&self, samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }

        let clipped_count = samples
            .iter()
            .filter(|&&x| x.abs() >= self.clipping_threshold)
            .count();

        clipped_count as f32 / samples.len() as f32
    }

    fn calculate_noise_floor(&self, samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return -60.0;
        }

        // Find quiet sections
        let window_size = (self.sample_rate * 0.1) as usize; // 100ms windows
        let mut min_rms = f32::MAX;

        for chunk in samples.chunks(window_size) {
            let rms = (chunk.iter().map(|x| x * x).sum::<f32>() / chunk.len() as f32).sqrt();
            if rms > 0.0 && rms < min_rms {
                min_rms = rms;
            }
        }

        if min_rms < f32::MAX && min_rms > 0.0 {
            20.0 * min_rms.log10()
        } else {
            -60.0
        }
    }

    fn calculate_dynamic_range(&self, samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }

        // Calculate RMS over windows
        let window_size = (self.sample_rate * 0.4) as usize; // 400ms windows
        let mut rms_values = Vec::new();

        for chunk in samples.chunks(window_size) {
            let rms = (chunk.iter().map(|x| x * x).sum::<f32>() / chunk.len() as f32).sqrt();
            if rms > 0.0 {
                rms_values.push(rms);
            }
        }

        if rms_values.len() < 2 {
            return 0.0;
        }

        rms_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Use 95th percentile vs 10th percentile
        let high_idx = (rms_values.len() as f32 * 0.95) as usize;
        let low_idx = (rms_values.len() as f32 * 0.10) as usize;

        let high_level = rms_values[high_idx.min(rms_values.len() - 1)];
        let low_level = rms_values[low_idx];

        if low_level > 0.0 {
            20.0 * (high_level / low_level).log10()
        } else {
            60.0
        }
    }

    fn analyze_frequency_response(&self, samples: &[f32]) -> f32 {
        // Simplified frequency response analysis
        // Returns score from 0-1, where 1 is perfectly flat

        use rustfft::{num_complex::Complex, FftPlanner};

        let fft_size = 4096.min(samples.len());
        if fft_size < 256 {
            return 0.5;
        }

        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(fft_size);

        let mut buffer: Vec<Complex<f32>> = samples
            .iter()
            .take(fft_size)
            .map(|&x| Complex::new(x, 0.0))
            .collect();

        fft.process(&mut buffer);

        // Analyze magnitude spectrum
        let magnitudes: Vec<f32> = buffer[1..fft_size / 2].iter().map(|c| c.norm()).collect();

        if magnitudes.is_empty() {
            return 0.5;
        }

        // Calculate variance
        let mean = magnitudes.iter().sum::<f32>() / magnitudes.len() as f32;
        let variance = magnitudes
            .iter()
            .map(|&x| (x - mean) * (x - mean))
            .sum::<f32>()
            / magnitudes.len() as f32;

        // Convert variance to score (lower variance = flatter response = higher score)
        let normalized_variance = variance / (mean * mean);
        (1.0 / (1.0 + normalized_variance * 10.0)).clamp(0.0, 1.0)
    }

    fn check_phase_coherence(&self, _samples: &[f32]) -> f32 {
        // Simplified phase coherence check
        // Would need stereo input for proper implementation
        // Return neutral score for mono
        0.75
    }

    fn detect_aliasing(&self, samples: &[f32]) -> f32 {
        // Detect high-frequency content near Nyquist that might indicate aliasing
        use rustfft::{num_complex::Complex, FftPlanner};

        let fft_size = 2048.min(samples.len());
        if fft_size < 256 {
            return 0.0;
        }

        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(fft_size);

        let mut buffer: Vec<Complex<f32>> = samples
            .iter()
            .take(fft_size)
            .map(|&x| Complex::new(x, 0.0))
            .collect();

        fft.process(&mut buffer);

        // Check energy near Nyquist frequency
        let nyquist_bin = fft_size / 2;
        let check_range = nyquist_bin / 10; // Check top 10% of spectrum

        let high_freq_energy: f32 = buffer[(nyquist_bin - check_range)..nyquist_bin]
            .iter()
            .map(|c| c.norm_sqr())
            .sum();

        let total_energy: f32 = buffer[1..nyquist_bin].iter().map(|c| c.norm_sqr()).sum();

        if total_energy > 0.0 {
            (high_freq_energy / total_energy).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }

    fn detect_dropouts(&self, samples: &[f32]) -> usize {
        let mut dropout_count = 0;
        let window_size = (self.sample_rate * 0.001) as usize; // 1ms windows
        let threshold = 0.0001; // Very low signal threshold

        for chunk in samples.chunks(window_size) {
            let max_val = chunk.iter().map(|x| x.abs()).fold(0.0, f32::max);
            if max_val < threshold {
                dropout_count += 1;
            }
        }

        dropout_count
    }

    fn assess_sample_rate_quality(&self) -> SampleRateQuality {
        if self.sample_rate >= 96000.0 {
            SampleRateQuality::UltraHigh
        } else if self.sample_rate >= 48000.0 {
            SampleRateQuality::High
        } else if self.sample_rate >= 44100.0 {
            SampleRateQuality::Standard
        } else {
            SampleRateQuality::Low
        }
    }

    fn assess_bit_depth_quality(&self, samples: &[f32]) -> BitDepthQuality {
        // Estimate bit depth from quantization levels
        let mut unique_values = std::collections::HashSet::new();

        // Sample a subset to avoid memory issues
        let sample_count = 10000.min(samples.len());
        for &sample in samples.iter().take(sample_count) {
            // Quantize to detect bit depth
            let quantized = (sample * 32768.0).round() / 32768.0;
            unique_values.insert(quantized.to_bits());
        }

        let unique_count = unique_values.len();

        if unique_count > 30000 {
            BitDepthQuality::UltraHigh // 32-bit float
        } else if unique_count > 10000 {
            BitDepthQuality::High // 24-bit
        } else if unique_count > 256 {
            BitDepthQuality::Standard // 16-bit
        } else {
            BitDepthQuality::Low // 8-bit
        }
    }

    fn detect_issues(&self, metrics: &QualityMetrics, samples: &[f32]) -> Vec<QualityIssue> {
        let mut issues = Vec::new();

        // Check for clipping
        if metrics.clipping_ratio > 0.001 {
            let severity = if metrics.clipping_ratio > 0.01 {
                IssueSeverity::Critical
            } else if metrics.clipping_ratio > 0.005 {
                IssueSeverity::High
            } else {
                IssueSeverity::Medium
            };

            issues.push(QualityIssue {
                issue_type: IssueType::Clipping,
                severity,
                description: format!(
                    "{:.2}% of samples are clipping",
                    metrics.clipping_ratio * 100.0
                ),
                time_ranges: self.find_clipping_regions(samples),
                impact: "Clipping causes distortion and affects all frequency-based analysis"
                    .to_string(),
            });
        }

        // Check SNR
        if metrics.snr_db < self.min_snr_db {
            let severity = if metrics.snr_db < 20.0 {
                IssueSeverity::High
            } else if metrics.snr_db < 30.0 {
                IssueSeverity::Medium
            } else {
                IssueSeverity::Low
            };

            issues.push(QualityIssue {
                issue_type: IssueType::HighNoiseFloor,
                severity,
                description: format!("Low SNR: {:.1} dB", metrics.snr_db),
                time_ranges: vec![],
                impact: "High noise floor reduces accuracy of pitch and onset detection"
                    .to_string(),
            });
        }

        // Check THD
        if metrics.thd_percent > self.max_thd_percent {
            let severity = if metrics.thd_percent > 5.0 {
                IssueSeverity::High
            } else if metrics.thd_percent > 2.0 {
                IssueSeverity::Medium
            } else {
                IssueSeverity::Low
            };

            issues.push(QualityIssue {
                issue_type: IssueType::Distortion,
                severity,
                description: format!("THD: {:.2}%", metrics.thd_percent),
                time_ranges: vec![],
                impact: "Harmonic distortion affects key detection and spectral analysis"
                    .to_string(),
            });
        }

        // Check DC offset
        if metrics.dc_offset.abs() > 0.01 {
            issues.push(QualityIssue {
                issue_type: IssueType::DCOffset,
                severity: IssueSeverity::Low,
                description: format!("DC offset: {:.3}", metrics.dc_offset),
                time_ranges: vec![],
                impact: "DC offset reduces headroom and may affect some algorithms".to_string(),
            });
        }

        // Check dropouts
        if metrics.dropout_count > 10 {
            let severity = if metrics.dropout_count > 100 {
                IssueSeverity::High
            } else if metrics.dropout_count > 50 {
                IssueSeverity::Medium
            } else {
                IssueSeverity::Low
            };

            issues.push(QualityIssue {
                issue_type: IssueType::Dropouts,
                severity,
                description: format!("{} dropouts detected", metrics.dropout_count),
                time_ranges: vec![],
                impact: "Dropouts cause discontinuities affecting temporal analysis".to_string(),
            });
        }

        // Check aliasing
        if metrics.aliasing_score > 0.1 {
            issues.push(QualityIssue {
                issue_type: IssueType::Aliasing,
                severity: IssueSeverity::Medium,
                description: format!("Possible aliasing: {:.1}%", metrics.aliasing_score * 100.0),
                time_ranges: vec![],
                impact: "Aliasing creates false frequencies affecting spectral analysis"
                    .to_string(),
            });
        }

        // Check sample rate
        if metrics.sample_rate_quality == SampleRateQuality::Low {
            issues.push(QualityIssue {
                issue_type: IssueType::LowSampleRate,
                severity: IssueSeverity::Medium,
                description: format!("Low sample rate: {:.1} kHz", self.sample_rate / 1000.0),
                time_ranges: vec![],
                impact:
                    "Low sample rate limits frequency range and may miss high-frequency content"
                        .to_string(),
            });
        }

        issues
    }

    fn find_clipping_regions(&self, samples: &[f32]) -> Vec<(f32, f32)> {
        let mut regions = Vec::new();
        let mut in_clip = false;
        let mut clip_start = 0.0;

        for (i, &sample) in samples.iter().enumerate() {
            let time = i as f32 / self.sample_rate;

            if sample.abs() >= self.clipping_threshold {
                if !in_clip {
                    in_clip = true;
                    clip_start = time;
                }
            } else if in_clip {
                in_clip = false;
                regions.push((clip_start, time));
            }
        }

        if in_clip {
            regions.push((clip_start, samples.len() as f32 / self.sample_rate));
        }

        // Limit to first 10 regions
        regions.truncate(10);
        regions
    }

    fn generate_recommendations(
        &self,
        metrics: &QualityMetrics,
        issues: &[QualityIssue],
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        for issue in issues {
            match issue.issue_type {
                IssueType::Clipping => {
                    recommendations.push("Reduce input gain to prevent clipping".to_string());
                }
                IssueType::HighNoiseFloor => {
                    recommendations.push(
                        "Apply noise reduction or use a higher quality recording".to_string(),
                    );
                }
                IssueType::DCOffset => {
                    recommendations
                        .push("Apply a high-pass filter to remove DC offset".to_string());
                }
                IssueType::Distortion => {
                    recommendations
                        .push("Check signal chain for sources of distortion".to_string());
                }
                IssueType::Dropouts => {
                    recommendations
                        .push("Check for buffer underruns or corrupted data".to_string());
                }
                IssueType::LowSampleRate => {
                    recommendations.push("Consider resampling to at least 44.1 kHz".to_string());
                }
                IssueType::Aliasing => {
                    recommendations
                        .push("Apply anti-aliasing filter before downsampling".to_string());
                }
                _ => {}
            }
        }

        // Add general recommendations based on metrics
        if metrics.dynamic_range_db < 20.0 {
            recommendations
                .push("Consider expanding dynamic range for better analysis".to_string());
        }

        if metrics.frequency_response_score < 0.5 {
            recommendations
                .push("Frequency response is uneven, consider EQ correction".to_string());
        }

        recommendations
    }

    fn calculate_overall_score(&self, metrics: &QualityMetrics, issues: &[QualityIssue]) -> f32 {
        let mut score = 1.0;

        // Deduct for issues based on severity
        for issue in issues {
            let deduction = match issue.severity {
                IssueSeverity::Critical => 0.3,
                IssueSeverity::High => 0.15,
                IssueSeverity::Medium => 0.08,
                IssueSeverity::Low => 0.03,
            };
            score -= deduction;
        }

        // Factor in SNR (more lenient)
        let snr_factor = if metrics.snr_db >= 40.0 {
            1.0
        } else if metrics.snr_db >= 20.0 {
            0.8 + (metrics.snr_db - 20.0) * 0.01
        } else {
            metrics.snr_db / 25.0
        };
        score *= snr_factor;

        // Factor in THD (more lenient for low distortion)
        let thd_factor = if metrics.thd_percent <= 0.5 {
            1.0
        } else if metrics.thd_percent <= 3.0 {
            1.0 - (metrics.thd_percent - 0.5) * 0.1
        } else {
            0.75 / (1.0 + (metrics.thd_percent - 3.0) / 10.0)
        };
        score *= thd_factor;

        // Factor in frequency response (weighted less)
        score *= 0.8 + metrics.frequency_response_score * 0.2;

        // Factor in sample rate quality
        let sr_factor = match metrics.sample_rate_quality {
            SampleRateQuality::UltraHigh => 1.0,
            SampleRateQuality::High => 0.98,
            SampleRateQuality::Standard => 0.95,
            SampleRateQuality::Low => 0.8,
        };
        score *= sr_factor;

        // Factor in clipping (critical issue)
        if metrics.clipping_ratio > 0.001 {
            score *= (1.0 - metrics.clipping_ratio * 10.0).max(0.3);
        }

        score.clamp(0.0, 1.0)
    }

    fn calculate_confidence(&self, metrics: &QualityMetrics, sample_count: usize) -> f32 {
        let mut confidence = 1.0;

        // Lower confidence for very short files
        if sample_count < self.sample_rate as usize {
            confidence *= sample_count as f32 / self.sample_rate;
        }

        // Lower confidence if metrics are at extremes
        if metrics.snr_db < 10.0 || metrics.snr_db > 100.0 {
            confidence *= 0.8;
        }

        confidence.clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quality_assessment_clean_signal() {
        // Generate clean sine wave
        let sample_rate = 44100.0;
        let frequency = 440.0;
        let duration = 1.0;
        let samples: Vec<f32> = (0..(sample_rate * duration) as usize)
            .map(|i| {
                let t = i as f32 / sample_rate;
                (2.0 * std::f32::consts::PI * frequency * t).sin() * 0.5
            })
            .collect();

        let analyzer = QualityAnalyzer::new(sample_rate);
        let assessment = analyzer.analyze(&samples).unwrap();

        assert!(
            assessment.overall_score > 0.7,
            "Clean signal should have high quality score (got {})",
            assessment.overall_score
        );
        assert!(
            assessment.metrics.clipping_ratio < 0.001,
            "No clipping expected"
        );
        assert!(assessment.metrics.snr_db > 30.0, "Good SNR expected");
    }

    #[test]
    fn test_clipping_detection() {
        let sample_rate = 44100.0;
        let mut samples: Vec<f32> = (0..44100)
            .map(|i| {
                let t = i as f32 / sample_rate;
                (2.0 * std::f32::consts::PI * 440.0 * t).sin()
            })
            .collect();

        // Add clipping
        for sample in samples.iter_mut() {
            if *sample > 0.99 {
                *sample = 1.0;
            } else if *sample < -0.99 {
                *sample = -1.0;
            }
        }

        let analyzer = QualityAnalyzer::new(sample_rate);
        let assessment = analyzer.analyze(&samples).unwrap();

        assert!(
            assessment.metrics.clipping_ratio > 0.0,
            "Clipping should be detected"
        );
        assert!(
            assessment
                .issues
                .iter()
                .any(|i| i.issue_type == IssueType::Clipping),
            "Clipping issue should be reported"
        );
    }

    #[test]
    fn test_dc_offset_detection() {
        let sample_rate = 44100.0;
        let dc_offset = 0.1;
        let samples: Vec<f32> = (0..44100)
            .map(|i| {
                let t = i as f32 / sample_rate;
                (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.3 + dc_offset
            })
            .collect();

        let analyzer = QualityAnalyzer::new(sample_rate);
        let assessment = analyzer.analyze(&samples).unwrap();

        assert!(
            (assessment.metrics.dc_offset - dc_offset).abs() < 0.01,
            "DC offset should be detected accurately"
        );
        assert!(
            assessment
                .issues
                .iter()
                .any(|i| i.issue_type == IssueType::DCOffset),
            "DC offset issue should be reported"
        );
    }
}
