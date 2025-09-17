use crate::analysis::spectral::StftProcessor;
use crate::utils::error::Result;
use std::collections::HashMap;

/// Audio fingerprint for similarity detection
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AudioFingerprint {
    /// Compact binary fingerprint
    pub fingerprint: Vec<u64>,

    /// Spectral hash sequences
    pub spectral_hashes: Vec<SpectralHash>,

    /// Temporal landmarks
    pub landmarks: Vec<Landmark>,

    /// Perceptual hash (overall signature)
    pub perceptual_hash: u64,

    /// Fingerprint metadata
    pub metadata: FingerprintMetadata,

    /// Sub-fingerprints for partial matching
    pub sub_fingerprints: Vec<SubFingerprint>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SpectralHash {
    /// Time position in seconds
    pub time: f32,

    /// Hash value derived from spectral peaks
    pub hash: u32,

    /// Frequency bins involved
    pub frequency_bins: Vec<u16>,

    /// Energy level
    pub energy: f32,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Landmark {
    /// Time of the landmark
    pub time: f32,

    /// Frequency of the peak
    pub frequency: f32,

    /// Magnitude of the peak
    pub magnitude: f32,

    /// Landmark type
    pub landmark_type: LandmarkType,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum LandmarkType {
    SpectralPeak,
    OnsetEvent,
    EnergyBurst,
    FrequencyShift,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FingerprintMetadata {
    /// Duration of the audio
    pub duration: f32,

    /// Sample rate
    pub sample_rate: f32,

    /// Average energy
    pub avg_energy: f32,

    /// Dominant frequencies
    pub dominant_frequencies: Vec<f32>,

    /// Fingerprint version
    pub version: u16,

    /// Algorithm parameters
    pub params: FingerprintParams,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FingerprintParams {
    /// FFT size used
    pub fft_size: usize,

    /// Hop size
    pub hop_size: usize,

    /// Number of frequency bands
    pub num_bands: usize,

    /// Peak picking threshold
    pub peak_threshold: f32,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SubFingerprint {
    /// Start time of sub-fingerprint
    pub start_time: f32,

    /// Duration
    pub duration: f32,

    /// Compact hash for this segment
    pub hash: u64,

    /// Energy profile
    pub energy_profile: Vec<f32>,
}

/// Result of comparing two fingerprints
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FingerprintMatch {
    /// Overall similarity score (0.0 to 1.0)
    pub similarity: f32,

    /// Confidence in the match
    pub confidence: f32,

    /// Time offset if audio is shifted
    pub time_offset: Option<f32>,

    /// Matched segments
    pub matched_segments: Vec<MatchedSegment>,

    /// Match type
    pub match_type: MatchType,

    /// Detailed scores
    pub scores: MatchScores,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MatchedSegment {
    /// Time in first audio
    pub time_a: f32,

    /// Time in second audio
    pub time_b: f32,

    /// Duration of match
    pub duration: f32,

    /// Match quality
    pub quality: f32,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum MatchType {
    Identical,
    VerySimilar,
    Similar,
    PartiallySimilar,
    Different,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MatchScores {
    /// Spectral similarity
    pub spectral: f32,

    /// Temporal pattern similarity
    pub temporal: f32,

    /// Energy profile similarity
    pub energy: f32,

    /// Landmark matching score
    pub landmark: f32,

    /// Perceptual hash similarity
    pub perceptual: f32,
}

/// Fingerprint database for searching
pub struct FingerprintDatabase {
    /// Stored fingerprints with IDs
    fingerprints: HashMap<String, AudioFingerprint>,

    /// Inverted index for fast searching
    hash_index: HashMap<u32, Vec<String>>,

    /// Perceptual hash index
    perceptual_index: HashMap<u64, Vec<String>>,
}

impl Default for FingerprintDatabase {
    fn default() -> Self {
        Self::new()
    }
}

impl FingerprintDatabase {
    pub fn new() -> Self {
        Self {
            fingerprints: HashMap::new(),
            hash_index: HashMap::new(),
            perceptual_index: HashMap::new(),
        }
    }

    pub fn insert(&mut self, id: String, fingerprint: AudioFingerprint) {
        // Index spectral hashes
        for hash in &fingerprint.spectral_hashes {
            self.hash_index
                .entry(hash.hash)
                .or_default()
                .push(id.clone());
        }

        // Index perceptual hash
        self.perceptual_index
            .entry(fingerprint.perceptual_hash)
            .or_default()
            .push(id.clone());

        // Store fingerprint
        self.fingerprints.insert(id, fingerprint);
    }

    pub fn search(
        &self,
        query: &AudioFingerprint,
        threshold: f32,
    ) -> Vec<(String, FingerprintMatch)> {
        let mut matches = Vec::new();
        let mut candidates = HashMap::new();

        // Find candidates using spectral hashes
        for hash in &query.spectral_hashes {
            if let Some(ids) = self.hash_index.get(&hash.hash) {
                for id in ids {
                    *candidates.entry(id.clone()).or_insert(0) += 1;
                }
            }
        }

        // Check perceptual hash
        let perceptual_candidates = self.find_similar_perceptual(query.perceptual_hash);
        for id in perceptual_candidates {
            *candidates.entry(id).or_insert(0) += 10; // Higher weight for perceptual match
        }

        // Score each candidate
        for (id, _score) in candidates {
            if let Some(fingerprint) = self.fingerprints.get(&id) {
                let match_result = FingerprintMatcher::new().compare(query, fingerprint);
                if match_result.similarity >= threshold {
                    matches.push((id, match_result));
                }
            }
        }

        // Sort by similarity
        matches.sort_by(|a, b| b.1.similarity.partial_cmp(&a.1.similarity).unwrap());

        matches
    }

    fn find_similar_perceptual(&self, hash: u64) -> Vec<String> {
        let mut similar = Vec::new();

        // Check exact match
        if let Some(ids) = self.perceptual_index.get(&hash) {
            similar.extend(ids.clone());
        }

        // Check near matches (1-2 bit differences)
        for (stored_hash, ids) in &self.perceptual_index {
            let diff = (hash ^ stored_hash).count_ones();
            if diff <= 2 && diff > 0 {
                similar.extend(ids.clone());
            }
        }

        similar
    }

    pub fn size(&self) -> usize {
        self.fingerprints.len()
    }
}

/// Audio fingerprinting engine
pub struct FingerprintGenerator {
    sample_rate: f32,
    fft_size: usize,
    hop_size: usize,
    num_bands: usize,
    peak_threshold: f32,
}

impl FingerprintGenerator {
    pub fn new(sample_rate: f32) -> Self {
        Self {
            sample_rate,
            fft_size: 2048,
            hop_size: 512,
            num_bands: 32,
            peak_threshold: 0.1,
        }
    }

    pub fn generate(&self, samples: &[f32]) -> Result<AudioFingerprint> {
        // Generate spectrogram
        let stft = StftProcessor::new(
            self.fft_size,
            self.hop_size,
            crate::analysis::spectral::WindowFunction::Hann,
        );
        let spectrogram = stft.process(samples);

        // Extract spectral hashes
        let spectral_hashes = self.generate_spectral_hashes(&spectrogram);

        // Find landmarks
        let landmarks = self.find_landmarks(&spectrogram);

        // Generate main fingerprint
        let fingerprint = self.generate_compact_fingerprint(&spectral_hashes, &landmarks);

        // Generate perceptual hash
        let perceptual_hash = self.generate_perceptual_hash(samples, &spectrogram);

        // Generate sub-fingerprints for partial matching
        let sub_fingerprints = self.generate_sub_fingerprints(samples, &spectrogram);

        // Calculate metadata
        let metadata = self.calculate_metadata(samples, &spectrogram);

        Ok(AudioFingerprint {
            fingerprint,
            spectral_hashes,
            landmarks,
            perceptual_hash,
            metadata,
            sub_fingerprints,
        })
    }

    fn generate_spectral_hashes(&self, spectrogram: &ndarray::Array2<f32>) -> Vec<SpectralHash> {
        let mut hashes = Vec::new();
        let num_frames = spectrogram.shape()[1];
        let num_bins = spectrogram.shape()[0];

        // Divide spectrum into bands
        let band_size = num_bins / self.num_bands;

        for frame_idx in 0..num_frames {
            let frame = spectrogram.column(frame_idx);
            let time = frame_idx as f32 * self.hop_size as f32 / self.sample_rate;

            // Find peak in each band
            let mut band_peaks = Vec::new();
            let mut frequency_bins = Vec::new();
            let mut total_energy = 0.0;
            let mut max_magnitude = 0.0;

            // First pass: find maximum magnitude in frame for relative threshold
            for i in 0..num_bins {
                let magnitude = frame[[i]];
                if magnitude > max_magnitude {
                    max_magnitude = magnitude;
                }
                total_energy += magnitude;
            }

            // Use relative threshold based on frame energy
            let _relative_threshold = (max_magnitude * 0.1).max(self.peak_threshold * 0.1);

            for band_idx in 0..self.num_bands {
                let start = band_idx * band_size;
                let end = ((band_idx + 1) * band_size).min(num_bins);

                if start < end {
                    // Find peak in this frequency band
                    let mut peak_idx = 0;
                    let mut peak_val = 0.0;

                    for i in start..end {
                        let magnitude = frame[[i]];
                        if magnitude > peak_val {
                            peak_val = magnitude;
                            peak_idx = i - start;
                        }
                    }

                    // Always include the peak, even if it's low energy
                    // This ensures we generate fingerprints even for quiet/simple signals
                    band_peaks.push((band_idx, peak_idx));
                    frequency_bins.push((start + peak_idx) as u16);
                }
            }

            // Generate hash from peak pattern
            let hash = self.peaks_to_hash(&band_peaks);

            hashes.push(SpectralHash {
                time,
                hash,
                frequency_bins,
                energy: if band_peaks.is_empty() {
                    0.0
                } else {
                    total_energy / num_bins as f32
                },
            });
        }

        hashes
    }

    fn peaks_to_hash(&self, peaks: &[(usize, usize)]) -> u32 {
        let mut hash = 0u32;

        for (band_idx, peak_idx) in peaks {
            // Combine band index and peak position into hash
            hash ^= ((*band_idx as u32) << 16) | (*peak_idx as u32);
            hash = hash.wrapping_mul(0x9e3779b9); // Golden ratio hash
        }

        hash
    }

    fn find_landmarks(&self, spectrogram: &ndarray::Array2<f32>) -> Vec<Landmark> {
        let mut landmarks = Vec::new();
        let num_frames = spectrogram.shape()[1];
        let num_bins = spectrogram.shape()[0];

        // Find spectral peaks
        for frame_idx in 1..num_frames - 1 {
            for bin_idx in 1..num_bins - 1 {
                let current = spectrogram[[bin_idx, frame_idx]];

                // Check if local maximum
                if current > self.peak_threshold
                    && current > spectrogram[[bin_idx - 1, frame_idx]]
                    && current > spectrogram[[bin_idx + 1, frame_idx]]
                    && current > spectrogram[[bin_idx, frame_idx - 1]]
                    && current > spectrogram[[bin_idx, frame_idx + 1]]
                {
                    let time = frame_idx as f32 * self.hop_size as f32 / self.sample_rate;
                    let frequency =
                        bin_idx as f32 * self.sample_rate / (2.0 * self.fft_size as f32);

                    landmarks.push(Landmark {
                        time,
                        frequency,
                        magnitude: current,
                        landmark_type: LandmarkType::SpectralPeak,
                    });
                }
            }
        }

        // Detect onset events
        let onsets = self.detect_onset_landmarks(spectrogram);
        landmarks.extend(onsets);

        // Sort by time
        landmarks.sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap());

        // Limit number of landmarks
        if landmarks.len() > 1000 {
            landmarks.truncate(1000);
        }

        landmarks
    }

    fn detect_onset_landmarks(&self, spectrogram: &ndarray::Array2<f32>) -> Vec<Landmark> {
        let mut landmarks = Vec::new();
        let num_frames = spectrogram.shape()[1];

        if num_frames < 3 {
            return landmarks;
        }

        // Calculate spectral flux
        for frame_idx in 1..num_frames {
            let current_frame = spectrogram.column(frame_idx);
            let prev_frame = spectrogram.column(frame_idx - 1);

            let flux: f32 = current_frame
                .iter()
                .zip(prev_frame.iter())
                .map(|(curr, prev)| (curr - prev).max(0.0))
                .sum();

            // Detect peaks in spectral flux
            if frame_idx > 0 && frame_idx < num_frames - 1 && flux > 0.5 {
                let time = frame_idx as f32 * self.hop_size as f32 / self.sample_rate;

                landmarks.push(Landmark {
                    time,
                    frequency: 0.0, // Onset doesn't have specific frequency
                    magnitude: flux,
                    landmark_type: LandmarkType::OnsetEvent,
                });
            }
        }

        landmarks
    }

    fn generate_compact_fingerprint(
        &self,
        hashes: &[SpectralHash],
        landmarks: &[Landmark],
    ) -> Vec<u64> {
        let mut fingerprint = Vec::new();

        // Combine spectral hashes into compact representation
        for chunk in hashes.chunks(8) {
            let mut combined = 0u64;
            for (i, hash) in chunk.iter().enumerate() {
                combined |= (hash.hash as u64 & 0xFF) << (i * 8);
            }
            fingerprint.push(combined);
        }

        // Add landmark information
        for chunk in landmarks.chunks(4) {
            let mut combined = 0u64;
            for (i, landmark) in chunk.iter().enumerate() {
                let freq_bits = (landmark.frequency / 10.0) as u16;
                combined |= (freq_bits as u64) << (i * 16);
            }
            fingerprint.push(combined);
        }

        fingerprint
    }

    fn generate_perceptual_hash(&self, samples: &[f32], spectrogram: &ndarray::Array2<f32>) -> u64 {
        let mut hash = 0u64;

        // Energy-based features
        let energy: f32 = samples.iter().map(|x| x * x).sum();
        let avg_energy = energy / samples.len() as f32;
        hash ^= (avg_energy * 1000.0) as u64;

        // Spectral centroid
        let mut weighted_sum = 0.0;
        let mut magnitude_sum = 0.0;
        for (bin_idx, row) in spectrogram.rows().into_iter().enumerate() {
            let mag: f32 = row.sum();
            weighted_sum += bin_idx as f32 * mag;
            magnitude_sum += mag;
        }

        if magnitude_sum > 0.0 {
            let centroid = weighted_sum / magnitude_sum;
            hash ^= ((centroid * 100.0) as u64) << 16;
        }

        // Zero crossing rate
        let mut zcr = 0u32;
        for i in 1..samples.len().min(10000) {
            if (samples[i - 1] >= 0.0) != (samples[i] >= 0.0) {
                zcr += 1;
            }
        }
        hash ^= (zcr as u64) << 32;

        // Spectral rolloff
        if magnitude_sum > 0.0 {
            let threshold = magnitude_sum * 0.85;
            let mut cumsum = 0.0;
            for (bin_idx, row) in spectrogram.rows().into_iter().enumerate() {
                cumsum += row.sum();
                if cumsum >= threshold {
                    hash ^= ((bin_idx as u64) & 0xFFFF) << 48;
                    break;
                }
            }
        }

        hash
    }

    fn generate_sub_fingerprints(
        &self,
        samples: &[f32],
        spectrogram: &ndarray::Array2<f32>,
    ) -> Vec<SubFingerprint> {
        let mut sub_fingerprints = Vec::new();

        // Create sub-fingerprints for 5-second segments
        let segment_duration = 5.0;
        let segment_samples = (segment_duration * self.sample_rate) as usize;
        let hop_samples = segment_samples / 2; // 50% overlap

        let mut pos = 0;
        while pos + segment_samples <= samples.len() {
            let segment = &samples[pos..pos + segment_samples];
            let start_time = pos as f32 / self.sample_rate;

            // Calculate energy profile
            let energy_profile = self.calculate_energy_profile(segment);

            // Generate hash for segment
            let hash = self.generate_segment_hash(segment, spectrogram, pos);

            sub_fingerprints.push(SubFingerprint {
                start_time,
                duration: segment_duration,
                hash,
                energy_profile,
            });

            pos += hop_samples;
        }

        sub_fingerprints
    }

    fn calculate_energy_profile(&self, samples: &[f32]) -> Vec<f32> {
        let window_size = 1024;
        let mut profile = Vec::new();

        for chunk in samples.chunks(window_size) {
            let energy: f32 = chunk.iter().map(|x| x * x).sum::<f32>() / chunk.len() as f32;
            profile.push(energy.sqrt());
        }

        profile
    }

    fn generate_segment_hash(
        &self,
        segment: &[f32],
        _spectrogram: &ndarray::Array2<f32>,
        _offset: usize,
    ) -> u64 {
        // Simplified segment hash based on energy and zero crossings
        let mut hash = 0u64;

        // Energy
        let energy: f32 = segment.iter().map(|x| x * x).sum::<f32>() / segment.len() as f32;
        hash ^= (energy * 10000.0) as u64;

        // Zero crossings
        let mut zcr = 0u32;
        for i in 1..segment.len() {
            if (segment[i - 1] >= 0.0) != (segment[i] >= 0.0) {
                zcr += 1;
            }
        }
        hash ^= (zcr as u64) << 32;

        hash
    }

    fn calculate_metadata(
        &self,
        samples: &[f32],
        spectrogram: &ndarray::Array2<f32>,
    ) -> FingerprintMetadata {
        let duration = samples.len() as f32 / self.sample_rate;

        // Average energy
        let avg_energy = samples.iter().map(|x| x.abs()).sum::<f32>() / samples.len() as f32;

        // Find dominant frequencies
        let dominant_frequencies = self.find_dominant_frequencies(spectrogram);

        FingerprintMetadata {
            duration,
            sample_rate: self.sample_rate,
            avg_energy,
            dominant_frequencies,
            version: 1,
            params: FingerprintParams {
                fft_size: self.fft_size,
                hop_size: self.hop_size,
                num_bands: self.num_bands,
                peak_threshold: self.peak_threshold,
            },
        }
    }

    fn find_dominant_frequencies(&self, spectrogram: &ndarray::Array2<f32>) -> Vec<f32> {
        let mut freq_energy = vec![0.0; spectrogram.shape()[0]];

        // Sum energy across time for each frequency bin
        for (row_idx, energy) in freq_energy
            .iter_mut()
            .enumerate()
            .take(spectrogram.shape()[0])
        {
            *energy = spectrogram.row(row_idx).sum();
        }

        // Find top frequencies
        let mut indexed: Vec<(usize, f32)> = freq_energy
            .iter()
            .enumerate()
            .map(|(i, &e)| (i, e))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        indexed
            .iter()
            .take(5)
            .map(|(idx, _)| *idx as f32 * self.sample_rate / (2.0 * self.fft_size as f32))
            .collect()
    }
}

/// Fingerprint matching and comparison
pub struct FingerprintMatcher {
    time_tolerance: f32,
    frequency_tolerance: f32,
}

impl Default for FingerprintMatcher {
    fn default() -> Self {
        Self::new()
    }
}

impl FingerprintMatcher {
    pub fn new() -> Self {
        Self {
            time_tolerance: 0.1,       // 100ms
            frequency_tolerance: 50.0, // 50Hz
        }
    }

    pub fn compare(&self, fp_a: &AudioFingerprint, fp_b: &AudioFingerprint) -> FingerprintMatch {
        // Calculate various similarity scores
        let spectral_score =
            self.compare_spectral_hashes(&fp_a.spectral_hashes, &fp_b.spectral_hashes);
        let landmark_score = self.compare_landmarks(&fp_a.landmarks, &fp_b.landmarks);
        let perceptual_score =
            self.compare_perceptual_hash(fp_a.perceptual_hash, fp_b.perceptual_hash);
        let energy_score = self.compare_energy(fp_a, fp_b);
        let temporal_score = self.compare_temporal_patterns(fp_a, fp_b);

        // Find matched segments
        let matched_segments = self.find_matched_segments(fp_a, fp_b);

        // Detect time offset
        let time_offset = self.detect_time_offset(&matched_segments);

        // Calculate overall similarity
        let similarity = (spectral_score * 0.3
            + landmark_score * 0.25
            + perceptual_score * 0.2
            + energy_score * 0.15
            + temporal_score * 0.1)
            .clamp(0.0, 1.0);

        // Determine match type
        let match_type = if similarity > 0.95 {
            MatchType::Identical
        } else if similarity > 0.85 {
            MatchType::VerySimilar
        } else if similarity > 0.7 {
            MatchType::Similar
        } else if similarity > 0.5 {
            MatchType::PartiallySimilar
        } else {
            MatchType::Different
        };

        // Calculate confidence based on amount of data matched
        let confidence = if matched_segments.is_empty() {
            0.0
        } else {
            let total_matched_duration: f32 = matched_segments.iter().map(|s| s.duration).sum();
            let max_duration = fp_a.metadata.duration.min(fp_b.metadata.duration);
            (total_matched_duration / max_duration).min(1.0)
        };

        FingerprintMatch {
            similarity,
            confidence,
            time_offset,
            matched_segments,
            match_type,
            scores: MatchScores {
                spectral: spectral_score,
                temporal: temporal_score,
                energy: energy_score,
                landmark: landmark_score,
                perceptual: perceptual_score,
            },
        }
    }

    fn compare_spectral_hashes(&self, hashes_a: &[SpectralHash], hashes_b: &[SpectralHash]) -> f32 {
        if hashes_a.is_empty() || hashes_b.is_empty() {
            // If both are empty, consider them identical
            return if hashes_a.is_empty() && hashes_b.is_empty() {
                1.0
            } else {
                0.0
            };
        }

        let mut matches = 0;

        // For each hash in A, find the best matching hash in B within time tolerance
        for hash_a in hashes_a {
            for hash_b in hashes_b {
                if (hash_a.time - hash_b.time).abs() < self.time_tolerance
                    && hash_a.hash == hash_b.hash
                {
                    matches += 1;
                    break; // Found exact match for this time frame
                }
            }
        }

        // Score based on percentage of hashes that matched
        matches as f32 / hashes_a.len().max(hashes_b.len()) as f32
    }

    fn compare_landmarks(&self, landmarks_a: &[Landmark], landmarks_b: &[Landmark]) -> f32 {
        if landmarks_a.is_empty() || landmarks_b.is_empty() {
            // If both have no landmarks, they're identical in this aspect
            return if landmarks_a.is_empty() && landmarks_b.is_empty() {
                1.0
            } else {
                0.0
            };
        }

        let mut matches = 0;

        // For each landmark in A, find best match in B
        for landmark_a in landmarks_a {
            let mut found_match = false;
            for landmark_b in landmarks_b {
                if (landmark_a.time - landmark_b.time).abs() < self.time_tolerance
                    && (landmark_a.frequency - landmark_b.frequency).abs()
                        < self.frequency_tolerance
                    && landmark_a.landmark_type == landmark_b.landmark_type
                {
                    found_match = true;
                    break; // Found a match for this landmark
                }
            }
            if found_match {
                matches += 1;
            }
        }

        // Score based on percentage of landmarks that matched
        let max_landmarks = landmarks_a.len().max(landmarks_b.len());
        (matches as f32 / max_landmarks as f32).min(1.0)
    }

    fn compare_perceptual_hash(&self, hash_a: u64, hash_b: u64) -> f32 {
        // Calculate Hamming distance
        let diff_bits = (hash_a ^ hash_b).count_ones();

        // Convert to similarity score
        1.0 - (diff_bits as f32 / 64.0)
    }

    fn compare_energy(&self, fp_a: &AudioFingerprint, fp_b: &AudioFingerprint) -> f32 {
        let diff = (fp_a.metadata.avg_energy - fp_b.metadata.avg_energy).abs();
        let max_energy = fp_a.metadata.avg_energy.max(fp_b.metadata.avg_energy);

        if max_energy > 0.0 {
            1.0 - (diff / max_energy).min(1.0)
        } else {
            1.0
        }
    }

    fn compare_temporal_patterns(&self, fp_a: &AudioFingerprint, fp_b: &AudioFingerprint) -> f32 {
        // Compare sub-fingerprint patterns
        if fp_a.sub_fingerprints.is_empty() || fp_b.sub_fingerprints.is_empty() {
            // If both have no sub-fingerprints, they match perfectly in this aspect
            return if fp_a.sub_fingerprints.is_empty() && fp_b.sub_fingerprints.is_empty() {
                1.0
            } else {
                0.0
            };
        }

        let mut total_similarity = 0.0;
        let mut comparisons = 0;

        for sub_a in &fp_a.sub_fingerprints {
            for sub_b in &fp_b.sub_fingerprints {
                if (sub_a.start_time - sub_b.start_time).abs() < 1.0 {
                    let hash_sim = self.compare_perceptual_hash(sub_a.hash, sub_b.hash);
                    total_similarity += hash_sim;
                    comparisons += 1;
                }
            }
        }

        if comparisons > 0 {
            total_similarity / comparisons as f32
        } else {
            0.0
        }
    }

    fn find_matched_segments(
        &self,
        fp_a: &AudioFingerprint,
        fp_b: &AudioFingerprint,
    ) -> Vec<MatchedSegment> {
        let mut segments = Vec::new();

        // Match sub-fingerprints
        for sub_a in &fp_a.sub_fingerprints {
            for sub_b in &fp_b.sub_fingerprints {
                let hash_similarity = self.compare_perceptual_hash(sub_a.hash, sub_b.hash);

                if hash_similarity > 0.7 {
                    // Compare energy profiles
                    let energy_sim =
                        self.compare_energy_profiles(&sub_a.energy_profile, &sub_b.energy_profile);

                    if energy_sim > 0.6 {
                        let quality = (hash_similarity + energy_sim) / 2.0;

                        segments.push(MatchedSegment {
                            time_a: sub_a.start_time,
                            time_b: sub_b.start_time,
                            duration: sub_a.duration.min(sub_b.duration),
                            quality,
                        });
                    }
                }
            }
        }

        // Remove overlapping segments, keeping highest quality
        segments.sort_by(|a, b| b.quality.partial_cmp(&a.quality).unwrap());
        let mut filtered = Vec::new();

        for segment in segments {
            let overlaps = filtered.iter().any(|s: &MatchedSegment| {
                (s.time_a <= segment.time_a && segment.time_a < s.time_a + s.duration)
                    || (s.time_b <= segment.time_b && segment.time_b < s.time_b + s.duration)
            });

            if !overlaps {
                filtered.push(segment);
            }
        }

        filtered
    }

    fn compare_energy_profiles(&self, profile_a: &[f32], profile_b: &[f32]) -> f32 {
        if profile_a.is_empty() || profile_b.is_empty() {
            return 0.0;
        }

        let len = profile_a.len().min(profile_b.len());
        let mut sum_diff = 0.0;
        let mut sum_total = 0.0;

        for i in 0..len {
            sum_diff += (profile_a[i] - profile_b[i]).abs();
            sum_total += profile_a[i] + profile_b[i];
        }

        if sum_total > 0.0 {
            1.0 - (sum_diff / sum_total).min(1.0)
        } else {
            1.0
        }
    }

    fn detect_time_offset(&self, segments: &[MatchedSegment]) -> Option<f32> {
        if segments.is_empty() {
            return None;
        }

        // Calculate median time offset
        let mut offsets: Vec<f32> = segments.iter().map(|s| s.time_b - s.time_a).collect();

        offsets.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let median_idx = offsets.len() / 2;
        Some(offsets[median_idx])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fingerprint_generation() {
        let sample_rate = 44100.0;
        let samples = vec![0.0; 44100]; // 1 second of silence

        let generator = FingerprintGenerator::new(sample_rate);
        let fingerprint = generator.generate(&samples).unwrap();

        assert!(!fingerprint.fingerprint.is_empty());
        assert_eq!(fingerprint.metadata.duration, 1.0);
        assert_eq!(fingerprint.metadata.sample_rate, sample_rate);
    }

    #[test]
    fn test_fingerprint_database() {
        let mut db = FingerprintDatabase::new();

        let generator = FingerprintGenerator::new(44100.0);
        let samples = vec![0.1; 44100];
        let fp = generator.generate(&samples).unwrap();

        db.insert("test_audio".to_string(), fp.clone());
        assert_eq!(db.size(), 1);

        let results = db.search(&fp, 0.5);
        assert!(!results.is_empty());
        assert_eq!(results[0].0, "test_audio");
        assert!(results[0].1.similarity > 0.9);
    }

    #[test]
    fn test_fingerprint_matching() {
        let generator = FingerprintGenerator::new(44100.0);

        // Same audio
        let samples = vec![0.1; 44100];
        let fp1 = generator.generate(&samples).unwrap();
        let fp2 = generator.generate(&samples).unwrap();

        let matcher = FingerprintMatcher::new();
        let result = matcher.compare(&fp1, &fp2);

        assert!(result.similarity > 0.9);
        assert_eq!(result.match_type, MatchType::Identical);
    }

    #[test]
    fn test_perceptual_hash() {
        let generator = FingerprintGenerator::new(44100.0);

        // Generate sine wave
        let samples: Vec<f32> = (0..44100)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin() * 0.5)
            .collect();

        let fp = generator.generate(&samples).unwrap();
        assert_ne!(fp.perceptual_hash, 0);
    }
}
