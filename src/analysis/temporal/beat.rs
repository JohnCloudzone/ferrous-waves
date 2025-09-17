use std::collections::HashMap;

pub struct BeatTracker {
    min_tempo: f32,
    max_tempo: f32,
}

impl BeatTracker {
    pub fn new() -> Self {
        Self {
            min_tempo: 60.0,
            max_tempo: 200.0,
        }
    }

    pub fn estimate_tempo(&self, onset_times: &[f32]) -> Option<f32> {
        if onset_times.len() < 2 {
            return None;
        }

        // Compute inter-onset intervals
        let mut intervals = Vec::new();
        for i in 1..onset_times.len() {
            intervals.push(onset_times[i] - onset_times[i - 1]);
        }

        // Build histogram of intervals
        let mut histogram = HashMap::new();
        let bin_width = 0.01; // 10ms bins

        for &interval in &intervals {
            let bin = (interval / bin_width).round() as i32;
            *histogram.entry(bin).or_insert(0) += 1;
        }

        // Find peaks in histogram corresponding to tempo range
        let min_interval = 60.0 / self.max_tempo;
        let max_interval = 60.0 / self.min_tempo;

        let mut best_interval = 0.0;
        let mut best_count = 0;

        for (&bin, &count) in &histogram {
            let interval = bin as f32 * bin_width;

            if interval >= min_interval && interval <= max_interval && count > best_count {
                best_interval = interval;
                best_count = count;
            }
        }

        if best_count > 0 {
            Some(60.0 / best_interval)
        } else {
            None
        }
    }

    pub fn track_beats(&self, onset_times: &[f32], tempo: f32) -> Vec<f32> {
        if onset_times.is_empty() {
            return Vec::new();
        }

        let beat_period = 60.0 / tempo;
        let mut beats = Vec::new();

        // Find the best phase offset
        let mut best_phase = 0.0;
        let mut best_score = 0.0;

        for &onset in onset_times.iter().take(10) {
            let mut score = 0.0;
            let mut beat_time = onset;

            while beat_time < onset_times[onset_times.len() - 1] {
                // Find closest onset
                for &o in onset_times {
                    let distance = (o - beat_time).abs();
                    if distance < beat_period * 0.2 {
                        score += 1.0 / (1.0 + distance);
                    }
                }
                beat_time += beat_period;
            }

            if score > best_score {
                best_score = score;
                best_phase = onset;
            }
        }

        // Generate beats
        let mut beat_time = best_phase;
        let duration = onset_times[onset_times.len() - 1];

        while beat_time <= duration {
            beats.push(beat_time);
            beat_time += beat_period;
        }

        beats
    }
}

impl Default for BeatTracker {
    fn default() -> Self {
        Self::new()
    }
}
