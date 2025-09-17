use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ferrous_waves::analysis::temporal::{BeatTracker, OnsetDetector};

fn benchmark_onset_detection(c: &mut Criterion) {
    let mut group = c.benchmark_group("onset_detection");
    let detector = OnsetDetector::new();

    let signal_lengths = vec![100, 500, 1000, 2000];

    for length in signal_lengths {
        // Create spectral flux signal with some peaks
        let mut flux = vec![0.1; length];
        for i in (0..length).step_by(50) {
            flux[i] = 1.0;
        }

        group.bench_function(format!("{}_frames", length), |b| {
            b.iter(|| detector.detect_onsets(black_box(&flux), 512, 44100));
        });
    }

    group.finish();
}

fn benchmark_tempo_estimation(c: &mut Criterion) {
    let mut group = c.benchmark_group("tempo_estimation");
    let tracker = BeatTracker::new();

    let onset_counts = vec![10, 50, 100, 200];

    for count in onset_counts {
        // Create regular onset pattern at 120 BPM
        let onsets: Vec<f32> = (0..count).map(|i| i as f32 * 0.5).collect();

        group.bench_function(format!("{}_onsets", count), |b| {
            b.iter(|| tracker.estimate_tempo(black_box(&onsets)));
        });
    }

    group.finish();
}

fn benchmark_beat_tracking(c: &mut Criterion) {
    let mut group = c.benchmark_group("beat_tracking");
    let tracker = BeatTracker::new();

    // Create onset times with slight variations
    let onset_counts = vec![20, 50, 100];
    let tempo = 120.0;

    for count in onset_counts {
        let onsets: Vec<f32> = (0..count)
            .map(|i| i as f32 * 0.5 + (i as f32 * 0.1).sin() * 0.05)
            .collect();

        group.bench_function(format!("{}_onsets", count), |b| {
            b.iter(|| tracker.track_beats(black_box(&onsets), tempo));
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_onset_detection,
    benchmark_tempo_estimation,
    benchmark_beat_tracking
);
criterion_main!(benches);
