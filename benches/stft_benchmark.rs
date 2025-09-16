use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use ferrous_waves::analysis::spectral::{StftProcessor, WindowFunction, MelFilterBank};
use ndarray::Array2;

fn benchmark_stft_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("stft_processing");

    let signal_lengths = vec![44100, 88200, 176400]; // 1s, 2s, 4s at 44.1kHz
    let window_size = 2048;
    let hop_size = 512;

    for length in signal_lengths {
        let processor = StftProcessor::new(window_size, hop_size, WindowFunction::Hann);
        let signal: Vec<f32> = (0..length).map(|i| (i as f32 * 0.001).sin()).collect();

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}samples", length)),
            &length,
            |b, _| {
                b.iter(|| processor.process(black_box(&signal)));
            },
        );
    }

    group.finish();
}

fn benchmark_window_hop_ratios(c: &mut Criterion) {
    let mut group = c.benchmark_group("window_hop_ratios");

    let configs = vec![
        (1024, 256),  // 75% overlap
        (1024, 512),  // 50% overlap
        (1024, 768),  // 25% overlap
        (1024, 1024), // No overlap
    ];

    let signal: Vec<f32> = (0..44100).map(|i| (i as f32 * 0.001).sin()).collect();

    for (window, hop) in configs {
        let processor = StftProcessor::new(window, hop, WindowFunction::Hann);

        group.bench_function(format!("w{}_h{}", window, hop), |b| {
            b.iter(|| processor.process(black_box(&signal)));
        });
    }

    group.finish();
}

fn benchmark_mel_filterbank(c: &mut Criterion) {
    let mut group = c.benchmark_group("mel_filterbank");

    let num_filters_list = vec![20, 40, 80, 128];
    let fft_size = 2048;
    let sample_rate = 44100;

    // Create a dummy spectrogram
    let num_frames = 100;
    let num_bins = fft_size / 2 + 1;
    let spectrogram = Array2::ones((num_bins, num_frames));

    for num_filters in num_filters_list {
        let filterbank = MelFilterBank::new(num_filters, sample_rate, fft_size);

        group.bench_function(format!("{}_filters", num_filters), |b| {
            b.iter(|| filterbank.apply(black_box(&spectrogram)));
        });
    }

    group.finish();
}

fn benchmark_db_conversion(c: &mut Criterion) {
    let processor = StftProcessor::new(2048, 512, WindowFunction::Hann);
    let signal: Vec<f32> = (0..44100).map(|i| (i as f32 * 0.001).sin()).collect();
    let spectrogram = processor.process(&signal);

    c.bench_function("stft_to_db", |b| {
        b.iter(|| processor.to_db(black_box(&spectrogram)));
    });
}

criterion_group!(
    benches,
    benchmark_stft_processing,
    benchmark_window_hop_ratios,
    benchmark_mel_filterbank,
    benchmark_db_conversion
);
criterion_main!(benches);