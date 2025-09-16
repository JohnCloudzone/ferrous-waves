use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use ferrous_waves::analysis::spectral::{FftProcessor, WindowFunction};

fn benchmark_fft_sizes(c: &mut Criterion) {
    let sizes = vec![256, 512, 1024, 2048, 4096, 8192];

    let mut group = c.benchmark_group("fft_sizes");

    for size in sizes {
        let processor = FftProcessor::new(size);
        let input: Vec<f32> = (0..size).map(|i| (i as f32).sin()).collect();

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| {
                processor.magnitude_spectrum(black_box(&input))
            });
        });
    }

    group.finish();
}

fn benchmark_window_functions(c: &mut Criterion) {
    let mut group = c.benchmark_group("window_functions");
    let size = 2048;
    let mut samples = vec![1.0; size];

    let windows = vec![
        ("Hann", WindowFunction::Hann),
        ("Hamming", WindowFunction::Hamming),
        ("Blackman", WindowFunction::Blackman),
        ("Nuttall", WindowFunction::Nuttall),
        ("Rectangular", WindowFunction::Rectangular),
    ];

    for (name, window) in windows {
        group.bench_function(name, |b| {
            b.iter(|| {
                window.apply(black_box(&mut samples));
            });
        });
    }

    group.finish();
}

fn benchmark_fft_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_operations");
    let processor = FftProcessor::new(2048);
    let input: Vec<f32> = (0..2048).map(|i| (i as f32 * 0.01).sin()).collect();

    group.bench_function("magnitude_spectrum", |b| {
        b.iter(|| processor.magnitude_spectrum(black_box(&input)));
    });

    group.bench_function("power_spectrum", |b| {
        b.iter(|| processor.power_spectrum(black_box(&input)));
    });

    group.bench_function("phase_spectrum", |b| {
        b.iter(|| processor.phase_spectrum(black_box(&input)));
    });

    group.finish();
}

criterion_group!(benches, benchmark_fft_sizes, benchmark_window_functions, benchmark_fft_operations);
criterion_main!(benches);