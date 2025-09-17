use ferrous_waves::analysis::spectral::{SimdFft, SimdLevel};

fn main() {
    // Detect and display CPU SIMD capabilities
    let simd_level = SimdLevel::detect();

    println!("Ferrous Waves CPU Feature Detection");
    println!("====================================");
    println!("Detected SIMD Level: {}", simd_level.name());
    println!(
        "Optimal Vector Size: {} floats",
        simd_level.optimal_vector_size()
    );
    println!();

    // Create FFT processor with automatic SIMD selection
    let fft_size = 2048;
    let mut simd_fft = SimdFft::new(fft_size);

    println!("FFT Processor Configuration:");
    println!("  FFT Size: {}", fft_size);
    println!("  SIMD Optimization: {}", simd_level.name());
    println!();

    // Test with sample data
    let input: Vec<f32> = (0..fft_size).map(|i| (i as f32 * 0.01).sin()).collect();

    println!("Processing {} samples...", fft_size);
    let spectrum = simd_fft.process(&input);
    let magnitudes = simd_fft.magnitude_spectrum(&spectrum);

    println!("Results:");
    println!("  Spectrum bins: {}", spectrum.len());
    println!(
        "  Peak magnitude: {:.3}",
        magnitudes.iter().fold(0.0f32, |a, &b| a.max(b))
    );
    println!();

    // Platform-specific information
    #[cfg(target_arch = "x86_64")]
    {
        println!("x86_64 CPU Features:");
        println!("  SSE2:    {}", is_x86_feature_detected!("sse2"));
        println!("  AVX:     {}", is_x86_feature_detected!("avx"));
        println!("  AVX2:    {}", is_x86_feature_detected!("avx2"));
        println!("  AVX-512: {}", is_x86_feature_detected!("avx512f"));
    }

    #[cfg(target_arch = "aarch64")]
    {
        println!("ARM64 CPU Features:");
        println!(
            "  NEON: {}",
            std::arch::is_aarch64_feature_detected!("neon")
        );
    }
}
