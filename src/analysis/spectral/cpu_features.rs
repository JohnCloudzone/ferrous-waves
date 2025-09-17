use std::sync::Once;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdLevel {
    None,
    Sse2,
    Avx,
    Avx2,
    Avx512,
    Neon,
}

static mut SIMD_LEVEL: SimdLevel = SimdLevel::None;
static INIT: Once = Once::new();

impl SimdLevel {
    pub fn detect() -> Self {
        unsafe {
            INIT.call_once(|| {
                SIMD_LEVEL = Self::detect_impl();
            });
            SIMD_LEVEL
        }
    }

    #[cfg(target_arch = "x86_64")]
    fn detect_impl() -> Self {
        if is_x86_feature_detected!("avx512f") {
            SimdLevel::Avx512
        } else if is_x86_feature_detected!("avx2") {
            SimdLevel::Avx2
        } else if is_x86_feature_detected!("avx") {
            SimdLevel::Avx
        } else if is_x86_feature_detected!("sse2") {
            SimdLevel::Sse2
        } else {
            SimdLevel::None
        }
    }

    #[cfg(target_arch = "x86")]
    fn detect_impl() -> Self {
        if is_x86_feature_detected!("avx2") {
            SimdLevel::Avx2
        } else if is_x86_feature_detected!("avx") {
            SimdLevel::Avx
        } else if is_x86_feature_detected!("sse2") {
            SimdLevel::Sse2
        } else {
            SimdLevel::None
        }
    }

    #[cfg(target_arch = "aarch64")]
    fn detect_impl() -> Self {
        if std::arch::is_aarch64_feature_detected!("neon") {
            SimdLevel::Neon
        } else {
            SimdLevel::None
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "x86", target_arch = "aarch64")))]
    fn detect_impl() -> Self {
        SimdLevel::None
    }

    pub fn optimal_vector_size(&self) -> usize {
        match self {
            SimdLevel::None => 1,
            SimdLevel::Sse2 => 4,    // 128-bit registers / 32-bit float = 4
            SimdLevel::Avx => 8,     // 256-bit registers / 32-bit float = 8
            SimdLevel::Avx2 => 8,    // 256-bit registers / 32-bit float = 8
            SimdLevel::Avx512 => 16, // 512-bit registers / 32-bit float = 16
            SimdLevel::Neon => 4,    // 128-bit registers / 32-bit float = 4
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            SimdLevel::None => "None",
            SimdLevel::Sse2 => "SSE2",
            SimdLevel::Avx => "AVX",
            SimdLevel::Avx2 => "AVX2",
            SimdLevel::Avx512 => "AVX-512",
            SimdLevel::Neon => "NEON",
        }
    }
}

pub fn log_cpu_features() {
    let level = SimdLevel::detect();
    tracing::info!(
        "CPU SIMD support detected: {} (vector size: {} floats)",
        level.name(),
        level.optimal_vector_size()
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_detection() {
        let level = SimdLevel::detect();
        println!("Detected SIMD level: {:?}", level);
        assert!(level.optimal_vector_size() >= 1);
    }

    #[test]
    fn test_simd_level_names() {
        assert_eq!(SimdLevel::None.name(), "None");
        assert_eq!(SimdLevel::Avx2.name(), "AVX2");
        assert_eq!(SimdLevel::Neon.name(), "NEON");
    }
}
