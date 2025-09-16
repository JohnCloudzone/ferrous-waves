use ferrous_waves::utils::error::{FerrousError, Result};
use ferrous_waves::audio::{AudioFile, AudioFormat};
use std::io;

#[test]
fn test_error_display() {
    let err = FerrousError::AudioDecode("Test error".to_string());
    assert_eq!(format!("{}", err), "Audio decode error: Test error");

    let err = FerrousError::Analysis("Analysis failed".to_string());
    assert_eq!(format!("{}", err), "Analysis error: Analysis failed");

    let err = FerrousError::Cache("Cache miss".to_string());
    assert_eq!(format!("{}", err), "Cache error: Cache miss");
}

#[test]
fn test_error_from_io() {
    let io_err = io::Error::new(io::ErrorKind::NotFound, "File not found");
    let ferrous_err: FerrousError = io_err.into();

    match ferrous_err {
        FerrousError::Io(_) => (),
        _ => panic!("Should be IO error"),
    }
}

#[test]
fn test_audio_file_nonexistent() {
    let result = AudioFile::load("/definitely/does/not/exist.wav");
    assert!(result.is_err());

    match result {
        Err(FerrousError::AudioDecode(_)) => (),
        _ => panic!("Should be AudioDecode error"),
    }
}

#[test]
fn test_unsupported_format() {
    // This test would need an actual file with unsupported extension
    let format = AudioFormat::from_path("test.xyz");
    assert_eq!(format, AudioFormat::Unknown);
    assert!(!format.is_supported());
}

#[test]
fn test_result_type() {
    fn test_function() -> Result<i32> {
        Ok(42)
    }

    let result = test_function();
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), 42);
}

#[test]
fn test_error_chaining() {
    fn inner_function() -> Result<()> {
        Err(FerrousError::Analysis("Inner error".to_string()))
    }

    fn outer_function() -> Result<()> {
        inner_function()?;
        Ok(())
    }

    let result = outer_function();
    assert!(result.is_err());

    if let Err(e) = result {
        assert!(format!("{}", e).contains("Inner error"));
    }
}