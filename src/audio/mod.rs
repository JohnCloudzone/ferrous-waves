pub mod buffer;
pub mod decoder;
pub mod formats;

use std::path::Path;
use crate::utils::error::Result;

pub use buffer::AudioBuffer;
pub use decoder::AudioDecoder;
pub use formats::AudioFormat;

pub struct AudioFile {
    pub buffer: AudioBuffer,
    pub format: AudioFormat,
    pub path: String,
}

impl AudioFile {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path_str = path.as_ref().to_string_lossy().to_string();
        let format = AudioFormat::from_path(&path);

        if !format.is_supported() {
            return Err(crate::utils::error::FerrousError::AudioDecode(
                format!("Unsupported audio format: {:?}", format)
            ));
        }

        let mut decoder = AudioDecoder::new(&path)?;
        let samples = decoder.decode_all()?;
        let sample_rate = decoder.sample_rate().unwrap_or(44100);
        let channels = decoder.num_channels().unwrap_or(2);

        let buffer = AudioBuffer::new(samples, sample_rate, channels);

        Ok(Self {
            buffer,
            format,
            path: path_str,
        })
    }
}