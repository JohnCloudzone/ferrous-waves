use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AudioFormat {
    Wav,
    Mp3,
    Flac,
    Ogg,
    M4a,
    Unknown,
}

impl AudioFormat {
    pub fn from_path<P: AsRef<Path>>(path: P) -> Self {
        path.as_ref()
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| match ext.to_lowercase().as_str() {
                "wav" | "wave" => Self::Wav,
                "mp3" => Self::Mp3,
                "flac" => Self::Flac,
                "ogg" | "oga" => Self::Ogg,
                "m4a" | "aac" => Self::M4a,
                _ => Self::Unknown,
            })
            .unwrap_or(Self::Unknown)
    }

    pub fn is_supported(&self) -> bool {
        !matches!(self, Self::Unknown)
    }
}