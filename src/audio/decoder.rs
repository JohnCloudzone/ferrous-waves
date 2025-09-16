use symphonia::core::audio::AudioBufferRef;
use symphonia::core::codecs::{Decoder, DecoderOptions};
use symphonia::core::formats::{FormatOptions, FormatReader};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use std::fs::File;
use std::path::Path;

use crate::utils::error::{FerrousError, Result};

pub struct AudioDecoder {
    format_reader: Box<dyn FormatReader>,
    decoder: Box<dyn Decoder>,
    track_id: u32,
}

impl AudioDecoder {
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(&path)
            .map_err(|e| FerrousError::AudioDecode(format!("Failed to open file: {}", e)))?;

        let mss = MediaSourceStream::new(Box::new(file), Default::default());

        // Create hint from file extension
        let mut hint = Hint::new();
        if let Some(ext) = path.as_ref().extension() {
            hint.with_extension(ext.to_string_lossy().as_ref());
        }

        // Probe the media source
        let probe_result = symphonia::default::get_probe()
            .format(&hint, mss, &FormatOptions::default(), &MetadataOptions::default())
            .map_err(|e| FerrousError::AudioDecode(format!("Failed to probe format: {}", e)))?;

        let format_reader = probe_result.format;

        // Find the first audio track
        let track = format_reader
            .tracks()
            .iter()
            .find(|t| t.codec_params.codec != symphonia::core::codecs::CODEC_TYPE_NULL)
            .ok_or_else(|| FerrousError::AudioDecode("No audio tracks found".to_string()))?;

        let track_id = track.id;

        // Create decoder
        let decoder = symphonia::default::get_codecs()
            .make(&track.codec_params, &DecoderOptions::default())
            .map_err(|e| FerrousError::AudioDecode(format!("Failed to create decoder: {}", e)))?;

        Ok(Self {
            format_reader,
            decoder,
            track_id,
        })
    }

    pub fn decode_all(&mut self) -> Result<Vec<f32>> {
        let mut samples = Vec::new();

        loop {
            let packet = match self.format_reader.next_packet() {
                Ok(packet) => packet,
                Err(symphonia::core::errors::Error::ResetRequired) => {
                    self.decoder.reset();
                    continue;
                }
                Err(symphonia::core::errors::Error::IoError(e))
                    if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(FerrousError::AudioDecode(format!("Packet read error: {}", e))),
            };

            if packet.track_id() != self.track_id {
                continue;
            }

            let decoded = match self.decoder.decode(&packet) {
                Ok(decoded) => decoded,
                Err(symphonia::core::errors::Error::DecodeError(_)) => continue,
                Err(e) => return Err(FerrousError::AudioDecode(format!("Decode error: {}", e))),
            };

            copy_samples(&decoded, &mut samples)?;
        }

        Ok(samples)
    }

    pub fn sample_rate(&self) -> Option<u32> {
        self.format_reader
            .tracks()
            .iter()
            .find(|t| t.id == self.track_id)
            .and_then(|t| t.codec_params.sample_rate)
    }

    pub fn num_channels(&self) -> Option<usize> {
        self.format_reader
            .tracks()
            .iter()
            .find(|t| t.id == self.track_id)
            .and_then(|t| t.codec_params.channels.map(|c| c.count()))
    }
}

fn copy_samples(decoded: &AudioBufferRef, samples: &mut Vec<f32>) -> Result<()> {
    match decoded {
        AudioBufferRef::F32(buf) => {
            for plane in buf.planes().planes() {
                samples.extend_from_slice(plane);
            }
        }
        AudioBufferRef::S16(buf) => {
            for plane in buf.planes().planes() {
                samples.extend(plane.iter().map(|&s| s as f32 / i16::MAX as f32));
            }
        }
        _ => {
            return Err(FerrousError::AudioDecode(
                "Unsupported sample format".to_string()
            ))
        }
    }
    Ok(())
}