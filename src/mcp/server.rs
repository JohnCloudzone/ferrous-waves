use dashmap::DashMap;
use rmcp::{
    model::{ErrorData as McpError, *},
    service::RequestContext,
    transport::stdio,
    RoleServer, ServerHandler, ServiceExt,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::borrow::Cow;
use std::sync::Arc;
use uuid::Uuid;

use crate::audio::AudioFile;
use crate::cache::Cache;
use crate::AnalysisEngine;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyzeAudioParams {
    /// Path to the audio file to analyze
    pub file_path: String,

    /// Optional: Specific analysis types to perform
    #[serde(skip_serializing_if = "Option::is_none")]
    pub analysis_types: Option<Vec<String>>,

    /// Optional: Output directory for results
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_dir: Option<String>,

    /// Return format: "full" | "summary" | "visual_only"
    #[serde(default = "default_return_format")]
    pub return_format: String,

    /// Analysis profile: "quick" | "standard" | "detailed" | "fingerprint" | "mastering"
    #[serde(default = "default_analysis_profile")]
    pub analysis_profile: String,

    /// Maximum number of data points in arrays (for pagination)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_data_points: Option<usize>,

    /// Pagination cursor for continuing from previous results
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cursor: Option<String>,

    /// Include visual data (base64 images) - defaults to false for MCP
    #[serde(default = "default_include_visuals")]
    pub include_visuals: bool,

    /// Include raw spectral data arrays
    #[serde(default = "default_include_spectral")]
    pub include_spectral: bool,

    /// Include temporal data arrays (beats, onsets)
    #[serde(default = "default_include_temporal")]
    pub include_temporal: bool,
}

fn default_return_format() -> String {
    "summary".to_string()
}

fn default_analysis_profile() -> String {
    "standard".to_string()
}

fn default_include_visuals() -> bool {
    false // Never include base64 images by default in MCP
}

fn default_include_spectral() -> bool {
    false // Don't include large arrays by default
}

fn default_include_temporal() -> bool {
    false // Don't include large arrays by default
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompareAudioParams {
    /// First audio file path
    pub file_a: String,

    /// Second audio file path
    pub file_b: String,

    /// Comparison metrics to calculate
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metrics: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct JobStatus {
    pub id: String,
    pub status: String,
    pub progress: f32,
    pub message: Option<String>,
    pub completed_modules: Vec<String>,
    pub pending_modules: Vec<String>,
    pub partial_results: Option<serde_json::Value>,
}

// Enhanced response structures
#[derive(Debug, Clone, Serialize)]
pub struct EnhancedSummary {
    pub audio: AudioInfo,
    pub content: ContentInfo,
    pub quality: QualityInfo,
    pub fingerprint: FingerprintInfo,
}

#[derive(Debug, Clone, Serialize)]
pub struct AudioInfo {
    pub duration: f32,
    pub format: String,
    pub sample_rate: u32,
    pub channels: usize,
    pub bit_depth: Option<u32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ContentInfo {
    #[serde(rename = "type")]
    pub content_type: String,
    pub confidence: f32,
    pub key: Option<String>,
    pub tempo: Option<f32>,
    pub time_signature: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct QualityInfo {
    pub score: f32,
    pub loudness_lufs: f32,
    pub true_peak_dbfs: f32,
    pub dynamic_range: f32,
    pub issues: Vec<String>,
    pub recommendation: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct FingerprintInfo {
    pub hash: String,
    pub unique_id: String,
    pub spectral_hashes: usize,
    pub landmarks: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct HierarchicalInsights {
    pub critical: Vec<String>,
    pub warnings: Vec<String>,
    pub info: Vec<String>,
    pub suggestions: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct EnhancedComparison {
    pub similarity: SimilarityMetrics,
    pub differences: DifferenceMetrics,
    pub match_classification: String,
    pub use_cases: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SimilarityMetrics {
    pub overall: f32,
    pub fingerprint: f32,
    pub spectral: f32,
    pub perceptual: f32,
    pub structural: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct DifferenceMetrics {
    pub loudness_delta: f32,
    pub tempo_delta: Option<f32>,
    pub quality_delta: f32,
    pub duration_delta: f32,
    pub key_change: Option<String>,
}

#[derive(Clone)]
pub struct FerrousWavesMcp {
    engine: AnalysisEngine,
    active_jobs: Arc<DashMap<String, JobStatus>>,
}

impl FerrousWavesMcp {
    pub fn new() -> Self {
        Self {
            engine: AnalysisEngine::new(),
            active_jobs: Arc::new(DashMap::new()),
        }
    }

    fn create_enhanced_summary(
        &self,
        result: &crate::analysis::engine::AnalysisResult,
        audio: &AudioFile,
    ) -> EnhancedSummary {
        EnhancedSummary {
            audio: AudioInfo {
                duration: audio.buffer.duration_seconds,
                format: std::path::Path::new(&audio.path)
                    .extension()
                    .and_then(|e| e.to_str())
                    .unwrap_or("unknown")
                    .to_string(),
                sample_rate: audio.buffer.sample_rate,
                channels: audio.buffer.channels,
                bit_depth: None, // Could be extracted from metadata if available
            },
            content: ContentInfo {
                content_type: format!("{:?}", result.classification.primary_type),
                confidence: result.classification.confidence,
                key: Some(result.musical.key.key.clone()),
                tempo: result.temporal.tempo,
                time_signature: None, // Could be extracted from beat pattern
            },
            quality: QualityInfo {
                score: result.quality.overall_score,
                loudness_lufs: result.perceptual.loudness_lufs,
                true_peak_dbfs: result.perceptual.true_peak_dbfs,
                dynamic_range: result.perceptual.dynamic_range,
                issues: result
                    .quality
                    .issues
                    .iter()
                    .map(|i| format!("{:?}", i.issue_type))
                    .collect(),
                recommendation: if result.quality.overall_score > 0.9 {
                    "professional".to_string()
                } else if result.quality.overall_score > 0.7 {
                    "good".to_string()
                } else if result.quality.overall_score > 0.5 {
                    "acceptable".to_string()
                } else {
                    "needs_improvement".to_string()
                },
            },
            fingerprint: FingerprintInfo {
                hash: format!("{:016x}", result.fingerprint.perceptual_hash),
                unique_id: format!("audio_{:016x}", result.fingerprint.perceptual_hash),
                spectral_hashes: result.fingerprint.spectral_hashes.len(),
                landmarks: result.fingerprint.landmarks.len(),
            },
        }
    }

    fn create_hierarchical_insights(
        &self,
        result: &crate::analysis::engine::AnalysisResult,
    ) -> HierarchicalInsights {
        let mut critical = Vec::new();
        let mut warnings = Vec::new();
        let mut info = Vec::new();
        let mut suggestions = Vec::new();

        // Analyze quality issues
        for issue in &result.quality.issues {
            match issue.severity {
                crate::analysis::quality::IssueSeverity::Critical => {
                    critical.push(format!("{:?}: {}", issue.issue_type, issue.description));
                }
                crate::analysis::quality::IssueSeverity::High => {
                    warnings.push(format!("{:?}: {}", issue.issue_type, issue.description));
                }
                crate::analysis::quality::IssueSeverity::Medium => {
                    warnings.push(format!("{:?}: {}", issue.issue_type, issue.description));
                }
                crate::analysis::quality::IssueSeverity::Low => {
                    info.push(format!("{:?}: {}", issue.issue_type, issue.description));
                }
            }
        }

        // Add perceptual warnings
        if result.perceptual.loudness_lufs > -14.0 {
            warnings.push("Audio may be too loud for streaming platforms".to_string());
        }
        if result.perceptual.dynamic_range < 6.0 {
            warnings.push("Low dynamic range detected".to_string());
        }

        // Add info from analysis
        if let Some(tempo) = result.temporal.tempo {
            info.push(format!("Tempo: {:.1} BPM", tempo));
        }
        info.push(format!(
            "Key: {} (confidence: {:.0}%)",
            result.musical.key.key,
            result.musical.key.confidence * 100.0
        ));
        info.push(format!(
            "Content type: {:?} ({:.0}% confidence)",
            result.classification.primary_type,
            result.classification.confidence * 100.0
        ));

        // Add suggestions from quality assessment
        for rec in &result.quality.recommendations {
            suggestions.push(rec.clone());
        }

        HierarchicalInsights {
            critical,
            warnings,
            info,
            suggestions,
        }
    }

    pub fn with_cache(cache: Cache) -> Self {
        Self {
            engine: AnalysisEngine::new().with_cache(cache),
            active_jobs: Arc::new(DashMap::new()),
        }
    }

    async fn analyze_audio_impl(
        &self,
        params: AnalyzeAudioParams,
    ) -> Result<serde_json::Value, McpError> {
        let job_id = Uuid::new_v4().to_string();

        // Start job
        self.active_jobs.insert(
            job_id.clone(),
            JobStatus {
                id: job_id.clone(),
                status: "processing".to_string(),
                progress: 0.0,
                message: Some("Loading audio file".to_string()),
                completed_modules: Vec::new(),
                pending_modules: Vec::new(),
                partial_results: None,
            },
        );

        // Load audio file
        let audio = AudioFile::load(&params.file_path)
            .map_err(|e| McpError::internal_error(format!("Failed to load audio: {}", e), None))?;

        // Determine which modules to run based on profile
        let modules_to_run = match params.analysis_profile.as_str() {
            "quick" => vec!["summary", "temporal"],
            "standard" => vec![
                "summary",
                "spectral",
                "temporal",
                "perceptual",
                "classification",
            ],
            "detailed" => vec![
                "summary",
                "spectral",
                "temporal",
                "perceptual",
                "classification",
                "musical",
                "quality",
                "segments",
                "fingerprint",
            ],
            "fingerprint" => vec!["summary", "fingerprint", "spectral"],
            "mastering" => vec!["summary", "perceptual", "quality", "spectral"],
            _ => vec![
                "summary",
                "spectral",
                "temporal",
                "perceptual",
                "classification",
            ], // default to standard
        };

        // Update job status with modules
        if let Some(mut status) = self.active_jobs.get_mut(&job_id) {
            status.progress = 0.2;
            status.message = Some("Starting analysis".to_string());
            status.pending_modules = modules_to_run.iter().map(|s| s.to_string()).collect();
            status.completed_modules = Vec::new();
        }

        // Use AnalysisEngine for comprehensive analysis
        let analysis_result = self
            .engine
            .analyze(&audio)
            .await
            .map_err(|e| McpError::internal_error(format!("Analysis failed: {}", e), None))?;

        // Update progress
        if let Some(mut status) = self.active_jobs.get_mut(&job_id) {
            status.progress = 1.0;
            status.status = "complete".to_string();
            status.message = Some("Analysis complete".to_string());
            status.completed_modules = modules_to_run.iter().map(|s| s.to_string()).collect();
            status.pending_modules.clear();
        }

        // Apply filtering based on parameters
        let mut filtered_result = analysis_result;

        // Filter visuals unless explicitly requested
        if !params.include_visuals {
            filtered_result.visuals.waveform = None;
            filtered_result.visuals.spectrogram = None;
            filtered_result.visuals.mel_spectrogram = None;
            filtered_result.visuals.power_curve = None;
        }

        // Filter spectral data unless requested
        if !params.include_spectral {
            filtered_result.spectral.spectral_centroid.clear();
            filtered_result.spectral.spectral_rolloff.clear();
            filtered_result.spectral.spectral_flux.clear();
            filtered_result.spectral.mfcc.clear();
            filtered_result.spectral.dominant_frequencies.clear();
        }

        // Filter temporal data unless requested
        if !params.include_temporal {
            filtered_result.temporal.beats.clear();
            filtered_result.temporal.onsets.clear();
        }

        // Parse cursor for pagination
        let offset = if let Some(cursor) = &params.cursor {
            cursor.parse::<usize>().unwrap_or(0)
        } else {
            0
        };

        // Apply pagination with max_data_points
        let mut next_cursor = None;
        if let Some(max_points) = params.max_data_points {
            if params.include_spectral {
                // Paginate spectral data
                let total_len = filtered_result.spectral.spectral_centroid.len();
                if offset < total_len {
                    let end = (offset + max_points).min(total_len);
                    filtered_result.spectral.spectral_centroid =
                        filtered_result.spectral.spectral_centroid[offset..end].to_vec();

                    if end < total_len {
                        next_cursor = Some(end.to_string());
                    }
                }

                // Apply same pagination to other spectral arrays
                if offset < filtered_result.spectral.spectral_flux.len() {
                    let end =
                        (offset + max_points).min(filtered_result.spectral.spectral_flux.len());
                    filtered_result.spectral.spectral_flux =
                        filtered_result.spectral.spectral_flux[offset..end].to_vec();
                }

                if offset < filtered_result.spectral.spectral_rolloff.len() {
                    let end =
                        (offset + max_points).min(filtered_result.spectral.spectral_rolloff.len());
                    filtered_result.spectral.spectral_rolloff =
                        filtered_result.spectral.spectral_rolloff[offset..end].to_vec();
                }

                // MFCC is special - limit number of frames
                if offset < filtered_result.spectral.mfcc.len() {
                    let end = (offset + max_points).min(filtered_result.spectral.mfcc.len());
                    filtered_result.spectral.mfcc =
                        filtered_result.spectral.mfcc[offset..end].to_vec();
                }
            }

            if params.include_temporal {
                // Paginate temporal data
                if offset < filtered_result.temporal.beats.len() {
                    let end = (offset + max_points).min(filtered_result.temporal.beats.len());
                    filtered_result.temporal.beats =
                        filtered_result.temporal.beats[offset..end].to_vec();

                    if end < filtered_result.temporal.beats.len() && next_cursor.is_none() {
                        next_cursor = Some(end.to_string());
                    }
                }

                if offset < filtered_result.temporal.onsets.len() {
                    let end = (offset + max_points).min(filtered_result.temporal.onsets.len());
                    filtered_result.temporal.onsets =
                        filtered_result.temporal.onsets[offset..end].to_vec();

                    if end < filtered_result.temporal.onsets.len() && next_cursor.is_none() {
                        next_cursor = Some(end.to_string());
                    }
                }
            }
        }

        // Format response based on return_format
        let response_data = match params.return_format.as_str() {
            "summary" => {
                let enhanced_summary = self.create_enhanced_summary(&filtered_result, &audio);
                let hierarchical_insights = self.create_hierarchical_insights(&filtered_result);

                let mut response = json!({
                    "job_id": job_id,
                    "status": "success",
                    "profile_used": params.analysis_profile,
                    "summary": enhanced_summary,
                    "insights": hierarchical_insights,
                });
                if let Some(cursor) = next_cursor {
                    response["next_cursor"] = json!(cursor);
                }
                response
            }
            "visual_only" => json!({
                "job_id": job_id,
                "status": "success",
                "visuals": filtered_result.get_visuals(),
            }),
            "full" => {
                let mut response = json!({
                    "job_id": job_id,
                    "status": "success",
                    "data": {
                        "summary": filtered_result.summary,
                        "spectral": if params.include_spectral {
                            Some(filtered_result.spectral)
                        } else {
                            None
                        },
                        "temporal": if params.include_temporal {
                            Some(filtered_result.temporal)
                        } else {
                            None
                        },
                        "visuals": if params.include_visuals {
                            Some(filtered_result.visuals)
                        } else {
                            None
                        },
                        "insights": filtered_result.insights,
                        "recommendations": filtered_result.recommendations,
                    }
                });
                if let Some(cursor) = next_cursor {
                    response["next_cursor"] = json!(cursor);
                    response["has_more"] = json!(true);
                } else {
                    response["has_more"] = json!(false);
                }
                response
            }
            _ => json!({
                "job_id": job_id,
                "status": "success",
                "summary": filtered_result.get_summary(),
                "insights": filtered_result.insights.iter().take(5).cloned().collect::<Vec<_>>(),
            }),
        };

        Ok(response_data)
    }

    async fn compare_audio_impl(
        &self,
        params: CompareAudioParams,
    ) -> Result<serde_json::Value, McpError> {
        // Load both audio files
        let audio_a = AudioFile::load(&params.file_a)
            .map_err(|e| McpError::internal_error(format!("Failed to load file A: {}", e), None))?;

        let audio_b = AudioFile::load(&params.file_b)
            .map_err(|e| McpError::internal_error(format!("Failed to load file B: {}", e), None))?;

        // Use AnalysisEngine for comparison and analysis
        let comparison_result = self.engine.compare(&audio_a, &audio_b).await;

        // Also run full analysis for more detailed comparison
        let analysis_a = self
            .engine
            .analyze(&audio_a)
            .await
            .map_err(|e| McpError::internal_error(format!("Analysis A failed: {}", e), None))?;
        let analysis_b = self
            .engine
            .analyze(&audio_b)
            .await
            .map_err(|e| McpError::internal_error(format!("Analysis B failed: {}", e), None))?;

        // Create enhanced comparison
        let fingerprint_similarity = comparison_result
            .comparison
            .fingerprint_similarity
            .unwrap_or(0.0);
        let match_type = comparison_result
            .comparison
            .fingerprint_match_type
            .clone()
            .unwrap_or_else(|| "Unknown".to_string());

        // Calculate additional similarity metrics
        let spectral_similarity = {
            let cent_a = analysis_a.spectral.spectral_centroid.iter().sum::<f32>()
                / analysis_a.spectral.spectral_centroid.len().max(1) as f32;
            let cent_b = analysis_b.spectral.spectral_centroid.iter().sum::<f32>()
                / analysis_b.spectral.spectral_centroid.len().max(1) as f32;
            1.0 - ((cent_a - cent_b).abs() / cent_a.max(cent_b).max(1.0)).min(1.0)
        };

        let perceptual_similarity = {
            let loudness_diff =
                (analysis_a.perceptual.loudness_lufs - analysis_b.perceptual.loudness_lufs).abs();
            let dynamic_diff =
                (analysis_a.perceptual.dynamic_range - analysis_b.perceptual.dynamic_range).abs();
            1.0 - ((loudness_diff / 40.0) + (dynamic_diff / 30.0)) / 2.0
        }
        .clamp(0.0, 1.0);

        let structural_similarity = if !analysis_a.segments.segments.is_empty()
            && !analysis_b.segments.segments.is_empty()
        {
            analysis_a
                .segments
                .segments
                .len()
                .min(analysis_b.segments.segments.len()) as f32
                / analysis_a
                    .segments
                    .segments
                    .len()
                    .max(analysis_b.segments.segments.len()) as f32
        } else {
            0.5
        };

        let overall_similarity = (fingerprint_similarity * 0.4
            + spectral_similarity * 0.2
            + perceptual_similarity * 0.2
            + structural_similarity * 0.2)
            .min(1.0);

        // Determine use cases based on similarity
        let use_cases = if fingerprint_similarity > 0.95 {
            vec!["Same recording", "Identical audio"]
        } else if fingerprint_similarity > 0.85 {
            vec!["Same recording, different encoding", "Minor edits"]
        } else if fingerprint_similarity > 0.7 {
            vec!["Same song, different master", "Remix or edit"]
        } else if fingerprint_similarity > 0.5 {
            vec!["Cover version", "Similar arrangement"]
        } else {
            vec!["Different recordings", "Unrelated audio"]
        };

        let enhanced_comparison = EnhancedComparison {
            similarity: SimilarityMetrics {
                overall: overall_similarity,
                fingerprint: fingerprint_similarity,
                spectral: spectral_similarity,
                perceptual: perceptual_similarity,
                structural: structural_similarity,
            },
            differences: DifferenceMetrics {
                loudness_delta: analysis_a.perceptual.loudness_lufs
                    - analysis_b.perceptual.loudness_lufs,
                tempo_delta: comparison_result.comparison.tempo_difference,
                quality_delta: analysis_a.quality.overall_score - analysis_b.quality.overall_score,
                duration_delta: comparison_result.comparison.duration_difference,
                key_change: if analysis_a.musical.key.key != analysis_b.musical.key.key {
                    Some(format!(
                        "{} → {}",
                        analysis_a.musical.key.key, analysis_b.musical.key.key
                    ))
                } else {
                    None
                },
            },
            match_classification: match_type,
            use_cases: use_cases.iter().map(|s| s.to_string()).collect(),
        };

        Ok(json!({
            "comparison": enhanced_comparison,
            "file_a": comparison_result.file_a,
            "file_b": comparison_result.file_b,
            "raw_comparison": comparison_result.comparison,
        }))
    }

    async fn get_job_status_impl(
        &self,
        params: serde_json::Value,
    ) -> Result<serde_json::Value, McpError> {
        let job_id = params
            .get("job_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| McpError::invalid_params("job_id is required", None))?;

        if let Some(status) = self.active_jobs.get(job_id) {
            Ok(json!({
                "status": status.clone()
            }))
        } else {
            Err(McpError::invalid_params("Job not found", None))
        }
    }

    pub async fn start(self) -> crate::utils::error::Result<()> {
        let service = self.serve(stdio()).await.map_err(|e| {
            crate::utils::error::FerrousError::Mcp(format!("Failed to start MCP server: {}", e))
        })?;

        service.waiting().await.map_err(|e| {
            crate::utils::error::FerrousError::Mcp(format!("MCP server error: {}", e))
        })?;

        Ok(())
    }
}

impl Default for FerrousWavesMcp {
    fn default() -> Self {
        Self::new()
    }
}

impl ServerHandler for FerrousWavesMcp {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            protocol_version: ProtocolVersion::V_2024_11_05,
            capabilities: ServerCapabilities::builder()
                .enable_tools()
                .build(),
            server_info: Implementation {
                name: "ferrous-waves".to_string(),
                title: None,
                version: env!("CARGO_PKG_VERSION").to_string(),
                website_url: None,
                icons: None,
            },
            instructions: Some(
                "Ferrous Waves audio analysis server. Tools: analyze_audio, compare_audio, get_job_status"
                    .to_string(),
            ),
        }
    }

    async fn initialize(
        &self,
        _request: InitializeRequestParam,
        _context: RequestContext<RoleServer>,
    ) -> std::result::Result<InitializeResult, McpError> {
        tracing::info!("Ferrous Waves MCP server initialized");
        Ok(self.get_info())
    }

    async fn list_tools(
        &self,
        _params: Option<PaginatedRequestParam>,
        _context: RequestContext<RoleServer>,
    ) -> std::result::Result<ListToolsResult, McpError> {
        let analyze_audio_schema = json!({
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the audio file to analyze"
                },
                "analysis_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional: Specific analysis types to perform"
                },
                "output_dir": {
                    "type": "string",
                    "description": "Optional: Output directory for results"
                },
                "return_format": {
                    "type": "string",
                    "enum": ["full", "summary", "visual_only"],
                    "description": "Return format (default: summary for MCP compatibility)",
                    "default": "summary"
                },
                "analysis_profile": {
                    "type": "string",
                    "enum": ["quick", "standard", "detailed", "fingerprint", "mastering"],
                    "description": "Analysis profile: quick (fast), standard (balanced), detailed (all features), fingerprint (similarity focus), mastering (audio quality focus)",
                    "default": "standard"
                },
                "max_data_points": {
                    "type": "integer",
                    "description": "Maximum number of data points in arrays (for pagination)",
                    "default": 1000
                },
                "cursor": {
                    "type": "string",
                    "description": "Pagination cursor from previous response's next_cursor field"
                },
                "include_visuals": {
                    "type": "boolean",
                    "description": "Include visual data (base64 images) - WARNING: very large",
                    "default": false
                },
                "include_spectral": {
                    "type": "boolean",
                    "description": "Include raw spectral data arrays",
                    "default": false
                },
                "include_temporal": {
                    "type": "boolean",
                    "description": "Include temporal data arrays (beats, onsets)",
                    "default": false
                }
            },
            "required": ["file_path"]
        });

        let compare_audio_schema = json!({
            "type": "object",
            "properties": {
                "file_a": {
                    "type": "string",
                    "description": "First audio file path"
                },
                "file_b": {
                    "type": "string",
                    "description": "Second audio file path"
                },
                "metrics": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Comparison metrics to calculate"
                }
            },
            "required": ["file_a", "file_b"]
        });

        let get_job_status_schema = json!({
            "type": "object",
            "properties": {
                "job_id": {
                    "type": "string",
                    "description": "Job ID to check status for"
                }
            },
            "required": ["job_id"]
        });

        Ok(ListToolsResult {
            tools: vec![
                Tool {
                    name: Cow::Borrowed("analyze_audio"),
                    title: None,
                    description: Some(Cow::Borrowed(
                        "Analyze an audio file and return comprehensive metrics",
                    )),
                    input_schema: Arc::new(analyze_audio_schema.as_object().unwrap().clone()),
                    output_schema: None,
                    icons: None,
                    annotations: None,
                },
                Tool {
                    name: Cow::Borrowed("compare_audio"),
                    title: None,
                    description: Some(Cow::Borrowed("Compare two audio files")),
                    input_schema: Arc::new(compare_audio_schema.as_object().unwrap().clone()),
                    output_schema: None,
                    icons: None,
                    annotations: None,
                },
                Tool {
                    name: Cow::Borrowed("get_job_status"),
                    title: None,
                    description: Some(Cow::Borrowed("Get the status of an analysis job")),
                    input_schema: Arc::new(get_job_status_schema.as_object().unwrap().clone()),
                    output_schema: None,
                    icons: None,
                    annotations: None,
                },
            ],
            next_cursor: None,
        })
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParam,
        _context: RequestContext<RoleServer>,
    ) -> std::result::Result<CallToolResult, McpError> {
        let tool_name = request.name.as_ref();
        let arguments = request.arguments.unwrap_or_default();

        match tool_name {
            "analyze_audio" => {
                let params: AnalyzeAudioParams =
                    serde_json::from_value(serde_json::Value::Object(arguments)).map_err(|e| {
                        McpError::invalid_params(format!("Invalid parameters: {}", e), None)
                    })?;

                match self.analyze_audio_impl(params).await {
                    Ok(result) => Ok(CallToolResult::success(vec![Content::text(
                        result.to_string(),
                    )])),
                    Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                        "Analysis failed: {}",
                        e
                    ))])),
                }
            }
            "compare_audio" => {
                let params: CompareAudioParams =
                    serde_json::from_value(serde_json::Value::Object(arguments)).map_err(|e| {
                        McpError::invalid_params(format!("Invalid parameters: {}", e), None)
                    })?;

                match self.compare_audio_impl(params).await {
                    Ok(result) => Ok(CallToolResult::success(vec![Content::text(
                        result.to_string(),
                    )])),
                    Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                        "Comparison failed: {}",
                        e
                    ))])),
                }
            }
            "get_job_status" => {
                match self
                    .get_job_status_impl(serde_json::Value::Object(arguments))
                    .await
                {
                    Ok(result) => Ok(CallToolResult::success(vec![Content::text(
                        result.to_string(),
                    )])),
                    Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                        "Status check failed: {}",
                        e
                    ))])),
                }
            }
            _ => Err(McpError::invalid_request("Unknown tool", None)),
        }
    }
}
