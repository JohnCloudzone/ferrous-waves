use rmcp::{
    model::{ErrorData as McpError, *},
    RoleServer, ServerHandler,
    service::RequestContext,
    ServiceExt, transport::stdio,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::sync::Arc;
use std::borrow::Cow;
use uuid::Uuid;
use dashmap::DashMap;

use crate::audio::AudioFile;
use crate::AnalysisEngine;
use crate::cache::Cache;

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
}

fn default_return_format() -> String {
    "full".to_string()
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

    pub fn with_cache(cache: Cache) -> Self {
        Self {
            engine: AnalysisEngine::new().with_cache(cache),
            active_jobs: Arc::new(DashMap::new()),
        }
    }

    async fn analyze_audio_impl(&self, params: AnalyzeAudioParams) -> Result<serde_json::Value, McpError> {
        let job_id = Uuid::new_v4().to_string();

        // Start job
        self.active_jobs.insert(
            job_id.clone(),
            JobStatus {
                id: job_id.clone(),
                status: "processing".to_string(),
                progress: 0.0,
                message: Some("Loading audio file".to_string()),
            },
        );

        // Load audio file
        let audio = AudioFile::load(&params.file_path)
            .map_err(|e| McpError::internal_error(format!("Failed to load audio: {}", e), None))?;

        // Update progress
        if let Some(mut status) = self.active_jobs.get_mut(&job_id) {
            status.progress = 0.2;
            status.message = Some("Performing comprehensive analysis".to_string());
        }

        // Use AnalysisEngine for comprehensive analysis
        let analysis_result = self.engine.analyze(&audio).await
            .map_err(|e| McpError::internal_error(format!("Analysis failed: {}", e), None))?;

        // Update progress
        if let Some(mut status) = self.active_jobs.get_mut(&job_id) {
            status.progress = 1.0;
            status.status = "complete".to_string();
            status.message = Some("Analysis complete".to_string());
        }

        // Format response based on return_format
        let response_data = match params.return_format.as_str() {
            "summary" => json!({
                "job_id": job_id,
                "status": "success",
                "summary": analysis_result.get_summary(),
            }),
            "visual_only" => json!({
                "job_id": job_id,
                "status": "success",
                "visuals": analysis_result.get_visuals(),
            }),
            _ => json!({
                "job_id": job_id,
                "status": "success",
                "data": analysis_result,
            }),
        };

        Ok(response_data)
    }

    async fn compare_audio_impl(&self, params: CompareAudioParams) -> Result<serde_json::Value, McpError> {
        // Load both audio files
        let audio_a = AudioFile::load(&params.file_a)
            .map_err(|e| McpError::internal_error(format!("Failed to load file A: {}", e), None))?;

        let audio_b = AudioFile::load(&params.file_b)
            .map_err(|e| McpError::internal_error(format!("Failed to load file B: {}", e), None))?;

        // Use AnalysisEngine for comparison
        let comparison_result = self.engine.compare(&audio_a, &audio_b).await;

        Ok(serde_json::to_value(comparison_result)
            .unwrap_or_else(|_| json!({"error": "Failed to serialize comparison"})))
    }

    async fn get_job_status_impl(&self, params: serde_json::Value) -> Result<serde_json::Value, McpError> {
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
        let service = self.serve(stdio()).await
            .map_err(|e| crate::utils::error::FerrousError::Mcp(format!("Failed to start MCP server: {}", e)))?;

        service.waiting().await
            .map_err(|e| crate::utils::error::FerrousError::Mcp(format!("MCP server error: {}", e)))?;

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
                version: env!("CARGO_PKG_VERSION").to_string(),
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
                    "description": "Return format",
                    "default": "full"
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
                    description: Some(Cow::Borrowed("Analyze an audio file and return comprehensive metrics")),
                    input_schema: Arc::new(
                        analyze_audio_schema.as_object().unwrap().clone()
                    ),
                    annotations: None,
                },
                Tool {
                    name: Cow::Borrowed("compare_audio"),
                    description: Some(Cow::Borrowed("Compare two audio files")),
                    input_schema: Arc::new(
                        compare_audio_schema.as_object().unwrap().clone()
                    ),
                    annotations: None,
                },
                Tool {
                    name: Cow::Borrowed("get_job_status"),
                    description: Some(Cow::Borrowed("Get the status of an analysis job")),
                    input_schema: Arc::new(
                        get_job_status_schema.as_object().unwrap().clone()
                    ),
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
        let arguments = request.arguments.unwrap_or_else(|| serde_json::Map::new());

        match tool_name {
            "analyze_audio" => {
                let params: AnalyzeAudioParams = serde_json::from_value(serde_json::Value::Object(arguments))
                    .map_err(|e| McpError::invalid_params(format!("Invalid parameters: {}", e), None))?;

                match self.analyze_audio_impl(params).await {
                    Ok(result) => Ok(CallToolResult {
                        is_error: Some(false),
                        content: vec![Content::text(result.to_string())],
                    }),
                    Err(e) => Ok(CallToolResult {
                        is_error: Some(true),
                        content: vec![Content::text(format!("Analysis failed: {}", e))],
                    }),
                }
            }
            "compare_audio" => {
                let params: CompareAudioParams = serde_json::from_value(serde_json::Value::Object(arguments))
                    .map_err(|e| McpError::invalid_params(format!("Invalid parameters: {}", e), None))?;

                match self.compare_audio_impl(params).await {
                    Ok(result) => Ok(CallToolResult {
                        is_error: Some(false),
                        content: vec![Content::text(result.to_string())],
                    }),
                    Err(e) => Ok(CallToolResult {
                        is_error: Some(true),
                        content: vec![Content::text(format!("Comparison failed: {}", e))],
                    }),
                }
            }
            "get_job_status" => {
                match self.get_job_status_impl(serde_json::Value::Object(arguments)).await {
                    Ok(result) => Ok(CallToolResult {
                        is_error: Some(false),
                        content: vec![Content::text(result.to_string())],
                    }),
                    Err(e) => Ok(CallToolResult {
                        is_error: Some(true),
                        content: vec![Content::text(format!("Status check failed: {}", e))],
                    }),
                }
            }
            _ => Err(McpError::invalid_request("Unknown tool", None)),
        }
    }
}