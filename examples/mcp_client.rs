// Example MCP client that would connect to the ferrous-waves MCP server
// This demonstrates how an AI assistant would use the MCP tools

use serde_json::json;

fn main() {
    println!("Example MCP Tool Calls:");
    println!();

    // Example 1: Analyze with different profiles
    let analyze_quick = json!({
        "tool": "analyze_audio",
        "arguments": {
            "file_path": "/path/to/audio.wav",
            "analysis_profile": "quick"  // Fast analysis
        }
    });
    println!("1. Analyze Audio (Quick Profile):");
    println!("{}", serde_json::to_string_pretty(&analyze_quick).unwrap());
    println!();

    // Example 1b: Mastering profile for quality focus
    let analyze_mastering = json!({
        "tool": "analyze_audio",
        "arguments": {
            "file_path": "/path/to/audio.wav",
            "analysis_profile": "mastering"  // Focus on loudness and quality
        }
    });
    println!("1b. Analyze Audio (Mastering Profile):");
    println!(
        "{}",
        serde_json::to_string_pretty(&analyze_mastering).unwrap()
    );
    println!();

    // Example 2: Analyze with detailed profile for complete analysis
    let analyze_detailed = json!({
        "tool": "analyze_audio",
        "arguments": {
            "file_path": "/path/to/audio.wav",
            "analysis_profile": "detailed",  // All modules
            "return_format": "full",
            "include_spectral": true,
            "include_temporal": true,
            "max_data_points": 100,
            "cursor": null
        }
    });
    println!("2. Analyze Audio (Detailed Profile with Full Data):");
    println!(
        "{}",
        serde_json::to_string_pretty(&analyze_detailed).unwrap()
    );
    println!();

    // Example 3: Continue pagination with cursor
    let analyze_next_page = json!({
        "tool": "analyze_audio",
        "arguments": {
            "file_path": "/path/to/audio.wav",
            "return_format": "full",
            "include_spectral": true,
            "max_data_points": 100,
            "cursor": "100"  // From previous response's next_cursor
        }
    });
    println!("3. Continue Pagination:");
    println!(
        "{}",
        serde_json::to_string_pretty(&analyze_next_page).unwrap()
    );
    println!();

    // Example 4: Compare two audio files
    let compare_request = json!({
        "tool": "compare_audio",
        "arguments": {
            "file_a": "/path/to/original.wav",
            "file_b": "/path/to/processed.wav"
        }
    });
    println!("4. Compare Audio:");
    println!(
        "{}",
        serde_json::to_string_pretty(&compare_request).unwrap()
    );
    println!();

    // Example 5: Check job status
    let status_request = json!({
        "tool": "get_job_status",
        "arguments": {
            "job_id": "550e8400-e29b-41d4-a716-446655440000"
        }
    });
    println!("5. Get Job Status:");
    println!("{}", serde_json::to_string_pretty(&status_request).unwrap());
    println!();

    println!("To start the MCP server, run:");
    println!("  cargo run --bin mcp_server");
    println!();
    println!("The server communicates over stdio and can be integrated");
    println!("with AI assistants that support the Model Context Protocol.");
}
