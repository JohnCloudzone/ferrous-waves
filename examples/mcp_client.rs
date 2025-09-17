// Example MCP client that would connect to the ferrous-waves MCP server
// This demonstrates how an AI assistant would use the MCP tools

use serde_json::json;

fn main() {
    println!("Example MCP Tool Calls:");
    println!();

    // Example 1: Analyze a single audio file
    let analyze_request = json!({
        "tool": "analyze_audio",
        "arguments": {
            "file_path": "/path/to/audio.wav",
            "return_format": "summary"
        }
    });
    println!("1. Analyze Audio:");
    println!(
        "{}",
        serde_json::to_string_pretty(&analyze_request).unwrap()
    );
    println!();

    // Example 2: Compare two audio files
    let compare_request = json!({
        "tool": "compare_audio",
        "arguments": {
            "file_a": "/path/to/original.wav",
            "file_b": "/path/to/processed.wav"
        }
    });
    println!("2. Compare Audio:");
    println!(
        "{}",
        serde_json::to_string_pretty(&compare_request).unwrap()
    );
    println!();

    // Example 3: Check job status
    let status_request = json!({
        "tool": "get_job_status",
        "arguments": {
            "job_id": "550e8400-e29b-41d4-a716-446655440000"
        }
    });
    println!("3. Get Job Status:");
    println!("{}", serde_json::to_string_pretty(&status_request).unwrap());
    println!();

    println!("To start the MCP server, run:");
    println!("  cargo run --bin mcp_server");
    println!();
    println!("The server communicates over stdio and can be integrated");
    println!("with AI assistants that support the Model Context Protocol.");
}
