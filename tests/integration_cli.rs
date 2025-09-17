use std::process::Command;

#[test]
fn test_cli_help() {
    let output = Command::new("cargo")
        .args(["run", "--bin", "ferrous-waves", "--", "--help"])
        .output()
        .expect("Failed to execute command");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("High-fidelity audio analysis bridge"));
    assert!(stdout.contains("serve"));
    assert!(stdout.contains("analyze"));
    assert!(stdout.contains("compare"));
}

#[test]
fn test_cli_version() {
    let output = Command::new("cargo")
        .args(["run", "--bin", "ferrous-waves", "--", "--version"])
        .output()
        .expect("Failed to execute command");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("ferrous-waves"));
}

#[test]
fn test_cache_stats_command() {
    let output = Command::new("cargo")
        .args(["run", "--bin", "ferrous-waves", "--", "cache-stats"])
        .output()
        .expect("Failed to execute command");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Cache Statistics"));
    assert!(stdout.contains("Total Entries"));
    assert!(stdout.contains("Total Size"));
}

#[test]
fn test_clear_cache_without_confirm() {
    let output = Command::new("cargo")
        .args(["run", "--bin", "ferrous-waves", "--", "clear-cache"])
        .output()
        .expect("Failed to execute command");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Please confirm cache clearing with --confirm flag"));
}

#[test]
fn test_analyze_help() {
    let output = Command::new("cargo")
        .args(["run", "--bin", "ferrous-waves", "--", "analyze", "--help"])
        .output()
        .expect("Failed to execute command");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Analyze a single audio file"));
    assert!(stdout.contains("--format"));
    assert!(stdout.contains("--output"));
}

#[test]
fn test_tempo_help() {
    let output = Command::new("cargo")
        .args(["run", "--bin", "ferrous-waves", "--", "tempo", "--help"])
        .output()
        .expect("Failed to execute command");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Extract tempo from audio file"));
    assert!(stdout.contains("--show-beats"));
}
