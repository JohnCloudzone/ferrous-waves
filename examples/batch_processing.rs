use ferrous_waves::{AnalysisEngine, AudioFile, Result};

#[tokio::main]
async fn main() -> Result<()> {
    // List of files to process
    let files = vec![
        "samples/track1.wav",
        "samples/track2.wav",
        "samples/track3.wav",
    ];

    // Create shared analysis engine
    let engine = AnalysisEngine::new();

    // Process files in parallel
    let tasks: Vec<_> = files
        .into_iter()
        .map(|path| {
            let engine = engine.clone();
            let path = path.to_string();
            tokio::spawn(async move {
                println!("Processing: {}", path);

                match AudioFile::load(&path) {
                    Ok(audio) => {
                        match engine.analyze(&audio).await {
                            Ok(result) => {
                                println!("  {} - Duration: {:.2}s, Tempo: {:?}",
                                    path,
                                    result.summary.duration,
                                    result.temporal.tempo
                                );
                                Ok(())
                            }
                            Err(e) => {
                                eprintln!("  {} - Analysis error: {}", path, e);
                                Err(e)
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("  {} - Load error: {}", path, e);
                        Err(e)
                    }
                }
            })
        })
        .collect();

    // Wait for all tasks to complete
    let results = futures::future::join_all(tasks).await;

    // Print summary
    let successful = results.iter().filter(|r| r.is_ok()).count();
    println!("\nProcessed {} files successfully out of {}", successful, results.len());

    Ok(())
}