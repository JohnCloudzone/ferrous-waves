use ferrous_waves::visualization::waveform::render_waveform_with_envelope;
use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Make sure samples directory exists
    fs::create_dir_all("samples")?;

    println!("Generating envelope visualization example...");

    // Create an interesting audio signal with varying dynamics
    let sample_rate = 44100;
    let duration = 3.0;
    let num_samples = (sample_rate as f32 * duration) as usize;
    let mut samples = Vec::with_capacity(num_samples);

    for i in 0..num_samples {
        let t = i as f32 / sample_rate as f32;

        // Create an ADSR-like envelope
        let envelope = if t < 0.1 {
            // Attack: quick rise
            t * 10.0
        } else if t < 0.3 {
            // Decay: fall to sustain level
            1.0 - (t - 0.1) * 2.0
        } else if t < 2.0 {
            // Sustain: hold at 60% with some variation
            0.6 + (t * 5.0).sin() * 0.1
        } else {
            // Release: fade out
            (3.0 - t).max(0.0) * 0.6
        };

        // Mix multiple frequencies for richer sound
        let carrier = (2.0 * std::f32::consts::PI * 440.0 * t).sin();
        let modulator = (2.0 * std::f32::consts::PI * 110.0 * t).sin() * 0.3;
        let high = (2.0 * std::f32::consts::PI * 880.0 * t).sin() * 0.2;

        let sample = (carrier + modulator + high) * envelope * 0.5;
        samples.push(sample);
    }

    // Generate the envelope visualization
    let output_path = "samples/envelope_example.png";
    render_waveform_with_envelope(&samples, output_path.as_ref(), 1200, 400)?;

    println!("✓ Created envelope visualization at: {}", output_path);
    println!();
    println!("The visualization shows:");
    println!("  - Blue filled area: Peak envelope (min/max values)");
    println!("  - Blue lines: Peak envelope outline");
    println!("  - Red lines: RMS envelope (average power)");
    println!("  - Black line: Zero crossing");
    println!();
    println!("Open {} to view the result!", output_path);

    Ok(())
}
