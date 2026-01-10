use anyhow::Result;
use clap::Parser;

use trade_signal::{
    data::{CsvType, get_samples},
    indicators::sma::SmaConfig,
    signal::{BreakoutConfig, FilterConfig, PullbackConfig, StrategyConfig},
};

use std::path::PathBuf;

const BREAKDOWN_LOOKBACK: usize = 5;
const PULLBACK_TOLERANCE_PCT: f64 = 0.003;

#[derive(Debug, Parser)]
struct Args {
    /// Path to the CSV file
    #[arg(long)]
    input: PathBuf,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Load raw samples from CSV
    let samples = get_samples(&args.input, CsvType::CryptoLogger)?;
    if samples.is_empty() {
        println!("No data found in CSV.");
        return Ok(());
    }

    // Resample to hourly closes
    let hourly = trade_signal::data::resample_to_hourly(&samples);
    println!(
        "Loaded {} raw points, {} hourly candles after resampling.",
        samples.len(),
        hourly.len()
    );
    if hourly.is_empty() {
        println!("No hourly data after resampling.");
        return Ok(());
    }

    // Extract prices and compute SMAs
    let prices: Vec<f64> = hourly.iter().map(|s| s.price).collect();
    let sma_config = SmaConfig::sma_20_50();
    let Some(smas) = trade_signal::indicators::compute_smas(&prices, sma_config) else {
        println!(
            "Not enough data: need at least 51 hourly candles for SMA20/50 logic, got {}.",
            prices.len()
        );
        return Ok(());
    };

    let strategy = StrategyConfig {
        breakouts: Some(BreakoutConfig {
            breakout_lookback: BREAKDOWN_LOOKBACK,
        }),
        enable_bias_only: true,
        enable_crossovers: true,
        pullbacks: Some(PullbackConfig {
            bounce_tolerance_pct: PULLBACK_TOLERANCE_PCT,
            reject_tolerance_pct: PULLBACK_TOLERANCE_PCT,
        }),
        sma_config,
        filters: FilterConfig {
            atr: None,
            regime: None,
            require_price_confirmation: true,
            require_trend_filter: true,
        },
    };

    // Perform final analysis
    let result = trade_signal::signal::analyze(&hourly, &prices, smas, strategy);

    // Print result.clone()
    trade_signal::output::print_analysis(&result, sma_config);

    Ok(())
}
