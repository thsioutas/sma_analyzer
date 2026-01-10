use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;

use trade_signal::{
    backtest::{
        ConfigSweep, find_best_strategy, generate_backtest_sweep_jobs,
        generate_pullback_pairs, generate_strategies,
        position::{PositionBacktester, buy_and_hold_equity, print_summary},
    },
    data::{get_samples, resample_to_n_hours},
};

#[derive(Debug, Parser)]
struct Args {
    /// config-file path
    #[arg(long)]
    config: PathBuf,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let config_path = args
        .config
        .into_os_string()
        .into_string()
        .expect("Failed to translate config file path into string");
    let config: ConfigSweep = config::Config::builder()
        .add_source(config::File::with_name(&config_path))
        .build()?
        .try_deserialize()?;

    let samples = get_samples(&config.setup.input, config.setup.csv_type)
        .with_context(|| format!("failed to load samples from {:?}", config.setup.input))?;

    if samples.is_empty() {
        println!("No data found in CSV.");
        return Ok(());
    }

    let resampled = resample_to_n_hours(&samples, config.setup.sample_hours);

    println!(
        "Loaded {} raw points, {} {}h-candles after resampling.",
        samples.len(),
        resampled.len(),
        config.setup.sample_hours,
    );

    let pullback_pairs = generate_pullback_pairs(
        config.min_pullback_pct,
        config.max_pullback_pct,
        0.001,
    );

    let strategies = generate_strategies(
        config.min_lookback,
        config.max_lookback,
        pullback_pairs,
    );

    let buy_sell_frac_steps = config.buy_sell_frac_steps;

    let jobs = generate_backtest_sweep_jobs(strategies, buy_sell_frac_steps);

    let best = find_best_strategy(
        jobs,
        config.max_buy_sell_fraction,
        buy_sell_frac_steps,
        &samples,
        || PositionBacktester::new(config.setup.initial_cash),
    );

    println!();
    if let Some((candidate, result)) = best {
        println!("=== Best configuration ===");
        println!(
            "strategy:          {}",
            candidate.strategy.describe_config()
        );
        println!("buy_fraction:      {:.2}", candidate.buy_sell_fraction);
        println!();
        print_summary(&result);

        if let Some(hold_equity) = buy_and_hold_equity(&samples, result.initial_equity) {
            println!();
            println!("Buy & hold final equity: {:.2}", hold_equity);
        }
    } else {
        println!("No valid backtest result produced.");
    }

    Ok(())
}
