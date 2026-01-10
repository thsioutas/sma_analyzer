use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use serde::Deserialize;

use trade_signal::{
    backtest::{
        ConfigSweep, find_best_strategy, generate_backtest_sweep_jobs, generate_pullback_pairs,
        generate_strategies,
        spot::{SpotBacktester, buy_and_hold_equity, print_summary},
    },
    data::{get_samples, resample_to_hourly},
};

#[derive(Debug, Parser)]
struct Args {
    /// config-file path
    #[arg(long)]
    config: PathBuf,
}

/// Sweep over backtest parameters (i.e. lookback, buy/sell fractions)
/// and report the best configuration.
#[derive(Deserialize)]
struct Config {
    #[serde(flatten)]
    common: ConfigSweep,

    /// Initial coin holdings (e.g. if you already own some SOL)
    initial_coin: f64,

    /// Trading fee in basis points (e.g. 10 = 0.10%)
    fee_bps: f64,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let config_path = args
        .config
        .into_os_string()
        .into_string()
        .expect("Failed to translate config file path into string");
    let config: Config = config::Config::builder()
        .add_source(config::File::with_name(&config_path))
        .build()?
        .try_deserialize()?;

    let samples = get_samples(&config.common.setup.input, config.common.setup.csv_type).expect("failed to load input CSV");
    let hourly = resample_to_hourly(&samples);

    println!(
        "Loaded {} raw samples -> {} hourly candles",
        samples.len(),
        hourly.len()
    );

    let pullback_pairs =
        generate_pullback_pairs(config.common.min_pullback_pct, config.common.max_pullback_pct, 0.001);

    let strategies = generate_strategies(config.common.min_lookback, config.common.max_lookback, pullback_pairs);

    let buy_sell_frac_steps = config.common.buy_sell_frac_steps;

    let jobs = generate_backtest_sweep_jobs(strategies, buy_sell_frac_steps);

    let best = find_best_strategy(
        jobs,
        config.common.max_buy_sell_fraction,
        buy_sell_frac_steps,
        &hourly,
        || SpotBacktester::new(config.common.setup.initial_cash, config.initial_coin, config.fee_bps),
    );

    println!();
    if let Some((candidate, result)) = best {
        println!("=== Best configuration ===");
        println!(
            "strategy:          {}",
            candidate.strategy.describe_config()
        );
        println!("buy_fraction:      {:.2}", candidate.buy_sell_fraction);
        println!("sell_fraction:     {:.2}", candidate.buy_sell_fraction);
        println!("fee_bps:           {:.2}", config.fee_bps);
        println!();
        print_summary(&result);

        if let Some(hold_equity) =
            buy_and_hold_equity(&hourly, config.common.setup.initial_cash, config.initial_coin)
        {
            println!();
            println!("Buy & hold final equity: {:.2}", hold_equity);
        }
    } else {
        println!("No valid backtest result produced.");
    }
    Ok(())
}
