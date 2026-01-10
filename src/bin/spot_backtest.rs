use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use serde::Deserialize;

use trade_signal::{
    backtest::spot::{SpotBacktester, buy_and_hold_equity, print_summary},
    backtest::{Backtester, Candidate, ConfigSingleRun},
    data::{get_samples, resample_to_hourly},
    indicators::sma::SmaConfig,
    indicators::{AtrFilter, RegimeFilter},
    signal::{BreakoutConfig, FilterConfig, PullbackConfig, StrategyConfig},
};

#[derive(Debug, Parser)]
struct Args {
    /// config-file path
    #[arg(long)]
    config: PathBuf,
}

#[derive(Deserialize)]
struct Config {
    #[serde(flatten)]
    common: ConfigSingleRun,

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

    let samples = get_samples(&config.common.setup.input, config.common.setup.csv_type)
        .with_context(|| format!("failed to load samples from {:?}", config.common.setup.input))?;

    if samples.is_empty() {
        println!("No data found in CSV.");
        return Ok(());
    }

    let hourly = resample_to_hourly(&samples);

    println!(
        "Loaded {} raw points, {} hourly candles after resampling.",
        samples.len(),
        hourly.len()
    );

    let pullbacks = match (
        config.common.pullback_bounce_tolerance_pct,
        config.common.pullback_rejection_tolerance_pct,
    ) {
        (Some(bounce_tolerance_pct), Some(reject_tolerance_pct)) => Some(PullbackConfig {
            bounce_tolerance_pct,
            reject_tolerance_pct,
        }),
        (None, None) => None,
        (Some(v), None) => {
            println!("Using given bounce_tolerance_pct as reject_tolerance_pct");
            Some(PullbackConfig {
                bounce_tolerance_pct: v,
                reject_tolerance_pct: v,
            })
        }
        (None, Some(v)) => {
            println!("Using given reject_tolerance_pct as bounce_tolerance_pct");
            Some(PullbackConfig {
                bounce_tolerance_pct: v,
                reject_tolerance_pct: v,
            })
        }
    };

    let strategy = StrategyConfig {
        breakouts: config.common.breakout_lookback.map(|v| BreakoutConfig {
            breakout_lookback: v,
        }),
        pullbacks,
        enable_crossovers: config.common.enable_crossovers,
        enable_bias_only: config.common.enable_bias_only,
        sma_config: SmaConfig {
            short_window: config.common.sma_short_window,
            long_window: config.common.sma_long_window,
        },
        filters: FilterConfig {
            require_price_confirmation: config.common.require_price_confirmation,
            require_trend_filter: config.common.require_trend_filter,
            atr: if config.common.atr_enabled {
                Some(AtrFilter::backtest())
            } else {
                None
            },
            regime: if config.common.regime_enabled {
                Some(RegimeFilter::backtest())
            } else {
                None
            },
        },
    };

    println!("Initial cash:      {}", config.common.setup.initial_cash);
    println!("Initial coin:      {}", config.initial_coin);
    println!("Fee bps:           {}", config.fee_bps);
    println!("Buy/Sell fraction: {}", config.common.setup.buy_sell_fraction);
    println!("Strategy:          {}", strategy.describe_config());

    let backtester = SpotBacktester::new(config.common.setup.initial_cash, config.initial_coin, config.fee_bps);
    let candidate = Candidate {
        buy_sell_fraction: config.common.setup.buy_sell_fraction,
        strategy,
    };
    let result = backtester.run_backtest(&hourly, &candidate).unwrap();

    print_summary(&result);
    if let Some(hold_equity) =
        buy_and_hold_equity(&hourly, config.common.setup.initial_cash, config.initial_coin)
    {
        println!();
        println!("Buy & hold final equity: {:.2}", hold_equity);
    }

    Ok(())
}
