use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;

use trade_signal::{
    backtest::{
        Backtester, Candidate, ConfigSingleRun, position::{NdjsonLogger, PositionBacktester, buy_and_hold_equity, print_summary}
    },
    data::{get_samples, resample_to_n_hours},
    indicators::{AtrFilter, RegimeFilter, sma::SmaConfig},
    signal::{BreakoutConfig, FilterConfig, PullbackConfig, StrategyConfig},
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
    let config: ConfigSingleRun = config::Config::builder()
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

    let pullbacks = match (
        config.pullback_bounce_tolerance_pct,
        config.pullback_rejection_tolerance_pct,
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
        breakouts: config.breakout_lookback.map(|v| BreakoutConfig {
            breakout_lookback: v,
        }),
        pullbacks,
        enable_crossovers: config.enable_crossovers,
        enable_bias_only: config.enable_bias_only,
        sma_config: SmaConfig {
            short_window: config.sma_short_window,
            long_window: config.sma_long_window,
        },
        filters: FilterConfig {
            require_price_confirmation: config.require_price_confirmation,
            require_trend_filter: config.require_trend_filter,
            atr: if config.atr_enabled {
                Some(AtrFilter::backtest())
            } else {
                None
            },
            regime: if config.regime_enabled {
                Some(RegimeFilter::backtest())
            } else {
                None
            },
        },
    };

    let candidate = Candidate {
        buy_sell_fraction: config.setup.buy_sell_fraction,
        strategy,
    };

    println!("Initial cash:      {}", config.setup.initial_cash);
    println!("Buy fraction:      {}", config.setup.buy_sell_fraction);
    println!("Strategy:          {}", strategy.describe_config());

    let log_path = log_path_unix("position_backtest");
    let position_logger = NdjsonLogger::new(log_path);
    let backtester = PositionBacktester::with_logger(config.setup.initial_cash, position_logger);
    let result = backtester.run_backtest(&resampled, &candidate).unwrap();

    print_summary(&result);
    if let Some(hold_equity) = buy_and_hold_equity(&resampled, config.setup.initial_cash) {
        println!();
        println!("Buy & hold final equity: {:.2}", hold_equity);
    }

    Ok(())
}

fn log_path_unix(prefix: &str) -> PathBuf {
    let now = chrono::Local::now();
    let fmt = now.format("%Y-%m-%d_%H:%M:%S");
    PathBuf::from("logs").join(format!("{prefix}_{fmt}.log"))
}
