use std::{
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};

use chrono::{DateTime, Utc};
use rayon::prelude::*;
use serde::Deserialize;

use crate::{
    data::{CsvType, Sample},
    indicators::{AtrFilter, RegimeFilter, sma::SmaConfig},
    signal::{BreakoutConfig, FilterConfig, PullbackConfig, StrategyConfig},
};

#[derive(Deserialize)]
pub struct ConfigSetup {
    /// Type of the CSV file
    pub csv_type: CsvType,

    /// Path to the CSV file (timestamp,price)pub
    pub input: PathBuf,

    /// Resample input to <sample_hours> hours (i.e. 1h, 4h, 6h, ...)
    pub sample_hours: i64,

    /// Initial cash for the backtest
    pub initial_cash: f64,

    /// Fraction of *available cash* to allocate on each BUY/SELL signal (0.0–1.0)
    pub buy_sell_fraction: f64,
}

#[derive(Deserialize)]
pub struct ConfigSingleRun {
    #[serde(flatten)]
    pub setup: ConfigSetup,

    /// Whether ATR gate filter should be used
    pub atr_enabled: bool,

    /// Whether regime filter should be used
    pub regime_enabled: bool,

    /// How many candles to lookback for a breakdown
    /// Do not set to not use breakout patterns
    pub breakout_lookback: Option<usize>,

    /// Do not set to not use pullback patterns
    pub pullback_bounce_tolerance_pct: Option<f64>,

    /// Do not set to not use pullback patterns
    pub pullback_rejection_tolerance_pct: Option<f64>,

    /// Whether sma crossover signals should be used
    pub enable_crossovers: bool,

    /// Whether bias_only signals should be used
    pub enable_bias_only: bool,

    /// SMA short window
    pub sma_short_window: usize,

    /// SMA long window
    pub sma_long_window: usize,

    /// Whether price confirmation is required
    pub require_price_confirmation: bool,

    /// Whether trend filter is required
    pub require_trend_filter: bool,
}

#[derive(Deserialize)]
pub struct ConfigSweep {
    #[serde(flatten)]
    pub setup: ConfigSetup,

    /// Min breakout lookback window (e.g. 3)
    pub min_lookback: usize,

    /// Max breakout lookback window (e.g. 10)
    pub max_lookback: usize,

    /// Min pullback tolerances (e.g. 0.001)
    pub min_pullback_pct: f64,

    /// Max pullback tolerances (e.g. 0.01)
    pub max_pullback_pct: f64,

    /// Maximum fraction for buy/sell (e.g. 0.5 = at most 50%)
    pub max_buy_sell_fraction: f64,

    /// Number of steps for buy/sell fraction (0–1).
    /// E.g. 100 => 0.01, 0.02, ..., 1.00
    pub buy_sell_frac_steps: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Signal {
    Buy,
    Sell,
}

pub fn suggestion_to_signal(s: &str) -> Option<Signal> {
    match s {
        "BUY" => Some(Signal::Buy),
        "SELL" => Some(Signal::Sell),
        _ => None,
    }
}

impl RegimeFilter {
    pub fn backtest() -> Self {
        Self::default()
    }
}

impl AtrFilter {
    pub fn backtest() -> Self {
        Self::new_fixed(5, 0.003)
    }
}

pub fn compute_max_drawdown(curve: &[(DateTime<Utc>, f64)]) -> f64 {
    if curve.is_empty() {
        return 0.0;
    }

    let mut peak = curve[0].1;
    let mut max_dd = 0.0;

    for &(_, equity) in curve {
        if equity > peak {
            peak = equity;
        }
        if peak > 0.0 {
            let dd = (peak - equity) / peak;
            if dd > max_dd {
                max_dd = dd;
            }
        }
    }

    max_dd
}

pub fn generate_strategies(
    min_lookback: usize,
    max_lookback: usize,
    pullback_pairs: Vec<(f64, f64)>,
) -> Vec<StrategyConfig> {
    let mut strategies = Vec::new();

    let short_candidates = [10, 20, 30];
    let long_candidates = [40, 60, 80, 100];

    let sma_configs: Vec<SmaConfig> = short_candidates
        .iter()
        .flat_map(|&short| {
            long_candidates.iter().filter_map(move |&long| {
                if long >= short * 2 {
                    Some(SmaConfig {
                        short_window: short,
                        long_window: long,
                    })
                } else {
                    None
                }
            })
        })
        .collect();

    for sma_config in sma_configs {
        // bit 0: breakouts
        // bit 1: pullbacks
        // bit 2: crossovers
        // bit 3: bias_only
        for mask in 0u8..16 {
            let enable_breakouts = (mask & 0b0001) != 0;
            let enable_pullbacks = (mask & 0b0010) != 0;
            let enable_crossovers = (mask & 0b0100) != 0;
            // No need to have this configurable (now)
            // let enable_bias_only = (mask & 0b1000) != 0;
            let enable_bias_only = true;

            match (enable_breakouts, enable_pullbacks) {
                (true, true) => {
                    for lookback in min_lookback..=max_lookback {
                        for (pullback_bounce_tol, pullback_rejection_tol) in &pullback_pairs {
                            let strategy = StrategyConfig {
                                breakouts: Some(BreakoutConfig {
                                    breakout_lookback: lookback,
                                }),
                                pullbacks: Some(PullbackConfig {
                                    bounce_tolerance_pct: *pullback_bounce_tol,
                                    reject_tolerance_pct: *pullback_rejection_tol,
                                }),
                                enable_crossovers,
                                enable_bias_only,
                                sma_config,
                                filters: FilterConfig {
                                    atr: None,
                                    regime: None,
                                    require_price_confirmation: true,
                                    require_trend_filter: true,
                                },
                            };

                            strategies.push(strategy);
                        }
                    }
                }
                (true, false) => {
                    for lookback in min_lookback..=max_lookback {
                        let strategy = StrategyConfig {
                            breakouts: Some(BreakoutConfig {
                                breakout_lookback: lookback,
                            }),
                            pullbacks: None,
                            enable_crossovers,
                            enable_bias_only,
                            sma_config,
                            filters: FilterConfig {
                                atr: None,
                                regime: None,
                                require_price_confirmation: true,
                                require_trend_filter: true,
                            },
                        };

                        strategies.push(strategy);
                    }
                }
                (false, true) => {
                    for (pullback_bounce_tol, pullback_rejection_tol) in &pullback_pairs {
                        let strategy = StrategyConfig {
                            breakouts: None,
                            pullbacks: Some(PullbackConfig {
                                bounce_tolerance_pct: *pullback_bounce_tol,
                                reject_tolerance_pct: *pullback_rejection_tol,
                            }),
                            enable_crossovers,
                            enable_bias_only,
                            sma_config,
                            filters: FilterConfig {
                                atr: None,
                                regime: None,
                                require_price_confirmation: true,
                                require_trend_filter: true,
                            },
                        };

                        strategies.push(strategy);
                    }
                }
                (false, false) => {
                    // Skip the totally empty strategy (nothing enabled).
                    if !enable_breakouts && !enable_pullbacks && !enable_crossovers {
                        continue;
                    }
                    let strategy = StrategyConfig {
                        breakouts: None,
                        pullbacks: None,
                        enable_crossovers,
                        enable_bias_only,
                        sma_config,
                        filters: FilterConfig {
                            atr: None,
                            regime: None,
                            require_price_confirmation: true,
                            require_trend_filter: true,
                        },
                    };

                    strategies.push(strategy);
                }
            }
        }
    }

    strategies
}

pub fn generate_pullback_pairs(min: f64, max: f64, step: f64) -> Vec<(f64, f64)> {
    let mut pairs = Vec::new();
    let mut bounce = min;
    while bounce <= max {
        let mut reject = bounce + step;
        while reject <= max {
            pairs.push((bounce, reject));
            reject += step;
        }
        bounce += step;
    }
    pairs
}

pub fn generate_backtest_sweep_jobs(
    strategies: Vec<StrategyConfig>,
    buy_sell_frac_steps: usize,
) -> Vec<(StrategyConfig, usize)> {
    strategies
        .iter()
        .flat_map(|&strategy| {
            (1..=buy_sell_frac_steps).map(move |buy_sell_frac_step| (strategy, buy_sell_frac_step))
        })
        .collect()
}

pub struct Candidate {
    pub buy_sell_fraction: f64,
    pub strategy: StrategyConfig,
}

pub fn find_best_strategy<B, F>(
    jobs: Vec<(StrategyConfig, usize)>,
    max_buy_sell_fraction: f64,
    buy_sell_frac_steps: usize,
    samples: &[Sample],
    // use factory instead of restricting with Sync
    make_backtester: F,
) -> Option<(Candidate, B::Output)>
where
    B: Backtester,
    F: Fn() -> B + Sync + Send,
{
    const EPS: f64 = 1e-9;

    let total_iters = jobs.len() as u64;
    let done = AtomicU64::new(0);
    let progress_every = (total_iters / 100).max(1);

    println!(
        "Running parameter sweep... ({} total combinations)",
        total_iters
    );

    let best_pair: Option<(Candidate, B::Output)> = jobs
        .into_par_iter()
        .map_init(
            || make_backtester(),
            |backtester, (strategy, buy_sell_frac_step)| {
                let current = done.fetch_add(1, Ordering::Relaxed) + 1;
                if progress_every != 0
                    && (current.is_multiple_of(progress_every) || current == total_iters)
                {
                    let pct = (current as f64 / total_iters as f64) * 100.0;
                    println!("Progress: {:6.2}% ({}/{})", pct, current, total_iters);
                }
                let buy_sell_fraction = (buy_sell_frac_step as f64 / buy_sell_frac_steps as f64)
                    * max_buy_sell_fraction;
                let candidate = Candidate {
                    buy_sell_fraction,
                    strategy,
                };
                let result = backtester
                    .run_backtest(samples, &candidate)
                    .inspect_err(|err| println!("Failed to get backtest result: {}", err))
                    .ok()?;
                Some((candidate, result))
            },
        )
        .filter_map(|x| x)
        .reduce_with(|res_a, res_b| {
            let a_ret = res_a.1.total_return_pct();
            let b_ret = res_b.1.total_return_pct();
            let a_dd = res_a.1.max_drawdown_pct();
            let b_dd = res_b.1.max_drawdown_pct();

            // "Better" = higher total return, tie-break by lower drawdown
            let pick_b = if b_ret > a_ret + EPS {
                true
            } else if (b_ret - a_ret).abs() < EPS {
                b_dd < a_dd
            } else {
                false
            };

            if pick_b { res_b } else { res_a }
        });

    best_pair
}

pub trait Backtester {
    type Output: TradingMetrics + Send;
    fn run_backtest(
        &self,
        samples: &[Sample],
        candidate: &Candidate,
    ) -> Result<Self::Output, String>;
}

pub trait TradingMetrics {
    fn total_return_pct(&self) -> f64;
    fn max_drawdown_pct(&self) -> f64;
}
