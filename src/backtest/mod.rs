mod common;
pub mod position;
pub mod spot;
pub use common::{
    Backtester, Candidate, ConfigSetup, ConfigSingleRun, ConfigSweep, TradingMetrics,
    find_best_strategy, generate_backtest_sweep_jobs, generate_pullback_pairs, generate_strategies,
};
