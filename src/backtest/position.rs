use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::Serialize;

use crate::backtest::{Backtester, Candidate, TradingMetrics};
use crate::data::Sample;
use crate::indicators::compute_smas;
use crate::signal::analyze;

use super::common::{Signal, suggestion_to_signal};

#[derive(Debug, Clone, Serialize)]
pub struct Position {
    pub side: PositionSide,
    pub entry_time: DateTime<Utc>,
    pub exit_time: Option<DateTime<Utc>>,
    pub entry_price: f64,
    pub exit_price: Option<f64>,
    pub entry_reason: String,
    pub exit_reason: Option<String>,
    pub size: f64,
    pub profit: Option<f64>,
    pub return_pct: Option<f64>,
    /// Gross collateral removed from cash at entry (before entry fee).
    pub entry_collateral_gross: f64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub enum PositionSide {
    Long,
    Short,
}

impl From<Signal> for PositionSide {
    fn from(s: Signal) -> Self {
        match s {
            Signal::Buy => Self::Long,
            Signal::Sell => Self::Short,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PositionBacktestResult {
    pub initial_equity: f64,
    pub positions: Vec<Position>,
    pub equity_curve: Vec<(DateTime<Utc>, f64)>,
    pub final_equity: f64,
    pub total_return_pct: f64,
    pub max_drawdown_pct: f64,
    pub win_rate_pct: f64,
}

fn position_liquidation_value(pos: &Position, price: f64) -> f64 {
    if price <= 0.0 || pos.size <= 0.0 {
        return 0.0;
    }

    match pos.side {
        PositionSide::Long => pos.size * price,
        PositionSide::Short => {
            let gross_pnl = (pos.entry_price - price) * pos.size;
            pos.entry_collateral_gross + gross_pnl
        }
    }
}

fn close_position(
    mut pos: Position,
    exit_price: f64,
    exit_time: DateTime<Utc>,
    exit_reason: String,
) -> Position {
    pos.exit_price = Some(exit_price);
    pos.exit_time = Some(exit_time);
    pos.exit_reason = Some(exit_reason);

    let gross_pnl = match pos.side {
        PositionSide::Long => (exit_price - pos.entry_price) * pos.size,
        PositionSide::Short => (pos.entry_price - exit_price) * pos.size,
    };

    let profit = gross_pnl;
    let ret = if pos.entry_collateral_gross > 0.0 {
        profit / pos.entry_collateral_gross
    } else {
        0.0
    };

    pos.profit = Some(profit);
    pos.return_pct = Some(ret);
    pos
}

fn open_position(
    side: PositionSide,
    price: f64,
    ts: DateTime<Utc>,
    cash: &mut f64,
    entry_frac: f64,
    reason: String,
) -> Option<Position> {
    if price <= 0.0 || *cash <= 0.0 || entry_frac <= 0.0 {
        return None;
    }

    let entry_collateral_gross = (*cash) * entry_frac;
    if entry_collateral_gross <= 0.0 {
        return None;
    }

    let size = entry_collateral_gross / price;
    if size <= 0.0 {
        return None;
    }

    *cash -= entry_collateral_gross;

    Some(Position {
        side,
        entry_time: ts,
        exit_time: None,
        entry_price: price,
        exit_price: None,
        entry_reason: reason,
        exit_reason: None,
        size,
        entry_collateral_gross,
        profit: None,
        return_pct: None,
    })
}

fn compute_max_drawdown(curve: &[(DateTime<Utc>, f64)]) -> f64 {
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

fn compute_win_rate(positions: &[Position]) -> f64 {
    if positions.is_empty() {
        return 0.0;
    }

    let wins = positions
        .iter()
        .filter(|p| p.profit.unwrap_or(0.0) > 0.0)
        .count() as f64;

    wins / positions.len() as f64
}

pub fn buy_and_hold_equity(hourly: &[Sample], initial_cash: f64) -> Option<f64> {
    if hourly.is_empty() {
        return None;
    }
    let first = hourly.first().unwrap().price;
    let last = hourly.last().unwrap().price;
    if first <= 0.0 {
        return None;
    }

    let qty = initial_cash / first;
    Some(qty * last)
}

/// Simple CLI-style summary you can reuse in a binary.
pub fn print_summary(result: &PositionBacktestResult) {
    println!("=== Backtest Summary ===");
    println!("Initial equity:  {:.2}", result.initial_equity);
    println!("Final equity:     {:.2}", result.final_equity);
    println!("Total return:     {:.2}%", result.total_return_pct * 100.0);
    println!("Max drawdown:     {:.2}%", result.max_drawdown_pct * 100.0);
    println!("Positions:           {}", result.positions.len());
    println!("Win rate:         {:.2}%", result.win_rate_pct * 100.0);
}

pub struct PositionBacktester<L> {
    initial_cash: f64,
    logger: L,
}

impl PositionBacktester<NoopLogger> {
    pub fn new(initial_cash: f64) -> Self {
        Self {
            initial_cash,
            logger: NoopLogger,
        }
    }
}

impl<L: PositionLogger> PositionBacktester<L> {
    pub fn with_logger(initial_cash: f64, logger: L) -> Self
    where
        L: PositionLogger,
    {
        Self {
            initial_cash,
            logger,
        }
    }
}

impl<L: PositionLogger> Backtester for PositionBacktester<L> {
    type Output = PositionBacktestResult;
    fn run_backtest(
        &self,
        samples: &[Sample],
        candidate: &Candidate,
    ) -> Result<Self::Output, String> {
        if samples.len() < candidate.strategy.sma_config.long_window + 1 {
            return Err("Not enough data".into());
        }

        let initial_equity = self.initial_cash;

        let mut prices: Vec<f64> = Vec::with_capacity(samples.len());
        let mut equity_curve: Vec<(DateTime<Utc>, f64)> = Vec::with_capacity(samples.len());
        let mut open: Option<Position> = None;
        let mut closed: Vec<Position> = Vec::new();

        // Initial portfolio state
        let mut cash = self.initial_cash;

        let buy_frac = candidate.buy_sell_fraction.clamp(0.0, 1.0);

        for (i, candle) in samples.iter().enumerate() {
            let price = candle.price;
            prices.push(price);

            let equity = cash
                + open
                    .as_ref()
                    .map(|p| position_liquidation_value(p, price))
                    .unwrap_or(0.0);
            equity_curve.push((candle.ts, equity));

            if prices.len() < candidate.strategy.sma_config.long_window + 1 {
                // Not enough data yet for SMAs
                continue;
            }

            let Some(smas) = compute_smas(&prices, candidate.strategy.sma_config) else {
                continue;
            };

            let analysis = analyze(&samples[..=i], &prices, smas, candidate.strategy);
            let signal = suggestion_to_signal(&analysis.suggestion);

            match signal {
                Some(signal) => {
                    let want_side = signal.into();
                    let same_side = open.as_ref().map(|p| p.side == want_side).unwrap_or(false);
                    if !same_side {
                        // close old if exists
                        if let Some(pos) = open.take() {
                            let closed_pos =
                                close_position(pos, price, candle.ts, analysis.reason.clone());
                            self.logger.log(&closed_pos)?;
                            cash += closed_pos.entry_collateral_gross
                                + closed_pos.profit.unwrap_or(0.0);
                            closed.push(closed_pos);
                        }
                        // open new
                        if let Some(pos) = open_position(
                            want_side,
                            price,
                            candle.ts,
                            &mut cash,
                            buy_frac,
                            analysis.reason,
                        ) {
                            open = Some(pos);
                        }
                    }
                }
                _ => {
                    // HOLD or suggestion that doesn't change position
                }
            }
        }

        // If a position is open close it
        if let Some(pos) = open.take() {
            let last = samples.last().unwrap();
            let closed_pos = close_position(pos, last.price, last.ts, "EOF".to_string());
            self.logger.log(&closed_pos)?;
            cash += closed_pos.entry_collateral_gross + closed_pos.profit.unwrap_or(0.0);
            closed.push(closed_pos);
        }
        let final_equity = cash;
        let total_return_pct = final_equity / initial_equity - 1.0;

        let max_drawdown_pct = compute_max_drawdown(&equity_curve);
        let win_rate_pct = compute_win_rate(&closed);

        Ok(PositionBacktestResult {
            initial_equity,
            positions: closed,
            equity_curve,
            final_equity,
            total_return_pct,
            max_drawdown_pct,
            win_rate_pct,
        })
    }
}

impl TradingMetrics for PositionBacktestResult {
    fn total_return_pct(&self) -> f64 {
        self.total_return_pct
    }

    fn max_drawdown_pct(&self) -> f64 {
        self.max_drawdown_pct
    }
}

pub trait PositionLogger: Sync {
    fn log(&self, position: &Position) -> Result<(), String>;
}

pub struct NdjsonLogger {
    pub path: PathBuf,
}

impl NdjsonLogger {
    pub fn new(path: PathBuf) -> Self {
        Self { path }
    }
}

impl PositionLogger for NdjsonLogger {
    fn log(&self, pos: &Position) -> Result<(), String> {
        let mut f = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .map_err(|err| err.to_string())?;
        let line = serde_json::to_string(pos).map_err(|err| err.to_string())?;
        writeln!(f, "{line}").map_err(|err| err.to_string())?;
        Ok(())
    }
}

pub struct NoopLogger;

impl PositionLogger for NoopLogger {
    fn log(&self, _pos: &Position) -> Result<(), String> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backtest::Candidate;
    use crate::indicators::sma::SmaConfig;
    use crate::signal::{FilterConfig, StrategyConfig};
    use chrono::TimeZone;

    fn sample(ts_secs: i64, price: f64) -> Sample {
        Sample {
            ts: Utc.timestamp_opt(ts_secs, 0).single().unwrap(),
            price,
        }
    }

    /// Creates a simple strategy config for testing.
    /// Uses SMA 5/10 with bias_only enabled (no breakouts/pullbacks/crossovers)
    /// and no filters to make behavior predictable.
    fn test_strategy() -> StrategyConfig {
        StrategyConfig {
            breakouts: None,
            pullbacks: None,
            enable_crossovers: false,
            enable_bias_only: true,
            sma_config: SmaConfig {
                short_window: 5,
                long_window: 10,
            },
            filters: FilterConfig {
                require_trend_filter: false,
                require_price_confirmation: false,
                atr: None,
                regime: None,
            },
        }
    }

    #[test]
    fn test_backtest_single_long_position_with_profit() {
        // Build price series that creates a clear uptrend (SMA5 > SMA10)
        // then holds until EOF.
        //
        // We need at least long_window + 1 = 11 samples for SMAs to compute.
        // Prices: start flat, then rise to create uptrend.
        //
        // Samples 0-9:  price = 100 (flat)
        // Samples 10+:  price rises to 110, 120, 130
        //
        // After sample 10, SMA5 and SMA10 can be computed.
        // With rising prices, SMA5 > SMA10 => BUY signal (bias_only)
        let mut samples = Vec::new();
        let base_ts = 1700000000i64;

        // First 10 samples at price 100
        for i in 0..10 {
            samples.push(sample(base_ts + i * 3600, 100.0));
        }

        // Next 5 samples rising: 110, 120, 130, 140, 150
        for (i, price) in [110.0, 120.0, 130.0, 140.0, 150.0].iter().enumerate() {
            samples.push(sample(base_ts + (10 + i as i64) * 3600, *price));
        }

        let backtester = PositionBacktester::new(10000.0);
        let candidate = Candidate {
            buy_sell_fraction: 1.0, // Use 100% of cash
            strategy: test_strategy(),
        };

        let result = backtester.run_backtest(&samples, &candidate).unwrap();

        // Verify results
        assert_eq!(result.initial_equity, 10000.0);
        assert_eq!(result.positions.len(), 1, "Expected exactly 1 position");

        let pos = &result.positions[0];
        assert_eq!(pos.side, PositionSide::Long);
        assert!(pos.exit_reason.as_ref().unwrap().contains("EOF"));

        // Entry should be when uptrend is detected (after we have enough data)
        // Exit at last price = 150
        assert_eq!(pos.exit_price.unwrap(), 150.0);
        assert!(pos.profit.unwrap() > 0.0, "Position should be profitable");

        // Final equity should be greater than initial
        assert!(
            result.final_equity > result.initial_equity,
            "Final equity {} should be > initial {}",
            result.final_equity,
            result.initial_equity
        );
        assert!(result.total_return_pct > 0.0);
    }

    #[test]
    fn test_backtest_single_short_position_with_profit() {
        // Build price series that creates a clear downtrend (SMA5 < SMA10)
        //
        // Samples 0-9:  price = 100 (flat)
        // Samples 10+:  price drops to 90, 80, 70, 60, 50
        let mut samples = Vec::new();
        let base_ts = 1700000000i64;

        // First 10 samples at price 100
        for i in 0..10 {
            samples.push(sample(base_ts + i * 3600, 100.0));
        }

        // Next 5 samples falling: 90, 80, 70, 60, 50
        for (i, price) in [90.0, 80.0, 70.0, 60.0, 50.0].iter().enumerate() {
            samples.push(sample(base_ts + (10 + i as i64) * 3600, *price));
        }

        let backtester = PositionBacktester::new(10000.0);
        let candidate = Candidate {
            buy_sell_fraction: 1.0,
            strategy: test_strategy(),
        };

        let result = backtester.run_backtest(&samples, &candidate).unwrap();

        assert_eq!(result.initial_equity, 10000.0);
        assert_eq!(result.positions.len(), 1, "Expected exactly 1 position");

        let pos = &result.positions[0];
        assert_eq!(pos.side, PositionSide::Short);
        assert!(pos.exit_reason.as_ref().unwrap().contains("EOF"));
        assert_eq!(pos.exit_price.unwrap(), 50.0);
        assert!(
            pos.profit.unwrap() > 0.0,
            "Short position should be profitable when price drops"
        );

        assert!(
            result.final_equity > result.initial_equity,
            "Final equity {} should be > initial {}",
            result.final_equity,
            result.initial_equity
        );
    }

    #[test]
    fn test_backtest_position_flip_long_to_short() {
        // Start with uptrend, then flip to downtrend
        //
        // Samples 0-9:   price = 100 (flat baseline)
        // Samples 10-14: price rises to 110, 120, 130, 140, 150 (uptrend -> LONG)
        // Samples 15-19: price drops to 100, 80, 60, 40, 20 (downtrend -> flip to SHORT)
        let mut samples = Vec::new();
        let base_ts = 1700000000i64;

        // Flat baseline
        for i in 0..10 {
            samples.push(sample(base_ts + i * 3600, 100.0));
        }

        // Rising prices (creates uptrend)
        for (i, price) in [110.0, 120.0, 130.0, 140.0, 150.0].iter().enumerate() {
            samples.push(sample(base_ts + (10 + i as i64) * 3600, *price));
        }

        // Falling prices (creates downtrend, should flip position)
        for (i, price) in [100.0, 80.0, 60.0, 40.0, 20.0].iter().enumerate() {
            samples.push(sample(base_ts + (15 + i as i64) * 3600, *price));
        }

        let backtester = PositionBacktester::new(10000.0);
        let candidate = Candidate {
            buy_sell_fraction: 1.0,
            strategy: test_strategy(),
        };

        let result = backtester.run_backtest(&samples, &candidate).unwrap();

        assert_eq!(result.initial_equity, 10000.0);
        assert!(
            result.positions.len() >= 2,
            "Expected at least 2 positions (long then short), got {}",
            result.positions.len()
        );

        // First position should be Long
        assert_eq!(result.positions[0].side, PositionSide::Long);

        // Last position should be Short (closed at EOF)
        let last_pos = result.positions.last().unwrap();
        assert_eq!(last_pos.side, PositionSide::Short);
        assert!(last_pos.exit_reason.as_ref().unwrap().contains("EOF"));
    }

    #[test]
    fn test_backtest_partial_allocation() {
        // Test that buy_sell_fraction correctly allocates partial cash
        let mut samples = Vec::new();
        let base_ts = 1700000000i64;

        for i in 0..10 {
            samples.push(sample(base_ts + i * 3600, 100.0));
        }
        for (i, price) in [110.0, 120.0, 130.0, 140.0, 150.0].iter().enumerate() {
            samples.push(sample(base_ts + (10 + i as i64) * 3600, *price));
        }

        let initial_cash = 10000.0;
        let backtester = PositionBacktester::new(initial_cash);
        let candidate = Candidate {
            buy_sell_fraction: 0.5, // Only use 50% of cash
            strategy: test_strategy(),
        };

        let result = backtester.run_backtest(&samples, &candidate).unwrap();

        assert_eq!(result.positions.len(), 1);
        let pos = &result.positions[0];

        // Entry collateral should be 50% of initial cash
        let expected_collateral = initial_cash * 0.5;
        assert!(
            (pos.entry_collateral_gross - expected_collateral).abs() < 0.01,
            "Expected collateral {}, got {}",
            expected_collateral,
            pos.entry_collateral_gross
        );
    }

    #[test]
    fn test_backtest_not_enough_data_returns_error() {
        // Only 5 samples, but we need long_window + 1 = 11
        let samples: Vec<Sample> = (0..5)
            .map(|i| sample(1700000000 + i * 3600, 100.0))
            .collect();

        let backtester = PositionBacktester::new(10000.0);
        let candidate = Candidate {
            buy_sell_fraction: 1.0,
            strategy: test_strategy(),
        };

        let result = backtester.run_backtest(&samples, &candidate);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Not enough data"));
    }

    #[test]
    fn test_backtest_equity_curve_length_matches_samples() {
        let mut samples = Vec::new();
        let base_ts = 1700000000i64;

        for i in 0..15 {
            samples.push(sample(base_ts + i * 3600, 100.0 + i as f64));
        }

        let backtester = PositionBacktester::new(10000.0);
        let candidate = Candidate {
            buy_sell_fraction: 1.0,
            strategy: test_strategy(),
        };

        let result = backtester.run_backtest(&samples, &candidate).unwrap();

        assert_eq!(
            result.equity_curve.len(),
            samples.len(),
            "Equity curve should have one entry per sample"
        );
    }

    #[test]
    fn test_backtest_max_drawdown_calculation() {
        // Create a scenario with a clear drawdown:
        // Price goes up (profit), then down (drawdown), then recovers
        let mut samples = Vec::new();
        let base_ts = 1700000000i64;

        // Baseline
        for i in 0..10 {
            samples.push(sample(base_ts + i * 3600, 100.0));
        }

        // Rise (creates profit in long position)
        for (i, price) in [110.0, 120.0, 130.0, 140.0, 150.0].iter().enumerate() {
            samples.push(sample(base_ts + (10 + i as i64) * 3600, *price));
        }

        // Drop (creates drawdown while still in long)
        for (i, price) in [140.0, 130.0, 120.0].iter().enumerate() {
            samples.push(sample(base_ts + (15 + i as i64) * 3600, *price));
        }

        let backtester = PositionBacktester::new(10000.0);
        let candidate = Candidate {
            buy_sell_fraction: 1.0,
            strategy: test_strategy(),
        };

        let result = backtester.run_backtest(&samples, &candidate).unwrap();

        // Max drawdown should be > 0 due to the price drop
        assert!(
            result.max_drawdown_pct > 0.0,
            "Expected positive drawdown, got {}",
            result.max_drawdown_pct
        );
    }

    #[test]
    fn test_backtest_win_rate_calculation() {
        // Create scenario with 2 positions: 1 winning, 1 losing
        // This requires careful price construction to flip positions
        let mut samples = Vec::new();
        let base_ts = 1700000000i64;

        // Baseline
        for i in 0..10 {
            samples.push(sample(base_ts + i * 3600, 100.0));
        }

        // Uptrend -> opens LONG
        for (i, price) in [110.0, 120.0, 130.0, 140.0, 150.0].iter().enumerate() {
            samples.push(sample(base_ts + (10 + i as i64) * 3600, *price));
        }

        // Downtrend -> closes LONG (with profit), opens SHORT
        for (i, price) in [140.0, 130.0, 120.0, 110.0, 100.0].iter().enumerate() {
            samples.push(sample(base_ts + (15 + i as i64) * 3600, *price));
        }

        // Slight uptick at end -> SHORT might lose a bit at EOF close
        samples.push(sample(base_ts + 20 * 3600, 105.0));

        let backtester = PositionBacktester::new(10000.0);
        let candidate = Candidate {
            buy_sell_fraction: 1.0,
            strategy: test_strategy(),
        };

        let result = backtester.run_backtest(&samples, &candidate).unwrap();

        // Should have at least 2 positions
        assert!(
            result.positions.len() >= 2,
            "Expected at least 2 positions, got {}",
            result.positions.len()
        );

        // Win rate should be between 0 and 1
        assert!(
            result.win_rate_pct >= 0.0 && result.win_rate_pct <= 1.0,
            "Win rate {} should be between 0 and 1",
            result.win_rate_pct
        );
    }
}
