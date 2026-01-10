use anyhow::{Context, Result};
use chrono::{DateTime, Duration, TimeZone, Utc};
use csv::ReaderBuilder;
use serde::Deserialize;

use std::collections::BTreeMap;
use std::fs::File;
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CsvType {
    CryptoLogger,
    KrakenOhlcvt,
}

#[derive(Debug, Deserialize)]
pub struct PriceRow {
    pub timestamp: String,
    pub price: f64,
}

/// Kraken OHLCVT row format: timestamp, open, high, low, close, volume, trades
/// No header row in Kraken CSV files
#[derive(Debug, Deserialize)]
pub struct KrakenOhlcvtRow {
    pub timestamp: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    pub trades: u64,
}

#[derive(Debug, Clone)]
pub struct Sample {
    pub ts: DateTime<Utc>,
    pub price: f64,
}

pub fn get_samples(input: &PathBuf, csv_type: CsvType) -> Result<Vec<Sample>> {
    match csv_type {
        CsvType::CryptoLogger => get_samples_from_input_file(input),
        CsvType::KrakenOhlcvt => get_samples_from_kraken_ohlcvt(input),
    }
}

fn get_samples_from_input_file(input: &PathBuf) -> Result<Vec<Sample>> {
    let file =
        File::open(input).with_context(|| format!("failed to open input file: {:?}", input))?;

    let mut rdr = ReaderBuilder::new().has_headers(true).from_reader(file);

    let mut samples: Vec<Sample> = Vec::new();

    for result in rdr.deserialize::<PriceRow>() {
        let row: PriceRow = result.with_context(|| "failed to deserialize CSV row")?;
        let ts = DateTime::parse_from_rfc3339(&row.timestamp)
            .with_context(|| format!("failed to parse timestamp: {}", row.timestamp))?
            .with_timezone(&Utc);
        samples.push(Sample {
            ts,
            price: row.price,
        });
    }
    Ok(samples)
}

/// Read samples from a Kraken OHLCVT CSV file.
/// Kraken format: timestamp (unix), open, high, low, close, volume, trades
/// Uses the close price as the sample price.
fn get_samples_from_kraken_ohlcvt(input: &PathBuf) -> Result<Vec<Sample>> {
    let file =
        File::open(input).with_context(|| format!("failed to open input file: {:?}", input))?;

    let mut rdr = ReaderBuilder::new().has_headers(false).from_reader(file);

    let mut samples: Vec<Sample> = Vec::new();

    for result in rdr.deserialize::<KrakenOhlcvtRow>() {
        let row: KrakenOhlcvtRow =
            result.with_context(|| "failed to deserialize Kraken CSV row")?;
        let ts = Utc
            .timestamp_opt(row.timestamp, 0)
            .single()
            .with_context(|| format!("failed to parse unix timestamp: {}", row.timestamp))?;
        samples.push(Sample {
            ts,
            price: row.close,
        });
    }
    Ok(samples)
}

/// Resample raw samples into fixed-size buckets (1h, 2h, 4h, ...),
/// keeping the *last* price available in each bucket.
/// - Bucket alignment is to Unix epoch (1970-01-01T00:00:00Z), so 4h buckets start at 00:00, 04:00, 08:00, ...
/// - The output Sample.ts is the timestamp of the last observation in that bucket (not the bucket start).
fn resample_to_close(samples: &[Sample], step: Duration) -> Vec<Sample> {
    assert!(step > Duration::zero(), "step must be positive");
    let step_secs = step.num_seconds();
    assert!(step_secs > 0, "step is too small (must be >= 1 second)");

    let mut buckets: BTreeMap<DateTime<Utc>, Sample> = BTreeMap::new();

    for s in samples {
        let t = s.ts.timestamp();
        let bucket_start_secs = t.div_euclid(step_secs) * step_secs;

        let bucket_start = Utc
            .timestamp_opt(bucket_start_secs, 0)
            .single()
            .expect("valid bucket start");

        buckets
            .entry(bucket_start)
            .and_modify(|prev| {
                // Keep the latest observation within the bucket
                if s.ts > prev.ts {
                    *prev = Sample {
                        ts: s.ts,
                        price: s.price,
                    };
                }
            })
            .or_insert_with(|| Sample {
                ts: s.ts,
                price: s.price,
            });
    }

    buckets.into_values().collect()
}

/// Convenience wrapper for 1h / 2h / 4h / ...
pub fn resample_to_n_hours(samples: &[Sample], hours: i64) -> Vec<Sample> {
    assert!(hours > 0, "hours must be >= 1");
    resample_to_close(samples, Duration::hours(hours))
}

pub fn resample_to_hourly(samples: &[Sample]) -> Vec<Sample> {
    resample_to_n_hours(samples, 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{TimeZone, Utc};

    fn sample(y: i32, m: u32, d: u32, h: u32, min: u32, s: u32, price: f64) -> Sample {
        let ts = Utc
            .with_ymd_and_hms(y, m, d, h, min, s)
            .single()
            .expect("valid datetime");
        Sample { ts, price }
    }

    #[test]
    fn test_resample_to_hourly_empty_input_returns_empty_vec() {
        let out = resample_to_hourly(&[]);
        assert!(out.is_empty());
    }

    #[test]
    fn test_resample_to_hourly_single_sample_is_preserved() {
        let s = sample(2025, 11, 28, 10, 15, 0, 100.0);
        let out = resample_to_hourly(&[s.clone()]);

        assert_eq!(out.len(), 1);
        assert_eq!(out[0].ts, s.ts);
        assert_eq!(out[0].price, s.price);
    }

    #[test]
    fn test_resample_to_hourly_multiple_samples_in_same_hour_keep_last_price_and_timestamp() {
        // All in the 10:00–10:59 hour
        let s1 = sample(2025, 11, 28, 10, 05, 00, 100.0);
        let s2 = sample(2025, 11, 28, 10, 30, 00, 101.0);
        let s3 = sample(2025, 11, 28, 10, 59, 59, 102.0);

        let samples = vec![s1, s2, s3.clone()];
        let out = resample_to_hourly(&samples);

        assert_eq!(out.len(), 1);
        assert_eq!(out[0].ts, s3.ts); // original timestamp of last tick in that hour
        assert_eq!(out[0].price, 102.0); // close price
    }

    #[test]
    fn test_resample_to_hourly_multiple_hours_keep_last_sample_per_hour_and_order_by_hour() {
        // Hour 10
        let h10_early = sample(2025, 11, 28, 10, 05, 00, 100.0);
        let h10_last = sample(2025, 11, 28, 10, 55, 00, 101.0);

        // Hour 11
        let h11_early = sample(2025, 11, 28, 11, 00, 00, 200.0);
        let h11_last = sample(2025, 11, 28, 11, 45, 00, 201.0);

        let samples = vec![h10_early, h10_last.clone(), h11_early, h11_last.clone()];
        let out = resample_to_hourly(&samples);

        assert_eq!(out.len(), 2);

        // First bucket: 10:00–10:59, last tick in that hour
        assert_eq!(out[0].ts, h10_last.ts);
        assert_eq!(out[0].price, 101.0);

        // Second bucket: 11:00–11:59, last tick in that hour
        assert_eq!(out[1].ts, h11_last.ts);
        assert_eq!(out[1].price, 201.0);
    }

    #[test]
    fn test_resample_to_hourly_amples_exactly_on_hour_boundary_form_separate_buckets() {
        // 10:00:00 and 10:30 in same hour
        let h10_start = sample(2025, 11, 28, 10, 00, 00, 100.0);
        let h10_mid = sample(2025, 11, 28, 10, 30, 00, 101.0);

        // 11:00:00 – new (exact) hour, its own bucket
        let h11_start = sample(2025, 11, 28, 11, 00, 00, 200.0);

        let samples = vec![h10_start, h10_mid.clone(), h11_start.clone()];
        let out = resample_to_hourly(&samples);

        assert_eq!(out.len(), 2);

        // Hour 10: we should keep the last sample in that hour (10:30)
        assert_eq!(out[0].ts, h10_mid.ts);
        assert_eq!(out[0].price, 101.0);

        // Hour 11: only one sample, kept as-is
        assert_eq!(out[1].ts, h11_start.ts);
        assert_eq!(out[1].price, 200.0);
    }

    #[test]
    fn test_resample_to_n_hours() {
        let s1 = sample(2025, 11, 28, 10, 05, 00, 100.0);
        let s2 = sample(2025, 11, 28, 10, 30, 00, 101.0);
        let s3 = sample(2025, 11, 28, 10, 59, 59, 103.0);
        let s4 = sample(2025, 11, 28, 11, 59, 59, 104.0);
        let s5 = sample(2025, 11, 28, 13, 59, 59, 102.0);
        let s6 = sample(2025, 11, 28, 15, 59, 59, 102.0);

        let samples = vec![s1, s2, s3, s4.clone(), s5, s6];
        let out = resample_to_n_hours(&samples, 2);

        assert_eq!(out.len(), 3);
        assert_eq!(out[0].ts, s4.ts); // original timestamp of last tick in that hour
        assert_eq!(out[0].price, 104.0); // close price
    }

    #[test]
    fn test_get_samples_from_kraken_ohlcvt() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        // Create a temporary CSV file with Kraken OHLCVT format
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "1623942000,42.0,42.0,33.7,33.95,3293.36201464,214").unwrap();
        writeln!(file, "1623945600,33.9,33.92,33.3,33.37,2438.65345165,193").unwrap();
        writeln!(file, "1623949200,33.44,33.44,32.2,32.22,1586.84243114,142").unwrap();

        let path = file.path().to_path_buf();
        let samples = get_samples_from_kraken_ohlcvt(&path).unwrap();

        assert_eq!(samples.len(), 3);

        // First sample: timestamp 1623942000 = 2021-06-17 13:00:00 UTC, close = 33.95
        assert_eq!(samples[0].ts.timestamp(), 1623942000);
        assert!((samples[0].price - 33.95).abs() < 0.001);

        // Second sample: timestamp 1623945600, close = 33.37
        assert_eq!(samples[1].ts.timestamp(), 1623945600);
        assert!((samples[1].price - 33.37).abs() < 0.001);

        // Third sample: timestamp 1623949200, close = 32.22
        assert_eq!(samples[2].ts.timestamp(), 1623949200);
        assert!((samples[2].price - 32.22).abs() < 0.001);
    }
}
