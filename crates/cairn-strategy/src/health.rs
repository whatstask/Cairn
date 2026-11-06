//! The health check: replay a team's history before anything goes live.
//!
//! Given daily send and outcome counts (from a CRM export and sender stats),
//! it finds where performance changed and when a sequential detector would
//! have flagged it, splits the change across outcome events (bounces versus
//! replies versus meetings), and says honestly whether the team's volume can
//! support live testing at all.

use std::collections::BTreeMap;

use chrono::NaiveDate;
use serde::{Deserialize, Serialize};

use crate::event::Shift;
use crate::stats::{Counts, glr_change, kl_bernoulli, samples_for_lift};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DailyRecord {
    pub date: NaiveDate,
    /// Actions taken that day (emails sent, calls made).
    pub sends: u64,
    /// Outcome counts by event, attributed to the day of the send.
    #[serde(default)]
    pub events: BTreeMap<String, u64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HistoricalChange {
    /// First day of the new level.
    pub began: NaiveDate,
    /// Day a sequential detector would have raised the alarm.
    pub detected: NaiveDate,
    pub before: f64,
    pub after: f64,
    /// Every outcome's shift around the change, largest first.
    pub shifts: Vec<Shift>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Power {
    pub weekly_volume: f64,
    pub base_rate: f64,
    /// Weeks for the detector to catch a sustained relative drop of 50% / 25%.
    pub weeks_to_detect_half_drop: f64,
    pub weeks_to_detect_quarter_drop: f64,
    /// Outcomes per strategy, and weeks with two strategies splitting the
    /// volume, to prove a +20% relative lift.
    pub outcomes_for_20pct_lift: f64,
    pub weeks_for_20pct_lift: f64,
    /// Proving a +20% lift fits in a quarter.
    pub testing_feasible: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HealthReport {
    pub days: usize,
    pub success_event: String,
    pub changes: Vec<HistoricalChange>,
    pub power: Power,
    pub summary: String,
}

/// Replay history with the GLR change detector (false-alarm level `delta`).
pub fn health_check(records: &[DailyRecord], success_event: &str, delta: f64) -> HealthReport {
    let mut records = records.to_vec();
    records.sort_by_key(|r| r.date);
    let bucket = |r: &DailyRecord, ev: &str| Counts::new(r.sends as f64, *r.events.get(ev).unwrap_or(&0) as f64);
    let buckets: Vec<Counts> = records.iter().map(|r| bucket(r, success_event)).collect();
    let total = buckets.iter().fold(Counts::default(), |mut a, b| {
        a.add(*b);
        a
    });
    let base_rate = total.rate().unwrap_or(0.0);
    let min_side = 30.0;

    let mut changes = Vec::new();
    let mut start = 0;
    let mut day = start + 1;
    while day < buckets.len() {
        if let Some(c) = glr_change(&buckets[start..=day], delta, min_side) {
            let began = start + c.split;
            let events: Vec<String> = {
                let mut names: Vec<String> = records.iter().flat_map(|r| r.events.keys().cloned()).collect();
                names.sort();
                names.dedup();
                names
            };
            let mut shifts: Vec<Shift> = events
                .iter()
                .filter_map(|ev| {
                    let sum = |rs: &[DailyRecord]| {
                        rs.iter().fold(Counts::default(), |mut a, r| {
                            a.add(bucket(r, ev));
                            a
                        })
                    };
                    let (b, a) = (sum(&records[start..began]), sum(&records[began..=day]));
                    let (pb, pa) = (b.rate()?, a.rate()?);
                    let pooled = (b.successes + a.successes) / (b.trials + a.trials);
                    let se = (pooled * (1.0 - pooled) * (1.0 / b.trials + 1.0 / a.trials)).sqrt();
                    Some(Shift {
                        event: ev.clone(),
                        before: pb,
                        after: pa,
                        z: if se > 0.0 { (pa - pb) / se } else { 0.0 },
                    })
                })
                .collect();
            shifts.sort_by(|x, y| y.z.abs().total_cmp(&x.z.abs()));
            changes.push(HistoricalChange {
                began: records[began].date,
                detected: records[day].date,
                before: c.before,
                after: c.after,
                shifts,
            });
            start = began;
            day = began.max(day) + 1;
        } else {
            day += 1;
        }
    }

    let days = records.len();
    let weekly_volume = if days == 0 { 0.0 } else { total.trials / days as f64 * 7.0 };
    let weeks_to_detect = |drop: f64| -> f64 {
        if base_rate <= 0.0 || weekly_volume <= 0.0 {
            return f64::INFINITY;
        }
        let kl = kl_bernoulli(base_rate * (1.0 - drop), base_rate);
        // n * kl must clear ln(N^1.5 / delta); solve by fixed-point iteration.
        let mut n = 100.0;
        for _ in 0..50 {
            n = ((total.trials + n).powf(1.5) / delta).ln() / kl;
        }
        n / weekly_volume
    };
    let outcomes_for_lift = if base_rate > 0.0 { samples_for_lift(base_rate, 0.2 * base_rate) } else { f64::INFINITY };
    let weeks_for_lift = if weekly_volume > 0.0 { 2.0 * outcomes_for_lift / weekly_volume } else { f64::INFINITY };
    let power = Power {
        weekly_volume,
        base_rate,
        weeks_to_detect_half_drop: weeks_to_detect(0.5),
        weeks_to_detect_quarter_drop: weeks_to_detect(0.25),
        outcomes_for_20pct_lift: outcomes_for_lift,
        weeks_for_20pct_lift: weeks_for_lift,
        testing_feasible: weeks_for_lift <= 13.0,
    };

    let mut summary = match changes.last() {
        Some(c) => {
            let lead = c
                .shifts
                .first()
                .map(|s| {
                    format!(
                        " The biggest shift was in `{}` ({:.2}% → {:.2}%).",
                        s.event,
                        s.before * 100.0,
                        s.after * 100.0
                    )
                })
                .unwrap_or_default();
            format!(
                "{} change(s) found. The latest began {} ({:.2}% → {:.2}% {success_event}) and would have been flagged on {}.{lead}",
                changes.len(),
                c.began,
                c.before * 100.0,
                c.after * 100.0,
                c.detected
            )
        }
        None => format!("No significant change in {success_event} over {days} days."),
    };
    summary.push_str(&if power.testing_feasible {
        format!(" At ~{:.0} sends/week, proving a +20% lift takes about {:.0} weeks: live testing is feasible.", weekly_volume, weeks_for_lift)
    } else {
        format!(
            " At ~{:.0} sends/week, proving a +20% lift would take about {:.0} weeks, so use monitoring and guardrails, and only test large changes.",
            weekly_volume, weeks_for_lift
        )
    });
    HealthReport { days, success_event: success_event.into(), changes, power, summary }
}
