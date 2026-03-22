#!/usr/bin/env python3
import csv
import math
import os
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from collections import deque
from dataclasses import dataclass


@dataclass
class Config:
    # Neutrales Beispielband
    freq_start_mhz: float = 433.050
    freq_end_mhz: float = 434.790
    bin_size_hz: int = 10000

    # rtl_power
    integration_seconds: int = 1
    gain_db: int = 20

    # Auto-Kalibrierung
    calibration_seconds: int = 30

    # Laufzeitlogik
    baseline_alpha: float = 0.02
    min_alert_seconds: float = 2.0
    cooldown_seconds: float = 8.0

    # Sicherheitsgrenzen für Auto-Tuning
    min_noise_margin_db: float = 3.0
    max_noise_margin_db: float = 12.0
    min_alert_on_score: float = 20.0
    max_alert_on_score: float = 120.0

    log_file: str = "activity_log.csv"
    print_debug: bool = True


class AdaptiveBandActivityMonitor:
    STATE_CALIBRATING = "CALIBRATING"
    STATE_IDLE = "IDLE"
    STATE_WATCH = "WATCH"
    STATE_ALERT = "ALERT"
    STATE_COOLDOWN = "COOLDOWN"

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.state = self.STATE_CALIBRATING

        self.baseline = None
        self.noise_margin_db = 5.0
        self.alert_on_score = 50.0
        self.alert_off_score = 28.0
        self.min_active_bins = 2

        self.watch_start = None
        self.cooldown_start = None
        self.recent_scores = deque(maxlen=8)

        self.calibration_started_at = time.time()
        self.calibration_samples = []

        self._ensure_log_header()

    def _ensure_log_header(self):
        if not os.path.exists(self.cfg.log_file):
            with open(self.cfg.log_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp",
                    "state",
                    "score",
                    "baseline_db",
                    "noise_margin_db",
                    "alert_on_score",
                    "alert_off_score",
                    "min_active_bins",
                    "peak_excess_db",
                    "avg_excess_db",
                    "active_bins",
                    "max_power_db",
                    "mean_power_db",
                    "std_power_db",
                ])

    def _clamp(self, value, low, high):
        return max(low, min(high, value))

    def _percentile(self, values, p):
        if not values:
            return 0.0
        values = sorted(values)
        if len(values) == 1:
            return values[0]
        idx = (len(values) - 1) * p
        lo = int(math.floor(idx))
        hi = int(math.ceil(idx))
        if lo == hi:
            return values[lo]
        frac = idx - lo
        return values[lo] * (1 - frac) + values[hi] * frac

    def _safe_stdev(self, values):
        if len(values) < 2:
            return 0.0
        return statistics.pstdev(values)

    def parse_metrics_from_power_values(self, power_values_db):
        if not power_values_db:
            return None

        mean_power_db = sum(power_values_db) / len(power_values_db)
        max_power_db = max(power_values_db)
        std_power_db = self._safe_stdev(power_values_db)

        return {
            "mean_power_db": mean_power_db,
            "max_power_db": max_power_db,
            "std_power_db": std_power_db,
            "power_values_db": power_values_db,
        }

    def update_baseline(self, mean_power_db):
        if self.baseline is None:
            self.baseline = mean_power_db
            return
        alpha = self.cfg.baseline_alpha
        self.baseline = (1 - alpha) * self.baseline + alpha * mean_power_db

    def collect_calibration_sample(self, raw_metrics):
        if raw_metrics is None:
            return

        power_values = raw_metrics["power_values_db"]
        mean_power = raw_metrics["mean_power_db"]
        max_power = raw_metrics["max_power_db"]
        std_power = raw_metrics["std_power_db"]

        self.calibration_samples.append({
            "mean_power_db": mean_power,
            "max_power_db": max_power,
            "std_power_db": std_power,
            "power_values_db": power_values,
        })

    def calibration_finished(self):
        return (time.time() - self.calibration_started_at) >= self.cfg.calibration_seconds

    def finalize_calibration(self):
        if not self.calibration_samples:
            self.baseline = -90.0
            self.noise_margin_db = 5.0
            self.alert_on_score = 50.0
            self.alert_off_score = 30.0
            self.min_active_bins = 2
            self.state = self.STATE_IDLE
            return

        mean_values = [s["mean_power_db"] for s in self.calibration_samples]
        max_values = [s["max_power_db"] for s in self.calibration_samples]
        std_values = [s["std_power_db"] for s in self.calibration_samples]

        # Baseline = Median der mittleren Leistung
        baseline = statistics.median(mean_values)

        # Typische Schwankung der Mittelwerte
        mean_std = self._safe_stdev(mean_values)
        typical_bin_std = statistics.median(std_values) if std_values else 0.0

        # Noise Margin dynamisch:
        # größer bei unruhiger Umgebung, aber in sinnvollen Grenzen
        noise_margin = baseline
        noise_margin = max(
            self.cfg.min_noise_margin_db,
            2.5 * mean_std + 1.5 * typical_bin_std
        )
        noise_margin = self._clamp(
            noise_margin,
            self.cfg.min_noise_margin_db,
            self.cfg.max_noise_margin_db
        )

        # Jetzt simulieren wir für alle Kalibriersamples den Aktivitäts-Score
        temp_scores = []
        temp_active_bins = []

        for s in self.calibration_samples:
            above = [p - baseline for p in s["power_values_db"]]
            positive = [x for x in above if x > noise_margin]
            active_bins = len(positive)
            peak_excess_db = max(above) if above else 0.0
            avg_excess_db = (sum(positive) / len(positive)) if positive else 0.0

            score = (
                active_bins * 6.0
                + max(0.0, peak_excess_db - noise_margin) * 2.0
                + avg_excess_db * 3.0
            )
            temp_scores.append(score)
            temp_active_bins.append(active_bins)

        # Alarm-Schwelle: deutlich über dem typischen Kalibrierungsniveau
        p95_score = self._percentile(temp_scores, 0.95)
        p99_score = self._percentile(temp_scores, 0.99)
        score_std = self._safe_stdev(temp_scores)

        alert_on_score = max(
            p99_score + 1.5 * score_std,
            p95_score + 2.0 * score_std,
            self.cfg.min_alert_on_score
        )
        alert_on_score = self._clamp(
            alert_on_score,
            self.cfg.min_alert_on_score,
            self.cfg.max_alert_on_score
        )

        # Hysterese
        alert_off_score = max(10.0, alert_on_score * 0.55)

        # Mindestzahl aktiver Bins:
        # leicht über normalem Kalibrierungsniveau
        p95_active = self._percentile(temp_active_bins, 0.95)
        min_active_bins = max(1, int(round(p95_active + 1)))

        self.baseline = baseline
        self.noise_margin_db = noise_margin
        self.alert_on_score = alert_on_score
        self.alert_off_score = alert_off_score
        self.min_active_bins = min_active_bins
        self.state = self.STATE_IDLE

        print("\nKalibrierung abgeschlossen:")
        print(f"  baseline         = {self.baseline:.2f} dB")
        print(f"  noise_margin_db  = {self.noise_margin_db:.2f} dB")
        print(f"  alert_on_score   = {self.alert_on_score:.2f}")
        print(f"  alert_off_score  = {self.alert_off_score:.2f}")
        print(f"  min_active_bins  = {self.min_active_bins}\n")

    def compute_runtime_metrics(self, raw_metrics):
        power_values_db = raw_metrics["power_values_db"]
        mean_power_db = raw_metrics["mean_power_db"]
        max_power_db = raw_metrics["max_power_db"]
        std_power_db = raw_metrics["std_power_db"]

        self.update_baseline(mean_power_db)

        above = [p - self.baseline for p in power_values_db]
        positive = [x for x in above if x > self.noise_margin_db]

        active_bins = len(positive)
        peak_excess_db = max(above) if above else 0.0
        avg_excess_db = (sum(positive) / len(positive)) if positive else 0.0

        score = (
            active_bins * 6.0
            + max(0.0, peak_excess_db - self.noise_margin_db) * 2.0
            + avg_excess_db * 3.0
        )

        return {
            "score": score,
            "peak_excess_db": peak_excess_db,
            "avg_excess_db": avg_excess_db,
            "active_bins": active_bins,
            "max_power_db": max_power_db,
            "mean_power_db": mean_power_db,
            "std_power_db": std_power_db,
        }

    def step(self, runtime_metrics):
        now = time.time()

        if self.state == self.STATE_CALIBRATING:
            return False, 0.0

        score = runtime_metrics["score"]
        active_bins = runtime_metrics["active_bins"]

        self.recent_scores.append(score)
        smoothed_score = sum(self.recent_scores) / len(self.recent_scores)

        triggered = False

        if self.state == self.STATE_IDLE:
            if (
                smoothed_score >= self.alert_on_score
                and active_bins >= self.min_active_bins
            ):
                self.state = self.STATE_WATCH
                self.watch_start = now

        elif self.state == self.STATE_WATCH:
            if (
                smoothed_score >= self.alert_on_score
                and active_bins >= self.min_active_bins
            ):
                if self.watch_start and (now - self.watch_start) >= self.cfg.min_alert_seconds:
                    self.state = self.STATE_ALERT
                    triggered = True
            else:
                self.state = self.STATE_IDLE
                self.watch_start = None

        elif self.state == self.STATE_ALERT:
            if smoothed_score <= self.alert_off_score:
                self.state = self.STATE_COOLDOWN
                self.cooldown_start = now

        elif self.state == self.STATE_COOLDOWN:
            if self.cooldown_start and (now - self.cooldown_start) >= self.cfg.cooldown_seconds:
                self.state = self.STATE_IDLE
                self.cooldown_start = None

        return triggered, smoothed_score

    def handle_alert(self):
        print("ALERT: Auffällige Band-Aktivität erkannt")

    def log(self, metrics, smoothed_score):
        with open(self.cfg.log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                int(time.time()),
                self.state,
                round(smoothed_score, 2),
                round(self.baseline if self.baseline is not None else -200.0, 2),
                round(self.noise_margin_db, 2),
                round(self.alert_on_score, 2),
                round(self.alert_off_score, 2),
                self.min_active_bins,
                round(metrics.get("peak_excess_db", 0.0), 2),
                round(metrics.get("avg_excess_db", 0.0), 2),
                metrics.get("active_bins", 0),
                round(metrics.get("max_power_db", -200.0), 2),
                round(metrics.get("mean_power_db", -200.0), 2),
                round(metrics.get("std_power_db", 0.0), 2),
            ])


def parse_rtl_power_line(line: str):
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 7:
        return None
    try:
        power_values = [float(x) for x in parts[6:] if x]
    except ValueError:
        return None
    return power_values


def build_rtl_power_command(cfg: Config, output_csv: str):
    return [
        "rtl_power",
        "-f", f"{cfg.freq_start_mhz}M:{cfg.freq_end_mhz}M:{cfg.bin_size_hz}",
        "-i", str(cfg.integration_seconds),
        "-g", str(cfg.gain_db),
        output_csv,
    ]


def follow_file(path: str):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        f.seek(0, os.SEEK_END)
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.1)
                continue
            yield line.rstrip("\n")


def main():
    cfg = Config()
    monitor = AdaptiveBandActivityMonitor(cfg)

    with tempfile.NamedTemporaryFile(prefix="rtl_power_", suffix=".csv", delete=False) as tmp:
        temp_csv = tmp.name

    cmd = build_rtl_power_command(cfg, temp_csv)
    print("Starte rtl_power:")
    print(" ".join(cmd))
    print(f"Kalibriere für {cfg.calibration_seconds} Sekunden...\n")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )

    def shutdown(*_args):
        print("\nBeende...")
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        if os.path.exists(temp_csv):
            os.remove(temp_csv)
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    time.sleep(2)

    try:
        for line in follow_file(temp_csv):
            power_values = parse_rtl_power_line(line)
            if power_values is None:
                continue

            raw_metrics = monitor.parse_metrics_from_power_values(power_values)
            if raw_metrics is None:
                continue

            if monitor.state == monitor.STATE_CALIBRATING:
                monitor.collect_calibration_sample(raw_metrics)

                elapsed = time.time() - monitor.calibration_started_at
                if cfg.print_debug:
                    print(f"Kalibrierung: {elapsed:5.1f}/{cfg.calibration_seconds}s", end="\r")

                if monitor.calibration_finished():
                    monitor.finalize_calibration()
                continue

            runtime_metrics = monitor.compute_runtime_metrics(raw_metrics)
            triggered, smoothed_score = monitor.step(runtime_metrics)
            monitor.log(runtime_metrics, smoothed_score)

            if cfg.print_debug:
                print(
                    f"state={monitor.state:10s} "
                    f"score={smoothed_score:6.1f} "
                    f"baseline={monitor.baseline:7.1f} dB "
                    f"noise={monitor.noise_margin_db:4.1f} dB "
                    f"bins={runtime_metrics['active_bins']:2d} "
                    f"peak={runtime_metrics['peak_excess_db']:5.1f} dB"
                )

            if triggered:
                monitor.handle_alert()

    finally:
        shutdown()


if __name__ == "__main__":
    main()