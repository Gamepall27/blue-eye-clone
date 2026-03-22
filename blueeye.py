#!/usr/bin/env python3
import csv
import os
import signal
import subprocess
import sys
import tempfile
import time
from collections import deque
from dataclasses import dataclass


@dataclass
class Config:
    # Neutrales Beispielband. Bei Bedarf selbst anpassen.
    freq_start_mhz: float = 433.050
    freq_end_mhz: float = 434.790
    bin_size_hz: int = 10000

    # rtl_power Einstellungen
    integration_seconds: int = 1
    gain_db: int = 20

    # Logik
    baseline_alpha: float = 0.03
    noise_margin_db: float = 5.0
    alert_on_score: float = 50.0
    alert_off_score: float = 28.0
    min_alert_seconds: float = 2.0
    cooldown_seconds: float = 8.0
    min_active_bins: int = 2

    # Anzeige / Logging
    log_file: str = "activity_log.csv"
    print_debug: bool = True


class BandActivityMonitor:
    STATE_IDLE = "IDLE"
    STATE_WATCH = "WATCH"
    STATE_ALERT = "ALERT"
    STATE_COOLDOWN = "COOLDOWN"

    def __init__(self, config: Config):
        self.cfg = config
        self.state = self.STATE_IDLE
        self.baseline = None
        self.watch_start = None
        self.cooldown_start = None
        self.recent_scores = deque(maxlen=10)

        self._ensure_log_header()

    def _ensure_log_header(self) -> None:
        if not os.path.exists(self.cfg.log_file):
            with open(self.cfg.log_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp",
                    "state",
                    "score",
                    "baseline_db",
                    "peak_excess_db",
                    "avg_excess_db",
                    "active_bins",
                    "max_power_db",
                ])

    def update_baseline(self, mean_power_db: float) -> None:
        if self.baseline is None:
            self.baseline = mean_power_db
            return

        alpha = self.cfg.baseline_alpha
        self.baseline = (1.0 - alpha) * self.baseline + alpha * mean_power_db

    def compute_metrics(self, power_values_db):
        if not power_values_db:
            return {
                "score": 0.0,
                "peak_excess_db": 0.0,
                "avg_excess_db": 0.0,
                "active_bins": 0,
                "max_power_db": -200.0,
                "mean_power_db": -200.0,
            }

        mean_power_db = sum(power_values_db) / len(power_values_db)
        self.update_baseline(mean_power_db)

        above = [p - self.baseline for p in power_values_db]
        positive = [x for x in above if x > self.cfg.noise_margin_db]

        active_bins = len(positive)
        peak_excess_db = max(above) if above else 0.0
        avg_excess_db = (sum(positive) / len(positive)) if positive else 0.0
        max_power_db = max(power_values_db)

        # Aktivitäts-Score:
        # - aktive Bins zählen
        # - Spitzen bewerten
        # - durchschnittlichen Überschuss bewerten
        score = (
            active_bins * 6.0
            + max(0.0, peak_excess_db - self.cfg.noise_margin_db) * 2.0
            + avg_excess_db * 3.0
        )

        return {
            "score": score,
            "peak_excess_db": peak_excess_db,
            "avg_excess_db": avg_excess_db,
            "active_bins": active_bins,
            "max_power_db": max_power_db,
            "mean_power_db": mean_power_db,
        }

    def step(self, metrics):
        now = time.time()
        score = metrics["score"]
        active_bins = metrics["active_bins"]

        self.recent_scores.append(score)
        smoothed_score = sum(self.recent_scores) / len(self.recent_scores)

        triggered = False

        if self.state == self.STATE_IDLE:
            if (
                smoothed_score >= self.cfg.alert_on_score
                and active_bins >= self.cfg.min_active_bins
            ):
                self.state = self.STATE_WATCH
                self.watch_start = now

        elif self.state == self.STATE_WATCH:
            if (
                smoothed_score >= self.cfg.alert_on_score
                and active_bins >= self.cfg.min_active_bins
            ):
                if self.watch_start and (now - self.watch_start) >= self.cfg.min_alert_seconds:
                    self.state = self.STATE_ALERT
                    triggered = True
            else:
                self.state = self.STATE_IDLE
                self.watch_start = None

        elif self.state == self.STATE_ALERT:
            if smoothed_score <= self.cfg.alert_off_score:
                self.state = self.STATE_COOLDOWN
                self.cooldown_start = now

        elif self.state == self.STATE_COOLDOWN:
            if self.cooldown_start and (now - self.cooldown_start) >= self.cfg.cooldown_seconds:
                self.state = self.STATE_IDLE
                self.cooldown_start = None

        return triggered, smoothed_score

    def log(self, metrics, smoothed_score):
        with open(self.cfg.log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                int(time.time()),
                self.state,
                round(smoothed_score, 2),
                round(self.baseline if self.baseline is not None else -200.0, 2),
                round(metrics["peak_excess_db"], 2),
                round(metrics["avg_excess_db"], 2),
                metrics["active_bins"],
                round(metrics["max_power_db"], 2),
            ])

    def handle_alert(self):
        # Später hier LED, Buzzer oder GPIO ergänzen.
        print("ALERT: Auffällige Band-Aktivität erkannt")


def parse_rtl_power_line(line: str):
    """
    Erwartetes Format von rtl_power (vereinfacht):
    date, time, hz_low, hz_high, hz_step, samples, db1, db2, db3, ...

    Gibt Liste der dB-Werte zurück.
    """
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
    """
    Einfacher 'tail -f' in Python.
    """
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
    monitor = BandActivityMonitor(cfg)

    with tempfile.NamedTemporaryFile(prefix="rtl_power_", suffix=".csv", delete=False) as tmp:
        temp_csv = tmp.name

    cmd = build_rtl_power_command(cfg, temp_csv)
    print("Starte rtl_power:")
    print(" ".join(cmd))

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

    # rtl_power braucht kurz zum Starten
    time.sleep(2)

    try:
        for line in follow_file(temp_csv):
            power_values = parse_rtl_power_line(line)
            if power_values is None:
                continue

            metrics = monitor.compute_metrics(power_values)
            triggered, smoothed_score = monitor.step(metrics)
            monitor.log(metrics, smoothed_score)

            if cfg.print_debug:
                print(
                    f"state={monitor.state:8s} "
                    f"score={smoothed_score:6.1f} "
                    f"baseline={monitor.baseline:7.1f} dB "
                    f"active_bins={metrics['active_bins']:2d} "
                    f"peak={metrics['peak_excess_db']:5.1f} dB"
                )

            if triggered:
                monitor.handle_alert()

    finally:
        shutdown()


if __name__ == "__main__":
    main()