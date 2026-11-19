#!/usr/bin/env python3
import csv
import argparse
import datetime
import math
import os
import random
import select
import signal
import shutil
import statistics
import subprocess
import sys
import tempfile
import termios
import time
import tty
from collections import deque
from dataclasses import dataclass


@dataclass
class Config:
    # Ein einzeln überwachtes Frequenzfenster
    freq_start_mhz: float = 380.000
    freq_end_mhz: float = 385.000
    band_label: str = ""
    bin_size_hz: int = 10000

    # rtl_power
    integration_seconds: int = 1
    gain_db: int = 20
    data_timeout_seconds: float = 30.0

    # Auto-Kalibrierung
    calibration_seconds: int = 30

    # Kontinuierliche Anpassung je rtl_power-Frequenz-Hop (in Wiederholungen)
    baseline_window_size: int = 15
    noise_window_size: int = 31
    threshold_window_size: int = 60
    min_alert_seconds: float = 2.0
    cooldown_seconds: float = 8.0

    # Sicherheitsgrenzen für Auto-Tuning
    min_noise_margin_db: float = 3.0
    max_noise_margin_db: float = 12.0
    min_alert_on_score: float = 20.0
    max_alert_on_score: float = 120.0

    log_file: str = "activity_log.csv"
    adaptation_log_file: str = "adaptive_calibration_log.csv"


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

        self.calibration_started_at = time.monotonic()
        self.calibration_samples = {}
        self.hop_models = {}
        self.last_record_timestamp = {}

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
        adaptation_header = [
            "timestamp", "band", "hop_start_mhz", "hop_end_mhz", "hop_step_hz",
            "baseline_db", "baseline_change_db", "noise_margin_db", "noise_margin_change_db",
            "alert_on_score", "alert_on_change", "alert_off_score", "alert_off_change",
            "min_active_bins", "min_active_change", "accepted_as_background",
        ]
        path = self.cfg.adaptation_log_file
        if os.path.exists(path):
            try:
                with open(path, "r", newline="") as f:
                    existing_header = next(csv.reader(f), [])
            except OSError:
                existing_header = []
            if existing_header != adaptation_header:
                root, extension = os.path.splitext(path)
                version = 2
                while True:
                    candidate = f"{root}_v{version}{extension}"
                    if not os.path.exists(candidate):
                        path = candidate
                        break
                    try:
                        with open(candidate, "r", newline="") as f:
                            candidate_header = next(csv.reader(f), [])
                    except OSError:
                        candidate_header = []
                    if candidate_header == adaptation_header:
                        path = candidate
                        break
                    version += 1
                self.cfg.adaptation_log_file = path
                print(f"Vorhandenes Anpassungs-CSV hat ein älteres Schema; neue Einträge gehen nach {os.path.abspath(path)}.", flush=True)
        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(adaptation_header)

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
        if not power_values_db or not all(math.isfinite(value) for value in power_values_db):
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

    def _row_center_and_spread(self, values):
        center = statistics.median(values)
        mad = statistics.median([abs(value - center) for value in values])
        return center, 1.4826 * mad

    def _seed_history(self, values, size, center):
        recent = list(values)[-size:]
        return deque([center] * max(0, size - len(recent)) + recent, maxlen=size)

    def _hop_key(self, first_hz, step_hz, count):
        return (int(round(first_hz)), int(round(step_hz)), int(count))

    def collect_calibration_sample(self, raw_metrics, hop_key):
        if raw_metrics is None:
            return
        values = raw_metrics["power_values_db"]
        center, spread = self._row_center_and_spread(values)
        self.calibration_samples.setdefault(hop_key, []).append({
            "center": center, "spread": spread, "values": values,
        })

    def calibration_finished(self):
        return (time.monotonic() - self.calibration_started_at) >= self.cfg.calibration_seconds

    def accept_record_timestamp(self, hop_key, observed_at):
        """Reject duplicate/late rtl_power rows and reset temporal state after a long gap."""
        if observed_at is None:
            return True
        previous = self.last_record_timestamp.get(hop_key)
        if previous is not None:
            elapsed = (observed_at - previous).total_seconds()
            if elapsed == 0:
                return False
            # rtl_power timestamps are local time. A daylight-saving or manual
            # clock rollback starts a new timestamp epoch rather than stale data.
            if elapsed < -300 or elapsed > self.cfg.data_timeout_seconds:
                model = self.hop_models.get(hop_key)
                if model is not None:
                    model["state"] = self.STATE_IDLE
                    model["watch_start"] = None
                    model["cooldown_start"] = None
                    model["recent_scores"].clear()
            elif elapsed < 0:
                return False
        self.last_record_timestamp[hop_key] = observed_at
        return True

    def finalize_calibration(self):
        if not self.calibration_samples:
            self.baseline = -90.0
            self.noise_margin_db = 5.0
            self.alert_on_score = 50.0
            self.alert_off_score = 30.0
            self.min_active_bins = 2
            self.state = self.STATE_IDLE
            return
        for hop_key, samples in self.calibration_samples.items():
            centers = [sample["center"] for sample in samples]
            spreads = [sample["spread"] for sample in samples]
            baseline = statistics.median(centers)
            noise_margin = self._noise_margin(spreads)
            scores, active_counts = self._calibration_scores(samples, baseline, noise_margin)
            alert_on, alert_off, min_active = self._alarm_thresholds(scores, active_counts)
            self.hop_models[hop_key] = {
                "baseline_window": self._seed_history(centers, self.cfg.baseline_window_size, baseline),
                "spread_window": self._seed_history(spreads, self.cfg.noise_window_size, statistics.median(spreads)),
                "background_scores": deque(scores[-self.cfg.threshold_window_size:], maxlen=self.cfg.threshold_window_size),
                "background_active": deque(active_counts[-self.cfg.threshold_window_size:], maxlen=self.cfg.threshold_window_size),
                "baseline": baseline, "noise_margin": noise_margin,
                "alert_on": alert_on, "alert_off": alert_off, "min_active": min_active,
                "state": self.STATE_IDLE, "watch_start": None, "cooldown_start": None,
                "recent_scores": deque(maxlen=8),
            }
        self._publish_model(next(iter(self.hop_models.values())))
        self.state = self.STATE_IDLE
        print(f"\nKalibrierung abgeschlossen [{self.cfg.band_label} MHz] · {len(self.hop_models)} Frequenz-Hops:")
        for hop_key, model in self.hop_models.items():
            print(f"  {self._hop_label(hop_key)}: Grundlinie {model['baseline']:.1f} dBFS, Rauschabstand {model['noise_margin']:.1f} dB, Alarmschwelle {model['alert_on']:.1f}")
        print("")

    def _hop_label(self, hop_key):
        first_hz, step_hz, count = hop_key
        last_hz = first_hz + step_hz * max(0, count - 1)
        return f"{first_hz / 1e6:.4f}–{last_hz / 1e6:.4f} MHz"

    def _noise_margin(self, spreads):
        typical_spread = statistics.median(spreads) if spreads else 0.0
        return self._clamp(max(self.cfg.min_noise_margin_db, 2.5 * typical_spread),
                           self.cfg.min_noise_margin_db, self.cfg.max_noise_margin_db)

    def _score_values(self, values, baseline, noise_margin):
        above = [value - baseline for value in values]
        positive = [value for value in above if value > noise_margin]
        active = len(positive)
        peak = max(above) if above else 0.0
        avg = sum(positive) / len(positive) if positive else 0.0
        score = active * 6.0 + max(0.0, peak - noise_margin) * 2.0 + avg * 3.0
        return score, active, peak, avg

    def _calibration_scores(self, samples, baseline, noise_margin):
        scores, active_counts = [], []
        for sample in samples:
            score, active, peak, _ = self._score_values(sample["values"], baseline, noise_margin)
            # Initial short or narrowband bursts are excluded from the noise reference.
            if active <= max(1, int(len(sample["values"]) * 0.10)) and peak <= 12.0:
                scores.append(score)
                active_counts.append(active)
        if not scores:
            return [0.0], [0]
        return scores, active_counts

    def _alarm_thresholds(self, scores, active_counts):
        p95 = self._percentile(scores, 0.95)
        p99 = self._percentile(scores, 0.99)
        spread = self._safe_stdev(scores)
        alert_on = self._clamp(max(p99 + 1.5 * spread, p95 + 2.0 * spread,
                                   self.cfg.min_alert_on_score),
                               self.cfg.min_alert_on_score, self.cfg.max_alert_on_score)
        alert_off = max(10.0, alert_on * 0.55)
        min_active = max(1, int(round(self._percentile(active_counts, 0.95) + 1)))
        return alert_on, alert_off, min_active

    def _publish_model(self, model):
        self.baseline = model["baseline"]
        self.noise_margin_db = model["noise_margin"]
        self.alert_on_score = model["alert_on"]
        self.alert_off_score = model["alert_off"]
        self.min_active_bins = model["min_active"]

    def _new_hop_model(self, values):
        center, spread = self._row_center_and_spread(values)
        baseline_window = self._seed_history([center], self.cfg.baseline_window_size, center)
        spread_window = self._seed_history([spread], self.cfg.noise_window_size, spread)
        return {
            "baseline_window": baseline_window, "spread_window": spread_window,
            "background_scores": deque(maxlen=self.cfg.threshold_window_size),
            "background_active": deque(maxlen=self.cfg.threshold_window_size),
            "baseline": center, "noise_margin": self._noise_margin([spread]),
            "alert_on": 50.0, "alert_off": 28.0, "min_active": 2,
            "state": self.STATE_IDLE, "watch_start": None, "cooldown_start": None,
            "recent_scores": deque(maxlen=8),
        }

    def compute_runtime_metrics(self, raw_metrics, hop_key):
        values = raw_metrics["power_values_db"]
        model = self.hop_models.get(hop_key)
        if model is None:
            model = self._new_hop_model(values)
            self.hop_models[hop_key] = model
        old_baseline, old_margin = model["baseline"], model["noise_margin"]
        center, spread = self._row_center_and_spread(values)
        model["baseline_window"].append(center)
        model["spread_window"].append(spread)
        model["baseline"] = statistics.median(model["baseline_window"])
        model["noise_margin"] = self._noise_margin(model["spread_window"])
        score, active, peak, avg = self._score_values(values, model["baseline"], model["noise_margin"])
        self._publish_model(model)
        return {
            "score": score, "peak_excess_db": peak, "avg_excess_db": avg,
            "active_bins": active, "max_power_db": raw_metrics["max_power_db"],
            "mean_power_db": raw_metrics["mean_power_db"], "std_power_db": raw_metrics["std_power_db"],
            "hop_key": hop_key, "hop_model": model, "baseline": model["baseline"],
            "baseline_change": model["baseline"] - old_baseline,
            "noise_margin": model["noise_margin"], "noise_change": model["noise_margin"] - old_margin,
            "power_values_db": values,
        }

    def step(self, runtime_metrics):
        now = time.monotonic()
        model = runtime_metrics["hop_model"]
        state = model["state"]
        score = runtime_metrics["score"]
        active_bins = runtime_metrics["active_bins"]
        model["recent_scores"].append(score)
        smoothed_score = sum(model["recent_scores"]) / len(model["recent_scores"])

        triggered = False

        threshold_before = model["alert_on"]
        threshold_off_before = model["alert_off"]
        min_active_before = model["min_active"]
        if state == self.STATE_IDLE:
            if (
                smoothed_score >= model["alert_on"]
                and active_bins >= model["min_active"]
            ):
                state = self.STATE_WATCH
                model["watch_start"] = now

        elif state == self.STATE_WATCH:
            if (
                smoothed_score >= model["alert_on"]
                and active_bins >= model["min_active"]
            ):
                if model["watch_start"] and (now - model["watch_start"]) >= self.cfg.min_alert_seconds:
                    state = self.STATE_ALERT
                    triggered = True
            else:
                state = self.STATE_IDLE
                model["watch_start"] = None

        elif state == self.STATE_ALERT:
            if smoothed_score <= model["alert_off"]:
                state = self.STATE_COOLDOWN
                model["cooldown_start"] = now

        elif state == self.STATE_COOLDOWN:
            if model["cooldown_start"] and (now - model["cooldown_start"]) >= self.cfg.cooldown_seconds:
                state = self.STATE_IDLE
                model["cooldown_start"] = None

        model["state"] = state
        accepted = (state in (self.STATE_IDLE, self.STATE_COOLDOWN)
                    and active_bins <= max(1, int(len(runtime_metrics["power_values_db"]) * 0.10))
                    and runtime_metrics["peak_excess_db"] <= 12.0)
        if accepted:
            model["background_scores"].append(score)
            model["background_active"].append(active_bins)
            if len(model["background_scores"]) >= 5:
                model["alert_on"], model["alert_off"], model["min_active"] = self._alarm_thresholds(
                    model["background_scores"], model["background_active"])
        runtime_metrics["alert_on_change"] = model["alert_on"] - threshold_before
        runtime_metrics["alert_off_change"] = model["alert_off"] - threshold_off_before
        runtime_metrics["min_active_change"] = model["min_active"] - min_active_before
        runtime_metrics["alert_on_score"] = model["alert_on"]
        runtime_metrics["alert_off_score"] = model["alert_off"]
        runtime_metrics["min_active_bins"] = model["min_active"]
        runtime_metrics["accepted_as_background"] = accepted
        self._publish_model(model)

        return triggered, smoothed_score, state

    def log(self, metrics, smoothed_score, state):
        hop = metrics["hop_key"]
        band_state = f"{state} [{self.cfg.band_label} hop {self._hop_label(hop)}]"
        with open(self.cfg.log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                int(time.time()),
                band_state,
                round(smoothed_score, 2),
                round(metrics["baseline"], 2),
                round(metrics["noise_margin"], 2),
                round(metrics["alert_on_score"], 2),
                round(metrics["alert_off_score"], 2),
                metrics["min_active_bins"],
                round(metrics.get("peak_excess_db", 0.0), 2),
                round(metrics.get("avg_excess_db", 0.0), 2),
                metrics.get("active_bins", 0),
                round(metrics.get("max_power_db", -200.0), 2),
                round(metrics.get("mean_power_db", -200.0), 2),
                round(metrics.get("std_power_db", 0.0), 2),
            ])
        with open(self.cfg.adaptation_log_file, "a", newline="") as f:
            writer = csv.writer(f)
            first_hz, step_hz, count = hop
            writer.writerow([
                int(time.time()), self.cfg.band_label, round(first_hz / 1e6, 6),
                round((first_hz + step_hz * max(0, count - 1)) / 1e6, 6), step_hz,
                round(metrics["baseline"], 2), round(metrics["baseline_change"], 2),
                round(metrics["noise_margin"], 2), round(metrics["noise_change"], 2),
                round(metrics["alert_on_score"], 2), round(metrics["alert_on_change"], 2),
                round(metrics["alert_off_score"], 2), round(metrics["alert_off_change"], 2),
                metrics["min_active_bins"], metrics["min_active_change"],
                metrics["accepted_as_background"],
            ])


def parse_rtl_power_csv_record(line: str):
    """Parse one complete rtl_power row, including its local wall-clock timestamp."""
    try:
        parts = next(csv.reader([line], skipinitialspace=True))
    except (csv.Error, StopIteration):
        return None
    if len(parts) < 7:
        return None
    try:
        observed_at = datetime.datetime.strptime(
            f"{parts[0].strip()} {parts[1].strip()}", "%Y-%m-%d %H:%M:%S")
        first_hz = float(parts[2])
        high_hz = float(parts[3])
        step_hz = float(parts[4])
        samples = int(parts[5])
        # Empty interior/trailing columns indicate a truncated or malformed row.
        if any(not value.strip() for value in parts[6:]):
            return None
        values = [float(x) for x in parts[6:]]
    except (ValueError, IndexError):
        return None
    if (not all(math.isfinite(value) for value in (first_hz, high_hz, step_hz))
            or high_hz <= first_hz or step_hz <= 0 or samples <= 0
            or not values or not all(math.isfinite(value) for value in values)):
        return None
    # rtl_power's low/high fields describe the first/last reported bin centers.
    expected_span = step_hz * (len(values) - 1)
    tolerance = max(2.0, step_hz * 0.02)
    if abs((high_hz - first_hz) - expected_span) > tolerance:
        return None
    return first_hz, step_hz, values, observed_at


def parse_rtl_power_record(line: str):
    """Compatibility helper returning (first bin Hz, bin width Hz, power values)."""
    record = parse_rtl_power_csv_record(line)
    return record[:3] if record is not None else None


def build_rtl_power_command(cfg: Config, output_csv: str):
    return [
        cfg.rtl_power_path,
        "-f", f"{cfg.freq_start_mhz}M:{cfg.freq_end_mhz}M:{cfg.bin_size_hz}",
        "-i", str(cfg.integration_seconds),
        "-g", str(cfg.gain_db),
        output_csv,
    ]


def process_power_sample(monitor, power_values, first_hz, step_hz, mode="LIVE"):
    """Feed one real or simulated spectrum through the shared monitor logic."""
    raw = monitor.parse_metrics_from_power_values(power_values)
    hop_key = monitor._hop_key(first_hz, step_hz, len(power_values))
    if monitor.state == monitor.STATE_CALIBRATING:
        monitor.collect_calibration_sample(raw, hop_key)
        if monitor.calibration_finished():
            monitor.finalize_calibration()
        return

    metrics = monitor.compute_runtime_metrics(raw, hop_key)
    triggered, smoothed_score, state = monitor.step(metrics)
    monitor.log(metrics, smoothed_score, state)
    peak_index = max(range(len(power_values)), key=power_values.__getitem__)
    peak_mhz = (first_hz + peak_index * step_hz) / 1_000_000
    band = monitor.cfg.band_label
    tuning = []
    if abs(metrics["baseline_change"]) >= 0.1:
        tuning.append(f"Grundlinie {metrics['baseline_change']:+.1f} dB")
    if abs(metrics["noise_change"]) >= 0.1:
        tuning.append(f"Rauschabstand {metrics['noise_change']:+.1f} dB")
    if abs(metrics["alert_on_change"]) >= 0.5:
        tuning.append(f"Alarmschwelle {metrics['alert_on_change']:+.1f}")
    if abs(metrics["alert_off_change"]) >= 0.5:
        tuning.append(f"Rückschaltschwelle {metrics['alert_off_change']:+.1f}")
    if metrics["min_active_change"]:
        tuning.append(f"Mindest-Bins {metrics['min_active_change']:+d}")
    tune_text = f" · Anpassung: {', '.join(tuning)}" if tuning else ""
    print(
        f"[{time.strftime('%H:%M:%S')}] {mode} [{band} · Hop {monitor._hop_label(hop_key)}] {state:9s} "
        f"Peak {peak_mhz:.5f} MHz ({power_values[peak_index]:.1f} dBFS) · "
        f"Aktiv {metrics['active_bins']} Bins · "
        f"Grundlinie {metrics['baseline']:.1f} dBFS · "
        f"Rauschabstand {metrics['noise_margin']:.1f} dB · "
        f"Alarm ab {metrics['alert_on_score']:.1f} / aus {metrics['alert_off_score']:.1f} · "
        f"Mindest-Bins {metrics['min_active_bins']} · "
        f"Ausschlag +{metrics['peak_excess_db']:.1f} dB{tune_text}",
        flush=True,
    )
    if triggered:
        print(f"{mode} [{band} · Hop {monitor._hop_label(hop_key)}]: ALARM – auffällige Bandaktivität; Quelle unbekannt.", flush=True)


def run_simulation(band_configs):
    if not sys.stdin.isatty():
        raise RuntimeError("Der Simulationstest benötigt ein interaktives Terminal (stdin ist kein TTY).")
    print("SIMULATION · kein SDR und kein rtl_power werden verwendet.")
    print("Tasten: s = kurzer Impuls · l = längeres Signal · u/d = Grundrauschen +/− 8 dB · q = beenden")
    cfg = band_configs[0]
    monitor = AdaptiveBandActivityMonitor(cfg)
    print(f"Simulierter Bereich: {cfg.band_label} MHz · Kalibrierung: {cfg.calibration_seconds} s · Mess-CSV: {os.path.abspath(cfg.log_file)} · Anpassungs-CSV: {os.path.abspath(cfg.adaptation_log_file)}", flush=True)
    rng = random.Random(380)
    pulse_until = 0.0
    noise_offset = 0.0
    noise_jitter = 0.45
    count = max(8, int((cfg.freq_end_mhz - cfg.freq_start_mhz) * 1_000_000 / cfg.bin_size_hz))
    first_hz = cfg.freq_start_mhz * 1_000_000
    old_settings = termios.tcgetattr(sys.stdin.fileno())
    try:
        tty.setcbreak(sys.stdin.fileno())
        while True:
            ready, _, _ = select.select([sys.stdin], [], [], 1.0)
            if ready:
                key = sys.stdin.read(1).lower()
                now = time.monotonic()
                if key == "q":
                    break
                if key == "s":
                    pulse_until = now + 1.0
                    print("SIMULATION: kurzer Impuls ausgelöst (ca. 1 Messung).", flush=True)
                elif key == "l":
                    pulse_until = now + 10.0
                    print("SIMULATION: längeres Signal ausgelöst (10 s).", flush=True)
                elif key == "u":
                    noise_offset += 8.0
                    noise_jitter = min(4.0, noise_jitter + 1.55)
                    print(f"SIMULATION: Grundrauschen angehoben; Offset {noise_offset:+.1f} dB, Streuung ±{noise_jitter:.2f} dB.", flush=True)
                elif key == "d":
                    noise_offset -= 8.0
                    noise_jitter = max(0.45, noise_jitter - 1.55)
                    print(f"SIMULATION: Grundrauschen abgesenkt; Offset {noise_offset:+.1f} dB, Streuung ±{noise_jitter:.2f} dB.", flush=True)
            values = [-95.0 + noise_offset + rng.uniform(-noise_jitter, noise_jitter) for _ in range(count)]
            if time.monotonic() < pulse_until:
                center = count // 2
                for i in range(max(0, center - 2), min(count, center + 3)):
                    values[i] += 24.0 - abs(i - center) * 1.5
            process_power_sample(monitor, values, first_hz, cfg.bin_size_hz, mode="SIMULATION")
            if monitor.state == monitor.STATE_CALIBRATING:
                elapsed = time.monotonic() - monitor.calibration_started_at
                print(f"[SIMULATION {cfg.band_label}] Kalibrierung {min(elapsed, cfg.calibration_seconds):.0f}/{cfg.calibration_seconds}s", flush=True)
    finally:
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old_settings)
    print("Simulation beendet.", flush=True)


def run_band_scan(cfg, monitor, rtl_power, env):
    cfg.rtl_power_path = rtl_power
    with tempfile.NamedTemporaryFile(prefix="rtl_power_", suffix=".csv", delete=False) as tmp:
        temp_csv = tmp.name
    proc = None
    stderr_log = None
    last_calibration_print = 0.0
    try:
        stderr_log = tempfile.TemporaryFile(mode="w+t")
        proc = subprocess.Popen(
            build_rtl_power_command(cfg, temp_csv),
            stdout=subprocess.DEVNULL,
            stderr=stderr_log,
            env=env,
        )
        last_data_at = time.monotonic()
        with open(temp_csv, "r", encoding="utf-8", errors="replace") as stream:
            while True:
                line = stream.readline()
                if line:
                    record = parse_rtl_power_csv_record(line)
                    if record is None:
                        continue
                    first_hz, step_hz, power_values, observed_at = record
                    hop_key = monitor._hop_key(first_hz, step_hz, len(power_values))
                    if not monitor.accept_record_timestamp(hop_key, observed_at):
                        continue
                    last_data_at = time.monotonic()
                    was_calibrating = monitor.state == monitor.STATE_CALIBRATING
                    process_power_sample(monitor, power_values, first_hz, step_hz)
                    if was_calibrating and monitor.state == monitor.STATE_CALIBRATING:
                        elapsed = time.monotonic() - monitor.calibration_started_at
                        if time.monotonic() - last_calibration_print >= 1:
                            print(f"[{time.strftime('%H:%M:%S')}] Kalibrierung [{cfg.band_label} MHz] {min(elapsed, cfg.calibration_seconds):.0f}/{cfg.calibration_seconds}s", flush=True)
                            last_calibration_print = time.monotonic()
                    continue
                if proc.poll() is not None:
                    stderr_log.flush()
                    stderr_log.seek(0)
                    error = stderr_log.read().strip()
                    raise RuntimeError(f"rtl_power für {cfg.band_label} MHz wurde unerwartet mit Status {proc.returncode} beendet. {error}")
                if time.monotonic() - last_data_at > cfg.data_timeout_seconds:
                    raise RuntimeError(f"rtl_power liefert für {cfg.band_label} MHz seit {cfg.data_timeout_seconds:.1f} Sekunden keine gültigen Messdaten.")
                time.sleep(0.1)
    finally:
        if proc is not None and proc.poll() is None:
            try:
                try:
                    proc.send_signal(signal.SIGINT)
                except ProcessLookupError:
                    pass
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        if stderr_log is not None:
            stderr_log.close()
        try:
            os.unlink(temp_csv)
        except OSError:
            pass


def main():
    parser = argparse.ArgumentParser(description="RTL-SDR Empfangspegel-Scan auf einem Frequenzbereich; keine Quellenidentifikation.")
    parser.add_argument("--start", type=float, help="Startfrequenz in MHz (nur zusammen mit --end; Standard: 380)")
    parser.add_argument("--end", type=float, help="Endfrequenz in MHz (nur zusammen mit --start)")
    parser.add_argument("--calibration", type=int, help="Kalibrierungsdauer in Sekunden (Standard: 30 live, 5 Simulation)")
    parser.add_argument("--log", help="CSV-Protokoll (Standard: activity_log.csv live, simulation_activity_log.csv Simulation)")
    parser.add_argument("--adapt-log", help="CSV-Protokoll für fortlaufende Grundlinien- und Schwellenanpassungen")
    parser.add_argument("--simulate", action="store_true", help="Hardwarefreier interaktiver Test mit synthetischen Signalen")
    parser.add_argument("--rtl-power", default=os.path.expanduser("~/.local/blue-eye-tools/bin/rtl_power"), help="Pfad zu rtl_power")
    args = parser.parse_args()
    if (args.start is None) != (args.end is None):
        parser.error("--start und --end müssen zusammen angegeben werden.")
    bands = [(380.0, 385.0)]
    if args.start is not None:
        bands = [(args.start, args.end)]
    for low, high in bands:
        if not math.isfinite(low) or not math.isfinite(high) or low <= 0 or high <= low or high - low > 100:
            parser.error("Frequenzen müssen positiv und aufsteigend sein; maximal 100 MHz Bandbreite.")
    if args.calibration is not None and args.calibration < 1:
        parser.error("Kalibrierungsdauer muss mindestens 1 Sekunde sein.")
    calibration_seconds = args.calibration or (5 if args.simulate else 30)
    log_file = args.log or ("simulation_activity_log.csv" if args.simulate else "activity_log.csv")
    adaptation_log_file = args.adapt_log or ("simulation_adaptation_log.csv" if args.simulate else "adaptive_calibration_log.csv")
    band_configs = [
        Config(freq_start_mhz=low, freq_end_mhz=high,
               band_label=f"{low:g}–{high:g}",
               calibration_seconds=calibration_seconds,
               log_file=log_file, adaptation_log_file=adaptation_log_file)
        for low, high in bands
    ]
    if args.simulate:
        try:
            run_simulation(band_configs)
        except (RuntimeError, OSError) as exc:
            parser.error(str(exc))
        return
    rtl_power = args.rtl_power if os.path.isfile(args.rtl_power) else shutil.which(args.rtl_power)
    if not rtl_power:
        parser.error(f"rtl_power nicht gefunden: {args.rtl_power} (Pfad mit --rtl-power angeben)")
    env = os.environ.copy()
    bundled_lib = os.path.join(os.path.dirname(os.path.dirname(rtl_power)), "lib")
    if os.path.isdir(bundled_lib):
        env["DYLD_LIBRARY_PATH"] = bundled_lib + os.pathsep + env.get("DYLD_LIBRARY_PATH", "")
    try:
        monitors = {cfg.band_label: AdaptiveBandActivityMonitor(cfg) for cfg in band_configs}
    except OSError as exc:
        parser.error(f"Protokolldatei kann nicht angelegt werden: {exc}")
    print(f"Starte Empfang auf {band_configs[0].band_label} MHz · anfängliche Kalibrierung: {calibration_seconds} s · Mess-CSV: {os.path.abspath(log_file)} · Anpassungs-CSV: {os.path.abspath(band_configs[0].adaptation_log_file)}")
    print("Leistungsspitzen zeigen Aktivität, keine verlässliche Geräte-, Organisations- oder Quellenidentität. Abbruch mit Ctrl-C.\n", flush=True)
    try:
        while True:
            for cfg in band_configs:
                monitor = monitors[cfg.band_label]
                print(f"[{time.strftime('%H:%M:%S')}] Erfasse {cfg.band_label} MHz", flush=True)
                run_band_scan(cfg, monitor, rtl_power, env)
    except KeyboardInterrupt:
        print("\nScan beendet.", flush=True)
    except (RuntimeError, OSError) as exc:
        print(f"\nFEHLER: {exc}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
