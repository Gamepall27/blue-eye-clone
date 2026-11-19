import csv
import os
import pty
import select
import subprocess
import sys
import tempfile
import termios
import time
import unittest
from pathlib import Path
from unittest import mock

import blueeye


ROOT = Path(__file__).resolve().parents[1]


class BlueEyeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="blueeye-tests-")
        self.addCleanup(self.temp.cleanup)
        self.temp_path = Path(self.temp.name)

    def config(self, **kwargs):
        options = dict(
            band_label="380–385",
            log_file=str(self.temp_path / "activity.csv"),
            adaptation_log_file=str(self.temp_path / "adaptation.csv"),
            calibration_seconds=1,
        )
        options.update(kwargs)
        return blueeye.Config(**options)

    @staticmethod
    def row(center=-95.0, count=500, jitter=0.0):
        return [center + (jitter if index % 2 else -jitter) for index in range(count)]

    def calibrated_monitor(self, first_hz=380000000, count=500):
        cfg = self.config()
        monitor = blueeye.AdaptiveBandActivityMonitor(cfg)
        key = monitor._hop_key(first_hz, 10000, count)
        for _ in range(3):
            raw = monitor.parse_metrics_from_power_values(self.row(count=count, jitter=0.2))
            monitor.collect_calibration_sample(raw, key)
        monitor.finalize_calibration()
        return monitor, key

    def test_rtl_power_csv_metadata_and_nonfinite_rejection(self):
        powers = [-100.0 + index / 10 for index in range(257)]
        line = "2026-11-19,12:00:00,380000000,382560000,10000,1," + ",".join(map(str, powers))
        record = blueeye.parse_rtl_power_record(line)
        self.assertIsNotNone(record)
        first, step, parsed = record
        self.assertEqual((first, step), (380000000.0, 10000.0))
        self.assertEqual(len(parsed), 257)
        self.assertEqual(first + (len(parsed) - 1) * step, 382560000.0)
        self.assertIsNone(blueeye.parse_rtl_power_record(line.replace(",10000,1,", ",nan,1,")))
        self.assertIsNone(blueeye.parse_rtl_power_record(line.replace(",-100.0,", ",inf,")))

    def test_hop_models_do_not_mix_baselines_or_state(self):
        monitor = blueeye.AdaptiveBandActivityMonitor(self.config())
        a = monitor._hop_key(380000000, 10000, 257)
        b = monitor._hop_key(382570000, 10000, 257)
        for _ in range(3):
            monitor.collect_calibration_sample(monitor.parse_metrics_from_power_values(self.row(-95, 257, .2)), a)
            monitor.collect_calibration_sample(monitor.parse_metrics_from_power_values(self.row(-72, 257, .2)), b)
        monitor.finalize_calibration()
        self.assertAlmostEqual(monitor.hop_models[a]["baseline"], -95.2)
        self.assertAlmostEqual(monitor.hop_models[b]["baseline"], -72.2)
        metrics = monitor.compute_runtime_metrics(monitor.parse_metrics_from_power_values(self.row(-95, 257, .2)), a)
        monitor.step(metrics)
        self.assertEqual(len(monitor.hop_models[b]["recent_scores"]), 0)
        self.assertEqual(monitor.hop_models[b]["state"], monitor.STATE_IDLE)

    def test_short_narrow_burst_is_not_learned_but_persistent_broad_shift_adapts_up_and_down(self):
        monitor, key = self.calibrated_monitor()
        model = monitor.hop_models[key]
        old_baseline = model["baseline"]
        narrow = self.row(count=500, jitter=.2)
        narrow[245:250] = [value + 28 for value in narrow[245:250]]
        m = monitor.compute_runtime_metrics(monitor.parse_metrics_from_power_values(narrow), key)
        self.assertAlmostEqual(m["baseline"], old_baseline, places=6)
        _, _, state = monitor.step(m)
        self.assertEqual(state, monitor.STATE_WATCH)
        self.assertFalse(m["accepted_as_background"])

        for _ in range(8):
            m = monitor.compute_runtime_metrics(monitor.parse_metrics_from_power_values(self.row(-87, 500, .2)), key)
        self.assertGreater(m["baseline"], -90)
        for _ in range(8):
            m = monitor.compute_runtime_metrics(monitor.parse_metrics_from_power_values(self.row(-95, 500, .2)), key)
        self.assertLess(m["baseline"], -92)

    def test_persistent_broad_noise_shift_is_learned_without_remaining_in_alarm(self):
        monitor, key = self.calibrated_monitor()
        observed_states = []
        last_metrics = None
        for index in range(32):
            with mock.patch.object(blueeye.time, "time", return_value=1000.0 + index):
                raw = monitor.parse_metrics_from_power_values(self.row(-87, 500, .2))
                last_metrics = monitor.compute_runtime_metrics(raw, key)
                _, _, state = monitor.step(last_metrics)
                observed_states.append(state)
        model = monitor.hop_models[key]
        self.assertIn(monitor.STATE_ALERT, observed_states)
        self.assertEqual(model["state"], monitor.STATE_IDLE)
        self.assertGreater(model["baseline"], -89)
        self.assertTrue(last_metrics["accepted_as_background"])

    def test_long_narrowband_signal_alarms_and_is_not_absorbed_as_background(self):
        monitor, key = self.calibrated_monitor()
        baseline = monitor.hop_models[key]["baseline"]
        active = self.row(count=500, jitter=.2)
        active[245:250] = [value + 28 for value in active[245:250]]
        states = []
        last_metrics = None
        for index in range(8):
            with mock.patch.object(blueeye.time, "time", return_value=2000.0 + index):
                last_metrics = monitor.compute_runtime_metrics(monitor.parse_metrics_from_power_values(active), key)
                _, _, state = monitor.step(last_metrics)
                states.append(state)
        model = monitor.hop_models[key]
        self.assertIn(monitor.STATE_ALERT, states)
        self.assertEqual(model["state"], monitor.STATE_ALERT)
        self.assertLess(abs(model["baseline"] - baseline), 0.5)
        self.assertFalse(last_metrics["accepted_as_background"])

    def test_noise_spread_window_tracks_increase_and_decrease(self):
        monitor, key = self.calibrated_monitor()
        model = monitor.hop_models[key]
        original_margin = model["noise_margin"]
        for _ in range(17):
            m = monitor.compute_runtime_metrics(monitor.parse_metrics_from_power_values(self.row(-95, 500, 4.0)), key)
        self.assertGreater(m["noise_margin"], original_margin)
        raised = m["noise_margin"]
        for _ in range(17):
            m = monitor.compute_runtime_metrics(monitor.parse_metrics_from_power_values(self.row(-95, 500, .2)), key)
        self.assertLess(m["noise_margin"], raised)
        self.assertGreaterEqual(m["noise_margin"], monitor.cfg.min_noise_margin_db)
        self.assertLessEqual(m["noise_margin"], monitor.cfg.max_noise_margin_db)

    def test_alarm_threshold_relearns_only_from_accepted_background_and_keeps_floor(self):
        monitor = blueeye.AdaptiveBandActivityMonitor(self.config())
        key = monitor._hop_key(380000000, 10000, 500)
        model = monitor._new_hop_model(self.row(count=500, jitter=.2))
        monitor.hop_models[key] = model
        self.assertEqual(model["alert_on"], 50.0)
        for index in range(8):
            with mock.patch.object(blueeye.time, "time", return_value=5000.0 + index):
                metrics = monitor.compute_runtime_metrics(
                    monitor.parse_metrics_from_power_values(self.row(count=500, jitter=.2)), key)
                _, _, state = monitor.step(metrics)
        self.assertEqual(state, monitor.STATE_IDLE)
        self.assertEqual(model["alert_on"], monitor.cfg.min_alert_on_score)
        self.assertGreaterEqual(model["alert_on"], 20.0)
        self.assertEqual(model["min_active"], 1)

    def test_alert_state_uses_existing_watch_duration_and_noise_event_does_not_freeze_it(self):
        monitor, key = self.calibrated_monitor()
        signal_row = self.row(count=500, jitter=.2)
        signal_row[245:250] = [value + 28 for value in signal_row[245:250]]
        states = []
        for tick in (100.0, 101.0, 102.1):
            with mock.patch.object(blueeye.time, "time", return_value=tick):
                metrics = monitor.compute_runtime_metrics(monitor.parse_metrics_from_power_values(signal_row), key)
                triggered, _, state = monitor.step(metrics)
                states.append((state, triggered))
        self.assertEqual(states[0][0], monitor.STATE_WATCH)
        self.assertFalse(states[1][1])
        self.assertEqual(states[2], (monitor.STATE_ALERT, True))

    def test_empty_calibration_fallback_and_empty_or_nonfinite_power(self):
        monitor = blueeye.AdaptiveBandActivityMonitor(self.config())
        monitor.finalize_calibration()
        self.assertEqual(monitor.state, monitor.STATE_IDLE)
        self.assertIsNone(monitor.parse_metrics_from_power_values([]))
        self.assertIsNone(monitor.parse_metrics_from_power_values([float("nan")]))
        self.assertIsNone(monitor.parse_metrics_from_power_values([float("inf")]))

    def test_csv_logs_current_values_and_adaptation_deltas_with_consistent_schemas(self):
        monitor, key = self.calibrated_monitor()
        values = self.row(-87, 500, .2)
        metrics = monitor.compute_runtime_metrics(monitor.parse_metrics_from_power_values(values), key)
        _, score, state = monitor.step(metrics)
        monitor.log(metrics, score, state)
        with open(self.config().log_file, newline="") as f:
            main_rows = list(csv.reader(f))
        with open(self.config().adaptation_log_file, newline="") as f:
            adaptation_rows = list(csv.reader(f))
        self.assertEqual(len(main_rows[0]), len(main_rows[1]))
        self.assertEqual(len(adaptation_rows[0]), len(adaptation_rows[1]))
        self.assertIn("380–385 hop", main_rows[1][1])
        self.assertEqual(adaptation_rows[1][1], "380–385")

    def test_old_adaptation_schema_is_preserved_and_new_log_is_versioned(self):
        old_path = self.temp_path / "adaptive.csv"
        old_path.write_text("timestamp,old_schema\n1,keep\n")
        cfg = self.config(adaptation_log_file=str(old_path))
        blueeye.AdaptiveBandActivityMonitor(cfg)
        self.assertEqual(old_path.read_text(), "timestamp,old_schema\n1,keep\n")
        self.assertTrue((self.temp_path / "adaptive_v2.csv").exists())

    def test_cli_rejects_invalid_and_nonfinite_frequencies(self):
        for arguments in (["--start", "380"], ["--start", "nan", "--end", "385"],
                          ["--start", "380", "--end", "inf"], ["--calibration", "0"]):
            with self.subTest(arguments=arguments), mock.patch.object(sys, "argv", ["blueeye.py"] + arguments):
                with self.assertRaises(SystemExit) as error:
                    blueeye.main()
                self.assertEqual(error.exception.code, 2)

    def test_default_cli_scans_only_380_to_385_without_launching_hardware(self):
        args = ["blueeye.py", "--log", str(self.temp_path / "default.csv"),
                "--adapt-log", str(self.temp_path / "default-adapt.csv")]
        with mock.patch.object(sys, "argv", args), \
                mock.patch.object(blueeye.os.path, "isfile", return_value=True), \
                mock.patch.object(blueeye, "run_band_scan", side_effect=KeyboardInterrupt) as scan:
            blueeye.main()
        self.assertEqual(scan.call_count, 1)
        cfg = scan.call_args[0][0]
        self.assertEqual((cfg.freq_start_mhz, cfg.freq_end_mhz), (380.0, 385.0))

    def test_simulation_requires_interactive_terminal(self):
        with mock.patch.object(sys, "argv", ["blueeye.py", "--simulate"]), \
                mock.patch.object(sys.stdin, "isatty", return_value=False):
            with self.assertRaises(SystemExit) as error:
                blueeye.main()
            self.assertEqual(error.exception.code, 2)

    def test_simulation_short_key_does_not_alarm_but_long_key_does(self):
        master, slave = pty.openpty()
        cfg = self.config(log_file=str(self.temp_path / "pulse-sim.csv"),
                          adaptation_log_file=str(self.temp_path / "pulse-adapt.csv"),
                          calibration_seconds=1)
        command = [sys.executable, str(ROOT / "blueeye.py"), "--simulate", "--calibration", "1",
                   "--log", cfg.log_file, "--adapt-log", cfg.adaptation_log_file]
        proc = subprocess.Popen(command, stdin=slave, stdout=slave, stderr=slave, cwd=str(ROOT))
        output = bytearray()

        def drain(duration):
            deadline = time.monotonic() + duration
            while time.monotonic() < deadline:
                readable, _, _ = select.select([master], [], [], .1)
                if readable:
                    try:
                        output.extend(os.read(master, 8192))
                    except OSError:
                        break

        try:
            drain(2.5)
            os.write(master, b"s")
            drain(2)
            self.assertIn("kurzer Impuls ausgelöst".encode(), output)
            self.assertNotIn("ALARM".encode(), output)
            os.write(master, b"l")
            drain(4)
            self.assertIn("längeres Signal ausgelöst".encode(), output)
            self.assertIn("ALARM".encode(), output)
            os.write(master, b"q")
            proc.wait(timeout=5)
            self.assertEqual(proc.returncode, 0)
        finally:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=3)
            os.close(master)
            os.close(slave)

    def _fake_rtl_power(self, script_body):
        executable = self.temp_path / "fake_rtl_power.py"
        executable.write_text("#!/usr/bin/env python3\n" + script_body)
        executable.chmod(0o755)
        return str(executable)

    def test_subprocess_failure_reports_stderr_and_cleans_temp_csv(self):
        tool = self._fake_rtl_power(
            "import pathlib,sys\n"
            "pathlib.Path(sys.argv[-1]).write_text('test')\n"
            "print('fake rtl_power failure', file=sys.stderr)\n"
            "sys.stderr.write('x' * 1000000)\n"
            "sys.exit(7)\n")
        cfg = self.config(data_timeout_seconds=.5)
        monitor = blueeye.AdaptiveBandActivityMonitor(cfg)
        with self.assertRaisesRegex(RuntimeError, "fake rtl_power failure"):
            blueeye.run_band_scan(cfg, monitor, tool, os.environ.copy())
        self.assertTrue(list(self.temp_path.glob("rtl_power_*.csv")) == [])

    def test_no_data_timeout_stops_fake_process_and_cleans_temp_csv(self):
        pid_marker = self.temp_path / "fake-pid.txt"
        tool = self._fake_rtl_power(
            "import os,pathlib,time\n"
            "pathlib.Path(os.environ['BLUEEYE_TEST_PID_FILE']).write_text(str(os.getpid()))\n"
            "time.sleep(60)\n")
        cfg = self.config(data_timeout_seconds=1.5)
        monitor = blueeye.AdaptiveBandActivityMonitor(cfg)
        start = time.monotonic()
        child_env = os.environ.copy()
        child_env["BLUEEYE_TEST_PID_FILE"] = str(pid_marker)
        with self.assertRaisesRegex(RuntimeError, "keine gültigen Messdaten"):
            blueeye.run_band_scan(cfg, monitor, tool, child_env)
        self.assertLess(time.monotonic() - start, 5)
        self.assertEqual(list(self.temp_path.glob("rtl_power_*.csv")), [])
        child_pid = int(pid_marker.read_text())
        with self.assertRaises(ProcessLookupError):
            os.kill(child_pid, 0)

    def test_spawn_setup_failures_remove_temp_csv(self):
        cfg = self.config()
        monitor = blueeye.AdaptiveBandActivityMonitor(cfg)
        with mock.patch.object(blueeye.tempfile, "TemporaryFile", side_effect=OSError("no temp storage")):
            with self.assertRaisesRegex(OSError, "no temp storage"):
                blueeye.run_band_scan(cfg, monitor, "/no/such/rtl_power", os.environ.copy())
        self.assertEqual(list(self.temp_path.glob("rtl_power_*.csv")), [])
        with self.assertRaises(FileNotFoundError):
            blueeye.run_band_scan(cfg, monitor, "/no/such/rtl_power", os.environ.copy())
        self.assertEqual(list(self.temp_path.glob("rtl_power_*.csv")), [])

    def test_simulation_terminal_restores_settings_and_adapts_both_directions(self):
        master, slave = pty.openpty()
        original = termios.tcgetattr(slave)
        cfg = self.config(log_file=str(self.temp_path / "sim.csv"),
                          adaptation_log_file=str(self.temp_path / "sim-adapt.csv"),
                          calibration_seconds=1)
        command = [sys.executable, str(ROOT / "blueeye.py"), "--simulate", "--calibration", "1",
                   "--log", cfg.log_file, "--adapt-log", cfg.adaptation_log_file]
        proc = subprocess.Popen(command, stdin=slave, stdout=slave, stderr=slave, cwd=str(ROOT))
        output = bytearray()

        def drain(duration):
            deadline = time.monotonic() + duration
            while time.monotonic() < deadline:
                readable, _, _ = select.select([master], [], [], .1)
                if readable:
                    try:
                        output.extend(os.read(master, 8192))
                    except OSError:
                        break

        try:
            drain(2.5)
            os.write(master, b"u")
            drain(10)
            with open(cfg.adaptation_log_file, newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertTrue(any(float(row["baseline_db"]) > -90 for row in rows))
            os.write(master, b"d")
            drain(10)
            with open(cfg.adaptation_log_file, newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertTrue(any(float(row["baseline_db"]) < -92 for row in rows))
            os.write(master, b"q")
            proc.wait(timeout=5)
            self.assertEqual(proc.returncode, 0)
            restored = termios.tcgetattr(slave)
            self.assertEqual(restored[:3], original[:3])
            # macOS may set PENDIN when queued cbreak input is reprocessed on canonical restore.
            pending_reprint_flag = getattr(termios, "PENDIN", 0)
            self.assertEqual(restored[3] & ~pending_reprint_flag,
                             original[3] & ~pending_reprint_flag)
            self.assertEqual(restored[6], original[6])
        finally:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=3)
            os.close(master)
            os.close(slave)

if __name__ == "__main__":
    unittest.main()
