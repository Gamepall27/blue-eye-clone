# Blue Eye

Blue Eye liest Empfangspegel eines RTL-SDR mit `rtl_power` und zeigt Aktivität im Terminal an. Es sendet nicht und identifiziert keine Geräte.

## Start

Im Terminal in diesen Projektordner wechseln und ausführen:

```sh
python3 blueeye.py
```

Standardmäßig erfasst das Programm ausschließlich **380–385 MHz**. `rtl_power` kann diesen 5-MHz-Bereich intern über mehrere Empfangs-Hops abtasten; das ist keine simultane oder garantiert lückenlose Erfassung.

Das Programm misst ausschließlich Empfangspegel. Es dekodiert keine Kommunikation und identifiziert oder ortet keine Quelle. Ein Alarm zeigt nur auffällige Aktivität im gerade erfassten Bereich an. `Ctrl-C` beendet den Scan sauber. Messwerte stehen in `activity_log.csv`; der Frequenzbereich und der jeweilige `rtl_power`-Hop sind im `state`-Feld mitgeführt. Laufende Kalibrierungsanpassungen werden zusätzlich in `adaptive_calibration_log.csv` protokolliert.

Ein anderer einzelner Frequenzbereich kann angegeben werden:

```sh
python3 blueeye.py --start 390 --end 395 --calibration 30 --log activity_log.csv
```

`--start` und `--end` müssen zusammen angegeben werden.

Falls `rtl_power` an einem anderen Ort liegt:

```sh
python3 blueeye.py --rtl-power /pfad/zu/rtl_power
```

## Hardwarefreier Simulationstest

Zum Testen ohne angeschlossenen SDR:

```sh
python3 blueeye.py --simulate
```

Die Simulation verwendet standardmäßig ebenfalls 380–385 MHz, kalibriert 5 Sekunden mit simuliertem Grundrauschen und nutzt die Erkennungslogik des Live-Empfangs. Während sie läuft, gelten diese Tasten:

- `s`: kurzer Signalausschlag (eine Messung; durch die Glättung normalerweise kein Alarm)
- `l`: längeres Signal (10 Sekunden; sollte den Zustandsautomaten bis zum Alarm führen)
- `q`: beenden

Alle Ausgaben sind mit `SIMULATION` und dem Frequenzbereich markiert. `u` und `d` heben oder senken das breitbandige simulierte Grundrauschen einschließlich seiner Streuung; die neue Lage bleibt bestehen, damit die Anpassung beobachtet werden kann. Das getrennte Messprotokoll heißt `simulation_activity_log.csv`, das Anpassungsprotokoll `simulation_adaptation_log.csv`. Kalibrierungsdauer und Protokollpfade lassen sich mit `--calibration`, `--log` und `--adapt-log` anpassen.

Die Live-Zeilen nennen Zeit, Zustand, stärksten Frequenz-Bin und dessen Pegel in dBFS. Der Ausschlag und die Aktiv-Bins vergleichen die Messung mit der automatisch ermittelten Grundlinie. Ein `ALERT` bedeutet auffällige Bandaktivität gemäß dieser Schwelle. Es belegt weder eine bestimmte Quelle noch deren Identität; unbekannte Quellen bleiben unbekannt.

Live werden zunächst standardmäßig 30 Sekunden Messdaten zur Kalibrierung gesammelt. Danach gibt es keine Kalibrierungspausen oder SDR-Neustarts: Jeder wiederkehrende `rtl_power`-Hop wird separat angepasst, damit Frequenzabschnitte nicht in eine gemeinsame Grundlinie geraten. Die Grundlinie ist der Median der letzten 15 Hop-Beobachtungen. Eine einzelne kurze Störung verschiebt diesen Median normalerweise nicht; bei einer anhaltenden breitbandigen Änderung bewegt sich die Grundlinie, sobald mehr als die Hälfte des Fensters die neue Lage enthält (bei vollem Fenster typischerweise nach etwa 8 wiederholten Beobachtungen dieses Hops). Rauschabstand und Alarmwerte nutzen robuste Fenster aus 31 Streuungsmessungen beziehungsweise bis zu 60 als Hintergrund akzeptierten Beobachtungen pro Hop. Die Rauschmarge bleibt auf 3–12 dB und die Alarmschwelle auf den bestehenden Grenzen von 20–120 begrenzt.

Als Hintergrund gelten nur Messungen mit höchstens 10 % aktiven Bins und ohne Ausschlag über 12 dB; Messungen im Zustand `WATCH` oder `ALERT` werden nicht in die Schwellenanpassung aufgenommen. Eine schmale starke Aussendung beeinflusst die räumliche Median-Grundlinie daher meist wenig und wird heuristisch aus dem Hintergrundfenster ausgeschlossen. Das ist keine perfekte Trennung: Ein lang anhaltendes Signal oder eine Änderung, die viele Bins belegt, kann nach und nach als neues Hintergrundniveau gelten. Bei neuen oder anders aufgeteilten Hop-Geometrien beginnt eine eigene Schätzung; nur identische Hop-Geometrien teilen Historie.

Die erste vollständige, plausible Messzeile muss innerhalb von 30 Sekunden eintreffen; unvollständige CSV-Zeilen werden verworfen. Wiederholte oder verspätete Zeilen desselben Hops werden anhand des `rtl_power`-Zeitstempels ignoriert. Nach einer Hop-Lücke von mehr als 30 Sekunden setzt Blue Eye die zeitliche Alarmhistorie dieses Hops zurück, damit veraltete WATCH-/ALERT-Messungen keinen unmittelbaren Alarm auslösen. Laufzeitdauern nutzen eine monotone Uhr und bleiben dadurch von Änderungen der Systemuhr unabhängig. Danach beendet Blue Eye den `rtl_power`-Prozess bei ausbleibenden gültigen Daten mit einer Fehlermeldung und räumt seine temporäre Datei auf. Falls `adaptive_calibration_log.csv` noch ein älteres Schema besitzt, bleibt es unverändert; das neue Format wird dann in `adaptive_calibration_log_v2.csv` geschrieben.

`rtl_power` muss installiert und der SDR angeschlossen sein. In der eingerichteten Umgebung wird standardmäßig `~/.local/blue-eye-tools/bin/rtl_power` verwendet.
