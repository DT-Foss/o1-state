# body/ — der handelnde Organism

**Auftrag (Co-Work-Session 2026-08-11):** die Architektur-Inventur nannte
als größte Lücke den handelnden Körper — alles im Repo ist Zuschauer. Der
visuelle Organism sieht Frames, wählt aber nie eine Aktion (P48s Walker
laufen Skripte, P52s Doom-Body würfelt seeded); der Sprecher sagt Tokens
voraus, greift nie ein. Damit fehlt die halbe Harnad-Schleife: Symbole, die
an Konsequenzen EIGENER Handlungen binden. `body/` schließt genau diese
Lücke, minimal und messbar, auf dem bestehenden Stack.

## Was hier steht

| Datei | Inhalt |
|---|---|
| `action_sources.py` | Gehandelte Welten mit `observe()`/`act(a)`-Contract: `ActedSyntheticSource` (64×128-Welt, 64er-Viewport, PAN_LEFT/PAN_RIGHT/HOLD — eigene Konsequenz und Weltdynamik per Konstruktion trennbar) und `ActedVizDoomSource` (basic.cfg, Frame-Skip 4, P52s Fresh-Engine-per-Episode-Factory). Beide loggen den VOLLEN Action-Trace: unter gelernter Politik ist die Aktionsfolge nicht mehr seed-ableitbar, also reist der Trace als Provenance-Träger mit. |
| `body_organism.py` | `ActionConditionedPredictor` — Frame-Encoder **+ Aktions-Embedding** (zero-init) → GSSMCore (importiert, nie kopiert) → Delta-Head (zero-init; die gemessene Phase-1-Lektion). `counterfactual()`: Vorhersage unter ALLEN Aktionen aus demselben (geklonten) Zustand — „was sehe ich, WENN ich das tue" als Forward-Pass. `BodyStreamer` — das F1/F2-Rezept an der (x, actions, y)-Signatur, Gate-Konstanten read-only aus `visual/frame_organism.py`. `LearningProgressPolicy` — Neugier als Motiv: pro Aktion die Verbesserungsrate der Vorhersagefehler (Oudeyer-Stil), softmax über z-scores, ε-Boden, Ignition uniform. KEIN Reward, kein Reward-Engineering — dieselbe Surprise, die bisher filterte, wählt jetzt. |
| `causal_records.py` | Stufe 2: „pressed left \| view shifted \| dx=−3" — gehandelte Ereignisse im livecausal-Schlüsselvokabular (trigger_key/mechanism/outcome_key), Quote mit Frame-Koordinate + sha256 beider Frames, versiegelt via unverändertem `src/livecausal/store.py` in einen Store **unter body/** (Entwurf, kein Merge in Bestands-Stores). `verify_records()` = Provenance-Gate: Replay aus (seed, trace), Hash-Vergleich bit-exakt, dx-Redetektion. |
| `run_body.py` | Lebensschleife: observe → **choose** → act → predict → gate/learn → record. Schreibt metrics.jsonl (inkl. Counterfactual-Separation & -Hit-Rate, Politik-Wahrscheinlichkeiten, LP-Schätzer), Status-Heartbeat, Snapshot, Summary-JSON, zwei GIF-Sorten. |
| `test_body.py` | 7 Smokes; laufen ohne vizdoom. Jeder Test bewacht einen tragenden Claim (Determinismus unter Trace, copy-last-Start, Counterfactual mutiert den Live-State nicht, Estimator-Ground-Truth, Record-Roundtrip bit-exakt, LP-Präferenz). |
| `PREDICTIONS_BODY.md` | Register-ENTWÜRFE P87–P90, vor den Scoring-Läufen geschrieben; Merge ins Hauptregister macht die Haupt-Session. |

## Sichtbar ab Tag 1

- `gifs/body_<arm>.gif` — Zeilen: real / |diff| / vorhergesagt / **Aktions-Streifen**
  (welche Aktion diesen Übergang erzeugt hat, Frame für Frame nachprüfbar).
- `gifs/body_cf_<arm>.gif` — pro Probe-Zustand die vorhergesagten
  Nächst-Frames unter JEDER Aktion nebeneinander + real; getane Aktion
  markiert. Sind die Spalten identisch, ist die Aktion im Modell nicht
  angekommen — auf einen Blick sichtbar, erwartbar Müll am Anfang.
- Neugier-Trajektorien: `probs`/`lp` pro Chunk in `results/*_metrics.jsonl`.

## Laufen lassen

```bash
python3 body/test_body.py                       # 7/7 erwartet, kein vizdoom nötig
python3 body/run_body.py --source acted_synthetic --frames 6000 \
    --policy curiosity --seed 42 --out-prefix body/results/body_sc
python3 body/run_body.py --source vizdoom --frames 2000 \
    --policy curiosity --seed 42 --out-prefix body/results/body_dc
```

Thread-Clamps (OMP/OpenBLAS/MKL=1, torch 1 Thread) sind gesetzt —
o1-maschine-last-threads-Konvention.

## Grenzen & ehrliche Lücken

- **Nur `body/`.** Importe aus `visual/`, `hsslm/`, `src/livecausal/` sind
  read-only; keine Bestandsdatei wird angefasst. Keine Server, keine Pushes.
- **Die Politik ist kontextfrei** (Bandit über Aktionen, nicht über
  Zustände). Zustandskonditionaler Lernfortschritt ist der benannte nächste
  Schritt — registriert, nicht eingeschmuggelt.
- **Der Konsequenz-Detektor ist absichtlich dumm** (globaler horizontaler
  Shift). Er trägt Pan/Strafe in beiden Welten; ATTACK-Konsequenzen
  (Mündungsfeuer, Monstertod) sieht er NICHT — deren Abwesenheit im
  Record-Strom ist die ehrliche Ausgabe, kein Feature. Besseres
  Konsequenz-Organ ⇒ erst registrieren.
- **Episodengrenzen** sind Stream-Realität fürs Training, aber keine
  Handlungskonsequenz — der Extraktor überspringt sie (gezählt in
  `episode_boundaries`).
- Der Grounding-Kernel (`visual/grounding/`) wird bewusst NICHT
  angebunden; seine Action/Transition-Contracts sind der natürliche
  nächste Andockpunkt, sobald P87/P88 stehen (intervention_necessity ist
  genau die Achse, die dieser Track bedienen soll).
