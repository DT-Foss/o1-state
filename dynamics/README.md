# dynamics/ — die Formeln treffen den Organismus

**Dritte Welle der Co-Work-Session 2026-08-11.** FERTIG.zip brachte
`Formeln/` (die Foss-2026-Serie) und das codex-lab-Rennen formelbasierter
Trainings-Dynamiken — im klassischen iid/Epochen-Labor. `dynamics/` geht
die zwei Wege, die dort fehlen: **unabhängige Replikation** der tiefsten
Formel und **Transfer ins Organismus-Regime**, gezielt auf die gemessenen
Wunden des body-Tracks.

## Bauteile

| Datei | Inhalt |
|---|---|
| `pslifted.py` | PS-Lifted-Consensus **frisch aus dem Paper implementiert** (nicht aus FERTIG-Code): Fiedler-Orientierung, verdoppelter Zustandsraum, Push-Sum mit Masse-Erhaltungs-Assert. Zwei vom Papiertext unterbestimmte Stellen sind explizit benannt (s ← Wᵀs; Randknoten-Reflexion). Baselines: Uniform-Max-Degree und Metropolis-Hastings. Benchmark: Karate + BA-Graphen n=100/1000/4000, 5 Trials. |
| `rapidity.py` | Die zwei stream-übertragbaren Dynamiken: `RapidityAdam` (Möbius-Momentum — Impuls in der Rapidität, Schritt = lr·tanh(w), strukturell \|Schritt\| < lr; dieselbe Möbius-Kopplung wie der GSSM-Kern, jetzt auf der Optimierer-Seite) und `lorentz_lr` (tau-Schedule, Peak auf Baseline-LR normiert). Re-Implementierung nach `_codex_lab/training_dynamics/race_dynamics.py`, mit eigenen Tests. |
| `organism_race.py` | Vier hart gepaarte Arme (adam / rapidity / lorentz / rapidity_lorentz) auf dem body-Organism-Task: identische Welt, Aktionsfolge, Init-Gewichte, Gate — nur die Update-Dynamik variiert. Bewertung auf fester Probe-Route (voice/-Design), copy-last der Route als Nullpunkt. |
| `test_dynamics.py` | 6 Smokes (W-Gesundheit, alle Methoden erreichen das Mittel, Pfad-Graph-Vorbedingung, Fiedler-Monotonie, Lorentz-Schranke + Lernen, Schedule-Form). |
| `PREDICTIONS_DYNAMICS.md` | Register-ENTWÜRFE P95–P96, vor den Scoring-Läufen; Verdicts danach darunter. |

## Ehrlich NICHT gebaut, mit Grund

- **BvN-Random-Reshuffling**: ein Leben hat keine Epochen — im
  Ein-Pass-Stream existiert die Operation nicht.
- **PS-Lifted-Gradient-Consensus / FLCA-Router**: bei B=1-Chunks gibt es
  höchstens eine Handvoll Micro-Gradienten; Konsens-nach-R-Runden ist vom
  exakten Mittel nicht unterscheidbar — das wäre Theater, kein Test.
- **PS-Lifted-Fleet** (N Organismen, Gossip-Sync): der O(1)-Runden-Vorteil
  zeigt sich erst bei großem n; bei unserer Fleet-Größe (≤8) ist jeder
  Prüfstand unfair im Sinne der Formel. Benannt für den Tag, an dem die
  Fleet dreistellig wird.

## Laufen lassen

```bash
python3 dynamics/test_dynamics.py          # 6/6
python3 dynamics/pslifted.py               # Replikation → results/pslifted_replication.json
python3 dynamics/organism_race.py --frames 6000
```

Grenzen wie in allen Wellen: nur `dynamics/` wird beschrieben; `body/`,
`voice/`, `visual/`, `hsslm/` read-only; Thread-Clamps; keine Server.
