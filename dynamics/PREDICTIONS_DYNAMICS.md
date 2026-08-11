# PREDICTIONS — dynamics/ (ENTWURF für den Merge ins Hauptregister)

Status: **DRAFT** im Stil von analysis/PREDICTIONS.md; Nummern P95–P96
sind VORSCHLÄGE (P91–P94 sind die voice/-Entwürfe derselben Session;
Merge und finale Nummerierung macht die Haupt-Session).

Kontext: FERTIG.zip (2026-08-11) enthält Formeln/ (Foss-2026-Serie) und
_codex_lab/training_dynamics/race_dynamics.py — das Formel-Rennen im
klassischen iid/Epochen-Regime (dynamics_race.json dort ist nur der
6-Step-Smoke, kein Verdict). dynamics/ geht die zwei ANDEREN Wege:
(1) die tiefste Formel unabhängig replizieren, (2) die übertragbaren
Dynamiken im Organismus-Regime antreten lassen, gezielt auf die
gemessenen Wunden P87b (copy-last ungeschlagen) und P88 (langsame
Aktionskanal-Ignition). Ehrlich NICHT übertragen: BvN-Reshuffling (ein
Leben hat keine Epochen), PS-Lifted-Gradient-Consensus und FLCA-Router
(bei B=1-Chunks ist Konsens vom exakten Mittel nicht unterscheidbar —
Theater), PS-Lifted-Fleet bei n≤8 Organismen (der O(1)-Vorteil zeigt
sich erst bei großem n; falscher Prüfstand).

Registriert 2026-08-11 (Co-Work-Session, dritte Welle), NACH 6/6
dynamics/test_dynamics.py und EINEM offengelegten Replikations-Smoke
(Karate + BA-100, 2 Trials: pslifted 9.0/15.5 vs uniform 221.5/68.0,
mh 150.0/33.5 Runden), aber VOR den Scoring-Läufen:

- `python3 dynamics/pslifted.py`  (Karate + BA 100/1000/4000, 5 Trials)
- `python3 dynamics/organism_race.py --frames 6000`  (4 Arme, gepaart)

Implementierung frisch aus dem Paper (nicht aus FERTIG-Code): zwei
unterbestimmte Stellen sind im Modul-Docstring benannt (Masse-Aktion
s ← Wᵀs; Randknoten-Reflexion) und reisen mit dem Ergebnis.

---

## P95 — Foss-Konvergenz repliziert unabhängig (ENTWURF)

Paper-Claim (Table 1 / Fig. b): PS-Lifted erreicht ‖x̂−x̄‖∞ < 0.01 in
12–34 Runden UNABHÄNGIG von n; reversibles Gossip skaliert diffusiv.
Fremde Implementierung, pc=0.95, ps=0.003, 5 Trials x~U[0,1]:

- (a) KARATE-ANKER: PS-Lifted ≤ 20 Runden UND ≥ 4× schneller als die
  beste reversible Baseline (Paper: 12 vs. MH 101 ≈ 8×; konservative
  Bar, weil Baseline-Gewichtswahlen die Absolutzahlen verschieben —
  der Smoke zeigt genau das: unsere Baselines sind LANGSAMER als die
  des Papers, PS-Lifted nicht).
- (b) n-UNABHÄNGIGKEIT (der Kern): rounds(BA-4000) ≤ 2 × rounds(BA-100)
  für PS-Lifted, WÄHREND die beste reversible Baseline über dieselbe
  Spanne ≥ 4× wächst. 40×-Spanne in n; konstant-gegen-wachsend ist die
  Foss-Konvergenz-Signatur, nicht eine Absolutzahl.
- (c) GESUNDHEIT: Masse-Erhaltung < 1e-6 relativ in jedem Lauf (ein
  Bruch wäre ein Implementierungs-Bug und annulliert (a)/(b), kein
  Befund über die Formel).
- Nicht gemessen, benannt: der 2×-Kommunikations-Overhead ist
  definitorisch (4 Skalare pro orientierter Kante) und wird nicht als
  Ergebnis verkauft; FDLA (SDP) ist bewusst ausgelassen, MH vertritt
  die reversible Klasse.

Falsifier: (a) reißt → die Formel repliziert nicht außerhalb ihrer
Heimat-Implementierung; (b) reißt bei haltendem (a) → der Vorteil ist
real, aber nicht n-unabhängig — dann stirbt genau der O(1)-Anspruch und
der Rest bleibt ein guter Beschleuniger.

**P95 GESCORT (2026-08-11, dynamics/results/pslifted_replication.json).**
(a) CONFIRMED, und zwar auf den Punkt: Karate 12.2 Runden in der fremden
Implementierung — das Paper sagt 12. Kontrast zur besten reversiblen
Baseline 15.9× (MH 194.2), Bar war 4×. (b) KERN CONFIRMED, NEBENKLAUSEL
FALSIFIED wie geschrieben: PS-Lifted ist über die 40×-Spanne KONSTANT
BIS FALLEND (BA-100: 14.0 → BA-1000: 11.0 → BA-4000: 11.0; Verhältnis
0.79 ≪ 2) — die Foss-Konvergenz-Signatur repliziert unabhängig. Aber
"die beste reversible Baseline wächst ≥ 4×" reißt: MH wächst nur 2.9×
(32.6 → 96.0, ~n^0.29 — BA-Graphen sind Expander-artig, die Bar war für
Bottleneck-Graphen kalibriert; Uniform wächst 9.8×, 63.6 → 621.0, und
erfüllt sie). Der registrierte Gegensatz konstant-vs-wachsend steht
trotzdem in beiden Baselines: bei n=4000 ist PS-Lifted 8.7× schneller
als MH und 56× schneller als Uniform, mit fallender eigener Kurve.
(c) CONFIRMED: Masse-Erhaltung hielt in jedem Lauf unter dem Assert
(kein Trial abgebrochen). Verdict in einem Satz: DIE FORMEL REPLIZIERT
DORT, WO SIE LEBT — eine fremde Implementierung aus dem Papiertext
reproduziert die konstante Rundenzahl quantitativ (12.2 vs. 12), die
Randklausel über die Gegner-Skalierung war zu grob kalibriert und wird
für Bottleneck-Graphen (SBM, Pfad/Ladder) nachregistriert, nicht
nachgebessert.

## P96 — Formel-Dynamiken im Organismus-Regime (ENTWURF)

Vier hart gepaarte Arme (identische Welt/Aktionen/Init/Gate, nur die
Update-Dynamik variiert): adam (Referenz = body-Lauf), rapidity
(Möbius-Momentum, |Schritt| < lr strukturell), lorentz (tau-LR-Schedule,
Peak = Baseline-LR 3e-4, Kühlung 0.95→0.5), rapidity_lorentz. 6000
Frames, Bewertung NUR auf der festen Probe-Route (Welt 999, Skript
9009), copy-last der Route als Nullpunkt (Konstante der Route).

- (a) DIE P87b-WUNDE SCHLIESST: mindestens ein Formel-Arm erreicht
  finale probe_L1 ≤ 0.95 × copy-last-der-Route UND < adams finaler
  probe_L1. (adam hat es im body-Lauf bei 6000 nicht geschafft; wenn
  eine Formel-Dynamik es schafft, ist das die erste konkrete
  Trainings-Verbesserung aus der Formeln-Linie im Organismus.)
- (b) IGNITION: bestes Formel-Arm-hit_AUC (Mittel über alle Checkpoints
  inkl. Frame 0) ≥ 1.15 × adams hit_AUC.
- (c) STABILITÄT: kein Arm divergiert (späte Lebens-L1 ≤ 1.5 × adam) —
  im codex-lab-Smoke war Möbius-Momentum der Ausreißer nach oben; ob
  die Lorentz-Grenze im Stream-Regime schützt oder schadet, ist genau
  die Frage.

Falsifier: (a) UND (b) reißen überall → die Formeln verbessern das
Organismus-Training in dieser Form nicht, und die ehrliche Antwort an
die FERTIG-Linie lautet: der Transfer Labor→Leben ist nicht gratis;
(c) reißt → die betroffene Dynamik ist im Stream-Regime schädlich und
wird so protokolliert.

**P96 GESCORT (2026-08-11, dynamics/results/organism_race.json).** DER
REGISTRIERTE FALSIFIER FEUERT VOLL: (a) FALSIFIED — kein Arm erreicht
0.95 × copy-last (Route-copy-last 0.0902; adam 0.0903, lorentz 0.0903,
rapidity_lorentz 0.0907, rapidity 0.0951). Die P87b-Wunde bleibt unter
allen vier Dynamiken offen; adam reproduziert sie auf der Probe-Route
aufs Zehntel-Promille (1.001×) — der Engpass ist NICHT die
Update-Dynamik. (b) FALSIFIED — bestes Formel-hit_AUC 0.557
(rapidity_lorentz) vs adam 0.543 = 1.026× ≪ 1.15×; lorentz' finaler Hit
0.646 vs adam 0.604 liegt innerhalb der Probe-σ (~0.05). (c) CONFIRMED
formal (kein Arm über 1.5×), aber mit dem protokollpflichtigen Befund:
PURES MÖBIUS-MOMENTUM SCHADET IM STREAM — rapidity ist in jeder Metrik
der schlechteste Arm (späte Lebens-L1 0.1298 = 1.38 × adam; hit_AUC
0.438 = 0.81 × adam), konsistent mit dem moebius-Ausreißer im
codex-lab-6-Step-Smoke. Plausible Mechanik, als Hypothese benannt: die
Rapidität akkumuliert über einen NICHTSTATIONÄREN Ein-Pass-Stream ohne
Epochen-Reset — tanh saturiert, der Optimierer fährt mit
Dauer-Vollimpuls durch Verteilungswechsel, die Lorentz-Grenze begrenzt
die Schrittweite, nicht die Sturheit. Der tau-Lorentz-Schedule allein
ist harmlos bis hauchdünn positiv (Lebens-L1 0.0930 vs 0.0943, −1.4%,
innerhalb Rauschen). EHRLICHE ANTWORT AN DIE FERTIG-LINIE, wie im
Falsifier vorformuliert: der Transfer Labor→Leben ist nicht gratis —
die Formel, die repliziert (P95), lebt im Konsensus-Raum; ihre
Optimierer-Ableger zahlen im Organismus-Regime (noch) nicht. Nächste
Registrierung, benannt: Rapiditäts-DECAY (w ← λw statt β₁w, λ<β₁) bzw.
Reset am Gate-Signal — die stream-gerechte Form des Möbius-Momentums,
falls es eine gibt.

---

## Scoring-Regel

Wie im Hauptregister: Verdicts mit Zahlen aus dynamics/results/
{pslifted_replication,organism_race}.json UNTER jedem Block; kein
Story-Fitting; Abweichungen sind Messungen.
