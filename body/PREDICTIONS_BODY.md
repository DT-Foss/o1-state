# PREDICTIONS — body/ (ENTWURF für den Merge ins Hauptregister)

Status: **DRAFT**. Diese Blöcke sind im Stil von analysis/PREDICTIONS.md
geschrieben und für den Merge dorthin bestimmt; die Nummern P87–P90 sind
VORSCHLÄGE (höchste vergebene Nummer beim Schreiben: P86) — beim Merge gegen
das Hauptregister prüfen. Der Merge selbst ist ausdrücklich Sache der
Haupt-Session (Grenzabsprache der Co-Work-Session vom 2026-08-11), nicht
dieses Verzeichnisses.

Registriert 2026-08-11 (Co-Work-Session, Cloud-Container, CPU), NACH
7/7 body/test_body.py und EINEM 400-Frame-Debug-Lauf
(body/results/mini_summary.json — Zahlen daraus: l1_late 0.0397,
cf_separation ~1.3e-5 flach, hit 0.40, 195 Records, Provenance 5/5), aber
VOR den Scoring-Läufen, gegen die diese Blöcke gewertet werden:

- Arm S-C: `run_body.py --source acted_synthetic --frames 6000 --policy curiosity --seed 42`
- Arm S-R: `run_body.py --source acted_synthetic --frames 6000 --policy random --seed 42`
- Arm D-C: `run_body.py --source vizdoom --frames 2000 --policy curiosity --seed 42`

Wissensstand für diese Vorhersagen: die Bestandsbefunde des Repos (P48:
deterministische Welt unter fixer Politik enthält nach Ignition keine
unableitbare Neuheit; P52: Engine bit-deterministisch unter (seed, actions);
Phase-1-Visual: copy-last dominiert, Residual-Head nötig) plus der
Debug-Lauf. Nichts aus den Scoring-Läufen.

---

## P87 — Der Körper läuft auf dem Stack (ENTWURF)

Aktions-konditionierte Frame-Vorhersage (Aktions-Embedding als zweiter
Encoder-Eingang, zero-init; Delta-Head zero-init) durch das UNVERÄNDERTE
F1/F2-Rezept (Gate-Konstanten read-only aus visual/frame_organism.py).
Registriert, je Arm S-C und S-R:

- (a) LERNEN ÜBER DIE LEBENSZEIT: späte mittlere Chunk-L1 (letzte 20%)
  fällt ≥ 30% unter das Ignition-Plateau (Mittel der ersten 15 Chunks).
- (b) DER KÖRPER SCHLÄGT DEN ZUSCHAUER-NULLPUNKT: späte L1 ≤ 0.8 × späte
  copy-last-L1 desselben Laufs. Begründung: copy-last zahlt bei jedem
  eigenen Pan die volle Konsequenz als Fehler; ein Modell, das die eigene
  Aktion kennt, muss genau diesen Anteil erklären können. (Phase-1-Kontext:
  beim Zuschauer-Organism war copy-last kaum schlagbar — der Unterschied
  IST die Aktionsinformation.)

Falsifier: (b) scheitert in beiden Armen, während (a) hält → das Modell
lernt Weltdynamik, nutzt die Aktionsinformation aber nicht einmal implizit;
P88 entscheidet dann, ob sie überhaupt ankommt.

**P87 GESCORT (2026-08-11, body/results/body_{sc,sr}_summary.json).**
(a) GESPALTEN: S-C PASS mit Raum (Plateau 0.1060 → spät 0.0429, Drop
59.5%); S-R FAIL (0.1060 → 0.0937, Drop 11.6%) — aber siehe die
(b)-Diagnose: S-Cs niedrige L1 ist zu großem Teil SELEKTION (die Politik
wählt HOLD-lastig, HOLD-Frames sind die leichten), nicht überlegene
Kompetenz. (b) FALSIFIED IN BEIDEN ARMEN: S-C 0.0429 vs copy-last 0.0390
(1.10×), S-R 0.0937 vs 0.0915 (1.02×) — der Körper schlägt den
Zuschauer-Nullpunkt bei 6000 Frames noch NICHT in roher L1. Der
registrierte Falsifier feuert NICHT (P88b hält, s.u.): die
Aktionsinformation kommt an, sie zahlt sich nur zuerst im
Counterfactual-Instrument aus, nicht in der Loss-Kurve — konsistent mit
der zero-init-Disziplin (der Kanal muss sich seinen Einfluss erst
verdienen). Budget-Befund im P88-Sinn, kein Mechanismus-Tod.

## P88 — Die Aktion ist im Modell angekommen (ENTWURF)

Counterfactual-Instrumente (drei 1-Schritt-Forwards aus DEMSELBEN
getragenen Zustand, Zustand geklont, nie mutiert — test_body.py Test 3):
Separation = mittlere paarweise L1 zwischen den A vorhergesagten
Nächst-Frames; Hit = argmin_a L1(cf_a, real) == getane Aktion. Registriert
auf Arm S-C (6000 Frames):

- (a) SEPARATION WÄCHST: cf_separation_late ≥ 5 × cf_separation_early
  (early = erste 20 Chunks, late = letzte 20%). Zero-init macht den
  Startwert ~0; jedes Wachstum ist verdienter Gradient, kein Prior.
- (b) HIT ÜBER ZUFALL: cf_hit_rate_late ≥ 0.45 bei Chance 1/3 — die
  Vorhersage unter der GETANEN Aktion passt besser zur Realität als die
  unter den ungetanen.

Falsifier (der Körper-Claim stirbt wie registriert): L1 fällt (P87a hält),
aber (a) UND (b) bleiben aus — dann ist das Netz ein Zuschauer geblieben,
der ein Aktions-Eingangsfeld ignoriert, und die Architektur-Lücke ist NICHT
geschlossen, nur verschoben. Ehrlicher Unsicherheitsvermerk: 6000 Frames
sind eine Schätzung, keine Messung; scheitert (a)/(b) knapp bei weiter
steigender Kurve, ist das ein Budget-Befund und wird als solcher
gekennzeichnet (P9-Präzedenz: "a stuck curriculum is a budget statement").

**P88 GESCORT (2026-08-11).** (a) FALSIFIED wie registriert: S-C-Separation
1.9e-5 → 3.1e-5 = 1.63× (Bar: 5×). (b) CONFIRMED: hit_rate_late 0.533 bei
Chance 0.333 — die Vorhersage unter der getanen Aktion passt besser zur
Realität als unter den ungetanen; die Aktion IST angekommen. Der
Quer-Arm-Befund ist schärfer als beide Klauseln: der RANDOM-Arm lehrt den
Aktionskanal SCHNELLER (Separation ×4.6 auf 8.7e-5, Hit 0.653) — die
LP-Politik hat mit ihrer frühen HOLD-Präferenz genau die Aktionen
ausgehungert, deren Konsequenzen das Embedding lernen muss
(Explorations-Starvation: das Motiv "folge dem Lernfortschritt" bremst den
Fortschritt dort, wo noch keiner messbar ist). Gegenstück zur P48-Lektion
(ein geteiltes Gate verdünnt 8 Körper): ein kontextfreier Bandit verdünnt
die eigene Neugier. Der zustandskonditionale LP-Kopf ist damit nicht mehr
nur "nächster Schritt", sondern die gemessene Notwendigkeit.

## P89 — Neugier als Politik (ENTWURF)

Learning-Progress-Bandit (Oudeyer-Stil; LP(a) = mean(ältere Fensterhälfte)
− mean(neuere) der per-Frame-L1 auf a-Frames; softmax über z-scores,
ε=0.1-Boden, Ignition uniform). Kontextfrei — Tag-1-Körper; die
zustandskonditionale Politik ist hiermit als NÄCHSTER Schritt benannt,
nicht eingeschmuggelt. Registriert, Arm S-C vs. Arm S-R:

- (a) EINE PRÄFERENZ EXISTIERT: post-Ignition verlässt mindestens eine
  Aktionswahrscheinlichkeit das Uniform-Band [0.28, 0.39] in ≥ 25% der
  Chunks (S-R liegt per Konstruktion bei exakt 1/3).
- (b) DIE PRÄFERENZ ENTWICKELT SICH: die bevorzugte Aktion (argmax p) ist
  über die Lebensdrittel NICHT konstant, ODER die maximale Abweichung von
  uniform schrumpft im letzten Drittel gegenüber dem mittleren um ≥ 30%
  (Habituation auf Aktionsebene: Interesse erlischt am Gemeisterten).
- (c) NEUGIER SCHADET NICHT: späte L1 von S-C ≤ 1.05 × späte L1 von S-R
  bei gleichem Frame-Budget. (Bewusst schwach: der Welt-Reichtum ist
  symmetrisch; ein ECHTER Vorteil ist hier nicht registriert, nur
  Nicht-Schaden. Wo Neugier zahlen müsste — ungleiche Lernbarkeit pro
  Aktion — ist Doom, siehe P90-Kontext, aber dort fehlt der Random-Twin
  in dieser Welle: bewusst klein gehalten.)

Falsifier: Politik bleibt uniform (Neugier wacht nie auf) ODER friert
statisch auf einer Ecke fest ohne (b)-Bewegung (Noise-Lock statt
Entwicklung) ODER (c) reißt (das Motiv kostet Kompetenz).

**P89 GESCORT (2026-08-11).** (a) CONFIRMED mit Maximum: 360/360
post-Ignition-Chunks (100%) außerhalb des Uniform-Bandes — eine Präferenz
existiert unmissverständlich (HOLD-dominant). (b) FALSIFIED wie
registriert: argmax über die Drittel konstant HOLD (mean-probs-Drittel:
maxdev 0.104 → 0.154 → 0.144; Schrumpfung 6.5% < 30%) — ABER der
End-Schnappschuss zeigt die Wende: finale Politik [0.09, 0.69, 0.22],
PAN_RIGHT übernimmt in den letzten Chunks, genau wenn das Embedding
anfängt, Pans erklärbar zu machen (LP wird dort positiv). Die Entwicklung
kam, nur später als die Drittel-Mittel-Bar sie messen konnte —
Budget-Befund, die Trajektorie liegt in body_curves.png offen. (c)
CONFIRMED (0.0429 ≤ 1.05 × 0.0937), mit dem im Block P87 benannten
Selektions-Caveat: L1-Vergleiche zwischen Politiken sind konfundiert, weil
jede Politik ihre eigene Testverteilung wählt — ein Politik-Vergleich auf
gemeinsamer Probe-Route ist das saubere Instrument der nächsten Welle und
wird dort registriert.

## P90 — Das Auge schreibt ins Weltbuch (Stufe 2, ENTWURF)

Gehandelte Ereignisse als Causal-Records im livecausal-Vokabular
(trigger_key "pressed:<aktion>" / mechanism "view_shift" / outcome_key
"view_shift:dx±k"), Quote = (env, base_seed, episode, frame, action,
dx, sha256 beider Frames); versiegelt via UNVERÄNDERTEM
src/livecausal/store.py in einen Store UNTER body/ (Entwurfs-Store, kein
Merge in Bestands-Stores). Weil unter gelernter Politik die Aktionsfolge
nicht mehr seed-ableitbar ist, reist der VOLLE Action-Trace als
Provenance-Träger mit (–_actions.json). Registriert:

- (a) SYNTHETIK, VOLLE WAHRHEIT: Arm S-C liefert ≥ 100 acted_event-Records
  und für JEDEN Record gilt dx_measured == dx_truth der Welt (die Welt
  exponiert die Wahrheit; ein einziger Widerspruch falsifiziert den
  Extraktor). [Debug-Lauf-Anker: 195 Records/400 Frames, 0 Widersprüche
  im Test — die Registrierung verlangt es bei 6000 erneut.]
- (b) REPLAY BIT-EXAKT: 5/5 gesampelte Records reproduzieren beide
  Frame-Hashes UND denselben dx durch Welt-Replay aus (seed, trace).
- (c) DOOM, ECHTER ENGINE-SHIFT: Arm D-C liefert ≥ 20 Records; unter den
  Records mit trigger pressed:move_left haben ≥ 80% dasselbe dx-Vorzeichen
  (negativ per Konvention f1(x)≈f0(x+s)), spiegelbildlich für move_right.
  (3D-Parallaxe macht den globalen Shift approximativ — deshalb 80%, nicht
  100%; attack/hold erzeugen erwartet KEINE Shift-Records, ihre Abwesenheit
  ist die ehrliche Ausgabe des dummen Detektors.)
- (d) DOOM-REPLAY BIT-EXAKT: 5/5 durch die Engine (P52s
  Fresh-Engine-Factory; scheitert dies, ist die Trace-Maschinerie kaputt,
  nicht die Engine — so herum wird es dann auch geschrieben).

Falsifier: (a)-Widerspruch (Extraktor erfindet Kausalität), (b)/(d)-Fail
(Provenance-Disziplin bricht unter Politik), (c) < 80% (der Shift-Detektor
trägt in 3D nicht und Stufe 2 braucht ein besseres Konsequenz-Organ, das
dann registriert — nicht nachgebessert — wird).

**P90 GESCORT (2026-08-11).** (a) CONFIRMED: 1522 Records im S-C-Arm,
**0 Widersprüche** zur Welt-Ground-Truth über ALLE Records (nicht nur
Stichprobe). (b) CONFIRMED: 5/5 bit-exakt (Hashes + dx-Redetektion), in
S-C UND S-R — die Provenance-Disziplin hält unter gelernter Politik, der
Action-Trace trägt. (c) FALSIFIED wie registriert, der Falsifier feuert
sauber: 0 Doom-Records. Diagnose (gemessen, forcierte Strafe-Proben):
NICHT der HUD (basic.cfg: render_hud=false), sondern (i)
Nearest-Neighbor-Downsampling zerstört Shift-Kohärenz (beste
Improvement-Ratio 0.88–0.94 vs. Bar 0.6) und (ii) Doom-Strafe hat
MOMENTUM — der erste Tastendruck bewegt kaum (dx 0..1), erst aufgebaute
Geschwindigkeit shiftet 2–3 px; die Konsequenz derselben Taste ist
zustandsabhängig. Die Vorzeichenstruktur ist in den Proben sichtbar
(move_left → dx<0, move_right → dx>0 bei Momentum) — der Mechanismus lebt,
das Organ ist zu grob. v2-Organ (hiermit BENANNT für die nächste
Registrierung, nicht nachgebessert): block-mean-Downsample statt
nearest-neighbor, Ratio-Bar auf forcierten Proben kalibriert,
Momentum-bewusste Attribution (Konsequenz als Funktion von Taste UND
Geschwindigkeitszustand). (d) NICHT WERTBAR (0 Records — vakant, nicht
bestanden).

---

## Scoring-Regel

Wie im Hauptregister: Verdicts (CONFIRMED / FALSIFIED / PARTIAL, mit
Zahlen aus body/results/*_summary.json) werden nach den Läufen UNTER jedem
Block ergänzt; kein Story-Fitting, Abweichungen sind Messungen.
