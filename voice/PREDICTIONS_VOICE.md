# PREDICTIONS — voice/ (ENTWURF für den Merge ins Hauptregister)

Status: **DRAFT** im Stil von analysis/PREDICTIONS.md; Nummern P91–P94 sind
VORSCHLÄGE (Stand beim Schreiben: origin/main endet bei P90; Merge macht
die Haupt-Session, nicht dieses Verzeichnis).

Registriert 2026-08-11 (Co-Work-Session, zweite Welle, Cloud-Container,
CPU), NACH 7/7 voice/test_voice.py und offengelegten Debug-Smokes auf
einem 1500-Frame-Leben (dabei gefunden und gefixt: Provenance-Verifier
replayte die falsche Weltklasse → eigener RichPan-Verifier; Tie-Guard für
die Hit-Metrik, weil zero-init-Counterfactuals identisch sind; Sprecher
braucht Schlaf-Replay-Pässe + Delta-Skalierung 4.0 — Sweep-Zahlen bei
1500 Frames: dir 0.77–0.83, mag 0.57–0.58, flip 0.40–0.55, comp ~0.03),
aber VOR dem Scoring-Lauf, gegen den gewertet wird:

- `python3 voice/run_voice.py --stage all --seed 42 --frames 8000 --b-seed 777 --b-frames 4000 --probe-every 500 --probe-len 96`

Wissensstand: die Debug-Smokes oben, P87–P90 (insb. die
Explorations-Starvation und der 6000-Frames-Budget-Befund), P47
(Schlaf-Replay: halb Inhalt, halb Regularisierung — deshalb existiert der
Gift-Arm), P90a (Extraktor == Ground-Truth auf Synthetik). Nichts aus dem
Scoring-Lauf.

Design-Anker (für den Leser des Registers): A lebt 8000 Frames unter
SEEDED RANDOM (P88-Lektion: Random lehrt den Aktionskanal am schnellsten —
As Job ist ehrliche Erfahrung, nicht Klugheit); der Sprecher sieht NUR
(f_t, Δf), nie die Aktion; B-Arme sind modell-gepaart (bit-identische
Init-Gewichte, identisches echtes Frame-Budget, identische Welt und
Aktionsfolge — Arme unterscheiden sich AUSSCHLIESSLICH im Gehörten);
Probe = feste Route in separater Welt (Seed 999), der P89c-Confound ist
damit konstruktiv tot.

---

## P91 — Der Sprecher: Sprache aus eigenem gelebtem Handeln (ENTWURF)

Supervision ausschließlich aus As eigenen acted Records (SILENCE sonst;
Aktions-Slot überall aus dem wahren Trace). Gewertet auf dem Held-out-Tail
(letzte 15% des Lebens, nie im Loss):

- (a) KOMPETENZ: dir_acc ≥ 0.85 UND mag_acc ≥ 0.60 auf Konsequenz-Schritten.
- (b) KOMPOSITION: Die Kombination (right, 2) war in KEINEM Loss-Schritt
  (maskiert, auch im Trainingsbereich). Satz-Accuracy auf diesen
  Schritten ≥ 0.30 (Zufall ≈ 1/12); Hoffnungs-Marke ≥ 0.5. Der Smoke
  sagt ~0.03 — DIES ist die riskanteste Klausel, und beide Ausgänge
  tragen: Bestehen ⇒ die Slots sind entkoppelt (Systematizität gratis);
  Scheitern bei gehaltenem (a) ⇒ die Köpfe binden konjunktiv, und
  Systematizität ist NICHT gratis — dann ist genau das der Befund.
- (c) INTERVENTION (die Kernel-Achse im Kleinen): erzwungene Konsequenz
  links vs. rechts am selben Zustand → das Richtungswort flippt in
  ≥ 0.85 der Proben (Punktschätzung 0.9+; Smoke bei 1500: 0.55, der
  Scoring-Lauf hat 5.7× mehr Trainingsschritte).
- (d) EPISTEMIK — Sprache reicht genau so weit wie sichtbare Evidenz:
  act_acc auf Konsequenz-Schritten ≥ 0.85 (Weltbijektion sichtbar),
  act_acc auf stillen Schritten ≤ prior + 0.10 (der Druck ist dort
  prinzipiell unsichtbar; Übertreffen des Priors wäre ein LECK — z.B.
  Weltzustands-Korrelationen — und würde als solches seziert, nicht
  gefeiert).

**P91 GESCORT (2026-08-11, voice/results/speaker_summary.json; A-Leben
8000 Frames, 5159 Records, Provenance 5/5).**
(a) CONFIRMED mit Raum: dir 0.991, mag 0.806, Satz 0.802 auf 752
Held-out-Konsequenz-Schritten. (b) FALSIFIED, deutlich: comp 0.022 auf 864
nie-im-Loss-Schritten — und genau die benannte zweite Lesart trägt:
(a) hält, also binden die Köpfe konjunktiv; SYSTEMATIZITÄT IST NICHT
GRATIS. Der Satz (right, 2) ist aus (right,1/3) und (left,2) nicht
komponierbar, obwohl beide Köpfe getrennt softmaxen — die Repräsentation
darunter ist verklumpt. Das ist der schärfste Einzelbefund dieses Blocks
und die Messlatte für jede künftige "kompositionale" Architektur-Behauptung
im Repo. (c) KNAPP FALSIFIED wie registriert: flip 0.817 vs Bar 0.85
(Punktschätzung 0.9+ war zu forsch; 60 Proben, σ≈0.05 — Budget/n-Befund,
Mechanismus sichtbar da weit über Zufall). (d) CONFIRMED in beiden
Klauseln, sauber: act_acc 0.992 auf Konsequenz vs 0.938 auf still bei
prior 0.929 (+0.009, kein Leck; silence_said 1.000 — der Mund schweigt
perfekt, wo nichts zu sehen war). Sprache reicht exakt so weit wie die
Evidenz — gemessen, nicht postuliert.

## P92 — Transmission: Hörensagen beschleunigt echte Verkörperung (ENTWURF)

B-hear-codec (Ground-Truth-Sätze) vs. B-silent, identisches echtes
Budget, Imagination offline (Kultur ist billig, Erfahrung teuer):

- (a) SOFORT-PRÄFERENZ: probe_hit(hear_codec) ≥ 0.30 bei Frame 0 (vor
  jedem echten Schritt; silent liegt per Tie-Guard bei ~0) — Hören
  allein erzeugt eine reale, richtige Aktionspräferenz.
- (b) VORSPRUNG ÜBER DAS LEBEN: probe_hit(hear_codec) ≥ probe_hit(silent)
  + 0.10 am ersten Post-Life-Checkpoint (~Frame 500) UND ≥ +0.05 am
  letzten Checkpoint.
- (c) IGNITION: erster Checkpoint mit hit ≥ 0.45 kommt bei hear_codec
  ≤ 0.5 × dem von silent (erreicht silent die Marke in 4000 Frames nie,
  zählt hear_codec ≤ 2000 als Bestehen).
- Ehrlicher Nicht-Anspruch: probe_L1 bei Frame 0 darf bei hear_codec
  SCHLECHTER sein als silent (die Rand-Lüge der Imagination ist im
  Modul benannt); registriert ist nur L1(hear) ≤ 1.02 × L1(silent) am
  letzten Checkpoint (die Lüge wird vom echten Leben ausgewaschen).

**P92 GESCORT (2026-08-11, voice/results/transmission_summary.json).**
(a) CONFIRMED: hit 0.333 bei Frame 0 vor jedem echten Schritt (silent per
Tie-Guard 0.000) — Hören allein erzeugt reale, richtige Präferenz.
(b) GESPALTEN: am letzten Checkpoint CONFIRMED mit Raum (0.594 vs 0.396,
+0.198 ≫ +0.05); am ersten Post-Life-Checkpoint FALSIFIED (0.417 vs
silent 0.521+0.10) — silent SPIKET bei 512 auf 0.521 und ZERFÄLLT dann
monoton auf 0.396, während alle hörenden Arme ihr Niveau halten. Die
Registrierung hat silents Nicht-Monotonie nicht vorhergesehen; der
eigentliche Befund ist schärfer als die Klausel: DAS STILLE LEBEN VERLIERT
SEINE PROBE-ÜBERTRAGUNG WIEDER (Welt-777-Spezialisierung), HÖREN IMPFT
DAGEGEN. (c) FALSIFIED als Metrik-Artefakt, offen seziert: silents
"Ignition 512" ist genau jener transiente Spike über eine Schwelle, unter
die es am Ende zurückfällt (0.396 < 0.45) — first-crossing auf einer
96-Schritt-Probe ist zu fragil, die Metrik (nicht der Mechanismus) ist
gestorben. L1-Klausel CONFIRMED (0.0623 ≤ 1.02×0.0618). WICHTIG: per
P93(b)-Vorbindung ist die registrierte BEDEUTUNG von P92 ("Inhalt ist der
Träger") durch P93s Ausgang mit-belastet — siehe dort, inklusive der
Gift-Diagnose, die diese Vorbindung selbst relativiert.

## P93 — Das Gift: verdrehte Sprache hilft nicht (ENTWURF)

B-scrambled hört konsistente Lügen (links↔rechts, Magnituden permutiert;
Stille bleibt Stille — gleiche Paarzahl, gleiche Gradientenschritte wie
hear_codec):

- (a) ab dem ersten Post-Life-Checkpoint gilt an JEDEM Checkpoint:
  probe_hit(scrambled) ≤ probe_hit(silent) + 0.03. (Frame 0 ist
  ausgenommen und wird berichtet: ein systematisch belogenes Modell hat
  ECHTE, falsche Präferenzen — auf Hold-Schritten trifft es trotzdem;
  der Tie-Guard-Nullpunkt von silent ist dort kein fairer Vergleich.)
- (b) DIE TÖDLICHE LESART IST BENANNT: Hilft scrambled dennoch
  (> silent + 0.03 nachhaltig), dann ist der P92-Effekt
  Regularisierung statt Inhalt (die P47c-Dekomposition), und P92 stirbt
  in seiner registrierten Bedeutung MIT — Inhalt muss der Träger sein,
  nicht Gradientenmasse.

**P93 GESCORT (2026-08-11) — FALSIFIED, UND DIE AUTOPSIE DREHT DEN
BEFUND.** (a) reißt an jedem Post-Life-Checkpoint (scrambled 0.552–0.646
vs silent+0.03) — das Gift half. ABER die Post-hoc-Diagnose (Frame-0-
Zerlegung nach Aktionstyp, voice/results/poison_v2_diagnostic.json) fand
den Konstruktionsfehler: im Codec leitet sich das Aktionswort per
Weltbijektion aus dem RICHTUNGSWORT ab — die registrierte Richtungslüge
tauscht daher Aktion UND Shift-Vorzeichen GEMEINSAM und lässt die
sensomotorische Zuordnung intakt (gelogen wurde effektiv nur über
Magnituden; Frame-0-Hits des "belogenen" Modells: PAN_LEFT 0.588,
PAN_RIGHT 0.727). Das registrierte Gift war selbstheilend; (b) ist damit
NICHT etabliert — der Lauf konnte Inhalt und Regularisierung nicht
trennen. Der klar etikettierte DIAGNOSE-Arm poison_v2 (wahres Aktionswort,
Vorzeichen geflippt — die Lüge, die die Bijektion nicht heilt; KEIN
registrierter Claim) zeigt die echte Struktur: Frame 0 = 0.208 (die Lüge
zeigt aktiv in die falsche Richtung: Wahrheit 0.333 > Lüge 0.208 >
Stille 0.000), aber das Leben wäscht den Inhalt aus und behält den
geweckten Kanal (Ende 0.521, immer noch +0.125 über silent). Gemessene
Dekomposition am letzten Checkpoint: Kanal-Weckung (jedes konsistente
aktions-konditionierte Vortraining) ≈ +0.13…0.16; Wahrheits-Marge
obendrauf ≈ +0.04…0.07 bei Probe-σ≈0.05 — Richtung stimmt, Signifikanz
bei n=1 nicht. DIE NÄCHSTE REGISTRIERUNG ist damit diktiert: poison_v2
als registrierter Arm, Seed-Sweep, längere Probe — Bedeutung von der
Weckung sauber getrennt. (Und die HOLD-Beobachtung fürs Protokoll: alle
Imaginations-Arme haben Frame-0-HOLD-Hit 0.0 — gelernt wird nur, wovon
gesprochen wurde; Stille lehrt nichts, auch das ist eine Eigenschaft von
Unterricht.)

## P94 — Der volle Kreis: der GELERNTE Mund trägt (ENTWURF)

B-hear-learned hört nicht die Codec-Wahrheit, sondern die tatsächliche
Erzählung des trainierten Sprechers über As Leben — Fehler, Lücken,
falsche Stillen inklusive:

- (a) am letzten Checkpoint: benefit(hear_learned) ≥ 0.6 ×
  benefit(hear_codec), mit benefit(arm) = probe_hit(arm) −
  probe_hit(silent). Gelebtes Wissen → gelernte Sprache → fremde
  Gewichte, Ende zu Ende.
- (b) Scheitert (a), entscheidet die Fehlerrechnung, WO der Kreis reißt:
  liegt Sprecher-Satz-Accuracy (P91a) über 0.8 und (a) reißt trotzdem,
  ist der Imaginations-Kanal fehlertoleranzarm (Lügen einzelner Sätze
  wiegen schwerer als Stille); liegt sie darunter, war der Mund zu jung
  — beides wird geschrieben, nichts wird gemittelt.

**P94 GESCORT (2026-08-11).** (a) FALSIFIED, knapp: benefit(learned)
0.104 vs 0.6 × benefit(codec) 0.119 (Verhältnis 0.525). (b) Die
Fehlerrechnung: Satz-Accuracy 0.802 liegt exakt AUF der 0.8-Grenze —
formal greift die erste Lesart (Kanal fehlertoleranzarm), aber die
Trajektorie widerspricht ihr im Kern: bei Frame 0 trägt der gelernte Mund
FAST VOLL (0.323 vs Codec 0.333), bei 512 führt er sogar (0.688, Maximum
aller Arme im ganzen Experiment), und erst der letzte Checkpoint (0.500
vs 0.594) entscheidet die registrierte Zahl — bei Probe-σ≈0.05 und n=1
Seed ist das Tail-Rauschen, nicht Kanal-Bruch. Ehrliches Verdict: DER
VOLLE KREIS TRÄGT ZU BEGINN NACHWEISLICH (gelebtes Wissen → gelernte
Sprache → fremde Gewichte, Frame-0-Parität mit der Wahrheit), und ob er
am Lebensende die 0.6-Marge hält, entscheidet erst der Seed-Sweep — der
hiermit, zusammen mit dem poison_v2-Arm, als nächste Registrierung
benannt ist. Nichts wird gemittelt, die 0.525 bleibt stehen.

---

## Scoring-Regel

Wie im Hauptregister: Verdicts mit Zahlen aus
voice/results/{speaker,transmission}_summary.json UNTER jedem Block,
nach dem Lauf; kein Story-Fitting; Abweichungen sind Messungen.
