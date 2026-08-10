# POS — Design Decisions (Phase A build log)

One line per decision, in order. Context: BRIEFING_POS.md + David's session directives.

- **Threads = 1** (not repo's ncpu−2): measured under the live SAT-solver load, 8 threads is ~30× slower than 1 for these tiny ops (fwd+bwd 1436 ms vs 46 ms idle-sample; median ~1.2 s under load either way), and 1 thread is bit-deterministic → G2.
- **B=8, K=64 kept** (exact streaming_train recipe) despite B=64 benching ~5× more tok/s: chunk-mean surprise over 512 tokens is the committed gating signal; batching 32+ parallel document streams would average the signal away, and G1 parity to streaming_train (8.69→5.22 @3M) is literal with B=8.
- **Expected volume ≈ 30M streamed tokens in 40h** under current machine load (~220 tok/s ensemble measured); thesis needs ~3M+ (the whole published streaming_train curve) → 10× headroom; tok/s logged live in status.json, Phase B reads actuals.
- **Eval cadence = wall-clock 900 s** (200k-token frozen WT-2 val slice, ~56 s for 4 arms measured): throughput varies with external load, token-scheduled evals would swing 15 min↔hours; smoke/G2 use `--eval-every-tokens` (deterministic token scheduling).
- **G2 determinism gate = sha256 digest** over all chunk/eval metric fields (wall-clock/RSS excluded), printed at run end; two smoke runs must produce identical digests.
- **A3 gating mechanics**: forward always no_grad (briefing wording); gated chunks recompute the forward WITH graph from the same pre-chunk state (identical values, weights unchanged) then backward+step; threshold = q-quantile (q=0.80) of a 500-chunk rolling window of chunk-mean surprises, window updated AFTER the gate decision; first 100 chunks always backward (ignition) and counted in gradient tokens.
- **Index warmup = 5M streamed tokens** (David's ~5M guideline; ≈6 h at current load, A3 has ~1.2M grad tokens by then and the surprise curve is past its steep fall) — final call confirmed after the smoke surprise curve.
- **Index/probe RNG isolated** in a dedicated torch.Generator (seed 42) so probes never touch arm determinism; probes run on cloned, row-sliced states (side-trips, never the live stream state).
- **Twin fork at wall-clock 24 h** (briefing): weights copied, Z=None + fresh Adam; twin INHERITS A3's rolling window and gates immediately (no ignition) — its elevated early gating is itself the measured warmup cost of a restart.
- **Reset-twin heldout == A3 at fork instant by construction** (stateless eval, same weights); the warmup cost shows in online stream surprise and post-fork gradient spend — both logged per chunk.
- **C4 stream resilience**: doc-counting wrapper with reconnect + ds.skip(docs) on network error (backoff ≤300 s); reconnects logged in status.json; exact data order guaranteed for smokes, best-effort across reconnects in the 40 h run.
- **Machine guard = pause, not kill** (briefing): RSS>12 GB or disk<5 GB → checkpoint, status "paused", re-check every 60 s; RSS-pause exits cleanly after 30 min (own RSS won't shrink by waiting); disk-pause waits indefinitely. NOTE: only ~6 GB free at launch — flagged to David.
- **Smoke outputs are tag-isolated** (`results/pos_<tag>_*`) so gates never clobber the real run's files.
- **q calibrated 0.80 → 0.75 after G3**: at q=0.80 the tail gate fraction measured 13.3% (the falling loss trend pushes chunks below the rolling window's quantile); recheck at q=0.75 measures 19.4% post-ignition / 15.3% tail — in the 15–30% band, and the fraction drifts toward nominal 25% as the curve flattens.
- **Probe logging 6 decimals + lookahead 32→16**: measured the injection effect on the young model at ~1e-4, concentrated in the ~6-token receptive field — 4dp logging would round it to zero and a 32-token lookahead dilutes it 2× (closed_loop precedent: 12); the state effect itself is real (max|ΔZ| = 1.29 between injected and random advance).
- **WP4 built and launched live** (David's go, revised from the earlier defer): src/holo_stream_recall.py marries the holographic complex write with detach-carried streaming state; equivalence full-vs-chunked < 1.2e-6; smoke P=1: carried recall 100% through gap 8 across chunk boundaries, zeroed-at-gap null at chance — the --full sweep (P∈{1,2,4}×G∈{0,8,32,128}, 2 seeds) runs nice-19 beside the long run; per David the historic ~9% MQAR ceiling is one agent-day of exploration, not a law — no ceiling assumption in the code.
- **Smoke checkpoints (47–67 MB each) deleted, not committed**: runtime artifacts only; every gate is evidenced by the committed jsonl/json logs; the long run's pos_ckpt.pt stays untracked (resume-only).
- **verify_pos.py two-tier**: hard PASS/FAIL only on data integrity + internal consistency; thesis numbers (A3/A2 ratio, injection deltas, twin warmup) are computed and printed as measured headlines — deviations are data points, not failures (briefing: Zielbild is orientation, not stop criterion).
- day4 ~10:30 — METHOD NOTE (rank-sweep builder's find): torch thread count
  changes TRAINING DYNAMICS near ignition boundaries, not just speed — the
  phase arm's 8.9% MQAR ceiling ignites under default multithreading (5/5
  historic seeds + 1/1 rebuild) but dies at threads=1 (0/4 seeds, ~4 sigma
  below reference). Reduction-order sensitivity in the early unstable phase.
  Consequences: (1) instruments measuring against a historic reference must
  reproduce the reference's threading regime; (2) all existing knee-arc /
  POS results remain internally valid (every arm-vs-arm comparison ran in
  ONE regime — fairness holds), but cross-regime numeric comparisons are
  not meaningful; (3) rank_sweep.py runs at torch default threads; the x86 runner
  striping compensates with fewer parallel stripes (4 stripes x 4 threads).

## Standing measurement rule: the cadence axis (2026-08-05, MS-G audit)

Any metric with per-chunk cost in its denominator — stall ratios, replay
speed, snapshot cost in "chunk slots", tok/s comparisons — is meaningless
unless (batch, chunk, d_model) are (a) set explicitly by the run and
(b) recorded IN THE SAME ARTIFACT the metric ships in. Width alone does not
make a chunk compute-heavy: chunk WEIGHT is B×K, and a subprocess that
inherits module defaults silently measures the defaults, not the system.
This killed P39 (a)/(c) twice — first as toy-cadence inflation (17.9×/3.79×
→ 7.1×/2.2× at real cadence), then as a real residual that falsified both
checks. Full audit: results/cadence_audit.json. Enforcement: portable_organism
and moebius_stage now expose and forward --batch/--chunk-size; every new
harness that reports a cost ratio must embed its cadence config in its
result JSON.

## 2026-08-05 — CPU time on multi-core machines is not a work measure

P50 measured it directly: on the x86 runner (16 cores), process CPU time runs at
13.6–15.5× wall for single-threaded torch work, and the inflation
survives torch.set_num_threads(1) AND OMP/MKL/OPENBLAS env pins set
before import. The pool spins regardless; RUSAGE_SELF counts every
spinning thread. Any historical CPU-time ratio from the x86 runner compared two
fictions (P39's "replay ≈2× live CPU" among them — decomposed and
retired by P50). Standing rule: cost claims use WALL time, per-chunk
MEDIANS after warmup, on both sides of any ratio; CPU time may be
recorded but never gates a verdict on a multi-core machine. Digest
design corollary (found by DECISIVE 1): run digests must hash only
chunk-indexed events, never wall-clock-driven ones (eval_every_s lines
made two bit-identical 50M runs hash differently).

## 2026-08-06 — Register scoring is machine-checkable by construction

The v2 auto-scorer audit (75 clause checks across 59 P-numbers, zero
mismatches, results/scorer_audit.json) measured where the register
resists machine verification, and the fixes become standing rules for
every FUTURE scoring: (1) every scoring artifact writes
p<N><clause>_pass booleans next to the raw numbers (P46/P47/P48/P52
are the pattern; 16 scored Ps carry their verdicts only in prose or
verdict strings); (2) every SCORED header names its results/*.json
path (P38/P42 did not); (3) every clause mark (a)/(b)/(c) in a scoring
text carries an explicit verdict word directly after it. Historical
entries stay as they are — the ledger is append-only; the scorer
handles them as prose.

## 2026-08-06 — every harvest writes stranger-verifiable coordinates

The P60 build measured it: the 40h POS index (results/pos_index.jsonl)
carries only a global token position (n) and NO doc_coord — a stranger
cannot re-instantiate the stream at the find without re-tokenizing the
whole prior life. knowledge_file_run and the vizdoom harness write full
coordinates and verify 5/5 and 10/10. Standing rule: every index/harvest
artifact records the coordinate tuple a stranger needs for direct stream
re-instantiation (doc_coord for text; (lane_seed, episode, frame,
offset) for simulated worlds). An entry without stranger coordinates is
storage, not knowledge.

## 2026-08-06 — chain triggers must key on status, not on a pgrep process pattern

Fehler diagnostiziert: die Wave-9-Ketten warteten mit
`while pgrep -f "tag s43_50_dXXX" >/dev/null; do sleep; done`. Das Muster
`tag s43_50_dXXX` steht wörtlich in der Kommandozeile des Wächters selbst,
daher matcht `pgrep -f` immer den Wächter — der Loop terminiert nie, auch
nachdem der eigentliche pos_run-Lauf längst `done` ist. Folge: fünf
beast-Glieder (P55–P61) und P57 auf core zündeten stundenlang nicht,
obwohl die Läufe fertig waren.
Regel: Chain-Trigger lesen den Zustand aus dem Artefakt
(`json.load(status.json)["phase"]=="done"`), nicht aus einem
Prozess-pgrep-Muster. Ein `pgrep`-Trigger ist nur zulässig, wenn das
Suchmuster garantiert NICHT in der eigenen Kommandozeile vorkommt
(z. B. Match auf den absoluten Interpreter-Pfad + Skriptname, nicht auf
ein CLI-Argument, das der Wächter selbst zitiert).

## 2026-08-06 — Cross-ISA-Spezifikation: Token-Quantisierung ist Integer-Arithmetik (P60)

Die x86→ARM-Rückverifikation (P60) fand genau eine divergente Koordinate
in 10 Stichproben: ein Block-Mittelwert auf einer Quantisierungs-Grenze
kippt zwischen den ISAs um ein Level (2 vs 1), weil `tokenize_frame`
einen FLOAT-Mean über 80 uint8-Pixel bildet und dann float-floor-dividiert
(256.0/12) — NumPys vektorisierte Reduktionsreihenfolge (NEON vs AVX)
unterscheidet sich um 1 ULP, und an der Bin-Grenze entscheidet das ULP
das Token. Innerhalb jeder ISA ist die Rechnung deterministisch
(Konsens 10/10 beidseitig).
Regel: Jede pixel-/sensor-zu-Token-Quantisierung in einem Harness, dessen
Dateien fremdverifizierbar sein sollen, wird INTEGER-EXAKT spezifiziert —
`level = (block_sum * LEVELS) // (n_pixels * 256)` auf der exakten
uint8-Summe, nie über einen Float-Mean. Bestehende Artefakte behalten
ihre gemessene v1-Map (Vergleichbarkeit); jeder NEUE Harness nutzt die
Integer-Form.

## 2026-08-06 — Harvest ohne Koordinaten ist ein Regelverstoß (P55-Lücke)

Der P55-Harness (`keyed_file_run.py`) friert Spans nur als Token-Listen
mit sha256 ein — ohne doc_coords und ohne Entries-jsonl. Folge: die
Datei ist NICHT fremdverifizierbar (die P60-Richtung-2-Verifikation
musste übersprungen werden, "substrate_absent"). Die bestehende Regel
("every harvest writes stranger-verifiable coordinates") gilt für JEDEN
Datei-Producer, auch wenn die Vorhersage selbst nur Speicherung misst.
Fix-Liste: keyed_file_run und filter_file_run schreiben beim nächsten
Anfassen ein Entries-jsonl mit doc_coord je Span.

## 2026-08-10 — Gescorte Store-Artefakte sind read-only (P73-Vorfall)

Ein Agent-Consult-Lauf schrieb ein use.ledger direkt in
results/p72_store_run1 (das gescorte P72-Artefakt), parallel zur
P73-Messung, die korrekt auf Kopien lief. Sofort entfernt (1 Zeile,
Segmente/Manifest unberührt, verify True danach; die P73-Kopien waren
nachweislich sauber — beide Ledger bit-identisch mit exakt den eigenen
14 Einträgen). Regel: Jeder Lauf, der einen gescorten Store konsumiert,
mountet eine KOPIE — das Originalverzeichnis eines gescorten Artefakts
wird nie als --store/Schreibziel übergeben. Gilt für Ledger-Anhänge
genauso wie für Segmente.
