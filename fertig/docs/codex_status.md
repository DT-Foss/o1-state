# Codex-Labor — integrierter Stand (2026-08-11, abends)

Was aus `_codex_lab/` übernommen und wo es im Hauptsystem lebt.

## GroundZero (integriert, offizielles Benchmark)

| Stufe | Ergebnis | Zugang |
|---|---|---|
| v1 | 10/10 Achsen, 3 Negativ-Kontrollen korrekt abgelehnt | `fertig bench groundzero` |
| Grade-3 | 8/8 Achsen, noncompensatory, isolierte Child-Prozesse, Commitment-Kette | `fertig bench groundzero --grade3` |
| P0 causal-v2 | 64/64 Blöcke; aktiv>random p=9.5e-12; aktiv>passiv p=1.7e-18; Brightness-Shortcut 0/64 | `fertig bench causal-v2` |
| Continuity | 4 Achsen + 14 noncompensatory Kontrollen, ein OS-Prozess, ein Checkpoint | läuft im Labor (Test-Suite) |

P0 causal-v2 im Detail: 8 Mechanismen aus rohen RGB-Übergängen induziert
(kein Outcome-Feld, kein Label), ungezeigter 3-Aktions-Plan durch
Komposition geerdeter Faktoren gelöst (Full-Lookup liefert falschen Code),
Ziel-Lokalisierung ohne Koordinaten (Brightness-Argmax 0/64).
Ehrlich: statistische Zertifizierung abgelehnt (public seed ist Replay-
Material, keine verborgene Attestations-Entropie); aktive vs. optimale
fixe Politik nicht signifikant (p=0.13); deranged-Shortcut 7/64 Blöcke.

## Training-Dynamics-Rennen (Sidequest: Optimierer aus der Formelsammlung)

6 Dynamiken, identisches Modell/Korpus/Init (Seed 42), CPU:
`data/dynamics_race2.json`

| Variante | final loss | auc | sec | Befund |
|---|---|---|---|---|
| **bvn_rr** (BvN Random-Reshuffling) | **4.578** | 763 | 105 | Sieger — Permutations-Epochen statt iid |
| baseline (AdamW iid) | 4.612 | 803 | 97 | Referenz |
| flca_router_v2 | 4.905 | 825 | 146 | 67× konservativ (Insuffizienz) |
| lorentz_v2 (τ-LR-Schedule) | 5.003 | 795 | 107 | — |
| pslift_r10 (PS-Lifted-Consensus) | 4.637 | 771 | 208 | — |
| moebius_v2 (RapidityAdam) | 14.017 | 2729 | 108 | divergiert (Doku, kein Fix) |

Konsequenz: künftige HSSLM-Trainings (z.B. `train_form_causal.py`) können
die bvn_rr-Batch-Dynamik übernehmen (≈0.7% besser, deterministischer).

## Wissenschafts-Infrastruktur

- THREAT_MODEL.md: Claim-Leiter Grade 1–4, Identifizierbarkeits-Grenze
  (Quotient, Permutationen, Automorphismen) — kopiert nach
  `docs/` (Referenz).
- RESULTS_*: post-freeze-Chronologie, Commitment-Ketten, noncompensatory
  Gates — Muster übernommen (Grade-3-Bench, Weltbuch-Zertifikat).
- `unified_grounder.py`: persistenter Lernender (induziert operationale
  Kategorien aus öffentlicher Evidenz) — Referenz-Kandidat für Grade-4,
  noch nicht ins Hauptsystem gehoben.
