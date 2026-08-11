# Moonshoot B — HSSLM-Form-Engine: Zwischenbilanz & Falsifikation

Stand: 2026-08-11. Diese Notiz hält fest, was Moonshoot B bewiesen hat —
und wo die Grenze liegt.

## Architektur-Beweis (erreicht)

1. **HSSLM-C (6,3M, Mamba/S6-Familie) vendored** — `fertig/hsslm/`,
   Trainingsschleife bewiesen (PURE-LM-Loss, kein Aux-Gradienten-Gift).
2. **Formen-Korpus**: verbalisierte Kausal-Prosa (multi-seed augmentiert)
   + Faraday-Kanon. BPE (400, auf dem Korpus gelernt, Wortgrenzen
   erhalten) statt Char-Level.
3. **Die Schleife**: Plan → HSSLM-Varianten → IR-Verifikation (alle
   Plan-Objekte müssen im Text stehen) → nur Freigegebene zählen.
   **Diese Schleife funktioniert exakt wie designed**: sie verweigert
   jede unbegründete Flüssigkeit.

## Trainings-Läufe (alle dokumentiert)

| Lauf | Daten | Steps | PPL | Generierung |
|---|---|---|---|---|
| char-level (200k) | causal+Faraday | 600 | 12.1 | `s s s s...` (degeneriert) |
| BPE (3M Prosa) | en_factual | 800 | 56.7 | `"Iffectshead...` (Wort-Fragmente) |
| BPE (246k) | +world.causal (Rauschen) | 800 | 36.9 | `—that—that` (Dash-Loop) |
| BPE (257k clean) | chained+faraday_clean | 800* | 32.3 | `ces.].].].` (Bracket-Loop) |

\* bei 700+ abgebrochen/beendet; PPL-Werte aus avg100.

## Diagnose

- **Kein Datenproblem allein**: vier Korpus-Varianten, vier Loop-Muster
  (`s`, `Iffects`, `—that`, `ces.]`) — die Degeneration ist systematisch.
- **Kein Sampling-Problem**: Zeno-Schedule UND deterministisch (τ=0.65,
  top_k=50) degenerieren identisch.
- **Ursache**: 2,2M-Parameter-Sequenzmodell + 137k Tokens + BPE-400 →
  lernt lokale Statistik (Wort-Fragmente), aber keine stabile
  autoregressive Fortsetzung über 8-12 Tokens. PPL ~33 klingt niedrig,
  ist aber token-lokal; die Kettenstruktur („tar buildup causes lung
  damage“) wird nicht über die Satzgrenze getragen.
- **Umgebung**: MPS, ~3-4s/Step, 800 Steps ≈ 45 min; 1600 Steps
  (1.8 Epochs) wäre ~90 min — die Skala der Flüssigkeit (≥1B Tokens,
  ≥100M Params) ist auf dieser Maschine nicht erreichbar.

## Falsifikation (ehrlich dokumentiert)

> **HSSLM-C @ 2,2M Params / 137k Tokens kann keine flüssigen
> Plan-Varianten generieren. 0/3..0/5 aller Varianten passieren das
> IR-Gate. „Generieren und hoffen“ bleibt — auch mit Mamba — an der
> Skalengrenze; das IR-Gate rettet die Abstinenz, nicht die Flüssigkeit.**

## WIDERRUFEN (2026-08-11, abends) — die Falsifikation war falsch

Codex fand die **Generierungs-Bugs** (training_dynamics/README.md R3-5):

1. `forward(current)` OHNE SSM-states -> kontextfreie Markov-1-Kette
   („the the the", „4.),—that" — ALLE Degenerations-Muster!).
2. `current = next_token` statt cat -> Kontext nach Schritt 1 verloren.
3. `torch.load(map_location='mps')` + in-place copy_ -> korrupte Gewichte.

Fix: voller Forward pro Schritt + cat + CPU-Load. Korpus-Design
(chained x16, Faraday 25%, kanonische Ketten) ergab v6@800:

**PPL 6.4 (Referenz 36.9). Generierung:**

    Smoking causes tar buildup. Consequently, tar buildup causes lung
    damage. Consequently, lung damage causes breathlessness. On the
    other hand, breathlessness reduces exercise...

**IR-Kreis: FREIGEGEBEN 2/2..2/3 — die erste Freigabe der
Projektgeschichte.** Die Wunde war die Implementierung, nicht die Skala.

## Was daraus folgt

1. **Die Rechenschafts-Schleife ist der bleibende Wert** — sie ist auf
   jede Form-Quelle übertragbar (auch auf LLMs!).
2. **Deterministische Konstruktion schlägt lokale Generierung**:
   Davids Form-Arena (FE3–FE5, `erweiterung/form_arena.py`) erzeugt
   Formvarianten konstruktiv (default/cleft/fronted/plain_the/consider/
   chain), alle mit Wahrheit als Konstruktionsbedingung, und wählt per
   IR-Gate → Ohr-Gate → UID-Varianz → Fluency: **21/21 Kanten, UID
   21/21 verbessert, Ohr-Richter pfeift 21× (falsifizierbar)**.
3. **Nächste Stufe (wenn überhaupt)**: HSSLM als Varianten-QUELLE für
   die Arena (Pool), nicht als Entscheider — die Arena misst und
   verwirft. Erst ab ~1B Tokens Training wäre das mehr als Dekoration.
4. **Weltbuch (Davids Welle 1)**: gelebte Evidenz mit bit-exakten
   Quittungen (SHA-256 + Frame) — FERTIGs erste direkt-sensorimotorische
   Kanten, 5/5 FREIGEGEBEN. Der Graph, dessen Kanten nicht gelesen,
   sondern GELEBT wurden.
