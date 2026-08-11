# FERTIG — das fertige symbolische Sprachsystem

**Gewicht-freie, deterministische Sprach-Erzeugung aus zwei gemessenen Quellen:**

- **`.causal`-Wissensgraphen** — Fakten exakt (Entitäten + Mechanismen aus dem
  Graphen, nie erfunden), der *Walk* darüber ist generierte Form.
- **Korpora** — gemessene Bigramm-/Trigramm-Übergänge, reines Zählen,
  kein Training, kein Backprop, keine Embeddings.

Gleiche Eingabe → gleiche Ausgabe. Das System besteht ausschließlich aus
deterministischer Mathematik und gemessenen Zählungen.

---

## Schnellstart

```bash
pip install -r requirements.txt          # numpy + msgpack

# Graph-Modus: Kausal-Walks aus chained.causal
python3 -m fertig info
python3 -m fertig graph                  # Walks (Entitäten exakt)
python3 -m fertig speech                 # Walks als gesprochene Prosa
python3 -m fertig chains                 # abgeleitete Ketten (3-Pass-Inferenz)

# Muster-Bank minen, dann mined-Modus (Form aus Korpus gemessen)
python3 scripts/extract_patterns.py -o data/faraday_bank.json
python3 -m fertig mined

# Korpus-Modus: Prompt gewicht-frei fortsetzen
python3 -m fertig corpus "the candle"

# --- Die Intent/Tool/Lern/Benchmark-Kette ---
# Verstehen: NL-Befehl -> Intent-Tupel (Parse-Baum sichtbar)
python3 -m fertig intent "explain how smoking affects health"
# Verstehen + Handeln: Intent -> Tool-Call ausführen
python3 -m fertig intent -x "how can i prevent exercise"
# Lernen: Lexikon wächst aus Korpora (deterministisches Zählen)
python3 -m fertig learn -o data/lexicon.json
# Messen: präregistrierter Selbst-Benchmark
python3 -m fertig arena
```

Tests: `python3 -m pytest tests/ -q`

---

## Architektur

```
.causal-Graph / Korpus
        │
        ▼
[load_graph / build_vocab]      gemessene Kanten: Adjazenz bzw. Trigramm-Zählungen
        │
        ▼
[inference.pass1/2/3]           (Graph-Modus) Jaro-Winkler + exakte Ketten + Richtung
        │
        ▼
[Walk]                          Kontraktions-Sampler (tau-kontrolliert, Zeno-Schedule),
        │                       Ginibre-Kern-Gewichtung, BvN-Pfad-Integral
        ▼
[Berry-Phasen-Wächter]          bphm: Schleifen-Erkennung auf Hyperboloid-Zuständen
        │
        ▼
[verbalize / verbalize_mined]   Form: Polaritäts-bewusste Verknüpfer,
                                handgeschrieben oder aus Korpus gemessen
```

### Module

| Modul | Quelle | Rolle |
|---|---|---|
| `sampler` | SYMBOLISCH/hsslm_s | tau→Temperatur (F36), Kontraktions-Sampling (F33–F40), Zeno/Anti-Zeno (F35), Ginibre-Kern (F38), BvN-Zerlegung (F63–F68) |
| `state_init` | fable language/hsslm_s | Hyperboloide Symbol-Zustände (Minkowski-Norm −1) |
| `bphm` | fable language/hsslm_s | Berry-Phasen-Wiederholungs-Erkennung, Fidelity, BvN-Modulation |
| `pattern_bank` | fable language/hsslm_s | Aus Korpora gemessene Satz-Skelette + Opener, F30-Weak-Signal-Backoff |
| `inference` | fable language/hsslm_s | Jaro-Winkler, Möbius-Konfidenz, 3-Pass-Ketten (exakt → Richtung → fuzzy) |
| `pipeline` | SYMBOLISCH run_causal/speak_causal | Graph laden, Walk, Verbalisierung (Polaritäts-Vorzeichenregel) |
| `corpus` | SYMBOLISCH run_symbolic | Vokabular + Trigramm/Bigramm-Kanten, Fortsetzung mit Rezenz-Penalty |
| `mined` | SYMBOLISCH speak_mined | Verbalisierung mit gemessener Muster-Bank (echter RNG nur in der Form) |
| `intent` | neu (FERTIG) | NeuroSymbolische Intent-Maschine: NL → Parse-Baum → Intent-Tupel (Aktion, Ziel, Konfidenz, Grounding) — Wirkung-Stil-Parsimonie-Scoring, Which-Path-Visibility, Jaro-Winkler-Ziel-Matching |
| `tools` | neu (FERTIG) | Tool-Schicht (Stufe 5 der Foss-Lernhierarchie): speech/chain/prevent/consult/help — Intents werden zu registrierten Tool-Calls |
| `learn` | neu (FERTIG) | Lern-Modul: gemessene Lexika wachsen aus Korpora (Verb-/Nomen-Slots, deterministisches Zählen) — erweitert die Intent-Coverage ohne Code-Änderung |
| `arena` | neu (FERTIG) | Präregistrierter Selbst-Benchmark: 12 Befehle, 100% deterministisch — die Zahlen sind Ledger-Einträge |
| `_vendor/dotcausal` | fable package/src/dotcausal | .causal-Format (core/io/inference), vendored, unverändert |

Alle Module sind unverändert aus ihren Quellen übernommen (David Tom Foss,
MIT) und nur um Verpackung (Paket-Imports, Pfade, CLI) ergänzt.

---

## Grundlagen — die Preprints, auf denen alles beruht

Die Mechanik des Systems ist die direkte Umsetzung der Foss-Preprints
(`SYMBOLISCH/preprints md/`):

| Baustein | Preprint | Umsetzung |
|---|---|---|
| **Kontraktion** | *Collapse Is Contraction* | `sampler.contraction_sample` — tau in [0, 0.95] ist der Kontraktionskoeffizient; bei tau→1 Phasenübergang (F35, `TAU_MAX`). Tau ist der Birkhoff-Ordnungsparameter: τ<1 = klassische Phase |
| **Möbius-Kopplung** | *MarkovChains to MinkowskiSpace* | `tau_to_temperature`: T(tau) = (1 − tau²)^(−1/2) − 1 — wörtlich der Lorentz-Faktor γ(λ) aus dem Paper; die Möbius-Kopplung f(λ,v) = (λ+v)/(1+λv) in Temperaturform |
| **Ginibre-Kerne** | *One Constant Rules All 2D Spectra* + *Universal Phase Transition GOE→Ginibre* | `ginibre_select_weights` (F38): w = s³·exp(−1.2·s²) — das s³ ist die kubische NND-Repulsion (β=3) aus dem Paper; Sinkhorn in `bvn_decompose` = KL-Gradientenfluss (Sättigung ≤10 Iterationen); `TAU_MAX` = kritischer Punkt |
| **BvN-Pfade** | *Non-Reversibility Is All You Need* | `bvn_decompose` / `bvn_path_integral_sample` (F63): 414 Permutations-Pfade („Parallel-Universen“), globale Summe in 11 Runden (`convergence_rounds`, F65) |
| **Spektrale Lücke** | *Linear Cheeger Improvement (Foss Gap Theorem)* + *Constant-Round Gossip* | `consensus_convergence_bound` (F67): (1 − λ₂/M)^t; und der Sampler-Default `tau=0.65` entspricht der optimalen Vorwärtswahrscheinlichkeit p_c ≈ 0.65 aus dem Gap-Theorem |
| **Topologischer Schutz** | *Unitarity Is the Boundary* | Berry-Phasen-Wächter in `bphm` = das dritte universelle Gesetz des Foss-Boundary-Theorems (topological protection) im Zustandsraum |
| **Lernhierarchie** | *Emergent Gravity / Foss Brain v2* | Die sechsstufige Lernhierarchie des Foss Brain (Habituation, Konditionierung, Raumgedächtnis, Grenz-Kommunikation) erscheint in FERTIG als: Rezenz-Penalty (Habituation), Konfidenz-Aktualisierung (Rescorla-Wagner), Hyperboloid-Zustände (Raumgedächtnis), ehrliche Graph-Sackgassen (Grenz-Kommunikation) |

**Determinismus-Trennung** (durchgängig): *Fakten* müssen deterministisch sein
(Graph-Lookup, Zählungen) — *Form* darf zufällig sein (Ver-knüpfer-Wahl mit
echtem RNG, tau-kontrolliert). Das ist die einzige erlaubte Zufallsquelle,
und sie ist auf die Form beschränkt.

---

## Daten

- `data/chained.causal` — der mitgelieferte Beispiel-Graph (3 KB), Quelle:
  `kimi/workspace/AI_Causal_Work/.../corpora/chained.causal`
- `data/faraday_candle.txt` — Faraday-Korpus für Korpus-Modus und
  Muster-Bank-Mining (223 KB)
- Eigene Graphen/Korpora: einfach `--graph`/`--corpus` auf eigene Dateien
  zeigen; das System ist format-getrieben, nicht daten-getrieben.

## Die Kette: labern → verstehen → lernen → toolcalls → benchmarks

FERTIG ist jetzt eine geschlossene Kette, jede Stufe deterministisch und gemessen:

1. **Labern** — `graph`/`speech`/`corpus`/`mined`: gewicht-freie Sprache aus
   gemessenen Übergängen (Fakten exakt, Form generiert).
2. **Verstehen** — `intent`: NL-Befehl → Intent-Tupel. Deterministischer Parse
   (Aktion aus gemessenem Lexikon, Ziel per Jaro-Winkler gegen das
   Graph-Vokabular), Parsimonie-Scoring im Wirkung-Stil
   (`score = log(fit) − 0.5·complexity`), Which-Path-Visibility als
   Ambiguitäts-Detektor. Fehler sind sichtbar (Parse-Baum), nie geraten.
3. **Lernen** — `learn`: gemessene Lexika wachsen aus Korpora (Verb-/Nomen-
   Slot-Zählen). Neue Verben werden Aktionen, neue Nomen Ziel-Kandidaten —
   Coverage compoundiert, ohne Code-Änderung.
4. **Tool-Calls** — `intent -x`: Intent → registrierter Tool-Call
   (speech/chain/prevent/consult/help). Nur grounded Intents werden
   ausgeführt; Fehlschläge sind ehrlich („kein Reduktor im Graphen").
5. **Benchmarks** — `arena`: präregistrierter Evaluations-Satz, 100%
   deterministisch. Stand (v1): 12/12 Volltreffer bei Aktion UND Ziel.

Der nächste Schritt der Vision: dieselbe Kette gegen ein LLM in einer
präregistrierten Arena (Intent-Präzision auf öffentlichen NLU-Sets).

## SOTA-Benchmarks — wo FERTIG steht (gemessen, nicht behauptet)

```bash
python3 -m fertig bench snips    # Intent-Klassifikation (SNIPS, 7 Intents)
python3 -m fertig bench blimp    # Grammatikalität (BLiMP, 8 Subtasks)
```

### SNIPS (Intent) — v2: 88.3% (Chance: 14.3%, DeepSeek gemessen: 98–100%)

Verb-Bigramme + Objekt-Slot-Signale (gemessene Interpolation): 5 Intents bei
92–98%. Die LLM-Arena (scripts/llm_arena.py) misst DeepSeek auf denselben
Daten: der Abstand ist das Goal.

### BLiMP (Grammatikalität) — v2: 52.9% über Chance (67 Subtasks, 67k Paare)

Struktur-Regeln (Kongruenz + Inseln) heben das System über die 50%-Chance:
  determiner_noun 87–90%, left_branch_island 7.9% → **82.6%**,
  wh_questions_subject_gap 74.9%. Anaphern (24.9%) und wh_with_gap
  (38.8%) sind die nächsten Ziele.

### HumanEval (Code) — v2: 0% ungesehen, 80.0% Retention

Nach dem Fix des Funktionsnamen-Parsings (Unterstriche!) erreicht der
Compounding-Loop 24/30 auf gelernten Problemen. Unseen bleibt 0% —
Fragment-Retrieval generalisiert nicht; das ist FORGEs Territorium.

### HellaSwag / WinoGrande / LAMBADA — die Statistik-Floor-Karte

  HellaSwag 26.7% (Chance 25%) | WinoGrande 50.9% (Chance 50%) |
  LAMBADA 0% (by Konstruktion n-gram-ausgefiltert)

Diese Benchmarks brauchen Weltwissen, nicht Struktur — die ehrliche
Grenze der Statistik-Schicht. WinoGrande-Kongruenz-Regeln wurden gebaut,
gemessen (48% auf gefeuerten Fällen) und verworfen: debiased gegen
Shortcuts.

### LLM-Arena (präregistriert, scripts/llm_arena.py)

DeepSeek auf denselben Samples: HellaSwag 45% vs 26.7% | WinoGrande 70%
vs 50.9% | BLiMP-Sample 85% vs (24.9%/82.6%) | SNIPS 100% vs 88.3%.
FERTIG gewinnt: Kosten (0€, offline, ms), Determinismus (bit-identisch),
Struktur-Subtasks (left_branch 82.6% ≈ DeepSeek-Niveau).

### Die Goals (aus den Zahlen abgeleitet, nicht aus Meinungen)

1. SNIPS: 88.3% → 93%+ (SearchScreeningEvent 68.6%: Film-Slots)
2. BLiMP: 52.9% → 58%+ (Anaphern 24.9% → 60%, wh_with_gap 38.8% → 60%)
3. HumanEval Retention: 80% → 90%+ (Imports der Bodies)
4. HellaSwag: 26.7% → 30%+ (4-gram-Erweiterung)

Jede Verbesserung ist ein Ledger-Eintrag: gleiche Benchmarks, gleiche
Argumente, deterministisch reproduzierbar.

## Der Gap-Loop: das Internet als Online-Learning

```bash
python3 -m fertig grow "vitamin d"          # ein Ziel, alle 7 Quellen
python3 -m fertig grow --gaps              # Lücken aus der Arena automatisch
python3 -m fertig grow sugar --sources wikipedia,pubmed   # Quellen wählen
```

**Lücke erkennen → Query formulieren → scrapen → extrahieren → speichern →
neu messen.** Der Kreislauf (F4: externer Index, on demand):

1. **Gap-Detection** existiert: `unknown-target`-Status + Arena-Fehlschläge
2. **8 Quellen** (`fertig/sources.py`, je fetch+parse getrennt):
   **web (generisch: Suche → beliebige Seiten → Text, Trafilatura +
   stdlib-Fallback)**, wikipedia (Infobox 0.7), wiktionary (0.45),
   pubmed (0.40), arxiv (0.35), semantic_scholar (0.35), openalex (0.35),
   duckduckgo-Snippets (0.30)
3. **Beliebige URL direkt**: `fertig crawl <url> --store` — eine Schicht
   für jede Website statt tausend Einzel-Adapter (Trafilatura für
   Boilerplate-Removal, stdlib-Fallback ohne Dependencies)
3. **Kausaler Satz-Extraktor**: (NP, Kausal-Verb, NP) mit Negations-Schutz —
   "no evidence that X causes Y" erzeugt keine Kante
4. **Evidenz-Aggregation**: gleiches Triplett aus n Quellen →
   conf = min(0.95, max_conf + 0.06·(n−1))
5. **Speicherung**: `data/world.causal` (CausalWriter) — wird von pipeline/
   intent/tools automatisch gemergt (`load_graph_merged`)
6. **Messbar**: derselbe Befehl vorher (unknown) vs. nachher (grounded)

Neue Quellen = eine parse-Funktion + ein Konfidenz-Tier + Registrierung in
`SOURCES`. Kein Repo-Import, kein Supply-Chain-Risiko — jeder Fetcher ist
~40 Zeilen audit-fähiger Code.

## Die Grounding-Schicht: Erdungs-Ebenen (ehrliche Terminologie)

```bash
python3 -m fertig ground cheetah          # Symbol verankern (Ebenen L1/L2)
python3 -m fertig ground --all            # alle Graph-Symbole + Coverage
python3 -m fertig quant --all             # quantitative QA (Anker-Beweis)
python3 -m fertig vision cheetah giraffe elephant -u   # Harnad-Ebene (L3)
```

**Wichtige Einschränkung**: Nichts davon ist „vollständig geerdet" im
Harnad-Sinn (1990). CLIP-Text-Encoder sind aus menschlichen Texten
trainiert, Wikipedia-Bilder sind menschenkuratiert, Zahlenstrings sind
menschliche Symbole — alles ist menschlich vermittelt. Die ehrlichen
Ebenen:

| Ebene | Anker | Vermittlung |
|---|---|---|
| L0 | nur Wort↔Wort-Kanten | reiner Regress |
| L1 | quantitative Anker (Zahlen+Einheiten) | menschlicher Text, aber nicht-lexikalische Denotation |
| L2 | perzeptuelle Bindung (CLIP/Commons-Bilder) | Pixel primär, Wort↔Bild-Zuordnung kuratiert |
| L3 | **unüberwachte Kategorien aus Pixel-Struktur** (fertig.vision) | Kategorien entstehen ohne Labels — die nächste Annäherung an Harnads sensorische Transduktion |

Die L3-Mechanik ist deterministisch (kein neuronales Netz):
Signatur(Bild) = HSV-Histogramme ⊕ Textur ⊕ Form-Gitter; k-means über
Signaturen findet die Kategorien; Wörter werden erst NACH der
Cluster-Bildung zugeordnet. Lokal verifiziert: 3 Arten, 12 Bilder,
keine Labels → Purity 100%.

Die kategorielle Wahrnehmung (Harnad 2005: „To cognize is to
categorize") ist messbar: within-category-Distanz < between-category-
Distanz (harnad_ratio < 1).

## Stream-Lernen: das Internet als permanenter Video-Lehrer (O(1))

```bash
python3 -m fertig stream data/puls.gif --seconds 10
python3 -m fertig stream "https://www.youtube.com/watch?v=..." --seconds 60
```

Die o1-state-Philosophie auf Video: **Der Stream ist ein Iterator, kein
Objekt.** `fertig/stream.py` verdichtet jeden Frame in gleitende
Statistiken — konstantes Memory bei unendlichem Input:

- **Prototyp**: EMA der Frame-Signaturen (was ist typisch?)
- **Bewegung**: EMA der Signatur-Distanz + Pixel-Differenz (zwei
  komplementäre Primitive — Histogramme sind translations-invariant!)
- **Grammatik**: begrenzte Code-Übergänge (Top-K pro Code, verdrängt)
- **Periodizität/Szenenwechsel**: Noether-Detektoren auf dem Fenster

Quellen: ffmpeg (Datei/URL) + **yt-dlp** (YouTube, Live-Streams).
Live getestet: „Me at the zoo" → yt-dlp → ffmpeg → 16 Frames gelernt,
Periodizität erkannt, 11 Grammatik-Kanten, stabile Zyklus-Fortsetzung.

**Der Lern-Zyklus schließt sich** (`--name` + `--recognize`):

```bash
python3 -m fertig stream video_a.gif --name puls     # Kategorie lernen
python3 -m fertig stream video_b.gif --name wander   # + speichern
python3 -m fertig stream video_x.gif --recognize     # gegen die Bank
```

Gelernte Streams werden VideoBank-Kategorien (data/video_bank.json) und
schreiben Struktur-Fakten in den Welt-Graphen (`hat_bewegung`,
`ist_periodisch`, `hat_szenenwechsel`) — der Stream wird Wissen.
Live getestet: „Me at the zoo" wurde als Kategorie `zoo` gelernt und
hat die Fakten `(zoo, hat_bewegung, 0.0416)` + `(zoo, ist_periodisch, 1)`
in den Graphen geschrieben.

**Sehen wird Sprache** — die VideoBank ist ein Intent-Tool:

```bash
python3 -m fertig intent -x --video data/puls.gif "erkenne dieses video"
# -> [Tool video] Das Video zeigt: puls
```

Unbekannte Videos werden ehrlich abgelehnt (Distanz über der Schwelle)
statt geraten. Stresstest: 5 Video-Arten, 10/10 ungesehene Varianten
korrekt erkannt.

## Interpolations-Lernen mit Stützrädern (fertig.interp)

```bash
python3 -m fertig interp data/puls48.gif            # Stützräder-Metrik
python3 -m fertig interp data/puls48.gif -s          # self-paced Curriculum
```

**Fahrradfahren**: Lücke k=1 (benachbarte Frames) ist fast gegeben;
mit wachsender Lücke muss das System echte Dynamik verstehen. Die
**beherrschte Lücke** (größte Lücke mit Interpolations-Fehler unter der
Schwelle) ist die Fortschritts-Metrik.

- **Kein Warp**: Interpolation im Code-Raum (gemessene Übergänge +
  Code-Prototypen) — Warp-Artefakte sind per Konstruktion unmöglich.
- **Richtungs-Bit**: Der Code trägt wächst/schrumpft — ohne es wäre er
  phasenblind und der Prototyp mittelte Anstieg+Abfall zu Unschärfe.
- **Self-paced (F1-Prinzip)**: Die eigene Surprise stellt die
  Stützräder ein — Fehler klein → Lücke wächst, Fehler groß → Lücke
  schrumpft. Live: gap-Verlauf [2,3,6,5,4,5,6,5,6,7] auf puls48.gif.
- **Ehrliche Grenzen**: wander (Translation) bleibt bei gap=2 — die
  Maschine weiß, was sie nicht kann, statt zu raten.

## Garantien

1. **Determinismus**: `info`, `chains`, `graph`, `speech`, `corpus` liefern bei
   gleichen Argumenten bit-identische Ausgaben (getestet).
2. **Keine erfundenen Fakten**: Im Graph-Modus stammen alle Entitäten und
   Mechanismen wörtlich aus dem `.causal`-Graphen; nur die Satzform wird
   generiert.
3. **Ehrliche Sackgassen**: Endet der Graph, endet der Walk — kein Füllsel.
4. **Kein Training**: keine Gradienten, keine Embeddings, keine Gewichte.
   Reines Zählen, Ziehen, und gemessene Übergänge.

**Ausnahme (explizit, nicht Teil der Kern-Garantie)**: Die optionale
Grounding-Schicht (`fertig.grounding`, `fertig ground`) nutzt für die
perzeptuelle Bindung (`clip_cross_modal_evidence`, vormals
`perceptual_anchor`) ein **vortrainiertes CLIP-Modell mit gelernten
Gewichten** — das widerspricht Garantie 4 für diesen einen, klar
abgegrenzten Modulteil. Die Kern-Sprach-Pipeline (Graph-Walk, Speech,
Corpus, Mined, Intent, Tools, Arena) bleibt vollständig gewichtsfrei;
CLIP wird ausschließlich als optionale `cross_modal_proxy`-Evidenzquelle
in der Grounding-Schicht verwendet (siehe
`_codex_lab/primitive_schema_snapshot/ADR-0001-grounding-is-a-certificate.md`
für die Evidenz-Tier-Trennung: `direct_sensorimotor` vs. `cross_modal_proxy`
vs. `textual_evidence` werden nie zu einem gemeinsamen Konfidenzwert
vermischt). Ohne `open_clip` installiert degradiert die Grounding-Schicht
sauber (perzeptuelle Anker liefern `None`, quantitative Anker und der
Rest des Systems funktionieren unverändert weiter).
