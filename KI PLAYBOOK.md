# FOSS-KI PLAYBOOK — Transformer-Ersatz aus vergessenen Paradigmen

**Ziel:** Eine vollständige KI die NICHT auf Transformern basiert. Kein Backprop (außer linearem Readout), kein GPU-Monopol, CPU-only. Architektonisch überall dort überlegen wo Transformer blind sind: Anti-Halluzination, Online-Lernen, dezentrale Ausführung.

**Szenario:** "KI-Holocaust" — wenn Transformer reguliert/verbannt werden, ist FOSS-KI die Alternative die auf jedem Laptop läuft.

---

## Architektur — Die 8 Layer

```
┌─ Vortex Language Model (3-Fiber PPM, 3-adische Gewichtung)
│    ├─ Fiber 1: Character PPM (Gewicht: 1)
│    ├─ Fiber 2: Structure PPM (Gewicht: 1/3)
│    └─ Fiber 3: Context PPM (Gewicht: 1/9)
│
├─ Knowledge Store (Modern Hopfield, Anti-Halluzination)
│    └─ Attractor-Distance = Confidence → "Ich weiß es nicht"
│
├─ MarkovReservoir (PS-Lifted, feste Gewichte)
│    └─ Z₂ Parity als zusätzliche Features
│
├─ Hopfield Memory (Pattern Storage/Recall)
│
├─ Ensemble Consensus (Foss-Gossip, 26-115× Speedup)
│
├─ Topology Evolver (MAP-Elites über Graphen)
│
├─ Spike Encoder (Z₂ Parity Events)
│
└─ Readout (lineares Ridge — der EINZIGE trainierte Teil)
```

**Kernprinzipien:**
- Online Adaptation: jeder Output fließt zurück (kein Einfrieren)
- Anti-Halluzination: Attractor-Distance sagt "Ich weiß es nicht"
- Kein Catastrophic Forgetting: PPM-Bäume sind additiv
- Foss Consensus WO ES FUNKTIONIERT: Bottleneck-Netzwerke
- 3-adische Hierarchie: Lokal > Global (algebraisch gewichtet)
- CPU-only: ~4-7ms pro Knowledge Query

---

## Was STEHT (verifiziert, mit Benchmarks)

| # | Komponente | Task | Ergebnis | Kernmetrik |
|---|-----------|------|----------|------------|
| 1 | PPM Language Model | T66b | ✅ Funktioniert auf echtem Text | 1.16 bpc (Order 3, 200K Zeichen) |
| 2 | Vortex 3-Fiber PPM | T83 | ✅ +9% über Standard-PPM | Konsistent auf Alice, Shakespeare, P&P |
| 3 | Galois-Heads | T84 | ✅ Preview | 11.7 bits KL-Diversität |
| 4 | Nichtlinearer Readout | T67b | ✅ Stacked Reservoir | +17.3% |
| 5 | Hopfield-Phrasen | T68b | ✅ Skaliert | 100% auf 1000 Phrasen, 256d |
| 6 | Faktenspeicher (S,R,O) | T71 | ✅ Komplett | 10/10 S+R→O, 6/6 S→(R,O) |
| 7 | Integration MVP | T79 | ✅ Alle Teile zusammen | 14/14 = 100%, 4-7ms, Anti-H funktioniert |
| 8 | Anyonisches PE | T85 | ✅ Distinguishability | 1.000 |
| 9 | Online-Adaptation | T86 | ✅ Domain-Switch | 73.1% Accuracy, nur 0.6% Forgetting |
| 10 | RG-Fluss Summarizer | T87 | ⚠️ Konzept ok | Braucht Tuning |
| 11 | Parser + Confidence | T88 | ✅ Bombenfest | P=1.0, R=1.0, F1=1.0, Gap=0.337 |
| 12 | Bernoulli-Shift Temp | T90 | ❌ NEGATIV | Softmax schlägt ×2mod9 bei gleicher Kohärenz |
| 13 | Hopfield Scaling | T91+S1 | ✅ bis 10K | 100% Retrieval + 100% Anti-H (Gumbel-Threshold) |
| 14 | Vortex Gesamtintegration | T89 | ❌ NEGATIV | PPM-3 solo=0.92 bpc, alle Vortex-Komp. verschlechtern |
| 15 | Kausale Inferenz | T92 | ✅ do-Calculus | Simpson korrekt (0.66 vs 0.52), Sprinkler, Smoking |
| 16 | CPU vs Transformer | T94 | ✅ 135× schneller | 1.6ms vs 218ms, 7/7 vs 0/7, Anti-H 4/4 |

### Anti-Halluzination — DER USP

Das architektonische Killer-Feature. Transformer können das NICHT:

- **SR-only Matching:** Vergleicht nur Subject+Relation Vektoren (Object-Slot ist bei Query unbekannt → Noise wenn mitgemessen)
- **Echte Matches:** SR-Similarity = 1.000 (exakte Treffer)
- **Fiktive Queries:** SR-Similarity max 0.796 bei N=10K
- **Gap:** 0.204+ Similarity-Punkte — robust trennbar bei ALLEN Skalen
- **Gumbel-Adaptive Threshold:** `low_t = mean + std * sqrt(2*log(N)) + margin` — skaliert mit Extremwert-Statistik
- **Dreistufenmodell:** HIGH (>high_t) → Antwort, MEDIUM → Warnung, REJECTED (<low_t) → Verweigerung

### Skalierung nach S1-Fix (Gumbel-Threshold)

| N Facts | dim=128 Anti-H | dim=128 Retrieval | Threshold (high/low) | Query-Zeit |
|---------|----------------|-------------------|---------------------|------------|
| 100 | 100% | 100% | 0.846 / 0.796 | ~5ms |
| 500 | 100% | 100% | 0.884 / 0.834 | ~10ms |
| 1000 | 100% | 100% | 0.897 / 0.847 | ~18ms |
| 5000 | 100% | 100% | 0.913 / 0.863 | ~48ms |
| 10000 | 100% | 100% | 0.858 / 0.808 | ~68ms |

**Gelöst:** Anti-Halluzination 100% bei ALLEN Skalen (N=5 bis N=10K) mit dim=128.
- Small-N: Basis-Thresholds 0.85/0.75 (echte Matches = 1.000, Fakes max ~0.60)
- Large-N: Gumbel-Extreme-Value Thresholds skalieren automatisch mit sqrt(2*log(N))
- Vorher: kollabiert ab N=500 + Narnia/Mordor als MEDIUM bei kleinem N
**Offen:** Query-Zeit linear O(N×d). Ab 100K+ braucht man Approximate Nearest Neighbor (FAISS/Annoy) oder hierarchisches Hopfield.

### Was NICHT funktioniert hat

- **Bernoulli ×2 mod 9 als Temperatur (T90):** Softmax gewinnt bei gleicher Kohärenz um 7.8% Diversität. Algebraische Struktur auf Character-Ebene bringt nichts. Tot.
- **PPM Exclusion Fix:** BPC verbessert, aber Generierung wird Müll. Reverted.
- **Bridge Strength > 0:** Fibers vollständig isoliert (bridge=0.0) ist optimal. ×3 Coupling verschlechtert.

---

## Phase 1: Offene Tasks — Sofort machbar

### T89 — Gesamtintegration aller Vortex-Komponenten
**Status:** ❌ NEGATIV (Alpha, 2026-03-13)
**Ergebnis:** PPM-3 allein = 0.9244 bpc (UNTER 1.0 Ziel!), aber alle Vortex-Komponenten verschlechtern: Fiber +13%, Galois +17%, p-adic +16%. Geometric-mean verwässert scharfe PPM-Distribution.
**Nächster Angriff:** Fibers als Kontext-Features statt unabhängige Prädiktoren.

### T92 — Kausale Inferenz: Pearl's do-Calculus auf PS-Lifted
**Status:** ✅ DONE (Bravo, 2026-03-13)
**Ergebnis:** Simpson's Paradox korrekt erkannt (P(R|do(T=yes))=0.660 vs P(R|do(T=no))=0.520). Sprinkler, Smoking-Cancer alle korrekt. Exakte Inferenz via Variable Elimination. PS-Lifted Speedup nur bei verteilter Inferenz relevant (T93).

### T94 — CPU vs Transformer Inference Benchmark
**Status:** ✅ DONE (Bravo, 2026-03-13)
**Ergebnis:** 135× schneller (1.6ms vs 218ms). 7/7 korrekt vs 0/7 GPT-2. Anti-H 4/4 rejected. Generation 21× schneller. BPC: GPT-2 gewinnt (1.65 vs 2.66) — erwartbar. Memory: 124MB vs 575MB.

### T93 — Dezentraler Consensus über TCP/UDP
**Status:** ✅ DONE (Alpha, 2026-03-13)
**Ergebnis:** 9 UDP-Agenten, 2.2-6.5× Speedup, überlebt 50% Paketverlust + Kill von 4/9 Agenten. PS-Lifted hat Consensus-Wert-Bias (konvergiert zu gewichtetem statt arithm. Mittel). David-Direktive: Zwei Modi — `consensus_mode=fast` (PS-Lifted, Bias egal) und `consensus_mode=exact` (Metropolis-korrigiert).

### T95 — Logik-Puzzles als Reasoning-Benchmark
**Status:** ✅ DONE (Alpha, 2026-03-13)
**Ergebnis:** 15/15 = 100%. Sudoku 4×4 (3/3), Zebra (German owns fish), Syllogismen (6/6 inkl. Modus Ponens/Tollens/Hypothetical/Disjunctive + invalid rejected), 3-SAT (4/4 inkl UNSAT), River Crossing (optimal 7 steps). Alles unter 2ms.

---

## Phase 2: Skalierung — Von Demo zu echtem System

### S1 — Hopfield Anti-Halluzination bei Skalierung fixen
**Status:** ✅ DONE (Bravo, 2026-03-13)
**Lösung:** SR-only Matching (Subject+Relation only, Object ignoriert) + Gumbel-basierter adaptiver Threshold: `low_t = mean_sim + std_sim * sqrt(2*log(N)) + 0.05`. Extremwert-Statistik statt Normalverteilungs-Annahme.
**Ergebnis:** 100% Anti-H + 100% Retrieval bei N=100 bis N=10.000, dim=128.
**Offen:** N>100K braucht FAISS/Annoy oder hierarchisches Hopfield für Query-Zeit.

### S2 — Echte Knowledge Graphs laden
**Status:** ✅ DONE (Bravo, 2026-03-13)
**Ergebnis:** FB15k-237 (272K Triplets, 14.5K Entities, 237 Relations). Anti-H = 100% bei ALLEN Skalen (N=500 bis N=10K). Accuracy: N=500→97%, N=1K→96%, N=2K→92%, N=5K→89%, N=10K→77%. Accuracy-Drop durch Mehrdeutigkeit im Dataset (same S,R → multiple Objects), nicht Hopfield-Kapazität. Dimension 64/128/256 irrelevant bei N=2K.

### S3 — PPM auf größeren Korpora
**Was:** OpenWebText oder The Pile Subset. Character-level PPM Order 5-8.
**Ziel:** BPC < 1.5 auf 1M+ Zeichen. Vergleich mit LSTM/Mamba gleicher Parameterzahl.

### S4 — Embedding-Raum statt Character-Raum
**Was:** PPM operiert aktuell auf Characters. Für echtes Language Modeling: Token-Embeddings (d=256-512), Transitionen im Embedding-Raum.
**Problem:** PPM mit Context Trees funktioniert nur für diskrete Symbole. Für kontinuierliche Embeddings braucht man einen anderen Ansatz (z.B. Reservoir als Sequence Processor im Embedding-Raum).
**Lösung:** Hybrid — PPM für lokale Vorhersage, Reservoir für Sequenzverarbeitung im Embedding-Raum, Hopfield für Knowledge Retrieval. Jede Komponente macht was sie am besten kann.

### S5 — Multi-GPU / Distributed Training des Readout
**Was:** Der lineare Readout ist der einzige trainierte Teil. Bei großen Reservoirs (>10K Nodes) wird die Ridge-Regression zum Bottleneck.
**Lösung:** Randomized SVD oder Online Ridge für inkrementelles Training. Kein Backprop nötig.

### Neue Tasks (David-Direktive, 2026-03-13)

#### Auto-Extraktion: Text → Triplets → Hopfield
**Status:** ✅ DONE (Alpha, 2026-03-13)
**Was:** Dependency Parsing + regelbasierte Patterns, kein Transformer. Text rein → Triplets extrahieren → sofort querybar.
**Ergebnis:** 15/15 Pattern Accuracy, 8/8 Query Accuracy. Online Loop funktioniert ohne Retraining.

#### Multi-Turn Dialog
**Status:** ✅ DONE (Bravo, 2026-03-13)
**Was:** DialogState Tracker mit PPM-Kontext + Hopfield-Referenz-Resolution. Kein NLU, kein Intent-Classifier.
**Ergebnis:** 26/28 = 93%. Reference Resolution 8/8, Direct Queries 6/6, Anti-H 4/5, Parse Robustness 6/6. Unter 2ms/Turn.

#### Vortex als System-Router
**Status:** ✅ DONE (Alpha, 2026-03-13)
**Was:** Fasern routen zu Subsystemen (PPM/Hopfield/Reasoning), nicht innerhalb PPM. ×3 Bridge Escalation bei low confidence.
**Ergebnis:** 15/15 Classification, 8/8 Answers. Fiber 1=PPM, Fiber 2=Hopfield, Fiber 3=Reasoning.

#### W5 — GloVe Semantic Encoder
**Status:** ✅ DONE (Bravo, 2026-03-13)
**Was:** GloVe 6B 100d als Drop-in Encoder. Phrase-Encoding via Wort-Durchschnitt, OOV-Fallback auf n-gram + Sub-word Splitting.
**Ergebnis:**
- "french" → France: GloVe OK (sim=0.90), n-gram REJECTED (sim=0.63) — Semantik funktioniert
- Austria↔Australia: GloVe 0.47 vs n-gram 0.69 — 32% bessere Separation
- Encoding: 360× schneller (dict-lookup vs MD5-Hashing)
- FB15k-237 N=2K: Accuracy gleich (92%), Anti-H 100% bei beiden
- Auto-Modus: `KnowledgeStore(encoder='auto')` — GloVe wenn verfügbar, sonst n-gram

#### Extractor-Bugfixes
**Status:** ✅ DONE (Bravo, 2026-03-13)
**Was:** Wikipedia Query Accuracy von 28% auf 86% gehoben.
**Bugs gefixt:**
1. "With"/"At"/"It" als Subject — fehlende Stopwords in Context-Tracker + Subject-Filter
2. "Its capital AND largest city" — Pattern matchte nur Komma-Variante
3. "hosts the capital, X" — neues Pattern für Brazil-Stil Sätze
**Ergebnis:** 12/14 Capitals korrekt (vorher 7/25). Anti-H 100% stabil.

#### Response Composer
**Status:** ✅ DONE (Alpha, 2026-03-13)
**Was:** Multi-Fakt Antwort-Generierung. "Tell me about Germany" → Capital + Population + Language + Borders in einem zusammenhängenden Text.
**Features:** Compare-Mode (zwei Entities), List-Mode, Priority-basiertes Ranking, Merge von List-Fakten (borders).
**Ergebnis:** Kohärente Multi-Fakt-Antworten, jeder Satz an einen Fakt gebunden. Kein Halluzinieren.

#### PPM-Formulierer
**Status:** ✅ DONE (Alpha, 2026-03-13)
**Was:** PPM-basierte Reformulierung. Composer-Output (Templates) wird via PPM-Scoring in flüssigen Text umgewandelt.
**Regel:** PPM für FORMULIERUNG, NICHT für Content. Anti-Halluzination stirbt wenn PPM Lücken füllt.
**Ergebnis:** Multiple Phrasings pro Relation, PPM wählt die natürlichste. Verification Pass prüft dass alle Facts im Output.

#### Metacognition — Self-Improving Feedback Loops
**Status:** ✅ DONE (Bravo, 2026-03-13)
**Was:** Drei Zyklen für automatische Selbstverbesserung:
1. **Gap-Fill (S10):** Query rejected → Corpus durchsuchen → 3 Extraction-Strategien → neue Fakten → Re-Query. Beweis: Brazil Capital automatisch aus Corpus gelernt.
2. **Self-Test (S11):** Benchmark-Suite mit JSON-Checkpoints, Diff-Analyse, Regressions/Improvements. CLI: `python cli.py selftest`.
3. **Pattern-Learner (S12):** Extraction-Failures clustern, gemeinsame Satzstrukturen erkennen, neue Regex vorschlagen.
4. **Gap-Prioritizer (S13):** Wissenslücken nach Häufigkeit ranken, Suggestions für Text-Ingestion generieren.
**Komponenten:** GapLog, GapFiller, SelfTester, PatternLearner, GapPrioritizer, MetacognitionEngine.

#### Full Pipeline Integration
**Status:** ✅ DONE (Bravo, 2026-03-13)
**Was:** Alle Komponenten verdrahtet: Dialog → Parser → (About? → Composer → Formulierer) | (Specific? → query_smart → Gap-Fill) → Response.
**Ergebnis:**
- "Tell me about France" → Multi-Fakt via Composer+Formulierer (Capital + Pop + Language + Borders)
- "Capital of Brazil?" → GAP-FILLED aus Corpus automatisch
- "Capital of Narnia?" → REJECTED (Anti-H intakt)
- "And Japan?" → Reference Resolution
- "Who wrote Hamlet?" → William Shakespeare (verb→relation mapping)

#### Wikidata Importer + Bootstrap
**Status:** ✅ DONE (Bravo, 2026-03-13)
**Was:** Strukturierte Daten direkt als Triplets importieren (Wikidata JSON, SPARQL, eigenes Format). Bootstrap: 174 Fakten out-of-the-box (20 Länder + 6 Elemente + 5 Personen + 7 Erfindungen + 3 Werke).
**CLI:** `python cli.py bootstrap --save data/world_knowledge.json`
**Ergebnis:** 100% Accuracy auf 101 auto-generierten Queries. System sofort nutzbar.

#### Komplementäre Knowledge-Features
**Status:** ✅ DONE (Bravo, 2026-03-13)
**Was:**
- `query_reverse(obj, rel)` — Object+Relation→Subject ("Who discovered radium?" → Marie Curie)
- `find_by_entity(entity)` — alle Fakten wo Entity als S oder O vorkommt
- `query_smart(subj, rel)` — Universal-Query mit Synonym + Reverse Fallback
- `add_synonym()` / `auto_synonyms()` — GloVe-basierte automatische Synonym-Discovery
- OR-Vector Cache — Performance-Optimierung für Reverse-Queries

---

## Phase 3: Transformer-Parität auf ausgewählten Tasks

### Wo FOSS-KI Transformer SCHLAGEN kann

1. **Faktenabruf mit Confidence:** "Weiß ich oder weiß ich nicht?" — Transformer haben das nicht.
2. **Latenz auf CPU:** 4-7ms vs 200-500ms pro Token. Faktor 50-100×.
3. **Online-Lernen:** Neues Wissen sofort integriert, kein Re-Training.
4. **Dezentrale Ausführung:** Kein zentraler Server nötig.
5. **Determinismus:** Gleicher Input → gleicher Output. Reproduzierbar.
6. **Erklärbarkeit:** PPM-Kontext sichtbar, Hopfield-Attractor identifizierbar, Confidence-Score transparent.

### Wo FOSS-KI Transformer NICHT schlagen wird (und das ist OK)

1. **Freie Textgenerierung auf Absatzebene:** PPM generiert Wörter, keine Romane. Das ist nicht das Ziel.
2. **MMLU / Wissens-Benchmarks > 50%:** Braucht Milliarden Parameter + Trillionen Tokens. Nicht unser Spielfeld.
3. **Code-Generierung (HumanEval):** Symbolischer Solver für Logik ja, aber nicht freie Code-Synthese.
4. **Multilinguales Reasoning:** Erfordert massive multilinguale Trainingsdaten.

### Realistisches Ziel

FOSS-KI wird kein GPT-4-Ersatz. Es wird ein **spezialisiertes System** das in seinen Domänen überlegen ist:
- Deterministische Frage-Antwort mit Confidence
- Echtzeit-Inference auf Consumer-Hardware
- Online-adaptives Lernen ohne Re-Training
- Dezentral, regulierungsresistent, auditierbar

---

## Phase 4: Produkt-Features (wenn Phase 1-3 stehen)

### P1 — Chat-Interface
**Was:** Terminal-basiert. User stellt Frage → System antwortet mit Confidence-Level.
**Nicht:** RLHF, DPO, Constitutional AI. Stattdessen: Knowledge Store + Anti-Halluzination + PPM-Generierung. Das Alignment kommt aus der Architektur (kann nur antworten was es weiß), nicht aus Post-Training.

### P2 — Tool Use
**Was:** Symbolischer Parser erkennt Intent → ruft passende Funktion auf.
**Nicht:** Generative Tool-Calls wie GPT. Stattdessen: deterministischer Dispatch basierend auf Knowledge Store Matching.

### P3 — Wissens-Import
**Was:** .causal Dateien, Wikipedia-Dumps, Wikidata → automatisch in Knowledge Store laden.
**Integration:** pipeline_dza (bereits vorhanden) extrahiert Triplets aus PDFs → direkt in Hopfield speichern.

### P4 — Federated Learning / Dezentraler Wissensaustausch
**Was:** Mehrere FOSS-KI Instanzen tauschen Knowledge Store Updates über Foss Consensus aus.
**Warum:** Jede Instanz lernt lokal, Wissen wird dezentral synchronisiert. Kein zentraler Server.

---

## Unterschiede zu Transformer-Entwicklung

| Aspekt | Transformer | FOSS-KI |
|--------|------------|---------|
| Training | End-to-end Backprop | PPM: Zählen. Reservoir: fest. Readout: Ridge. |
| Hardware | GPU-Cluster (A100/H100) | Einzelner CPU-Kern |
| Skalierung | Parameter × Tokens (Chinchilla) | Kontext-Tiefe × Fakten × Dimension |
| Alignment | RLHF / DPO / Constitutional | Architektonisch (Anti-Halluzination = eingebaut) |
| Halluzination | Unvermeidlich (generativ) | Architektonisch verhindert (Attractor-Distance) |
| Online-Lernen | Catastrophic Forgetting | Additiv (PPM-Bäume wachsen, vergessen nicht) |
| Kontextfenster | O(n²) Attention, feste Länge | O(1) Hidden State, theoretisch unbegrenzt |
| Deployment | vLLM, TGI, Quantisierung | Einzelnes Python-Script, kein Server nötig |

---

## Codebase-Übersicht

```
foss-ki/
├── core/
│   ├── engine.py          ← FossKI Hauptklasse (alle 8 Layer)
│   ├── language.py         ← PPMModel + HierarchicalLanguageModel
│   ├── vortex.py           ← VortexLanguageModel (3-Fiber) + SymmetryDetector
│   ├── knowledge.py        ← KnowledgeStore + ModernHopfield + PhraseEncoder
│   ├── reservoir.py        ← MarkovReservoir (PS-Lifted)
│   ├── markov.py           ← PSLifted Markov Chain
│   ├── memory.py           ← HopfieldMemory
│   ├── consensus.py        ← FossConsensus + EnsembleConsensus
│   ├── evolution.py        ← TopologyEvolver + GraphGenome (MAP-Elites)
│   ├── causal.py           ← CausalGraph + do-Calculus (T92)
│   ├── dialog.py           ← DialogSystem + QueryParser + EntityTracker
│   ├── extractor.py        ← Auto-Extraktion Text→Triplets (Alpha+Bravo)
│   ├── router.py           ← VortexRouter (ℤ/9ℤ Fiber→Subsystem, Alpha)
│   ├── composer.py         ← ResponseComposer (Multi-Fakt-Antworten, Alpha)
│   ├── formulierer.py      ← PPM-Formulierer (Template→Prose, Alpha)
│   ├── metacognition.py    ← Gap-Fill + Self-Test + Pattern-Learner (Bravo)
│   ├── reasoning.py        ← Why/How/Prove/Solve Reasoning (Alpha)
│   ├── wikidata.py         ← Wikidata Importer + Bootstrap (Bravo)
│   └── spikes.py           ← SpikeEncoder + EventDrivenProcessor
├── benchmarks/
│   ├── benchmark_integration.py    ← Gesamtsystem-Test
│   ├── benchmark_vortex_integrated.py ← Vortex +9% Beweis
│   ├── benchmark_language_real.py  ← PPM auf Gutenberg
│   ├── benchmark_bernoulli.py      ← T90 (NEGATIV)
│   ├── benchmark_bernoulli_fair.py ← T90 fairer Vergleich
│   ├── benchmark_hopfield_scale.py ← T91+S1 Skalierung
│   ├── benchmark_causal.py         ← T92 do-Calculus
│   ├── benchmark_cpu_vs_transformer.py ← T94 FOSS-KI vs GPT-2
│   ├── benchmark_decentralized_consensus.py ← T93 UDP-Consensus
│   ├── benchmark_logic_puzzles.py  ← T95 Sudoku/Zebra/SAT
│   ├── benchmark_ppm_real_corpora.py ← S3 PPM auf 1M+ chars
│   ├── benchmark_knowledge_graph.py ← S2 FB15k-237 real KG
│   ├── benchmark_dialog.py          ← Multi-Turn Dialog 93%
│   ├── benchmark_glove_vs_ngram.py  ← W5 GloVe vs n-gram Encoder
│   ├── benchmark_vortex_router.py   ← Vortex Router Classification
│   ├── benchmark_auto_extraction.py ← Auto-Extraction 15/15
│   ├── benchmark_full_demo.py       ← Full Pipeline Demo (5 Capabilities)
│   └── benchmark_wikipedia_scale.py ← Wikipedia 100 Artikel Scale Test
├── data/
│   ├── alice.txt (144K)
│   ├── shakespeare.txt (96K)
│   ├── pride.txt (728K)
│   ├── world_knowledge.json   ← Bootstrap-Daten (174 Fakten, 41 Entities)
│   ├── gap_log.json           ← Persistente Wissenslücken
│   └── selftest_checkpoints/  ← Self-Test JSON-Checkpoints
├── cli.py                  ← CLI (ask/feed/chat/selftest/bootstrap/gaps/patterns/status)
└── KI PLAYBOOK.md          ← Dieses Dokument
```

---

## Quick Commands

```bash
# Gesamtsystem testen
python benchmarks/benchmark_integration.py

# Vortex Language Model (+9%)
python benchmarks/benchmark_vortex_integrated.py

# Hopfield Skalierung
python benchmarks/benchmark_hopfield_scale.py

# Interaktiv (Engine)
python -c "
from core.engine import FossKI
fki = FossKI()
fki.store_facts([('France','capital','Paris'), ('Water','formula','H2O')])
print(fki.query_fact('France', 'capital'))
print(fki.query_fact('Narnia', 'capital'))  # → REJECTED
"

# Full Pipeline: Text → Extract → Dialog
python -c "
from core.extractor import TripletExtractor
from core.dialog import DialogSystem
text = 'Paris is the capital of France. Berlin is the capital of Germany.'
ext = TripletExtractor()
ds = DialogSystem()
ds.load_knowledge(ext.extract_from_text(text))
print(ds.turn('What is the capital of France?')['response'])
print(ds.turn('And Germany?')['response'])
print(ds.turn('What is the capital of Narnia?')['response'])
"
```

---

*Stand: 2026-03-13 09:40. FOSS-KI ist jetzt ein VOLLSTÄNDIGER AGENT (nicht mehr nur QA-Bot). Kann: Fragen beantworten, Code schreiben, Dateien lesen/schreiben, Mathe lösen, Texte generieren, Konzepte erklären. Agent Gamma im Nexus (Gedächtnis + Fakten-Checker). 521 Bootstrap-Fakten, 34 Tests ALL PASS, Plugin-Architektur für Self-Extension. 9.6K LOC in 24 Core-Modulen.*

## Neue Capabilities (Phase 3 — Agent Layer)

### Agent (core/agent.py)
- **IntentClassifier**: QUESTION / CODE_REQUEST / REASONING / GENERATION / TOOL_USE / CONVERSATION
- **CodeGenerator**: Template-basiert + AST-Validierung (Sort, Prime, Fibonacci, Search, etc.)
- **TextGenerator**: Erklärungen aus Knowledge + PPM + Templates
- **MathSolver**: Sichere arithmetische Auswertung
- **ToolExecutor**: File read/write + Shell execution (sandboxed)
- **Agent.process()**: Universeller Einstiegspunkt für beliebige User-Eingaben

### Tool Plugin System (core/tool_loader.py + tools/)
- Auto-Discovery: Drop .py in tools/ → sofort verfügbar
- 4 Tools: file_read, file_write, shell, count_lines
- Self-Extension möglich: Metacognition erkennt Lücken → schreibt neue Tools

### Agent Gamma — Nexus Adapter (core/nexus.py)
- FOSS-KI als 3. Agent im Multi-Agent-Netzwerk
- MEMORY: Auto-Extraktion aus Nexus-Nachrichten → Hopfield
- FACT-CHECKER: Claims gegen Knowledge verifizieren
- TASK-TRACKER: Tasks aus Nachrichten extrahieren und tracken
- CLI: `python3 cli.py nexus --status` / `--watch` / `--tasks`
