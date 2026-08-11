# Primitive-Schema-Lab: Befund und Transfer

Diese Kopie ist absichtlich vom Live-Paket getrennt. Sie enthält keine
Migration des produktiven `data/world.causal`; alle Persistenztests verwenden
temporäre Pfade.

## Architekturentscheidung

Die größte Locality-Verletzung war dieselbe Relationsbedeutung in Extraktion,
Quellenadaptern, Graph-Merge, ARC und Migration. Das Lab vertieft diese Grenze:

1. `fertig.primitives` besitzt Namen, Aliase, Familien, Muster, Marker und
   Inversen.
2. `fertig.relations` ist der einzige allgemeine Text-zu-Relations-Pfad.
3. `fertig.pipeline.KnowledgeGraph` ist die verlustfreie Graph-Fassade.
4. Schreibgrenzen (`grow_world`, `migrate_world_graph`) kanonisieren,
   deduplizieren und quarantänisieren.
5. ARC und Evolve konsumieren dieselben kanonischen Relationen und trennen
   Adaptation von Evaluation.

Das Register ist bewusst **nicht als geschlossen oder semantisch vollständig**
bezeichnet. Neue unbekannte Relationen bleiben sichtbar; Coverage, Unknown-Rate,
held-out accuracy/coverage/calibration und Beschreibungslänge sind geeignete
Messgrößen für Weiterentwicklung.

## Formel-Audit

- Sinnvoll: geordnete Relationskomposition als stochastische Kernel
  `K_(r;s) = K_r K_s`, sofern Domäne/Codomäne typisiert sind.
- Sinnvoll: Mehrdeutigkeit über kalibrierte Wahrscheinlichkeiten, Entropie und
  Top-2-Margin; Schwellen auf einem Holdout festlegen.
- Sinnvoll: Zeno als explizite Staleness-/Stop-Policy im Evolve-Loop.
- Nur diagnostisch: TwoNN/Levina-Bickel auf deduplizierten kontinuierlichen
  Repräsentationen mit Metrik-/k-Sweep und Bootstrap. Nicht auf one-hot
  Relationslabels und nicht als Vollständigkeitsbeweis.
- Nicht semantisch integrieren: Digitalwurzeln sind Mod-9-Buckets. Die eigenen
  Vortex-Notizen berichten Nachteile für Sprache und schlechtere PPM-Werte;
  als Synonym-Tiebreaker wären sie ein kollisionsreicher Hash.
- Nicht als Relationskomposition: Möbius-Skalaraddition ist kommutativ;
  Relationsfolgen sind geordnet. Allenfalls für kalibrierte, unabhängige
  Evidenzfusion in Log-Odds-Darstellung verwenden.
- Noether-Parität/Skalen-/Periodizitätsdetektoren sind Signaldiagnostik, keine
  linguistischen Primitive.

## Sicherer Transfer

Vor einer Übernahme zuerst gegen einen frischen Live-Snapshot diffen, weil das
Live-Repo während des Lab-Baus aktiv weiterlief. Empfohlene Reihenfolge:

1. `primitives.py`, `relations.py`, `diagnostics.py` plus Tests.
2. Typed `KnowledgeGraph` und Merge-Regressionstests.
3. Quellenadapter und sichere World-Persistenz.
4. ARC/Evolve und die Consumer (`intent`, `tools`, CLI).
5. Dry-Run-Migration eines **kopierten** World-Graphen prüfen; erst danach ein
   separates kanonisches Ziel schreiben.

Beispiel auf einer Kopie:

```python
from pathlib import Path
from fertig.gaps import migrate_world_graph

report = migrate_world_graph(
    Path("data/world-copy.causal"),
    Path("data/world-copy.canonical.causal"),
    dry_run=True,
)
print(report.final_count, report.rejected, report.quarantine)
```
