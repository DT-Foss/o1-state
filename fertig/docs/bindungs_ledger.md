# GSM8K Bindungs-Parser — Fortschritts-Ledger (Runde 1-3)

Stand 2026-08-11 (Nacht): **11/300 (3.7%)** via Bindungs-Parser allein.
Runde 5-6: they-total (320), more2x (25), Synonyme (pieces==slices),
letzter Frage-Satz, Rate x Dauer (120). Nächste: End-minus-Muster
(Janice 536-318+40), Prozent-Subtraktion (Baez 20%), mehrteilige Dauer.
(vorher 0-3/100 mit Templates — Falsifikation der Muster-Route bestätigt).
Runde 4 hinzugekommen: Futterketten (1080), with-Mengen (49),
Wort-Zahlen-Digitalisierung, Variablen-Gleichungen (10).

## Architektur

`fertig/bindings.py`: Zahl → Objekt → Einheit → Rolle, dann
Relations-Graph, dann Resolver. Abstinenz: unvollständige Bindung → None.

```
Textaufgabe
  -> _find_quantities   (Zahlen + Zahlwörter, NP-Phrasen)
  -> _find_relations    (jede/more/fewer/times/assign — ITERATIV)
  -> _apply_relations   (Relationen -> absolute Mengen, dedupe)
  -> _effective_quantities (Roh-Mengen ersetzen, die in Relationen stecken)
  -> _bind_roles        (qty/partitive/ratio/price/duration, 1:1-Transfer)
  -> _solve_variables   (Personen-Gleichungen, vorwärts+rückwärts)
  -> Resolver-Fälle A/B/B' (sum/ratio/partitiv/left/diff)
```

## Gelernte Klassen (fixiert in tests/test_bindings.py)

| Klasse | Beispiel | Ergebnis |
|---|---|---|
| 1:1-Transfer + Ratio | Natalia: sold clips to 48 friends, half as many | 72 ✓ |
| Objekt-Trennung | 5 apples + 3 oranges, "how many apples" | 5 ✓ |
| Summe gleicher Objekte | 12 cakes + 15 cakes total | 27 ✓ |
| jede-Relation (Pro-Stück) | 2×16 + 2×8 Slices | 48 ✓ |
| more/fewer-Kette (iterativ) | 11 snowflake, +9 truck, -13 rose | 38 ✓ |
| Variablen-Gleichungen | Mina=6×Carlos, Mina=24, Sam=Carlos+6 | 10 ✓ |
| Wort-Zahlen | "six times"/"six more" (digitize) | — |

## Nächste Klassen (43 falsch, kategorisiert)

1. **Futterkette**: "each bird eats 12 beetles, each snake eats 3 birds,
   each jaguar eats 5 snakes, 6 jaguars" -> 6×5×3×12 (Ketten-each)
2. **each-of-N**: "each of the first four houses has 3 gnomes"
3. **Zeiträume**: "once every hundred years ... over 700 years" (Division),
   "Every hour ... for N hours" (Summe über Schleife)
4. **with-Mengen**: "7 starfish with 5 arms each and one seastar with 14
   arms" = 7×5 + 14
5. **Variable + Basis aus Kontext**: "Tim has 30 less than Martha, Harry
   half as many as Tim" (Martha-Wert fehlt im Satz -> Kontext)
6. **Typ-Phrasen mit Adjektiv**: "2 large pizzas" → "a large pizza has
   16 slices" (Holder-Matching über Adjektiv hinweg)

## Messprotokoll

- 300 Trainings-Fragen (erstes Drittel), exakter Gold-Abgleich
- Falsch-gebundene Fälle (43) sind LERNMATERIAL, nicht Bug-Fälle
  (jede Klasse = nächste Regel)
- Determinismus: gleiche Frage -> gleiche Bindung -> gleiche Antwort
