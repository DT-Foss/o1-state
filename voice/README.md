# voice/ — das Weltbuch wird gesprochen

**Moonshot (Co-Work-Session 2026-08-11, zweite Welle):** Die Organe des
Repos waren getrennte Tiere — der Sprecher hat nie etwas erlebt, der
Körper (seit heute, body/) kann nicht sagen was er tat, und niemand hat je
vom Erlebten eines ANDEREN profitiert. `voice/` schließt den Kreis durch
alle: **A lebt → As Records lehren einen Sprecher, aus Sehen allein zu
sagen was geschah → ein fremder Körper B hört nur die Sätze, stellt sie
sich auf der eigenen Sicht vor und wird messbar schneller verkörpert.**
Sprache als Träger verkörperten Wissens zwischen Körpern, Ende zu Ende,
bit-provenant an jeder Naht — und exakt die intervention_necessity-Achse,
die der Grounding-Kernel prüft, im Kleinen: die Äußerung muss der
Intervention folgen.

## Der Kreis, Bauteil für Bauteil

| Datei | Inhalt |
|---|---|
| `vocab.py` | Proto-Grammatik mit drei Slots (Richtung/Magnitude/Aktionswort), deterministischer Codec Record↔Satz, und `RichPanSource` — body/s Welt, read-only subklassiert, mit zyklischer Pan-Magnitude 1,2,3 (sonst wäre der Magnitude-Slot eine Konstante). Provenance-Contract (seed, trace) → Frames bleibt byte-identisch erhalten. |
| `speaker.py` | `SpeakerNet`: (f_t, Δf·4) → GSSMCore d128 → drei Köpfe. **Die Aktion ist bewusst KEIN Input** — aus Sehen allein wird das Aktionswort zum Bericht statt zum Echo, und die Epistemik wird messbar: auf stillen Schritten (Hold, geklemmte Pans) ist der Druck prinzipiell unsichtbar → Accuracy-Decke beim Prior (P91d). Training = Schlaf-Replay über das eigene aufgezeichnete Leben (6 Pässe; P47-Präzedenz), kompositionaler Holdout (right, 2) in keinem Loss. |
| `imagination.py` | Hörensagen → Pseudo-Erfahrung: gehörten Satz dekodieren, beschriebene Konsequenz per Kanten-replizierendem Shift auf die EIGENE aktuelle Sicht anwenden, (Frame, beschriebene Aktion) → imaginierter Nächst-Frame als Trainingsmaterial. Die Rand-Lüge (enthüllte Spalten sind unwissbar) ist im Docstring benannt und im Register eingepreist. `scramble_utterances` = konsistente Lüge für den Gift-Arm. |
| `run_voice.py` | Stufen life/speak/transmit. Leben unter seeded Random (P88-Lektion), modellfreier Extraktor, VOLLER Trace als Provenance-Träger, eigener RichPan-Verifier (der Debug-Smoke fing den Weltklassen-Mismatch als 0/5). Transmission: **modell-gepaartes Design** — bit-identische Init-Gewichte, identische Welt/Aktionsfolge/Budget über alle vier Arme (silent / hear_codec / hear_learned / scrambled), Unterschied AUSSCHLIESSLICH das Gehörte. Probe = feste Route in separater Welt: kein Arm wird auf einem Test benotet, den er selbst gewählt hat (P89c-Confound konstruktiv geschlossen). Tie-Guard: keine Counterfactual-Differenz = kein Treffer. |
| `test_voice.py` | 7 Smokes (Codec-Bijektion, Provenance der reichen Welt, Imagination == Welt-Op via Extraktor-Estimator, konsistente Lüge, Sprecher-Mechanik, Arm-Paarung bit-identisch, Pair-Cap). |
| `PREDICTIONS_VOICE.md` | Register-ENTWÜRFE P91–P94, vor dem Scoring-Lauf; Verdicts danach darunter. Merge ins Hauptregister macht die Haupt-Session. |

## Sichtbar ab Tag 1

- `gifs/voice_narrated_life.gif` — das Leben, vom eigenen Mund erzählt:
  Frame für Frame „said: right 2 (pan_right)" gegen „trth: …".
- Neugier auf die Zahlen: `results/speaker_summary.json` (Kompetenz,
  Komposition, Interventions-Flip, Epistemik-Decke),
  `results/transmission_summary.json` (vier Arme, Checkpoints,
  Ignitionszeiten), Kurven in `results/voice_curves.png`.

## Laufen lassen

```bash
python3 voice/test_voice.py     # 7/7, kein vizdoom nötig
python3 voice/run_voice.py --stage all --seed 42 --frames 8000 \
    --b-seed 777 --b-frames 4000 --probe-every 500 --probe-len 96
```

## Grenzen & ehrliche Lücken

- Nur `voice/` wird beschrieben; `body/`, `visual/`, `hsslm/`,
  `src/livecausal/` sind read-only Importe. Keine Server, Thread-Clamps.
- Die Sprache ist ein 3-Slot-Protokoll, kein offenes Vokabular — der Punkt
  ist Falsifizierbarkeit der Übertragung, nicht linguistische Breite.
  Offenes Vokabular (Sprecher erfindet Wörter, Hörer erdet sie im
  Naming-Game des Kernels) ist der benannte nächste Schritt.
- Imagination nutzt den Codec auch beim GELERNTEN Sprecher (die Wörter
  sind geteilt; gelernt ist, WANN welches Wort gilt). Privates Vokabular
  + gelernter Hörer = Folgearbeit.
- Benannt-nicht-gebaut bleiben außerdem: der zustandskonditionale
  Zweifel/Neugier-Kopf (P88-Notwendigkeit) und das Doom-v2-Konsequenz-Organ
  (P90c-Falsifier).
