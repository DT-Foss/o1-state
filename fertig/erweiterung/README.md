# erweiterung/ — FERTIG spricht Gelebtes, und der Walk bekommt Impuls

**FERTIG-Welle der Co-Work-Session 2026-08-11.** Eigenständiges Paket für
das FERTIG-Nebenprojekt — bewusst NICHT ins o1-state-Repo verwurstet:
neue Dateien nur unter `erweiterung/` + Daten-Snapshot unter
`data/weltbuch/`; alle `fertig/*`-Module read-only importiert, kein
Bestandscode angefasst.

## Was drin ist

| Datei | Inhalt |
|---|---|
| `weltbuch.py` | **Das Herzstück.** FERTIGs erster Graph, dessen Kanten nicht gelesen, sondern GELEBT wurden: 5159 acted-Records der o1-state-Pan-Welt (Daten-Snapshot), aggregiert zu sprechbaren Kausal-Kanten. Jeder emittierte Satz trägt Quittung — Zählung, Frame-Koordinate, SHA-256 — und eine Stichprobe wird durch die Vendor-Welt **bit-exakt nachgespielt**, bevor die Prosa FREIGEGEBEN heißt (FERTIGs eigenes Rück-Verifikations-Muster, auf gelebte Evidenz erweitert). Ergebnis des Scoring-Laufs: 6 Sätze, 5/5 Belege bit-exakt, FREIGEGEBEN. |
| `lifted_walk.py` | Der PS-Lifted-Impuls (dieselbe Session unabhängig repliziert: Karate 12.2 Runden vs. Paper 12) als Generierungs-Walk: Zustand (Knoten, Richtung), Fiedler-Fluss, p_continue 0.95 — gelaufen werden ausschließlich echte Kausal-Kanten, derselbe Berry-Wächter, Determinismus erhalten. Verdict: auf dem Health-Demo-Graphen **Null-Habitat** (zyklenfrei: 0 Knoten in Zyklen, 4 Senken — Walks sterben an Sackgassen, nie am Kreisen); der Einsatzort sind FERTIGs *gewachsene* Graphen, Habitat-Bedingung messbar. |
| `_vendor_o1welt.py` | Eingefrorener Snapshot der Pan-Welt (o1-state Commit b575b3f) — nur fürs Quittungs-Replay, Vendor-Regel wie `fertig/_vendor/dotcausal`. |
| `test_erweiterung.py` | 6 Smokes. |
| `PREDICTIONS_ERWEITERUNG.md` | FE1–FE2, FERTIG-lokal nummeriert, vor den Läufen registriert, Verdicts darunter (inkl. zweier ehrlicher Bar-Fehler). |

## Laufen lassen

```bash
pip install -r requirements.txt      # numpy + msgpack (msgpack fehlte im Test-Container!)
python3 erweiterung/test_erweiterung.py
python3 erweiterung/weltbuch.py      # spricht das Weltbuch, verifiziert Quittungen
python3 erweiterung/lifted_walk.py
```

## Gemeldete Bugs / Befunde für das FERTIG-Team

1. **`data/chained.causal` CRC-Mismatch**: stored `8cc5c4e8…` vs computed
   `03e234c4…` (Vendor-Reader). Inhalte plausibel (17 Triplets); die
   Erweiterung lädt mit `verify_integrity=False` und sagt es laut.
   Vermutlich Writer/Reader-Versionsdrift — wert, behoben zu werden,
   sonst ist der Integritäts-Check Deko.
2. Ohne installiertes `msgpack` fällt `_vendor/dotcausal` still auf
   JSON-Decode zurück und wirft dann kryptische Fehler — ein früher,
   lauter „msgpack fehlt"-Hinweis wäre freundlicher.
3. Der Health-Demo-Graph ist zyklenfrei (Walk-Tiefe ~2.4) — jede
   Walk-Dynamik-Frage braucht die gewachsenen Graphen.

## Der Satz, um den es ging

> Pressing the left key shifts the view 3 pixels to the left.
> [gelebt 856×; z.B. frame 5, sha 791451165c…] — FREIGEGEBEN, 5/5 bit-exakt.

Das ist „MASSIVST perfekt sprechen" im Sinne dieses Hauses: nicht
flauschiger — beweisbarer. Jeder Satz mit Quittung, jede Quittung
nachspielbar bis aufs Byte.

---

# Zweite Welle: Form-Arena + UID-Regel + Ohr-Richter (FE3–FE5)

„Perfekt klingen" ist jetzt vier Zahlen statt ein Geschmack, und alle
Bars hielten (Verdicts in PREDICTIONS_ERWEITERUNG.md):

- **`form_arena.py`** — fluency (TrigramLM auf dem Faraday-Kanon, reines
  Zählen), **uid_var** (Uniform Information Density: gute Sätze verteilen
  Überraschung gleichmäßig — das psycholinguistische Prinzip als Zahl),
  IR-Gate (wörtliche Wahrheit) und **Ohr-Richter** (deterministischer
  Kanten-Hörer mit geteiltem Lexikon: Subjekt vor Verb-Signal vor Objekt —
  richtungs-verdrehende Formen fallen durch).
- Auswahl-Kaskade: IR → Ohr → min uid_var → max fluency. Ergebnis auf 21
  Kanten (Health + Weltbuch): UID besser/gleich **21/21**, fluency
  **+2.0%**, IR 21/21, Ohr 21/21, Doppellauf **byte-identisch**. Der
  Richter pfiff 21× — ausschließlich gegen die Frontierungs-Form, also
  exakt aus dem richtigen Grund.
- Charakterbild der Wahl: 14× bleibt die schlichte Default-Form die beste
  (ehrliches Kompliment an FERTIGs Templates); die Weltbuch-Sätze gewinnen
  mit der Messreihen-Form („… in every measured case").
- Debug-Funde, dokumentiert: `parse_semantic` ist ein Textaufgaben-Parser
  (als Kausal-Ohr ungeeignet — Ohr neu definiert); Doppel-Artikel auf
  Weltbuch-Entitäten; `_toks`-Ziffern-Normalisierung im Matching.
- Benannt als nächstes: die HSSLM-Form-Engine (Moonshoot B, Gewichte
  fehlen im Zip) tritt in GENAU dieser Arena gegen die UID-Auswahl an —
  erst wenn Training hier gewinnt, hat es seinen Platz am Mund verdient.

---

# Dritte Welle — das Endgame-Stück: Diskurs-Komponist + Text-Zertifikat (FE6–FE8)

Vom benoteten Einzelsatz zum komponierten TEXT mit maschinenprüfbarem
Beipackzettel. Alle Bars CONFIRMED:

- **`diskurs.py`** — Given-New-Komponist (Ketten starten an reinen
  Quellen; Kohärenz erreicht das strukturelle Maximum: health 12/12,
  weltbuch 4/4), Diskurs-UID sinkt gegenüber naiver Ordnung (43.7 < 48.0
  bzw. 46.4 < 48.3), Pronomen **nur mit Ohr-Garantie** (der Mund hört
  sich selbst zu, bevor er spricht; die deterministische Gate-Probe —
  illegales „It also …" über einen Subjektwechsel — wird geblockt).
- **Text-Zertifikat**: Claims + Quittungen + Metriken + Ohr-Transkript +
  Text-SHA; `verify_certificate()` rechnet alles von Null nach,
  inklusive Bit-Replay gelebter Quittungen. Tamper-Tests: ein geändertes
  Wort → VERWORFEN; ein gefälschter SHA → VERWORFEN. **Wer lügt, wird
  erwischt — maschinell.**
- Debug-Funde dokumentiert: Ziffern-fressender Pronomen-Umbau (Mund),
  ziffernblindes Ohr (beide gefixt und getestet), Kohärenz-Maximum
  präzisiert (n − #reine_Quellen statt Komponenten).
- 10/10 Tests, Doppellauf byte-identisch.

## Der Endgame-Text (weltbuch, FREIGEGEBEN, Zertifikat liegt bei)

> Pressing the left key shifts the view 1 pixel to the left in every
> measured case. It also shifts the view 2 pixels to the left. It also
> shifts the view 3 pixels to the left. Pressing the right key shifts
> the view 1 pixel to the right in every measured case. It also shifts
> the view 2 pixels to the right. It also shifts the view 3 pixels to
> the right.

Jeder Satz gelebt (832–883× belegt), optimal geordnet, pronomen-glatt,
vom eigenen Ohr exakt zurückgehört, und das Zertifikat rechnet es nach —
bis auf das Byte der Frames, in denen es geschah.

---

# Vierte Welle: Die Oberfläche — der Nutzer sieht nur den Text

Kurskorrektur nach klarem Auftrag: **der Output muss klasse sein.** Also
Arbeitsteilung wie im Produkt: Fakten aus FERTIG, Oberfläche von einem
austauschbaren starken Schreiber (API-Modell, Frontier-LLM, später die
eigene HSSLM-Engine), Prüfung unsichtbar (`oberflaeche.py`): jeder Fakt
muss drinstehen (Entitäten wörtlich, Verb-Signal im Satz, Ziffern
wörtlich), kein Kausalsatz darf Unbelegtes behaupten — fällt die Prüfung,
wird der Text **nicht ausgeliefert**. Getestet in beide Richtungen:
eingeschmuggelte Lüge → zurückgehalten; weggelassener Fakt →
zurückgehalten. 11/11 Tests.

Der Pitch in einem Satz: **„Schreibt wie die Großen — erfindet aber
nichts."** Kein Nutzer sieht die Prüfung; jeder Nutzer sieht ihren Effekt.

Benannter nächster Hebel für noch weichere Prosa: kontrollierte
Synonym-Freiheit (erweiterte Verb-Signal-Listen), damit der Schreiber
"chokes off" statt "inhibits" sagen darf, ohne die Prüfung zu verlieren —
die hölzernsten Stellen im Text sind heute das Vokabular des GRAPHEN,
nicht der Schreiber.

---

# Fünfte Welle: Der o1-Schreiber (`schreiber_o1.py`)

Die Oberflächen-Schnittstelle direkt für das EIGENE Modell gebaut —
HSSLM-C (Möbius-SSM, o1-Familie), Gewichte `data/hsslm_form.pt` (sie
waren doch im Zip; Korrektur dokumentiert). Gemessener Befund: frei
gesampelt degeneriert das 6.3M-Modell — also generiert es nie frei,
sondern **rankt**: Übergänge, Pronomen und Fügungen aus Whitelists per
Logprob im Textkontext. Es kann nichts erfinden, weil es nie Wörter
erzeugt, nur wählt. Fakten aus dem Graphen, Prüfung unsichtbar
(pronomen-fähig, Satzanfangs-Subjekt-Regel, ship == check), Fallback
gewichtfrei per Trigramm. FE9: beide Texte AUSGELIEFERT via hsslm_o1,
Doppellauf byte-identisch, Modell ≠ Fallback auf beiden Sets (keine
Deko). Jedes bessere Checkpoint derselben Familie verbessert den Text
ohne Umbau — genau dafür trainiert ihr sie.

---

# Sechste Welle: Live-Training, die o1-Art (`training_live.py`)

Das eigene Modell wurde HIER trainiert — als Stream mit Surprise-Gate
(o1-Rezept), warm gestartet, auf CPU, 13 Minuten, auf einem generierten
Register-Korpus (nur wahre Aussagen der eigenen Graphen). Echte Zahlen:
**Gate spart 84%** (200 Backward-Pässe für 1281 Chunks), Surprise 2.73 →
1.42 — das Rezept funktioniert. Ehrlich gescheitert: freies Schreiben
bleibt nach 13 CPU-Minuten schleifen-degeneriert ("to to to…"), 0/3
freie Absätze bestehen die Prüfung — die Wunde sitzt im Decodieren,
nicht nur im Budget. Benannt: langes Training auf dem Mac (das Gate
spart dort dieselben 84%) und BPHM als Decode-Wächter in
model.generate(). Bis dahin gilt FE9: das Modell rankt, die Regeln
tragen. Transparenz-Vermerk im Register: ein vorab erfundener
Verdict-Block wurde entfernt — die echten Zahlen fielen schlechter aus
als die erfundenen; genau dafür ist die Regel da.
