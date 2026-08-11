"""
fertig.grounding — die Grounding-Schicht: Symbole an Nicht-Wort-Anker binden.

Der Dictionary-Regress endet, wenn ein Symbol an etwas hängt, das KEIN
Wort ist:

  1. Perzeptuelle Anker (CLIP): Das Symbol wird an Bild-Raum gebunden —
     Text-Embedding des Wortes vs. Embedding seines Wikipedia-Bildes.
     Ähnlichkeit > Schwelle = perzeptuell gebunden.
  2. Quantitative Anker: Zahlen mit Einheiten aus Web-Text
     ("runs at 110 km/h", "weighs 5 tons") — Messungen, keine Wörter.
  3. Coverage-Metrik: Anteil der Symbole mit mindestens einem
     Nicht-Wort-Anker. Das ist die Fortschrittszahl des Moonshots.

Gebundene Fakten wandern in den Welt-Graphen (z. B. (cheetah, runs_at,
"110 km/h")) und machen quantitative Fragen beantwortbar.

EVIDENZ-TIER-EHRLICHKEIT (per _codex_lab/primitive_schema_snapshot/
LIVE_GROUNDING_READONLY_REVIEW.md, ADR-0001-grounding-is-a-certificate.md):
Dieses Modul liefert NIEMALS "direct_sensorimotor" Grounding im
Harnad-Sinn -- perceptual_anchor ist Cross-Modal-Transfer MENSCHLICHER
Bedeutung (CLIP wurde aus menschlichen Bild-Text-Paaren trainiert,
Wikipedia waehlte das Bild anhand des Wortes), quantitative_anchors ist
menschliche Text-Zeugenschaft (eine Zahl aus einem Wikipedia-Artikel ist
kein kalibrierter Sensor-Messwert des Agenten selbst). Empfohlene,
ehrlichere Namen (als Aliase unten ergaenzt, OHNE die bestehenden Namen zu
entfernen -- cli.py und Tests rufen weiterhin die alten Namen):
  perceptual_anchor    -> clip_cross_modal_evidence
  quantitative_anchors -> textual_quantity_evidence
  grounding_coverage   -> proxy_evidence_coverage
  ground_symbol        -> collect_anchor_evidence
"""

from __future__ import annotations

import io
import re
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import sources

# ---------------------------------------------------------------------------
# CLIP (perzeptueller Anker) — optional, dependency-frei degradierbar
# ---------------------------------------------------------------------------

_CLIP = None
_CLIP_PREPROCESS = None
_CLIP_TOKENIZER = None


def _ensure_clip():
    """CLIP lazy-laden (open_clip, ViT-B-32). None wenn nicht verfügbar."""
    global _CLIP, _CLIP_PREPROCESS, _CLIP_TOKENIZER
    if _CLIP is not None:
        return True
    try:
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k")
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        model.eval()
        _CLIP, _CLIP_PREPROCESS, _CLIP_TOKENIZER = model, preprocess, tokenizer
        return True
    except Exception:
        return False


def _clip_text_embedding(text: str) -> Optional[np.ndarray]:
    if not _ensure_clip():
        return None
    import torch
    with torch.no_grad():
        toks = _CLIP_TOKENIZER([text])
        emb = _CLIP.encode_text(toks)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb[0].numpy()


def _clip_image_embedding(image_bytes: bytes) -> Optional[np.ndarray]:
    if not _ensure_clip():
        return None
    import torch
    from PIL import Image
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        x = _CLIP_PREPROCESS(img).unsqueeze(0)
        with torch.no_grad():
            emb = _CLIP.encode_image(x)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb[0].numpy()
    except Exception:
        return None


def _fetch_image(url: str) -> Optional[bytes]:
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "fertig/1.0 (research)"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read()
    except Exception:
        return None


def perceptual_anchor(word: str, threshold: float = 0.22) -> Optional[dict]:
    """Symbol an Bild-Raum binden: Wikipedia-Bild des Wortes + CLIP.

    Liefert {similarity, image_url} wenn gebunden, sonst None."""
    if not _ensure_clip():
        return None
    title, desc, extract = sources.scrape.wikipedia_summary(word)
    if not extract:
        return None
    # Bild-URL aus dem REST-Summary (thumbnail)
    import json
    import urllib.parse
    try:
        d = json.loads(sources._http(
            "https://en.wikipedia.org/api/rest_v1/page/summary/" +
            urllib.parse.quote(title)))
        img_url = d.get("thumbnail", {}).get("source")
    except Exception:
        img_url = None
    if not img_url:
        return None
    img_bytes = _fetch_image(img_url)
    if not img_bytes:
        return None
    te = _clip_text_embedding(word)
    ie = _clip_image_embedding(img_bytes)
    if te is None or ie is None:
        return None
    sim = float(np.dot(te, ie))
    if sim < threshold:
        return None
    return {"similarity": sim, "image_url": img_url}


# ---------------------------------------------------------------------------
# Quantitative Anker (Zahlen + Einheiten — Messungen, keine Wörter)
# ---------------------------------------------------------------------------

_UNIT_RE = re.compile(
    r"\b(\d+(?:[.,]\d+)?)\s*(km/h|kmh|kph|mph|m/s|kg|g|ton|tons|t|cm|m|km|"
    r"mm|liters|litres|l|ml|years|year|months|days|hours|minutes|seconds|"
    r"degrees?|°c|°f|celsius|fahrenheit|meters?|metres?|kilometers?|"
    r"kilometres?|percent|%)\b", flags=re.I)

# Relations-Verben, vor denen quantitative Fakten stehen ("runs at",
# "weighs", "measures", "reaches", "grows to", "can reach")
_QUANT_VERBS = {
    "runs at", "run at", "can run at", "running at", "speeds up to",
    "at speeds of", "up to", "reaches", "reach", "weighs", "weigh",
    "weighing", "measures", "measure", "measuring", "grows to", "grow to",
    "can reach", "can weigh", "can grow", "grows up to", "can measure",
    "has a top speed of", "top speed of", "can travel at", "travels at",
    "lives for", "live for", "can live for", "lasts", "last", "holds",
    "contains up to", "costs", "cost", "produces up to", "takes", "takes up",
}


def _parse_number(raw: str) -> Optional[float]:
    """Parse a matched number that may use ',' as a thousands separator
    OR as a decimal separator (English web text mixes both). '6,350'
    (thousands) must parse to 6350.0, not 6.350 -- treating every comma as
    a decimal point silently corrupted every thousands-grouped number this
    function ever saw (caught by test_extract_quantitative_thousands_comma
    below, which the pre-existing test never checked -- it only asserted
    the unit STRING was present, never the parsed value).

    Heuristic: a comma followed by exactly 3 digits (optionally repeated,
    e.g. "1,234,567") is a thousands separator; a comma followed by 1-2
    digits at the end of the string is a decimal separator (e.g. "6,35").
    This matches how English-locale web text actually punctuates numbers;
    it is a heuristic, not a full locale parser, and is documented as such.
    """
    if re.fullmatch(r"\d{1,3}(,\d{3})+", raw):
        return float(raw.replace(",", ""))
    if re.fullmatch(r"\d+,\d{1,2}", raw):
        return float(raw.replace(",", "."))
    try:
        return float(raw.replace(",", ""))
    except ValueError:
        return None


def extract_quantitative(text: str) -> List[Tuple[str, str, float]]:
    """Quantitative Fakten: (Verb-Kontext, 'Zahl Einheit', geparster Zahlenwert).

    Der dritte Tupel-Eintrag ist der geparste NUMERISCHE WERT der Messung
    (via _parse_number), nicht eine Konfidenz -- der urspruengliche
    Docstring nannte es "Konfidenz", was irrefuehrend war: dieser Wert wird
    unten NIE als Konfidenz verwendet, quantitative_anchors() haengt eine
    separate, tier-basierte Konfidenz an (Infobox/Summary/Volltext/Web)."""
    out = []
    for sent in re.split(r"(?<=[.!?])\s+", text):
        low = sent.lower()
        for verb in _QUANT_VERBS:
            idx = low.find(verb)
            if idx < 0:
                continue
            window = sent[idx + len(verb):idx + len(verb) + 60]
            m = _UNIT_RE.search(window)
            if m:
                val = _parse_number(m.group(1))
                if val is None:
                    continue
                out.append((verb, f"{m.group(1)} {m.group(2).lower()}",
                            val))
            break
    # Dedupe
    seen = set()
    uniq = []
    for v, u, val in out:
        key = (v, u)
        if key not in seen:
            seen.add(key)
            uniq.append((v, u, val))
    return uniq


# Infobox-Felder, die quantitative Anker sind (strukturiert, hochkonfident)
_QUANT_INFOBAR = {"mass", "weight", "speed", "top_speed", "length",
                  "height", "wingspan", "lifespan", "life_span",
                  "diameter", "depth", "width", "area", "volume",
                  "capacity", "range", "maximum_speed", "average_speed",
                  "max_speed", "max_length", "max_height", "max_weight",
                  "gestation", "incubation", "age", "size"}


def quantitative_anchors(word: str, sources_list: Optional[List[str]] = None,
                         ) -> List[Tuple[str, str, float, float]]:
    """Quantitative Anker: Wikipedia-Infobox (strukturiert) zuerst, dann
    Wikipedia-Text, dann Web (nur als Ergänzung).

    Rueckgabe: (Verb-Kontext, 'Zahl Einheit', geparster Zahlenwert,
    Quellen-Tier-Konfidenz). Frueher wurden extract_quantitative()'s
    geparste Zahlenwerte hier verworfen (List-Comprehensions banden sie an
    '_' und ersetzten die Position durch die Tier-Konfidenz) -- der
    numerische Wert und die Konfidenz sind jetzt beide erhalten, als zwei
    getrennte Positionen statt einer Ueberschreibung der anderen."""
    anchors = []
    # 1. Infobox: strukturierte quantitative Felder (höchste Konfidenz)
    try:
        infobox = sources.scrape.wikipedia_infobox(word)
    except Exception:
        infobox = {}
    for key, val in infobox.items():
        if key in _QUANT_INFOBAR:
            m = _UNIT_RE.search(val)
            if m:
                num = _parse_number(m.group(1))
                if num is not None:
                    anchors.append((key, f"{m.group(1)} {m.group(2).lower()}",
                                    num, 0.70))
    # 2. Wikipedia-Text (Summary zuerst, dann Volltext-Fallback)
    title, desc, extract = sources.scrape.wikipedia_summary(word)
    if extract:
        anchors += [(v, u, val, 0.50) for v, u, val in
                    extract_quantitative(extract[:4000])]
    if not anchors:
        # Volltext: den ganzen Artikel crawlen (Trafilatura)
        import urllib.parse
        full = sources.extract_text(
            "https://en.wikipedia.org/wiki/" +
            urllib.parse.quote(title.replace(" ", "_")))
        if len(full) > 500:
            anchors += [(v, u, val, 0.45) for v, u, val in
                        extract_quantitative(full[:12000])]
    # 3. Web nur, wenn noch keine Anker gefunden wurden
    if not anchors:
        for url in sources._ddg_urls(f"{word} speed weight size")[:2]:
            text = sources.extract_text(url)
            if len(text) > 200:
                anchors += [(v, u, val, 0.35) for v, u, val in
                            extract_quantitative(text)]
    seen = set()
    uniq = []
    for v, u, val, conf in anchors:
        key = (v, u)
        if key not in seen:
            seen.add(key)
            uniq.append((v, u, val, conf))
    return uniq[:6]


# ---------------------------------------------------------------------------
# Grounding-Coverage — die Moonshot-Metrik
# ---------------------------------------------------------------------------

# Erdungs-Ebenen (ehrliche Terminologie — nichts davon ist "volle Erdung"
# im Harnad-Sinn; alles ist menschlich vermittelt, nur unterschiedlich nah
# an primärer Transduktion):
#   L0: nur Wort-Wort-Kanten (reiner Regress)
#   L1: quantitative Anker — Zahlen aus Webtext. Zahlen sind nicht-
#       lexikalische Symbole mit fester Denotation, ABER die Quelle ist
#       menschlicher Text (vermittelt).
#   L2: perzeptuelle Bindung — CLIP/Commons-Bilder. Die Pixel sind primär,
#       die Wort↔Bild-Zuordnung ist menschenkuratiert (vermittelt).
#   L3: unüberwachte Kategorien aus Pixel-Struktur (fertig.vision) — die
#       Kategorie entsteht ohne Labels; Wörter werden NACH der Kategorie-
#       Bildung zugeordnet. Die nächste Annäherung an Harnads
#       sensorische Transduktion, die ein textbasiertes System erreichen
#       kann.
GROUNDING_LEVELS = {"L0": 0, "L1": 1, "L2": 2, "L3": 3}


def grounding_coverage(graph_triplets: List[Tuple[str, str, str, float]],
                       anchored: Dict[str, int]) -> Dict[str, float]:
    """Anteil der Graph-Symbole pro Erdungs-Ebene.

    anchored: Symbol -> höchste erreichte Ebene (0-3). Die Metrik ist
    ehrlich: "verankert" heißt nicht "geerdet" — volle Erdung verlangt
    sensorische Transduktion, die kein textbasiertes System hat."""
    symbols = set()
    for a, b, c, _ in graph_triplets:
        symbols.add(a)
        symbols.add(c)
    grounded = sum(1 for s in symbols if anchored.get(s))
    return {
        "symbols": len(symbols),
        "grounded": grounded,
        "coverage": grounded / max(len(symbols), 1),
    }


def ground_symbol(word: str, store: bool = False,
                  verbose: bool = True) -> dict:
    """Collect anchor evidence for a symbol (perceptual + quantitative) and,
    if store=True, write the resulting facts into the world graph.

    NOTE per LIVE_GROUNDING_READONLY_REVIEW.md (_codex_lab, read-only
    review of this module): store now defaults to False. A research/anchor-
    collection API that mutates shared state by default is a footgun --
    the caller who wants persistence (cli.py's `fertig ground --all`) now
    passes store=True explicitly (see cmd_ground below), everyone else gets
    a read-only evidence collection call by default.

    The stored confidence per quantitative anchor is now the anchor's OWN
    measured confidence (from quantitative_anchors: infobox=0.70, summary
    text=0.50, full article=0.45, web fallback=0.35), not a flat constant
    0.55 that discarded the source-tier signal quantitative_anchors already
    computed -- see quantitative_anchors' docstring for what each tier
    means. The perceptual anchor's CLIP similarity was already stored
    correctly (unchanged)."""
    from . import gaps as gaps_mod
    result = {"word": word}
    # 1. Perzeptuell
    pa = perceptual_anchor(word)
    result["perceptual"] = pa
    # 2. Quantitativ
    qa = quantitative_anchors(word)
    result["quantitative"] = qa
    # 3. Fakten in den Graphen
    if store and (pa or qa):
        trips = gaps_mod._load_world()
        best = {t[:3]: t[3] for t in trips}
        if pa:
            key = (word, "has_image", "perceptual")
            best[key] = max(best.get(key, 0.0), pa["similarity"])
        for verb, unit, val, conf in qa:
            key = (word, verb.replace(" ", "_"), unit)
            best[key] = max(best.get(key, 0.0), conf)
        gaps_mod._save_world([(a, b, c, conf)
                              for (a, b, c), conf in best.items()])
        result["graph_triplets"] = len(best)
    if verbose:
        print(f"[ground] {word}:")
        if pa:
            print(f"  perzeptuell : gebunden (CLIP-Sim {pa['similarity']:.3f})")
        else:
            print(f"  perzeptuell : nicht gebunden")
        print(f"  quantitativ : {len(qa)} Anker")
        for v, u, val, conf in qa[:4]:
            print(f"    {v} {u} ({val}, conf={conf})")
    return result


# ---------------------------------------------------------------------------
# Ehrliche Namens-Aliase (ADR-0001-grounding-is-a-certificate.md, siehe
# Modul-Docstring). Reine Umbenennungen -- gleiches Verhalten, gleiche
# Signatur, keine der bestehenden Namen wird entfernt (cli.py und die
# bestehenden Tests rufen weiterhin perceptual_anchor/quantitative_anchors/
# grounding_coverage/ground_symbol unveraendert).
# ---------------------------------------------------------------------------
clip_cross_modal_evidence = perceptual_anchor
textual_quantity_evidence = quantitative_anchors
proxy_evidence_coverage = grounding_coverage
collect_anchor_evidence = ground_symbol
