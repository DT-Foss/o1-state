"""pytest-Tests fuer graph_to_text_v3.py (Sprecher-Datenbasis v2, dringende
Nachbesserung: Satzgrenzen-Snapping + Content-Guard gegen den vom DeepSeek-
Grader-Schwarm gemessenen v2-Defekt: 44% broken, dominiert von FRAGMENT
(87% der Fenster beginnen mitten im Satz) und MISMATCH (Fenster ohne den
Record-Inhalt).

Laeuft gegen results/p72_store_local (klein, lokal verfuegbar) fuer die
teuren Store-Rekonstruktions-Tests, plus reine Unit-Tests der Snap-/Guard-
Funktionen ohne Store-Abhaengigkeit.

Run:
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \
    python3 -m pytest hsslm/data/test_graph_to_text_v3.py -q
"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest

from graph_to_text_v3 import (
    snap_to_sentence_boundaries, content_guard_ok, build_pairs_v3,
    P72_CADENCE, SNAP_PAD_TOKENS,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STORE_DIR = os.path.join(REPO_ROOT, "results", "p72_store_local")


# ---------------------------------------------------------------------------
# Reine Unit-Tests der Snap-/Guard-Funktionen (kein Store noetig)
# ---------------------------------------------------------------------------

def test_snap_moves_fragment_start_to_sentence_boundary():
    """Der v2-FRAGMENT-Defekt selbst: ein Fenster, das mitten im Satz
    beginnt, muss nach dem Snap am naechsten Satzanfang beginnen."""
    padded = ('the album gained no radio airplay . " Desolation Row " , '
              'backed by acoustic guitar and understated bass , offers '
              'the sole exception , with Dylan alluding to figures')
    orig_start = padded.index('Desolation') - 2  # window began right at the quote/period
    orig_end = len(padded)
    snapped = snap_to_sentence_boundaries(padded, orig_start, orig_end)
    assert snapped.startswith('"')
    assert not snapped.startswith(".")


def test_snap_ends_at_sentence_boundary_not_mid_sentence():
    # Zwei Satzgrenzen im Puffer: eine VOR orig_start (verbraucht vom
    # Vorwaerts-Snap), eine NACH orig_start aber VOR orig_end (die als
    # Ende-Grenze gefunden werden soll) -- sonst hat der Rueckwaerts-Snap
    # keine eigene Grenze mehr zur Verfuegung (siehe der Regressions-Test
    # fuer den Single-Boundary-Fall unten).
    padded = ('Prior sentence ends here . Released in January 2011 in Japan , '
              'it is the third game in the Valkyria series . Employing the '
              'same fusion of tactical')
    orig_start = padded.index('Released')
    orig_end = padded.index('Employing') + 5
    snapped = snap_to_sentence_boundaries(padded, orig_start, orig_end)
    assert snapped.endswith(".")
    assert "Employ" not in snapped
    assert snapped.startswith("Released")


def test_snap_single_boundary_in_buffer_does_not_collapse():
    """Regressions-Test fuer den gefundenen Bug: wenn NUR EINE Satzgrenze
    im Puffer liegt und orig_start_char genau auf sie faellt, darf der
    Vorwaerts- und Rueckwaerts-Match nicht denselben Treffer benutzen und
    dadurch auf den unveraenderten (fragmentierten) Rohbereich zurueckfallen."""
    padded = ('while their live shows showcased the same organ riffs . '
              '" Desolation Row " , backed by acoustic guitar and understated '
              'bass , offers the sole exception , with Dylan alluding to '
              'figures in Western culture')
    orig_start = padded.index('" Desolation') - 1
    orig_end = len(padded) - 5
    snapped = snap_to_sentence_boundaries(padded, orig_start, orig_end)
    assert snapped.startswith('"')
    assert not snapped.startswith(".")


def test_snap_falls_back_when_no_boundary_found():
    """Kein Satzzeichen im Puffer -- Fallback auf den Rohbereich statt
    eines Absturzes oder eines leeren Strings."""
    padded = "a very long sequence of words with no sentence punctuation at all here"
    start_idx, end_idx = 5, len(padded) - 5
    snapped = snap_to_sentence_boundaries(padded, start_idx, end_idx)
    assert snapped  # nichtleer
    assert snapped == padded[start_idx:end_idx].strip()


def test_content_guard_accepts_matching_window():
    assert content_guard_ok(
        "the animals", "metapopulation of hybrids",
        "animals formed a metapopulation of hybrids with varying morphology",
    ) is True


def test_content_guard_rejects_mismatch():
    """Der v2-MISMATCH-Defekt selbst: Fenster ohne den Record-Inhalt muss
    verworfen werden."""
    assert content_guard_ok(
        "many frozen", "any prehistoric animal",
        "2015 study suggested that the animals in the range where M. columbi "
        "and M. primigenius overlapped formed a metapopulation of hybrids",
    ) is False


def test_content_guard_short_words_dont_block():
    # nur kurze Woerter (<=3 Zeichen) -- nichts zu pruefen, nicht verwerfen
    assert content_guard_ok("a it is", "an of to", "completely unrelated text") is True


# ---------------------------------------------------------------------------
# Store-basierte Property-Tests (teuer -- p72_store_local, kleine Cadence)
# ---------------------------------------------------------------------------

pytestmark_store = pytest.mark.skipif(
    not os.path.isdir(STORE_DIR),
    reason="results/p72_store_local not present (read-only fixture store)",
)


@pytest.fixture(scope="module")
def v3_pairs_and_report():
    if not os.path.isdir(STORE_DIR):
        pytest.skip("results/p72_store_local not present")
    cadence = dict(P72_CADENCE)
    return build_pairs_v3(STORE_DIR, cadence)


def test_v3_report_has_expected_keys(v3_pairs_and_report):
    pairs, report = v3_pairs_and_report
    assert set(report.keys()) == {
        "n_raw", "n_skipped_missing_coverage", "n_content_guard_rejected",
        "n_junk_filtered", "n_dupe_filtered", "n_final",
    }


def test_v3_report_accounting_is_consistent(v3_pairs_and_report):
    pairs, report = v3_pairs_and_report
    assert report["n_final"] == len(pairs)
    assert (report["n_raw"] - report["n_content_guard_rejected"]
            - report["n_junk_filtered"] - report["n_dupe_filtered"]) == report["n_final"]


def test_v3_sentence_start_property(v3_pairs_and_report):
    """Property-Check: der weit ueberwiegende Teil der ausgegebenen Fenster
    beginnt an einer Satzgrenze (Grossbuchstabe oder Anfuehrungszeichen als
    erstes Zeichen). Dies ist die direkte Messung, die der Lead fuer den
    Report angefordert hat ('Anteil Satzanfang').

    KEIN Anspruch auf exakt 100%: snap_to_sentence_boundaries faellt auf
    den unveraenderten Rohbereich zurueck, wenn im +/-SNAP_PAD_TOKENS-
    Kontextpuffer keine Satzgrenze liegt (z.B. sehr nah am Tape-Anfang
    oder bei ungewoehnlich langen satzlosen Passagen) -- dokumentierter,
    kein stiller Fehler. Schwelle 95% ist konservativ unter der auf dem
    vollen Store gemessenen 97.5% (siehe Report an team-lead), damit
    dieser Test bei einer echten Regression (Snap bricht grossflaechig)
    trotzdem feuert."""
    pairs, report = v3_pairs_and_report
    violations = [p for p in pairs if p["text"] and not
                  (p["text"][0].isupper() or p["text"][0] in '"“‘\'(')]
    rate = 1.0 - len(violations) / len(pairs)
    assert rate >= 0.95, (
        f"only {rate*100:.1f}% sentence-start compliance "
        f"({len(violations)}/{len(pairs)} violations), e.g. "
        f"{violations[0]['text'][:60]!r}" if violations else ""
    )


def test_v3_content_guard_property_holds_for_all_output(v3_pairs_and_report):
    """Property-Check: JEDES ausgegebene Paar muss den Content-Guard
    bestehen -- kein Paar im finalen Output darf ein Mismatch sein (die
    Guard-Funktion selbst wurde bereits beim Filtern angewendet, dieser
    Test verifiziert das UNABHAENGIG erneut gegen die Ausgabe)."""
    pairs, report = v3_pairs_and_report
    for p in pairs:
        assert content_guard_ok(p["trigger"], p["outcome"], p["text"]), (
            f"content guard violated in output: trigger={p['trigger']!r} "
            f"outcome={p['outcome']!r} text={p['text'][:80]!r}"
        )


def test_v3_output_is_deduplicated(v3_pairs_and_report):
    pairs, report = v3_pairs_and_report
    keys = [(p["structure"], p["text"]) for p in pairs]
    assert len(keys) == len(set(keys))


def test_v3_output_is_sorted_by_citation(v3_pairs_and_report):
    pairs, report = v3_pairs_and_report
    keys = [(p["citation"]["sha"], p["citation"]["idx"]) for p in pairs]
    assert keys == sorted(keys)


def test_v3_schema_matches_v1_v2(v3_pairs_and_report):
    pairs, report = v3_pairs_and_report
    required = {"structure", "text", "citation", "doc_coord",
                "trigger", "mechanism", "outcome"}
    for p in pairs[:20]:
        assert required <= set(p.keys())
        assert p["structure"].startswith("<fact> ")
        assert p["structure"].endswith(" <say>")


def test_v3_determinism_two_runs_identical(v3_pairs_and_report):
    pairs1, report1 = v3_pairs_and_report
    cadence = dict(P72_CADENCE)
    pairs2, report2 = build_pairs_v3(STORE_DIR, cadence)
    assert pairs1 == pairs2
    assert report1 == report2
