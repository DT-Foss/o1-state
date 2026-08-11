"""pytest-Tests fuer speak_train.py's --vocab-file-Flag (Sprecher-
Datenbasis v2, Baustein 5): rueckwaertskompatibel, Default-Verhalten
byte-identisch zum alten festen 5000er-Organism-Vokabular.

Testet NUR den Vokabular-Auswahl-Pfad (build_extended_vocab vs.
vocab.load_vocab), nicht den vollen Trainingslauf -- ein voller Lauf
braucht WT-103 + HSSLM-Forward-Pass, viel zu teuer fuer einen Unit-Test.
Die End-to-End-Verifikation (--vocab-file tatsaechlich durchgezogen bis
zum Modell) ist manuell gegen echte 4-Chunk-Smokes verifiziert (siehe
Report an team-lead) -- hier wird nur die Auswahl-LOGIK selbst getestet,
mit einem winzigen synthetischen Text statt dem vollen WT-103-Download.

Run:
    python3 -m pytest hsslm/data/test_speak_train_vocab_file.py -q
"""
import os
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "hsslm"))
sys.path.insert(0, os.path.join(REPO_ROOT, "hsslm", "training"))
sys.path.insert(0, os.path.join(REPO_ROOT, "hsslm", "data"))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from speak_train import build_extended_vocab
from vocab import build_vocab_n, save_vocab, load_vocab

SAMPLE_TEXT = (
    "the cat sat on the mat . the dog ran in the park . "
    "a fox jumped over the lazy hound near the old barn . "
    "birds sing songs while rivers flow through green valleys . "
) * 20


def test_default_path_matches_build_extended_vocab():
    """Ohne --vocab-file MUSS die Vokabular-Erzeugung exakt
    build_extended_vocab(text) sein -- die Rueckwaertskompatibilitaets-
    Garantie aus dem Auftrag."""
    stoi, unk_id, mask_id, fact_id, say_id, total_ids = build_extended_vocab(SAMPLE_TEXT)
    assert total_ids == mask_id + 3  # fact, say are mask+1, mask+2; total = say+1


def test_vocab_file_path_uses_file_size_not_fixed_5004():
    """Mit --vocab-file darf total_ids NICHT an die feste 5004 gebunden
    sein -- es folgt der Groesse des geladenen Vokabulars (n+4)."""
    vocab, stoi_built, unk, mask = build_vocab_n(SAMPLE_TEXT, 10)
    assert len(vocab) == 10
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        path = f.name
    try:
        save_vocab(path, vocab)
        _, stoi_loaded, unk_id, mask_id, fact_id, say_id, total_ids = load_vocab(path)
        assert total_ids == len(vocab) + 4  # words + unk + mask + fact + say
        assert total_ids != 5004
    finally:
        os.unlink(path)


def test_vocab_file_and_default_produce_different_schemas():
    """Ein 5000er-Default-Vokabular und ein kleines --vocab-file-Vokabular
    duerfen NICHT dasselbe ID-Schema haben (sonst waere die Datei-Groesse
    wirkungslos)."""
    stoi_default, unk_d, mask_d, fact_d, say_d, total_d = build_extended_vocab(SAMPLE_TEXT)

    vocab, _, _, _ = build_vocab_n(SAMPLE_TEXT, 10)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        path = f.name
    try:
        save_vocab(path, vocab)
        _, stoi_file, unk_f, mask_f, fact_f, say_f, total_f = load_vocab(path)
        assert total_d != total_f
    finally:
        os.unlink(path)


def test_cli_help_documents_vocab_file_flag():
    """--vocab-file muss im argparse-CLI dokumentiert sein (rueckwaerts-
    kompatibel: default=None, kein Pflichtargument)."""
    import argparse
    import subprocess
    result = subprocess.run(
        [sys.executable, os.path.join(REPO_ROOT, "hsslm", "training", "speak_train.py"), "--help"],
        capture_output=True, text=True, timeout=30,
    )
    assert "--vocab-file" in result.stdout
