"""Plain-assert tests for LiveStore. Run: python3 src/livecausal/test_store.py"""

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from livecausal.store import LiveStore, MANIFEST_NAME


def make_record(i, trigger="t{}".format(0)):
    return {
        "trigger": "trigger_{}".format(i),
        "mechanism": "mechanism_{}".format(i),
        "outcome": "outcome_{}".format(i),
        "trigger_key": "TK{}".format(i),
        "outcome_key": "OK{}".format(i),
        "doc_coord": i,
        "evidence_count": 1,
        "use_count": 0,
        "meta": {"idx": i},
    }


def with_tmpdir(fn):
    d = tempfile.mkdtemp(prefix="livecausal-test-")
    try:
        fn(d)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_append_verify_iter_roundtrip(d):
    store = LiveStore(d)
    records = [make_record(0), make_record(1), make_record(2)]
    sha = store.append_segment(records)

    assert store.segments() == [sha]
    assert store.verify() is True
    assert store.verify(sha) is True

    got = list(store.iter_records())
    assert len(got) == 3
    for (seg_sha, idx, rec), expected in zip(got, records):
        assert seg_sha == sha
        assert rec == expected

    got_single = list(store.iter_records(sha))
    assert len(got_single) == 3


def test_drop_removes_exactly_that_segment(d):
    store = LiveStore(d)
    sha1 = store.append_segment([make_record(0)])
    sha2 = store.append_segment([make_record(1)])
    sha3 = store.append_segment([make_record(2)])

    assert store.segments() == [sha1, sha2, sha3]

    store.drop_segments([sha2])

    assert store.segments() == [sha1, sha3]
    assert not os.path.exists(os.path.join(d, "{}.seg".format(sha2)))
    assert os.path.exists(os.path.join(d, "{}.seg".format(sha1)))
    assert os.path.exists(os.path.join(d, "{}.seg".format(sha3)))


def test_manifest_order_stable(d):
    store = LiveStore(d)
    shas = [store.append_segment([make_record(i)]) for i in range(5)]

    assert store.segments() == shas

    # Re-mounting must preserve order.
    store2 = LiveStore(d)
    assert store2.segments() == shas


def test_duplicate_append_same_records_same_sha(d):
    store = LiveStore(d)
    records = [make_record(0), make_record(1)]

    sha_a = store.append_segment(records)
    sha_b = store.append_segment(list(records))  # fresh list, identical content

    assert sha_a == sha_b
    # Manifest must not carry a duplicate entry.
    assert store.segments() == [sha_a]

    # Same records, different insertion order into the source list is a
    # different sequence -> must NOT collide unless truly identical content.
    reordered = [make_record(1), make_record(0)]
    sha_c = store.append_segment(reordered)
    assert sha_c != sha_a
    assert store.segments() == [sha_a, sha_c]


def test_verify_fails_on_tampered_file(d):
    store = LiveStore(d)
    sha = store.append_segment([make_record(0), make_record(1)])

    seg_path = os.path.join(d, "{}.seg".format(sha))
    with open(seg_path, "r", encoding="utf-8") as f:
        content = f.read()
    tampered = content.replace("trigger_1", "trigger_TAMPERED")
    assert tampered != content
    with open(seg_path, "w", encoding="utf-8") as f:
        f.write(tampered)

    assert store.verify(sha) is False
    assert store.verify() is False


def test_manifest_write_is_atomic(d):
    store = LiveStore(d)
    store.append_segment([make_record(0)])
    store.append_segment([make_record(1)])
    store.drop_segments([store.segments()[0]])

    # No leftover tmp files after any completed operation.
    entries = os.listdir(d)
    tmp_leftovers = [e for e in entries if e.startswith(".tmp-")]
    assert tmp_leftovers == [], "leftover tmp files: {}".format(tmp_leftovers)
    assert MANIFEST_NAME in entries


def run_all():
    tests = [
        test_append_verify_iter_roundtrip,
        test_drop_removes_exactly_that_segment,
        test_manifest_order_stable,
        test_duplicate_append_same_records_same_sha,
        test_verify_fails_on_tampered_file,
        test_manifest_write_is_atomic,
    ]
    for t in tests:
        with_tmpdir(t)
        print("OK  {}".format(t.__name__))
    print("ALL {} TESTS PASSED".format(len(tests)))


if __name__ == "__main__":
    run_all()
