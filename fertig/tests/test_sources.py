"""FERTIG — Tests des Quellen-Registry (offline, gemockte Antworten)."""

from __future__ import annotations

from fertig import sources


def test_extract_causal_positive():
    trips = sources.extract_causal(
        "Smoking causes lung cancer. Exercise improves heart health.")
    assert ("smoking", "causes", "lung cancer", 0.35) in trips
    assert ("exercise", "improves", "heart health", 0.35) in trips


def test_extract_causal_negation_guard():
    trips = sources.extract_causal(
        "The study found no evidence that coffee causes cancer.")
    assert not any("causes" == t[1] for t in trips)


def test_extract_causal_dedupe():
    trips = sources.extract_causal(
        "Sugar causes cavities. Sugar causes cavities in children.")
    keys = [t[:3] for t in trips]
    assert len(keys) == len(set(keys))


def test_extract_causal_too_short():
    assert sources.extract_causal("It causes pain.") == []


def test_fetch_all_merges_max_conf(monkeypatch):
    calls = {}

    def fake_wiki(t):
        return [("a", "causes", "b", 0.7)]

    def fake_ddg(t):
        return [("a", "causes", "b", 0.3), ("x", "reduces", "y", 0.3)]

    monkeypatch.setitem(sources.SOURCES, "wikipedia", fake_wiki)
    monkeypatch.setitem(sources.SOURCES, "duckduckgo", fake_ddg)
    trips = sources.fetch_all("test", sources=["wikipedia", "duckduckgo"])
    best = {t[:3]: t[3] for t in trips}
    # max (0.7) + Evidenz-Boost für die 2. Quelle (0.06)
    assert best[("a", "causes", "b")] == 0.76
    assert ("x", "reduces", "y") in best


def test_duckduckgo_snippet_parse():
    page = ('<div class="result">'
            '<a class="result__snippet">Sugar causes cavities in '
            'children.</a>'
            '<a class="result__snippet">Exercise reduces stress levels.</a>'
            "</div>")
    snips = sources.re.findall(r'class="result__snippet"[^>]*>(.*?)</a>',
                               page, flags=sources.re.S)
    text = " ".join(sources.html.unescape(sources.re.sub(r"<[^>]+>", "", s))
                    for s in snips)
    trips = sources.extract_causal(text, base_conf=0.30)
    assert ("sugar", "causes", "cavities in children", 0.30) in trips


def test_stdlib_text_extractor_skips_scripts():
    page = ("<html><head><style>.x{}</style><script>var x=1;</script></head>"
            "<body><p>Smoking causes lung cancer.</p>"
            "<p>Exercise improves heart health.</p></body></html>")
    text = sources._extract_text_stdlib(page)
    assert "Smoking causes lung cancer." in text
    assert "var x=1" not in text
    assert ".x{}" not in text


def test_ddg_urls_extraction():
    page = ('<a class="result__a" href="//duckduckgo.com/l/?uddg='
            'https%3A%2F%2Fexample.com%2Farticle&amp;rut=123">t</a>'
            '<a class="result__a" href="https://duckduckgo.com">x</a>')
    urls = sources._parse_ddg_urls(page)
    assert "https://example.com/article" in urls
    assert all("duckduckgo.com" not in u for u in urls)
