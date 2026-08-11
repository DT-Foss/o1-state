#!/usr/bin/env python3
"""
Benchmark: Web & API Access Module
====================================
Tests FOSS-KI's ability to fetch real-world data.

Tests:
  1. WebClient GET (httpbin or similar)
  2. Content extraction (HTML → text)
  3. RSS feed parsing
  4. Cache behavior
  5. DuckDuckGo instant answer
  6. Wikipedia summary
  7. REST Countries
  8. GitHub search
  9. Exchange rates
  10. Smart search routing
  11. URL fetch and extract
  12. Tool integration
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.web import (
    WebClient, ContentExtractor, APIClient, LRUCache,
    TextExtractor, API_REGISTRY,
    _tool_web_search, _tool_fetch_url,
)


def test(name, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    return 1 if passed else 0


def run():
    print("=" * 60)
    print("BENCHMARK: Web & API Access")
    print("=" * 60)
    total = 0
    passed = 0

    # === 1. LRU Cache ===
    print("\n--- Cache ---")
    total += 1
    cache = LRUCache(maxsize=3, ttl=10)
    cache.put('a', 1)
    cache.put('b', 2)
    cache.put('c', 3)
    cache.put('d', 4)  # Should evict 'a'
    ok = cache.get('a') is None and cache.get('d') == 4 and cache.get('b') == 2
    passed += test("LRU eviction", ok, f"a={cache.get('a')}, d={cache.get('d')}")

    # === 2. HTML Text Extraction ===
    print("\n--- Content Extraction ---")
    total += 1
    html = """<html><head><title>Test</title></head>
    <body><h1>Hello World</h1>
    <p>This is a <b>test</b> page.</p>
    <script>var x = 1;</script>
    <div>Some content here.</div>
    </body></html>"""
    text = ContentExtractor.html_to_text(html)
    ok = 'Hello World' in text and 'test page' in text and 'var x' not in text
    passed += test("HTML → text extraction", ok, f"'{text[:80]}...'")

    # === 3. Link Extraction ===
    total += 1
    html_links = '<a href="https://example.com">Example</a> <a href="/about">About</a>'
    links = ContentExtractor.extract_links(html_links, 'https://base.com')
    ok = len(links) >= 1 and links[0]['url'] == 'https://example.com'
    passed += test("Link extraction", ok, f"{len(links)} links")

    # === 4. RSS Parsing ===
    total += 1
    rss = """<?xml version="1.0"?>
    <rss><channel>
    <item><title>Story 1</title><link>https://example.com/1</link>
    <description>First story</description></item>
    <item><title>Story 2</title><link>https://example.com/2</link>
    <description><![CDATA[<b>Second</b> story]]></description></item>
    </channel></rss>"""
    items = ContentExtractor.extract_rss(rss)
    ok = len(items) == 2 and items[0]['title'] == 'Story 1'
    passed += test("RSS feed parsing", ok, f"{len(items)} items")

    # === 5. JSON Extraction ===
    total += 1
    json_text = '{"name": "test", "value": 42, "nested": {"key": "deep"}}'
    data = ContentExtractor.extract_json(json_text)
    ok = data is not None and data['value'] == 42
    passed += test("JSON extraction", ok)

    # === 6. API Registry ===
    total += 1
    ok = len(API_REGISTRY) >= 10 and 'wikipedia' in API_REGISTRY and 'github_api' in API_REGISTRY
    passed += test("API registry populated", ok, f"{len(API_REGISTRY)} APIs")

    # === LIVE TESTS (require network) ===
    print("\n--- Live API Tests ---")
    client = APIClient()

    # 7. DuckDuckGo
    total += 1
    try:
        r = client.query('duckduckgo', 'search', query='Python programming language')
        ok = r['success'] and isinstance(r.get('data'), dict)
        detail = f"status={r.get('success')}"
        if ok and r['data'].get('Abstract'):
            detail += f", abstract={r['data']['Abstract'][:60]}..."
    except Exception as e:
        ok = False
        detail = str(e)
    passed += test("DuckDuckGo instant answer", ok, detail)

    # 8. Wikipedia
    total += 1
    try:
        r = client.query('wikipedia', 'summary', title='Albert_Einstein')
        ok = r['success'] and isinstance(r.get('data'), dict)
        extract = ''
        if ok:
            extract = r['data'].get('extract', '')[:80]
        detail = f"extract='{extract}...'" if extract else f"raw={str(r)[:80]}"
    except Exception as e:
        ok = False
        detail = str(e)
    passed += test("Wikipedia summary", ok, detail)

    # 9. REST Countries
    total += 1
    try:
        r = client.query('restcountries', 'name', query='Norway')
        ok = r['success'] and isinstance(r.get('data'), list)
        if ok and r['data']:
            name = r['data'][0].get('name', {}).get('common', '?')
            detail = f"country={name}"
        else:
            detail = str(r.get('error', '?'))
    except Exception as e:
        ok = False
        detail = str(e)
    passed += test("REST Countries (Norway)", ok, detail)

    # 10. Exchange Rate
    total += 1
    try:
        r = client.query('exchangerate', 'latest', currency='NOK')
        ok = r['success'] and isinstance(r.get('data'), dict)
        if ok:
            rates = r['data'].get('rates', {})
            eur = rates.get('EUR', '?')
            detail = f"1 NOK = {eur} EUR"
        else:
            detail = str(r.get('error', '?'))
    except Exception as e:
        ok = False
        detail = str(e)
    passed += test("Exchange rate (NOK)", ok, detail)

    # 11. GitHub Search
    total += 1
    try:
        r = client.query('github_api', 'search_repos', query='foss-ki')
        ok = r['success']
        detail = f"status={r.get('success')}"
    except Exception as e:
        ok = False
        detail = str(e)
    passed += test("GitHub search", ok, detail)

    # 12. Smart Search
    total += 1
    try:
        r = client.search('what is the capital of France')
        ok = r.get('success', False)
        answer = r.get('answer', '')[:100]
        detail = f"answer='{answer}...'" if answer else f"err={r.get('error', '?')}"
    except Exception as e:
        ok = False
        detail = str(e)
    passed += test("Smart search (capital of France)", ok, detail)

    # 13. Smart Search — weather intent
    total += 1
    try:
        r = client.search('weather in Oslo')
        ok = r.get('success', False) and 'temperature' in str(r.get('answer', '')).lower()
        detail = r.get('answer', str(r.get('error', '?')))[:100]
    except Exception as e:
        ok = False
        detail = str(e)
    passed += test("Smart search (weather Oslo)", ok, detail)

    # 14. Cache hit
    total += 1
    try:
        # Second call should be cached
        r = client.query('wikipedia', 'summary', title='Albert_Einstein')
        ok = r.get('cached', False)
        detail = f"cached={r.get('cached')}"
    except Exception as e:
        ok = False
        detail = str(e)
    passed += test("Cache hit on repeat request", ok, detail)

    # 15. Tool functions
    print("\n--- Tool Integration ---")
    total += 1
    try:
        result = _tool_web_search('Python programming')
        ok = len(result) > 20 and 'No results' not in result
        detail = f"len={len(result)}, preview='{result[:60]}...'"
    except Exception as e:
        ok = False
        detail = str(e)
    passed += test("web_search tool function", ok, detail)

    # Summary
    print("\n" + "=" * 60)
    pct = (passed / total * 100) if total else 0
    print(f"RESULT: {passed}/{total} ({pct:.0f}%)")
    print("=" * 60)
    return passed, total


if __name__ == '__main__':
    run()
