#!/usr/bin/env python3
"""Fetch Wikipedia articles for FOSS-KI knowledge extraction benchmark.

Downloads plain text from Wikipedia API (no dependencies beyond stdlib).
Targets fact-dense articles: countries, scientists, cities, inventions.
"""

import urllib.request
import json
import os
import time
import re

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

# Fact-dense Wikipedia articles — countries, scientists, cities, inventions
ARTICLES = [
    # Countries (capitals, population, languages, borders)
    "Germany", "France", "Japan", "United_Kingdom", "Italy", "Spain",
    "Brazil", "Canada", "Australia", "India", "China", "Russia",
    "Mexico", "Egypt", "South_Africa", "Argentina", "Sweden", "Norway",
    "Switzerland", "Netherlands", "South_Korea", "Turkey", "Poland",
    "Greece", "Portugal", "Austria", "Belgium", "Denmark", "Finland",
    "Ireland", "New_Zealand", "Thailand", "Vietnam", "Indonesia",
    "Nigeria", "Kenya", "Morocco", "Chile", "Colombia", "Peru",

    # Scientists
    "Albert_Einstein", "Isaac_Newton", "Marie_Curie", "Charles_Darwin",
    "Nikola_Tesla", "Galileo_Galilei", "Louis_Pasteur", "Ada_Lovelace",
    "Alexander_Fleming", "Niels_Bohr", "Max_Planck", "Werner_Heisenberg",
    "Richard_Feynman", "Stephen_Hawking", "Alan_Turing",

    # Cities
    "Berlin", "Paris", "Tokyo", "London", "Rome", "New_York_City",
    "Moscow", "Beijing", "Sydney", "Mumbai", "Cairo", "Istanbul",
    "Los_Angeles", "Toronto", "São_Paulo",

    # Inventions & Technology
    "Telephone", "Internet", "Printing_press", "Steam_engine",
    "Electricity", "Penicillin", "Television", "Radio",
    "Automobile", "Airplane", "Computer", "Transistor",

    # Literature & Arts
    "William_Shakespeare", "Leonardo_da_Vinci", "Ludwig_van_Beethoven",
    "Wolfgang_Amadeus_Mozart", "Johann_Sebastian_Bach",

    # Elements & Science
    "Water", "Gold", "Iron", "Oxygen", "Carbon", "Hydrogen",
    "Periodic_table", "DNA", "Photosynthesis", "Gravity",

    # Landmarks
    "Eiffel_Tower", "Great_Wall_of_China", "Colosseum", "Taj_Mahal",
    "Statue_of_Liberty", "Pyramids_of_Giza",
]


def fetch_article(title, max_chars=5000):
    """Fetch plain text extract from Wikipedia API."""
    url = (
        f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
    )
    try:
        req = urllib.request.Request(url, headers={
            'User-Agent': 'FOSS-KI/1.0 (research; david@foss.com.de)'
        })
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            extract = data.get('extract', '')
            return extract[:max_chars]
    except Exception as e:
        print(f"  SKIP {title}: {e}")
        return None


def fetch_full_article(title, max_chars=15000):
    """Fetch full article text from Wikipedia API (TextExtracts)."""
    url = (
        f"https://en.wikipedia.org/w/api.php?"
        f"action=query&titles={title}&prop=extracts&explaintext=1"
        f"&exlimit=1&format=json"
    )
    try:
        req = urllib.request.Request(url, headers={
            'User-Agent': 'FOSS-KI/1.0 (research; david@foss.com.de)'
        })
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            pages = data.get('query', {}).get('pages', {})
            for pid, page in pages.items():
                extract = page.get('extract', '')
                return extract[:max_chars]
    except Exception as e:
        print(f"  SKIP {title}: {e}")
        return None


def clean_text(text):
    """Clean Wikipedia text for extraction."""
    # Remove section headers
    text = re.sub(r'==+\s*[^=]+\s*==+', '', text)
    # Remove parenthetical pronunciation guides
    text = re.sub(r'\([^)]*pronunciation[^)]*\)', '', text)
    text = re.sub(r'\([^)]*listen[^)]*\)', '', text)
    # Remove very short lines
    lines = [l.strip() for l in text.split('\n') if len(l.strip()) > 30]
    return '\n'.join(lines)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    out_path = os.path.join(DATA_DIR, 'wikipedia_facts.txt')

    print(f"Fetching {len(ARTICLES)} Wikipedia articles...")
    print(f"Output: {out_path}")

    all_text = []
    success = 0
    total_chars = 0

    for i, title in enumerate(ARTICLES):
        # Use full article for more facts
        text = fetch_full_article(title)
        if text:
            cleaned = clean_text(text)
            if len(cleaned) > 100:
                all_text.append(f"# {title.replace('_', ' ')}\n{cleaned}\n")
                success += 1
                total_chars += len(cleaned)
                print(f"  [{i+1}/{len(ARTICLES)}] {title}: {len(cleaned):,} chars")
        # Be nice to Wikipedia API
        time.sleep(0.5)

    # Write combined file
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(all_text))

    print(f"\nDone: {success}/{len(ARTICLES)} articles, {total_chars:,} total chars")
    print(f"Saved to: {out_path}")


if __name__ == '__main__':
    main()
