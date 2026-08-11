#!/usr/bin/env python3
"""
Knowledge Extraction Pipeline — Scale KB from 722 to 100K+ triplets
====================================================================
Extracts structured triplets from:
  1. wikidata_facts.txt (541 natural language facts)
  2. wikipedia_facts.txt (2682 lines of article text)
  3. fb15k237/ (knowledge graph dataset)

All rule-based, no LLM needed. Pattern matching + NLP heuristics.
"""

import sys
import os
import re
import json
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, 'data')


def extract_wikidata_facts(path: str) -> list:
    """Extract triplets from 'X is the Y of Z' style sentences."""
    triplets = []
    patterns = [
        # "X is the capital of Y" → (Y, capital, X)
        (r'^(.+?)\s+is\s+the\s+(capital|currency|language|continent)\s+of\s+(.+?)\.?$',
         lambda m: (m.group(3).strip(), m.group(2), m.group(1).strip(), 4.0)),
        # "X is a/an Y" → (X, is_a, Y)
        (r'^(.+?)\s+is\s+(?:a|an)\s+(.+?)\.?$',
         lambda m: (m.group(1).strip(), 'is_a', m.group(2).strip(), 3.0)),
        # "X is located in Y" → (X, location, Y)
        (r'^(.+?)\s+is\s+located\s+in\s+(.+?)\.?$',
         lambda m: (m.group(1).strip(), 'location', m.group(2).strip(), 3.5)),
        # "X has a population of Y" → (X, population, Y)
        (r'^(.+?)\s+has\s+a\s+population\s+of\s+(.+?)\.?$',
         lambda m: (m.group(1).strip(), 'population', m.group(2).strip(), 3.0)),
        # "The X of Y is Z" → (Y, X, Z)
        (r'^The\s+(\w+)\s+of\s+(.+?)\s+is\s+(.+?)\.?$',
         lambda m: (m.group(2).strip(), m.group(1).lower(), m.group(3).strip(), 3.5)),
        # "X borders Y" → (X, borders, Y)
        (r'^(.+?)\s+borders?\s+(.+?)\.?$',
         lambda m: (m.group(1).strip(), 'borders', m.group(2).strip(), 3.0)),
        # "X was founded in Y" → (X, founded, Y)
        (r'^(.+?)\s+was\s+founded\s+in\s+(\d{4})\.?$',
         lambda m: (m.group(1).strip(), 'founded', m.group(2), 3.5)),
        # "X invented Y" / "X discovered Y" / "X created Y"
        (r'^(.+?)\s+(invented|discovered|created|wrote|composed)\s+(.+?)\.?$',
         lambda m: (m.group(3).strip(), 'creator', m.group(1).strip(), 3.5)),
    ]

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            for pattern, builder in patterns:
                m = re.match(pattern, line, re.I)
                if m:
                    try:
                        triplet = builder(m)
                        if triplet and len(triplet[0]) > 1 and len(triplet[2]) > 1:
                            triplets.append(list(triplet))
                    except Exception:
                        pass
                    break
    return triplets


def extract_wikipedia_facts(path: str) -> list:
    """Extract triplets from Wikipedia article paragraphs."""
    triplets = []

    with open(path) as f:
        content = f.read()

    # Split by article headers
    articles = re.split(r'^#\s+(.+)$', content, flags=re.M)
    # articles = ['', 'Germany', 'text...', 'France', 'text...', ...]

    for i in range(1, len(articles), 2):
        title = articles[i].strip()
        text = articles[i + 1] if i + 1 < len(articles) else ''

        # Extract from sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 20 or len(sent) > 500:
                continue

            extracted = extract_from_sentence(title, sent)
            triplets.extend(extracted)

    return triplets


def extract_from_sentence(context_entity: str, sentence: str) -> list:
    """Extract triplets from a single sentence with context."""
    triplets = []

    # Pattern: "X is/was the Y of Z"
    m = re.search(r"(?:^|\.\s)(.+?)\s+(?:is|was)\s+the\s+(\w+(?:\s+\w+)?)\s+of\s+(.+?)(?:\.|,|;|$)", sentence)
    if m:
        obj, rel, subj = m.group(1).strip(), m.group(2).lower().strip(), m.group(3).strip()
        if len(subj) < 80 and len(obj) < 80:
            triplets.append([subj, rel, obj, 3.0])

    # Pattern: "X, the Y of Z"
    m = re.search(r"(.+?),\s+the\s+(\w+(?:\s+\w+)?)\s+of\s+(.+?)(?:\.|,|;|$)", sentence)
    if m:
        subj, rel, obj = m.group(1).strip(), m.group(2).lower().strip(), m.group(3).strip()
        if len(subj) < 80 and len(obj) < 80 and rel in ('capital', 'president', 'king', 'queen', 'leader',
                                                          'founder', 'inventor', 'author', 'creator'):
            triplets.append([obj, rel, subj, 3.0])

    # Pattern: "X borders Y, Z, and W"
    m = re.search(r"(?:^|\.\s)(.+?)\s+borders?\s+(.+?)(?:\.|$)", sentence)
    if m:
        subj = m.group(1).strip()
        borders_text = m.group(2)
        countries = re.split(r',\s*(?:and\s+)?|\s+and\s+', borders_text)
        for c in countries:
            c = c.strip().rstrip('.')
            if len(c) > 1 and len(c) < 50:
                triplets.append([subj if subj != context_entity else context_entity,
                               'borders', c, 3.0])

    # Pattern: population numbers
    m = re.search(r"population\s+of\s+(?:over\s+|about\s+|approximately\s+)?(\d[\d,.]+(?:\s+(?:million|billion))?)",
                  sentence, re.I)
    if m:
        triplets.append([context_entity, 'population', m.group(1).strip(), 3.0])

    # Pattern: "founded in YEAR" / "established in YEAR"
    m = re.search(r"(?:founded|established|created|formed)\s+(?:in\s+)?(\d{4})", sentence)
    if m:
        triplets.append([context_entity, 'founded', m.group(1), 3.0])

    # Pattern: "X is a Y" (categorization)
    m = re.search(r"(?:^|\.\s)(.{2,40}?)\s+is\s+(?:a|an)\s+(.{3,60}?)(?:\.|,|;|\s+(?:that|which|in|with|located))",
                  sentence)
    if m:
        subj, cat = m.group(1).strip(), m.group(2).strip()
        if not any(w in cat.lower() for w in ('the', 'this', 'that', 'which', 'where')):
            triplets.append([subj, 'is_a', cat, 2.5])

    # Pattern: "X is known for Y" / "X is famous for Y"
    m = re.search(r"(.{2,40}?)\s+is\s+(?:known|famous|noted|renowned)\s+for\s+(.{3,80}?)(?:\.|,|$)", sentence)
    if m:
        triplets.append([m.group(1).strip(), 'known_for', m.group(2).strip(), 3.0])

    # Pattern: "largest/biggest/smallest X in Y"
    m = re.search(r"(?:the\s+)?(?:largest|biggest|smallest|tallest|longest|highest)\s+(.+?)\s+in\s+(.+?)(?:\.|,|$)",
                  sentence, re.I)
    if m:
        triplets.append([context_entity, 'notable_for',
                        f"{'largest' if 'largest' in sentence.lower() else 'notable'} {m.group(1).strip()} in {m.group(2).strip()}",
                        2.5])

    return triplets


def load_fb15k237(path: str) -> list:
    """Load FB15k-237 knowledge graph dataset."""
    triplets = []
    for split in ['train.txt', 'valid.txt', 'test.txt']:
        fpath = os.path.join(path, split)
        if not os.path.exists(fpath):
            continue
        with open(fpath) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 3:
                    s, r, o = parts
                    # Clean FB15k entities: /m/0123 → readable names
                    # Relations: /people/person/nationality → nationality
                    rel_clean = r.split('/')[-1].replace('_', ' ')
                    triplets.append([s, rel_clean, o, 2.0])
    return triplets


def deduplicate(triplets: list) -> list:
    """Remove duplicate triplets."""
    seen = set()
    unique = []
    for t in triplets:
        key = (t[0].lower(), t[1].lower(), t[2].lower())
        if key not in seen:
            seen.add(key)
            unique.append(t)
    return unique


def main():
    print("=" * 60)
    print("  KNOWLEDGE EXTRACTION PIPELINE")
    print("=" * 60)

    all_triplets = []

    # Load existing KB
    kb_path = os.path.join(DATA, 'knowledge_full.json')
    with open(kb_path) as f:
        existing = json.load(f)
    existing_triplets = existing.get('triplets', [])
    print(f"\nExisting KB: {len(existing_triplets)} triplets")
    all_triplets.extend(existing_triplets)

    # Source 1: Wikidata facts
    wikidata_path = os.path.join(DATA, 'wikidata_facts.txt')
    if os.path.exists(wikidata_path):
        wd = extract_wikidata_facts(wikidata_path)
        print(f"Wikidata extraction: {len(wd)} triplets")
        all_triplets.extend(wd)

    # Source 2: Wikipedia articles
    wiki_path = os.path.join(DATA, 'wikipedia_facts.txt')
    if os.path.exists(wiki_path):
        wp = extract_wikipedia_facts(wiki_path)
        print(f"Wikipedia extraction: {len(wp)} triplets")
        all_triplets.extend(wp)

    # Source 3: FB15k-237
    fb_path = os.path.join(DATA, 'fb15k237')
    if os.path.exists(fb_path):
        fb = load_fb15k237(fb_path)
        print(f"FB15k-237: {len(fb)} triplets")
        all_triplets.extend(fb)

    # Deduplicate
    unique = deduplicate(all_triplets)
    print(f"\nTotal after dedup: {len(unique)} triplets")

    # Categorize
    categories = defaultdict(int)
    for t in unique:
        rel = t[1].lower()
        if rel in ('is_a', 'type', 'instance of'):
            categories['taxonomy'] += 1
        elif rel in ('capital', 'location', 'borders', 'continent'):
            categories['geography'] += 1
        elif rel in ('creator', 'inventor', 'author', 'founded'):
            categories['history'] += 1
        elif rel in ('has_property', 'known_for', 'notable_for'):
            categories['properties'] += 1
        elif rel in ('population', 'area', 'elevation'):
            categories['statistics'] += 1
        else:
            categories['other'] += 1

    print("\nCategories:")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")

    # Save expanded KB
    expanded = {
        'version': 2.0,
        'source': 'foss-ki-extraction-pipeline',
        'categories': dict(categories),
        'count': len(unique),
        'triplets': unique,
    }

    output_path = os.path.join(DATA, 'knowledge_full.json')
    with open(output_path, 'w') as f:
        json.dump(expanded, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to {output_path}")
    print(f"KB growth: {len(existing_triplets)} → {len(unique)} "
          f"({len(unique)/max(len(existing_triplets),1):.1f}×)")

    return unique


if __name__ == '__main__':
    main()
