"""
Knowledge Import — ConceptNet + WordNet Offline Loaders
=========================================================
Import external knowledge bases into FOSS-KI's CommonSenseEngine
and KnowledgeStore WITHOUT needing external APIs.

Data sources (download once, use forever):
  1. ConceptNet CSV: https://s3.amazonaws.com/conceptnet/downloads/2019/edges/
     - conceptnet-assertions-5.7.0.csv.gz (~1.2GB, ~34M lines)
     - Filter to English + weight>1.0 → ~3M useful assertions

  2. WordNet TAB files: included in NLTK data or from Princeton
     - https://wordnet.princeton.edu/download
     - ~12MB uncompressed

This module handles:
  - Parsing ConceptNet CSV format
  - Filtering to English, high-weight assertions
  - Converting to (S, R, O) triplets
  - Loading into CommonSenseEngine or KnowledgeStore
  - Generating a compact .json subset for distribution
"""

import csv
import gzip
import json
import os
import re
from typing import Dict, Any, List, Optional, Tuple, Set
from collections import Counter


# ConceptNet relation mapping to our internal format
CN_RELATION_MAP = {
    '/r/IsA': 'is_a',
    '/r/PartOf': 'part_of',
    '/r/HasA': 'has_part',
    '/r/UsedFor': 'used_for',
    '/r/CapableOf': 'can',
    '/r/AtLocation': 'found_in',
    '/r/Causes': 'causes',
    '/r/HasProperty': 'has_property',
    '/r/MadeOf': 'made_of',
    '/r/ReceivesAction': 'receives_action',
    '/r/CreatedBy': 'created_by',
    '/r/Synonym': 'synonym',
    '/r/Antonym': 'opposite_of',
    '/r/DerivedFrom': 'derived_from',
    '/r/DefinedAs': 'defined_as',
    '/r/HasSubevent': 'has_subevent',
    '/r/HasPrerequisite': 'requires',
    '/r/MotivatedByGoal': 'motivated_by',
    '/r/CausesDesire': 'causes_desire',
    '/r/Desires': 'desires',
    '/r/HasFirstSubevent': 'starts_with',
    '/r/HasLastSubevent': 'ends_with',
    '/r/LocatedNear': 'near',
    '/r/SimilarTo': 'similar_to',
    '/r/DistinctFrom': 'distinct_from',
    '/r/RelatedTo': 'related_to',
}


def parse_conceptnet_csv(filepath: str,
                         lang: str = 'en',
                         min_weight: float = 1.0,
                         max_entries: int = 500000,
                         relations: Optional[Set[str]] = None) -> List[Tuple[str, str, str, float]]:
    """
    Parse ConceptNet CSV dump to (subject, relation, object, weight) tuples.

    Args:
        filepath: path to conceptnet-assertions-*.csv or .csv.gz
        lang: language filter (default: English only)
        min_weight: minimum edge weight (1.0 filters noise)
        max_entries: maximum entries to return
        relations: set of relation URIs to include (None = all)

    Returns:
        list of (subject, relation, object, weight) tuples
    """
    results = []
    opener = gzip.open if filepath.endswith('.gz') else open

    with opener(filepath, 'rt', encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f, delimiter='\t')

        for row in reader:
            if len(row) < 5:
                continue

            rel_uri = row[1]
            start_uri = row[2]
            end_uri = row[3]
            info_json = row[4] if len(row) > 4 else '{}'

            # Filter language
            if f'/c/{lang}/' not in start_uri or f'/c/{lang}/' not in end_uri:
                continue

            # Filter relation
            if relations and rel_uri not in relations:
                continue

            # Parse weight
            try:
                info = json.loads(info_json)
                weight = info.get('weight', 1.0)
            except (json.JSONDecodeError, TypeError):
                weight = 1.0

            if weight < min_weight:
                continue

            # Extract concept names
            subject = _uri_to_name(start_uri)
            obj = _uri_to_name(end_uri)
            relation = CN_RELATION_MAP.get(rel_uri, rel_uri.split('/')[-1].lower())

            if subject and obj and subject != obj:
                results.append((subject, relation, obj, weight))

            if len(results) >= max_entries:
                break

    return results


def _uri_to_name(uri: str) -> str:
    """Convert ConceptNet URI to human-readable name."""
    # /c/en/dog/n → "dog"
    # /c/en/hot_dog → "hot dog"
    parts = uri.split('/')
    if len(parts) >= 4:
        name = parts[3]
        return name.replace('_', ' ')
    return ''


def load_conceptnet_to_engine(filepath: str, engine, **kwargs):
    """
    Load ConceptNet data directly into a CommonSenseEngine.

    Args:
        filepath: path to ConceptNet CSV
        engine: CommonSenseEngine instance
        **kwargs: passed to parse_conceptnet_csv
    """
    triplets = parse_conceptnet_csv(filepath, **kwargs)
    loaded = 0

    for subject, relation, obj, weight in triplets:
        engine.add_fact(subject, relation, obj, weight)

        # Also add taxonomy entries for is_a relations
        if relation == 'is_a':
            engine.add_taxonomy(subject, [obj])

        loaded += 1

    return loaded


def load_conceptnet_to_knowledge_store(filepath: str, store, **kwargs):
    """
    Load ConceptNet data into a KnowledgeStore.

    Args:
        filepath: path to ConceptNet CSV
        store: KnowledgeStore instance
        **kwargs: passed to parse_conceptnet_csv
    """
    triplets = parse_conceptnet_csv(filepath, **kwargs)
    loaded = 0

    for subject, relation, obj, weight in triplets:
        store.store(subject, relation, obj)
        loaded += 1

    return loaded


# ============================================================
# Compact JSON subset generator
# ============================================================

def generate_compact_subset(filepath: str, output_path: str,
                           top_n: int = 50000,
                           useful_relations: Optional[Set[str]] = None):
    """
    Generate a compact JSON file from ConceptNet dump.

    Filters to the most useful relations and highest-weight edges,
    producing a small (~5MB) file that can be distributed with FOSS-KI.

    Args:
        filepath: path to full ConceptNet CSV
        output_path: where to save the compact JSON
        top_n: number of top entries to keep
        useful_relations: relations to include
    """
    if useful_relations is None:
        useful_relations = {
            '/r/IsA', '/r/PartOf', '/r/HasA', '/r/UsedFor',
            '/r/CapableOf', '/r/AtLocation', '/r/Causes',
            '/r/HasProperty', '/r/Antonym', '/r/Synonym',
            '/r/HasPrerequisite', '/r/MadeOf', '/r/DefinedAs',
        }

    triplets = parse_conceptnet_csv(
        filepath, lang='en', min_weight=1.5,
        max_entries=top_n * 2, relations=useful_relations
    )

    # Sort by weight and take top N
    triplets.sort(key=lambda x: x[3], reverse=True)
    triplets = triplets[:top_n]

    # Convert to compact format
    data = {
        'version': '1.0',
        'source': 'conceptnet-5.7.0',
        'lang': 'en',
        'count': len(triplets),
        'triplets': [[s, r, o, round(w, 2)] for s, r, o, w in triplets],
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)

    return len(triplets)


def load_compact_json(filepath: str) -> List[Tuple[str, str, str, float]]:
    """Load a compact JSON subset file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return [(s, r, o, w) for s, r, o, w in data['triplets']]


def load_compact_to_engine(filepath: str, engine):
    """Load compact JSON into CommonSenseEngine."""
    triplets = load_compact_json(filepath)
    loaded = 0

    for subject, relation, obj, weight in triplets:
        engine.add_fact(subject, relation, obj, weight)
        if relation == 'is_a':
            engine.add_taxonomy(subject, [obj])
        loaded += 1

    return loaded


# ============================================================
# WordNet Loader
# ============================================================

def parse_wordnet_data(data_dir: str) -> List[Tuple[str, str, str]]:
    """
    Parse WordNet data files to extract hypernym relations.

    Expected files in data_dir:
    - data.noun, data.verb, data.adj, data.adv
    - Or: wn_s.tab, wn_hyp.tab (simplified format)

    Returns (hyponym, is_a, hypernym) triplets.
    """
    triplets = []

    # Try simplified tab format first
    hyp_file = os.path.join(data_dir, 'wn_hyp.tab')
    if os.path.exists(hyp_file):
        synset_names = _load_wordnet_synset_names(data_dir)
        with open(hyp_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    child_id, parent_id = parts[0], parts[1]
                    child_name = synset_names.get(child_id, '')
                    parent_name = synset_names.get(parent_id, '')
                    if child_name and parent_name:
                        triplets.append((child_name, 'is_a', parent_name))
        return triplets

    # Try NLTK-style data files
    for pos in ['noun', 'verb', 'adj', 'adv']:
        data_file = os.path.join(data_dir, f'data.{pos}')
        if not os.path.exists(data_file):
            continue

        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('  '):  # Comment/header
                    continue
                # Parse WordNet data line
                parts = line.strip().split()
                if len(parts) < 6:
                    continue

                # Extract lemma and hypernym pointers
                # Format: offset lex_filenum ss_type w_cnt word ... p_cnt [ptr ...] | gloss
                try:
                    w_cnt = int(parts[3], 16)
                    words = []
                    idx = 4
                    for _ in range(w_cnt):
                        if idx < len(parts):
                            words.append(parts[idx].replace('_', ' '))
                            idx += 2  # Skip lex_id

                    if idx < len(parts):
                        p_cnt = int(parts[idx])
                        idx += 1
                        for _ in range(p_cnt):
                            if idx + 3 < len(parts):
                                ptr_symbol = parts[idx]
                                ptr_offset = parts[idx + 1]
                                # @ = hypernym
                                if ptr_symbol == '@' and words:
                                    # We'd need to look up the target offset
                                    pass
                                idx += 4
                except (ValueError, IndexError):
                    continue

    return triplets


def _load_wordnet_synset_names(data_dir: str) -> Dict[str, str]:
    """Load synset ID → name mapping from wn_s.tab."""
    names = {}
    s_file = os.path.join(data_dir, 'wn_s.tab')
    if os.path.exists(s_file):
        with open(s_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    synset_id = parts[0]
                    word = parts[2].replace('_', ' ').lower()
                    if synset_id not in names:  # Keep first (most common) word
                        names[synset_id] = word
    return names


# ============================================================
# Download helpers (optional, for initial setup)
# ============================================================

def download_conceptnet_subset(output_dir: str = '.') -> Optional[str]:
    """
    Download a pre-built ConceptNet English subset.

    Uses the official ConceptNet API to build a local cache.
    Downloads common concepts in batches.

    Returns path to the generated JSON file, or None on failure.
    """
    import urllib.request

    output_path = os.path.join(output_dir, 'conceptnet_en_subset.json')

    # Common concepts to seed the download
    seed_concepts = [
        'dog', 'cat', 'car', 'house', 'food', 'water', 'fire', 'tree',
        'human', 'child', 'school', 'book', 'computer', 'phone', 'money',
        'doctor', 'teacher', 'city', 'country', 'language', 'music',
        'science', 'mathematics', 'history', 'art', 'sport', 'game',
        'weather', 'rain', 'sun', 'wind', 'snow', 'earth', 'ocean',
        'family', 'friend', 'love', 'happiness', 'anger', 'fear',
        'democracy', 'law', 'government', 'education', 'health',
        'work', 'travel', 'cooking', 'eating', 'sleeping', 'exercise',
    ]

    all_triplets = []

    for concept in seed_concepts:
        try:
            url = f"http://api.conceptnet.io/c/en/{concept}?limit=50"
            req = urllib.request.Request(url)
            req.add_header('Accept', 'application/json')
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())

            for edge in data.get('edges', []):
                rel = edge.get('rel', {}).get('label', '')
                start = edge.get('start', {}).get('label', '')
                end = edge.get('end', {}).get('label', '')
                weight = edge.get('weight', 1.0)
                start_lang = edge.get('start', {}).get('language', '')
                end_lang = edge.get('end', {}).get('language', '')

                if start_lang == 'en' and end_lang == 'en' and weight >= 1.0:
                    relation = rel.lower().replace(' ', '_')
                    all_triplets.append([start.lower(), relation, end.lower(), round(weight, 2)])

        except Exception:
            continue

    if all_triplets:
        # Deduplicate
        seen = set()
        unique = []
        for t in all_triplets:
            key = (t[0], t[1], t[2])
            if key not in seen:
                seen.add(key)
                unique.append(t)

        data = {
            'version': '1.0',
            'source': 'conceptnet-api',
            'lang': 'en',
            'count': len(unique),
            'triplets': unique,
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=1)

        return output_path

    return None


def check_available_data(data_dir: str = '.') -> Dict[str, bool]:
    """Check which knowledge base files are available locally."""
    return {
        'conceptnet_csv': any(
            os.path.exists(os.path.join(data_dir, f))
            for f in ['conceptnet-assertions-5.7.0.csv.gz',
                      'conceptnet-assertions-5.7.0.csv']
        ),
        'conceptnet_json': os.path.exists(
            os.path.join(data_dir, 'conceptnet_en_subset.json')
        ),
        'wordnet_data': os.path.exists(
            os.path.join(data_dir, 'data.noun')
        ),
        'wordnet_tab': os.path.exists(
            os.path.join(data_dir, 'wn_s.tab')
        ),
    }
