"""
ConceptNet Loader — Import 500K ConceptNet Assertions into KnowledgeStore
==========================================================================
Maps ConceptNet relation types to FOSS-KI relations and bulk-loads them.

Filters:
  - Skip DerivedFrom, FormOf, Synonym, SimilarTo (morphological, not knowledge)
  - Skip single-character subjects/objects
  - Weight threshold: only import w >= 1.0
  - Multi-word entities: replace spaces with underscores

Usage:
    from core.conceptnet_loader import load_conceptnet
    count = load_conceptnet(knowledge_store)
"""

import json
import os
from typing import Optional

# ConceptNet → FOSS-KI relation mapping
# Only map relations that carry real-world knowledge
CN_RELATION_MAP = {
    'IsA': 'is_a',
    'AtLocation': 'found_in',
    'CapableOf': 'can',
    'RelatedTo': 'related_to',
    'Causes': 'causes',
    'Antonym': 'opposite_of',
    'PartOf': 'part_of',
    'UsedFor': 'used_for',
    'CausesDesire': 'causes',
    'HasPrerequisite': 'requires',
    'HasSubevent': 'has_part',
    'HasProperty': 'has_property',
    'HasA': 'has_part',
    'Desires': 'needs',
    'MotivatedByGoal': 'used_for',
    'ReceivesAction': 'used_for',
    'CreatedBy': 'creator',
    'MadeOf': 'made_of',
    'HasFirstSubevent': 'happens_before',
    'DefinedAs': 'description',
    'LocatedNear': 'found_in',
    'MannerOf': 'is_a',
    # Skip: DerivedFrom, FormOf, Synonym, SimilarTo, HasContext,
    #        Entails, NotDesires, DistinctFrom, HasLastSubevent
}

# Relations to SKIP (morphological/linguistic, not world knowledge)
CN_SKIP_RELATIONS = {
    'DerivedFrom', 'FormOf', 'Synonym', 'SimilarTo', 'HasContext',
    'Entails', 'NotDesires', 'DistinctFrom', 'HasLastSubevent',
}


def load_conceptnet(knowledge_store, data_path: Optional[str] = None,
                    min_weight: float = 1.0, max_facts: Optional[int] = None,
                    verbose: bool = True) -> int:
    """
    Load ConceptNet assertions into a KnowledgeStore.

    Args:
        knowledge_store: KnowledgeStore instance
        data_path: Path to conceptnet_en_500k.json (auto-detected if None)
        min_weight: Minimum ConceptNet weight to include (default 1.0)
        max_facts: Maximum facts to import (None = all)
        verbose: Print progress

    Returns:
        Number of facts imported
    """
    if data_path is None:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_path = os.path.join(base, 'data', 'conceptnet_en_500k.json')

    if not os.path.exists(data_path):
        if verbose:
            print(f"ConceptNet data not found at {data_path}")
        return 0

    if verbose:
        print(f"Loading ConceptNet from {data_path}...")

    with open(data_path) as f:
        data = json.load(f)

    relations = data.get('relations', [])
    if verbose:
        print(f"  Total assertions: {len(relations)}")

    imported = 0
    skipped_relation = 0
    skipped_weight = 0
    skipped_short = 0
    batch = []

    for item in relations:
        cn_rel = item['r']
        subject = item['s'].strip()
        obj = item['o'].strip()
        weight = item.get('w', 0)

        # Skip relations we don't map
        if cn_rel in CN_SKIP_RELATIONS:
            skipped_relation += 1
            continue

        # Weight filter
        if weight < min_weight:
            skipped_weight += 1
            continue

        # Skip single-char or empty entities
        if len(subject) < 2 or len(obj) < 2:
            skipped_short += 1
            continue

        # Map relation
        foss_rel = CN_RELATION_MAP.get(cn_rel)
        if not foss_rel:
            skipped_relation += 1
            continue

        # Normalize: replace spaces with underscores for multi-word entities
        subject_norm = subject.replace(' ', '_')
        obj_norm = obj.replace(' ', '_')

        batch.append((subject_norm, foss_rel, obj_norm))
        imported += 1

        if max_facts and imported >= max_facts:
            break

    # Bulk import
    if batch:
        knowledge_store.store_facts_fast(batch)

    if verbose:
        actual = len(knowledge_store.facts)
        print(f"  Imported: {imported} facts")
        print(f"  Skipped: {skipped_relation} (unmapped relations), "
              f"{skipped_weight} (low weight), {skipped_short} (too short)")
        print(f"  KB total: {actual} facts")

    return imported
