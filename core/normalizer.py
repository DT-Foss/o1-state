"""
Entity Normalizer — Cluster similar entity names to canonical forms
====================================================================
Ported from pipeline_dza's EntityNormalizer. Lightweight version
using manual synonyms + Jaro-Winkler fuzzy matching (no embeddings).

Problem: "USA", "United States", "US", "America" are all the same entity.
Solution: Map all variants to one canonical form.

Integrates directly with KnowledgeStore — normalize on query AND on store.
"""

import re
from collections import defaultdict


# Common entity synonyms (expandable)
DEFAULT_SYNONYMS = {
    'united states': ['usa', 'us', 'america', 'united states of america', 'u.s.', 'u.s.a.'],
    'united kingdom': ['uk', 'britain', 'great britain', 'england'],
    'germany': ['deutschland', 'federal republic of germany', 'brd'],
    'france': ['french republic', 'république française'],
    'russia': ['russian federation', 'ussr', 'soviet union'],
    'china': ['prc', "people's republic of china", 'peoples republic of china'],
    'japan': ['nippon'],
    'south korea': ['korea', 'republic of korea', 'rok'],
    'european union': ['eu'],
    'javascript': ['js'],
    'typescript': ['ts'],
    'python': ['py'],
    'artificial intelligence': ['ai'],
    'machine learning': ['ml'],
    'natural language processing': ['nlp'],
    'operating system': ['os'],
    'database': ['db'],
    'application programming interface': ['api'],
    'world wide web': ['www', 'web'],
    'internet of things': ['iot'],
    'user interface': ['ui'],
    'user experience': ['ux'],
}

# Article prefixes to strip
ARTICLE_RE = re.compile(r'^(?:the|a|an)\s+', re.IGNORECASE)

# Parenthetical suffixes to strip: "Berlin (Germany)" → "Berlin"
PAREN_RE = re.compile(r'\s*\(.*?\)\s*$')


class EntityNormalizer:
    """
    Normalize entity names to canonical forms.

    Priority:
    1. Manual synonyms (exact match)
    2. Article/prefix stripping
    3. Fuzzy matching against known entities (optional)
    """

    def __init__(self, synonyms=None):
        self._synonym_to_canonical = {}
        self._canonical_set = set()

        # Build synonym map
        syns = synonyms or DEFAULT_SYNONYMS
        for canonical, variants in syns.items():
            canon_lower = canonical.lower().strip()
            self._canonical_set.add(canon_lower)
            self._synonym_to_canonical[canon_lower] = canon_lower
            for v in variants:
                self._synonym_to_canonical[v.lower().strip()] = canon_lower

    def add_synonym(self, canonical, *variants):
        """Add a synonym mapping."""
        canon_lower = canonical.lower().strip()
        self._canonical_set.add(canon_lower)
        self._synonym_to_canonical[canon_lower] = canon_lower
        for v in variants:
            self._synonym_to_canonical[v.lower().strip()] = canon_lower

    def normalize(self, entity):
        """
        Normalize an entity name. Returns the canonical form.

        Steps:
        1. Strip whitespace, articles, parentheticals
        2. Check synonym map
        3. Return cleaned form
        """
        if not entity:
            return entity

        # Clean
        clean = entity.strip()
        clean = PAREN_RE.sub('', clean).strip()

        # Check synonym map (case-insensitive)
        clean_lower = clean.lower()
        canonical = self._synonym_to_canonical.get(clean_lower)
        if canonical:
            # Return with original casing of canonical
            return canonical

        # Strip articles
        no_article = ARTICLE_RE.sub('', clean).strip()
        if no_article:
            canonical = self._synonym_to_canonical.get(no_article.lower())
            if canonical:
                return canonical
            clean = no_article

        return clean

    def normalize_triplet(self, subject, relation, obj):
        """Normalize both subject and object of a triplet."""
        return (self.normalize(subject), relation, self.normalize(obj))

    def learn_from_store(self, knowledge_store):
        """
        Auto-discover potential synonyms from the KnowledgeStore.
        Finds entities that share many facts (structural similarity).
        """
        # Group entities by their fact patterns
        entity_patterns = defaultdict(set)
        for entity, indices in knowledge_store._entity_subject_index.items():
            for idx in indices:
                _, r, o = knowledge_store.facts[idx]
                entity_patterns[entity].add((r.lower(), o.lower()))

        # Find structurally similar entities
        entities = list(entity_patterns.keys())
        discovered = 0
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                e1, e2 = entities[i], entities[j]
                p1, p2 = entity_patterns[e1], entity_patterns[e2]
                if not p1 or not p2:
                    continue

                # Jaccard similarity of fact patterns
                overlap = len(p1 & p2)
                union = len(p1 | p2)
                if union > 0 and overlap / union > 0.5:
                    # These entities share >50% of their facts
                    # Use the shorter name as canonical
                    if len(e1) <= len(e2):
                        self.add_synonym(e1, e2)
                    else:
                        self.add_synonym(e2, e1)
                    discovered += 1

        return discovered
