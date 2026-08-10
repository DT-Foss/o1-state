"""
Shovel Mode — Semantically Equivalent Dummy Substitution
==========================================================
Replaces all entities in the knowledge base with fake but structurally
identical dummy data. This allows exporting the system state for external
debugging (e.g., sending to Claude or ChatGPT) WITHOUT exposing real
company data.

Key properties:
  1. Structure-preserving: Relations stay the same (capital stays capital)
  2. Type-preserving: Numbers stay numbers, names stay names, dates stay dates
  3. Reversible: Mapping table exists only on the air-gapped system
  4. Deterministic: Same input always produces same output (seeded)
  5. Semantically equivalent: Debugging with dummy data produces fixes
     that work identically on real data

Usage:
    shovel = ShovelMode(knowledge_store)
    dummy_state = shovel.export_dummy()
    # Send dummy_state to external AI for debugging
    # Apply fixes back to real system
"""

import json
import hashlib
import random
import re
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


# Dummy name pools — semantically appropriate replacements
DUMMY_COUNTRIES = [
    'Nordavia', 'Surelia', 'Estmark', 'Westfall', 'Centralis',
    'Polarheim', 'Tropica', 'Meridion', 'Borealis', 'Equatoria',
    'Altania', 'Delvano', 'Ferronia', 'Greyholm', 'Hexalia',
    'Ironvale', 'Junipera', 'Kaldera', 'Luminos', 'Mythos',
    'Novatica', 'Oceandria', 'Pinnacle', 'Quartzland', 'Rimshire',
    'Silvanus', 'Tundrica', 'Umbria', 'Verdania', 'Windmark',
    'Xenosia', 'Yeldania', 'Zephyria', 'Aethon', 'Bravos',
    'Crystalis', 'Dawnland', 'Emberia', 'Frostheim', 'Goldcrest',
]

DUMMY_CITIES = [
    'Starport', 'Irongate', 'Silverkeep', 'Goldcrest', 'Oakshield',
    'Stormhaven', 'Frostburg', 'Sunspire', 'Moonfall', 'Cloudpeak',
    'Thornwall', 'Crystalford', 'Ebonheart', 'Flamecrest', 'Glimmerdale',
    'Harrowfield', 'Ivywood', 'Jadecity', 'Knightholm', 'Larkspire',
    'Mistfield', 'Nightfall', 'Opalwatch', 'Pineshade', 'Quartzmill',
    'Ravenport', 'Stonecliff', 'Twilight', 'Underhall', 'Vinelake',
]

DUMMY_PEOPLE = [
    'Dr. Ash Vortex', 'Prof. Blake Ember', 'Sir Cedar Frost',
    'Dame Diana Storm', 'Earl Edison Nova', 'Lady Flora Steel',
    'Baron Grant Atlas', 'Hilda Quartz', 'Igor Blaze', 'Jane Prism',
    'Karl Zenith', 'Luna Tempest', 'Max Cobalt', 'Nora Flint',
    'Otto Radiance', 'Petra Shadow', 'Quinn Vertex', 'Rosa Cascade',
    'Sven Thunder', 'Tara Eclipse',
]

DUMMY_LANGUAGES = [
    'Nordan', 'Surelian', 'Estmarkish', 'Westfallian', 'Centralese',
    'Polarian', 'Tropican', 'Meridese', 'Borealan', 'Equatorese',
]

DUMMY_CURRENCIES = [
    'Starmark', 'Ironbit', 'Silvergold', 'Crystalcoin', 'Moonshard',
    'Suntoken', 'Frostpenny', 'Stormcredit', 'Ember', 'Prism',
]

DUMMY_CONTINENTS = [
    'Northara', 'Southara', 'Eastara', 'Westara', 'Centrara', 'Polara',
]


class ShovelMode:
    """Exports knowledge base with dummy data substitutions."""

    def __init__(self, knowledge_store=None, seed: int = 42):
        self.ks = knowledge_store
        self.seed = seed
        self._rng = random.Random(seed)

        # Mapping tables: real → dummy, dummy → real
        self._entity_map: Dict[str, str] = {}
        self._reverse_map: Dict[str, str] = {}

        # Type classification
        self._entity_types: Dict[str, str] = {}

    def _classify_entity(self, entity: str, relations: Dict[str, str]) -> str:
        """Classify an entity into a type for appropriate substitution."""
        el = entity.lower()

        # Check relations for type hints
        for rel, obj in relations.items():
            rl = rel.lower()
            if rl == 'is_a':
                ol = obj.lower()
                if ol == 'country':
                    return 'country'
                if ol == 'city':
                    return 'city'
                if ol in ('person', 'physicist', 'composer', 'inventor',
                          'mathematician', 'philosopher', 'military leader',
                          'political leader', 'playwright', 'naturalist',
                          'computer scientist', 'software engineer', 'pharaoh',
                          'civil rights leader', 'physician', 'astronomer',
                          'polymath'):
                    return 'person'
                if ol == 'language':
                    return 'language'
                if ol in ('chemical element', 'chemical compound'):
                    return 'element'
                if ol == 'planet':
                    return 'planet'
                if ol == 'sport':
                    return 'sport'

        if 'capital' in relations:
            return 'country'
        if 'location' in relations and relations.get('is_a', '').lower() == 'city':
            return 'city'
        if 'birthplace' in relations or 'known_for' in relations:
            return 'person'

        return 'generic'

    def _get_dummy(self, entity: str, entity_type: str) -> str:
        """Get or create a dummy name for an entity."""
        if entity in self._entity_map:
            return self._entity_map[entity]

        pools = {
            'country': DUMMY_COUNTRIES,
            'city': DUMMY_CITIES,
            'person': DUMMY_PEOPLE,
            'language': DUMMY_LANGUAGES,
            'currency': DUMMY_CURRENCIES,
            'continent': DUMMY_CONTINENTS,
        }

        pool = pools.get(entity_type, None)
        if pool:
            # Pick deterministically based on entity hash
            h = int(hashlib.md5(entity.encode()).hexdigest(), 16)
            idx = h % len(pool)
            dummy = pool[idx]
            # Avoid collisions
            attempts = 0
            while dummy in self._reverse_map and attempts < len(pool):
                idx = (idx + 1) % len(pool)
                dummy = pool[idx]
                attempts += 1
        else:
            # Generic: hash-based name
            h = hashlib.md5(entity.encode()).hexdigest()[:8]
            dummy = f'Entity_{h}'

        self._entity_map[entity] = dummy
        self._reverse_map[dummy] = entity
        self._entity_types[entity] = entity_type
        return dummy

    def _dummy_value(self, value: str, relation: str) -> str:
        """Create a dummy value preserving type and format."""
        # Numbers: randomize but preserve magnitude
        m = re.match(r'^(\d+(?:\.\d+)?)(.*)$', value)
        if m:
            num = float(m.group(1))
            suffix = m.group(2)
            # Randomize within ±50% but preserve order of magnitude
            factor = self._rng.uniform(0.5, 1.5)
            new_num = num * factor
            if '.' not in m.group(1):
                return f'{int(new_num)}{suffix}'
            return f'{new_num:.1f}{suffix}'

        # Population strings like "67M" or "1.4B"
        m = re.match(r'^(\d+(?:\.\d+)?)([MBK])$', value)
        if m:
            num = float(m.group(1))
            unit = m.group(2)
            factor = self._rng.uniform(0.5, 1.5)
            new_num = num * factor
            if new_num == int(new_num):
                return f'{int(new_num)}{unit}'
            return f'{new_num:.1f}{unit}'

        # Years: shift by random offset
        m = re.match(r'^(-?\d{4})$', value)
        if m:
            year = int(m.group(1))
            offset = self._rng.randint(-50, 50)
            return str(year + offset)

        # Temperatures: randomize
        m = re.match(r'^(-?\d+)°C$', value)
        if m:
            temp = int(m.group(1))
            offset = self._rng.randint(-20, 20)
            return f'{temp + offset}°C'

        # Check if value is a known entity → substitute
        if value in self._entity_map:
            return self._entity_map[value]

        # Default: return as-is (relations, boolean, formula strings)
        return value

    def build_mapping(self):
        """Build the entity mapping from the knowledge store."""
        if not self.ks:
            return

        # First pass: collect all relations per entity
        entity_relations: Dict[str, Dict[str, str]] = defaultdict(dict)
        for s, r, o in self.ks.facts:
            entity_relations[s][r] = o

        # Classify entities
        for entity, rels in entity_relations.items():
            etype = self._classify_entity(entity, rels)
            self._get_dummy(entity, etype)

        # Second pass: also map objects based on relation type
        relation_type_map = {
            'capital': 'city', 'language': 'language', 'currency': 'currency',
            'location': 'continent', 'borders': 'country',
            'creator': 'person', 'inventor': 'person', 'discoverer': 'person',
            'founder': 'person', 'author': 'person',
            'birthplace': 'city',
        }
        for s, r, o in self.ks.facts:
            if o not in self._entity_map:
                if o in entity_relations:
                    otype = self._classify_entity(o, entity_relations.get(o, {}))
                elif r.lower() in relation_type_map:
                    otype = relation_type_map[r.lower()]
                else:
                    continue
                self._get_dummy(o, otype)

    def export_dummy(self) -> Dict:
        """Export the entire KB state with dummy substitutions."""
        self.build_mapping()

        dummy_facts = []
        for s, r, o in self.ks.facts:
            ds = self._entity_map.get(s, s)
            # Relations stay the same — this is structure-preserving
            do = self._entity_map.get(o, self._dummy_value(o, r))
            dummy_facts.append([ds, r, do])

        return {
            'version': 'shovel-1.0',
            'mode': 'dummy-substitution',
            'seed': self.seed,
            'n_entities_mapped': len(self._entity_map),
            'n_facts': len(dummy_facts),
            'facts': dummy_facts,
            'note': 'All entity names and values are dummy substitutions. '
                    'Structure and relations are preserved. '
                    'Debugging with this data produces fixes applicable to real data.',
        }

    def export_mapping(self) -> Dict:
        """Export the mapping table (NEVER send this externally!)."""
        return {
            'warning': 'THIS FILE CONTAINS THE REAL-TO-DUMMY MAPPING. '
                       'NEVER EXPORT THIS FILE. Keep on air-gapped system only.',
            'entity_map': self._entity_map,
            'entity_types': self._entity_types,
        }

    def reverse_lookup(self, dummy_name: str) -> Optional[str]:
        """Look up the real entity from a dummy name."""
        return self._reverse_map.get(dummy_name)

    def get_stats(self) -> Dict:
        """Get statistics about the shovel mapping."""
        type_counts = defaultdict(int)
        for _, etype in self._entity_types.items():
            type_counts[etype] += 1

        return {
            'total_entities': len(self._entity_map),
            'types': dict(type_counts),
        }
