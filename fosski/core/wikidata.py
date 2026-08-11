"""
Wikidata Importer — Structured Knowledge at Scale
====================================================
Import facts directly from Wikidata's structured format instead
of extracting from prose text. Every Wikidata claim is already
a (Subject, Relation, Object) triplet — no extraction needed.

Data source: Wikidata JSON dumps, SPARQL queries, or LIVE API.
Output: Direct storage into KnowledgeStore.

This is exponential scaling: one import = thousands of facts,
compared to prose extraction which gives ~5 facts per paragraph.
"""

import json
import os
import re
import time
import urllib.request
import urllib.parse
import urllib.error

# Wikidata property ID → human-readable relation name
# These are the most commonly useful properties
PROPERTY_MAP = {
    'P36': 'capital',
    'P1082': 'population',
    'P37': 'language',
    'P47': 'borders',
    'P30': 'location',        # continent
    'P17': 'country',
    'P131': 'part_of',        # located in administrative entity
    'P38': 'currency',
    'P35': 'leader',          # head of state
    'P6': 'leader',           # head of government
    'P112': 'founder',
    'P170': 'creator',
    'P50': 'author',
    'P61': 'discoverer',
    'P571': 'founded',        # inception date
    'P19': 'birthplace',
    'P20': 'deathplace',
    'P569': 'born',           # date of birth
    'P570': 'died',           # date of death
    'P27': 'nationality',
    'P106': 'occupation',
    'P31': 'type',            # instance of
    'P279': 'subclass_of',
    'P1376': 'capital_of',
    'P274': 'formula',        # chemical formula
    'P246': 'symbol',         # element symbol
    'P1566': 'geonames_id',
    'P625': 'coordinates',
    'P856': 'website',
    'P18': 'image',
    'P154': 'logo',
}

# Relations to skip (too noisy or not useful for QA)
SKIP_RELATIONS = {
    'geonames_id', 'coordinates', 'website', 'image', 'logo',
}


class WikidataImporter:
    """
    Import structured knowledge from Wikidata into KnowledgeStore.

    Supports:
    1. Wikidata JSON entity files (from dumps)
    2. Simple dict format: {label, claims: [{property, value}, ...]}
    3. Pre-built fact lists
    """

    def __init__(self, knowledge_store):
        self.knowledge = knowledge_store
        self.stats = {
            'entities_processed': 0,
            'facts_added': 0,
            'facts_skipped': 0,
            'unknown_properties': set(),
        }

    def import_entity(self, entity_data):
        """
        Import a single Wikidata entity.

        Args:
            entity_data: dict with keys:
                id: str (Q-ID, e.g., "Q183")
                label: str (e.g., "Germany")
                claims: list of dicts with:
                    property: str (P-ID, e.g., "P36")
                    value: str (resolved label, e.g., "Berlin")
                    or
                    property_label: str (e.g., "capital")
                    value: str

        Returns: number of facts added
        """
        label = entity_data.get('label', '')
        if not label:
            return 0

        claims = entity_data.get('claims', [])
        added = 0

        for claim in claims:
            # Resolve property name
            prop_id = claim.get('property', '')
            prop_label = claim.get('property_label', '')

            if prop_id in PROPERTY_MAP:
                relation = PROPERTY_MAP[prop_id]
            elif prop_label:
                relation = prop_label.lower().replace(' ', '_')
            else:
                self.stats['unknown_properties'].add(prop_id)
                continue

            if relation in SKIP_RELATIONS:
                self.stats['facts_skipped'] += 1
                continue

            value = claim.get('value', '')
            if not value or len(str(value)) < 1:
                continue

            # Store fact
            value_str = str(value).strip()
            self.knowledge.store_fact(label, relation, value_str)
            added += 1

        self.stats['entities_processed'] += 1
        self.stats['facts_added'] += added
        return added

    def import_entities(self, entities):
        """Import multiple entities."""
        total = 0
        for entity in entities:
            total += self.import_entity(entity)
        return total

    def import_json_file(self, path):
        """
        Import entities from a JSON file.

        Supports formats:
        1. List of entities: [{id, label, claims}, ...]
        2. Wikidata dump entity: {id, labels, claims}
        3. SPARQL results: {results: {bindings: [...]}}
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, list):
            return self.import_entities(data)
        elif 'results' in data and 'bindings' in data['results']:
            return self._import_sparql_results(data)
        elif 'labels' in data and 'claims' in data:
            return self._import_wikidata_entity(data)
        else:
            return self.import_entity(data)

    def _import_wikidata_entity(self, entity):
        """Import a raw Wikidata entity JSON (from dump)."""
        # Get label (prefer English)
        labels = entity.get('labels', {})
        label = ''
        for lang in ('en', 'de', 'fr', 'es'):
            if lang in labels:
                label = labels[lang].get('value', '')
                if label:
                    break

        if not label:
            return 0

        # Process claims
        claims = []
        for prop_id, claim_list in entity.get('claims', {}).items():
            for claim in claim_list:
                mainsnak = claim.get('mainsnak', {})
                if mainsnak.get('snaktype') != 'value':
                    continue

                datavalue = mainsnak.get('datavalue', {})
                value_type = datavalue.get('type', '')
                value = datavalue.get('value', '')

                # Resolve value based on type
                if value_type == 'wikibase-entityid':
                    # Need to resolve Q-ID to label — store placeholder
                    qid = value.get('id', '')
                    resolved = self._resolve_qid(qid)
                    if resolved:
                        claims.append({
                            'property': prop_id,
                            'value': resolved,
                        })
                elif value_type == 'string':
                    claims.append({
                        'property': prop_id,
                        'value': value,
                    })
                elif value_type == 'quantity':
                    amount = value.get('amount', '')
                    if amount.startswith('+'):
                        amount = amount[1:]
                    claims.append({
                        'property': prop_id,
                        'value': amount,
                    })
                elif value_type == 'time':
                    time_val = value.get('time', '')
                    # Extract year from "+2023-01-01T00:00:00Z"
                    m = re.match(r'[+-]?(\d{4})', time_val)
                    if m:
                        claims.append({
                            'property': prop_id,
                            'value': m.group(1),
                        })
                elif value_type == 'monolingualtext':
                    claims.append({
                        'property': prop_id,
                        'value': value.get('text', ''),
                    })

        return self.import_entity({
            'id': entity.get('id', ''),
            'label': label,
            'claims': claims,
        })

    def _import_sparql_results(self, data):
        """Import from SPARQL query results."""
        bindings = data['results']['bindings']
        added = 0

        for binding in bindings:
            subject = self._sparql_value(binding, 'subjectLabel', 'subject')
            predicate = self._sparql_value(binding, 'predicateLabel', 'predicate')
            obj = self._sparql_value(binding, 'objectLabel', 'object')

            if subject and predicate and obj:
                relation = predicate.lower().replace(' ', '_')
                if relation not in SKIP_RELATIONS:
                    self.knowledge.store_fact(subject, relation, obj)
                    added += 1

        self.stats['entities_processed'] += len(bindings)
        self.stats['facts_added'] += added
        return added

    @staticmethod
    def _sparql_value(binding, *keys):
        """Extract value from SPARQL binding, trying multiple keys."""
        for key in keys:
            if key in binding:
                return binding[key].get('value', '')
        return ''

    def _resolve_qid(self, qid):
        """Resolve a Wikidata Q-ID to a label. Placeholder for now."""
        # In a full implementation, this would query Wikidata API
        # or use a local Q-ID → label cache
        return None

    def import_countries_simple(self):
        """
        Pre-built dataset: world countries with basic facts.
        No API needed — hardcoded core knowledge.

        This is the bootstrap: enough facts for the system to be
        immediately useful, then it learns more from text.
        """
        countries = [
            {'label': 'France', 'claims': [
                {'property_label': 'capital', 'value': 'Paris'},
                {'property_label': 'language', 'value': 'French'},
                {'property_label': 'population', 'value': '67 million'},
                {'property_label': 'location', 'value': 'Europe'},
                {'property_label': 'currency', 'value': 'Euro'},
                {'property_label': 'borders', 'value': 'Germany'},
                {'property_label': 'borders', 'value': 'Spain'},
                {'property_label': 'borders', 'value': 'Belgium'},
                {'property_label': 'borders', 'value': 'Italy'},
                {'property_label': 'borders', 'value': 'Switzerland'},
            ]},
            {'label': 'Germany', 'claims': [
                {'property_label': 'capital', 'value': 'Berlin'},
                {'property_label': 'language', 'value': 'German'},
                {'property_label': 'population', 'value': '83 million'},
                {'property_label': 'location', 'value': 'Europe'},
                {'property_label': 'currency', 'value': 'Euro'},
                {'property_label': 'borders', 'value': 'France'},
                {'property_label': 'borders', 'value': 'Poland'},
                {'property_label': 'borders', 'value': 'Denmark'},
                {'property_label': 'borders', 'value': 'Netherlands'},
                {'property_label': 'borders', 'value': 'Austria'},
            ]},
            {'label': 'Japan', 'claims': [
                {'property_label': 'capital', 'value': 'Tokyo'},
                {'property_label': 'language', 'value': 'Japanese'},
                {'property_label': 'population', 'value': '125 million'},
                {'property_label': 'location', 'value': 'Asia'},
                {'property_label': 'currency', 'value': 'Yen'},
                {'property_label': 'type', 'value': 'island country'},
            ]},
            {'label': 'Brazil', 'claims': [
                {'property_label': 'capital', 'value': 'Brasilia'},
                {'property_label': 'language', 'value': 'Portuguese'},
                {'property_label': 'population', 'value': '203 million'},
                {'property_label': 'location', 'value': 'South America'},
                {'property_label': 'currency', 'value': 'Real'},
            ]},
            {'label': 'India', 'claims': [
                {'property_label': 'capital', 'value': 'New Delhi'},
                {'property_label': 'language', 'value': 'Hindi'},
                {'property_label': 'population', 'value': '1.4 billion'},
                {'property_label': 'location', 'value': 'Asia'},
                {'property_label': 'currency', 'value': 'Rupee'},
            ]},
            {'label': 'United States', 'claims': [
                {'property_label': 'capital', 'value': 'Washington D.C.'},
                {'property_label': 'language', 'value': 'English'},
                {'property_label': 'population', 'value': '331 million'},
                {'property_label': 'location', 'value': 'North America'},
                {'property_label': 'currency', 'value': 'Dollar'},
                {'property_label': 'borders', 'value': 'Canada'},
                {'property_label': 'borders', 'value': 'Mexico'},
            ]},
            {'label': 'China', 'claims': [
                {'property_label': 'capital', 'value': 'Beijing'},
                {'property_label': 'language', 'value': 'Mandarin'},
                {'property_label': 'population', 'value': '1.4 billion'},
                {'property_label': 'location', 'value': 'Asia'},
                {'property_label': 'currency', 'value': 'Yuan'},
            ]},
            {'label': 'Russia', 'claims': [
                {'property_label': 'capital', 'value': 'Moscow'},
                {'property_label': 'language', 'value': 'Russian'},
                {'property_label': 'population', 'value': '144 million'},
                {'property_label': 'location', 'value': 'Europe'},
                {'property_label': 'currency', 'value': 'Ruble'},
            ]},
            {'label': 'United Kingdom', 'claims': [
                {'property_label': 'capital', 'value': 'London'},
                {'property_label': 'language', 'value': 'English'},
                {'property_label': 'population', 'value': '67 million'},
                {'property_label': 'location', 'value': 'Europe'},
                {'property_label': 'currency', 'value': 'Pound'},
            ]},
            {'label': 'Australia', 'claims': [
                {'property_label': 'capital', 'value': 'Canberra'},
                {'property_label': 'language', 'value': 'English'},
                {'property_label': 'population', 'value': '26 million'},
                {'property_label': 'location', 'value': 'Oceania'},
                {'property_label': 'currency', 'value': 'Dollar'},
            ]},
            {'label': 'Canada', 'claims': [
                {'property_label': 'capital', 'value': 'Ottawa'},
                {'property_label': 'language', 'value': 'English'},
                {'property_label': 'population', 'value': '38 million'},
                {'property_label': 'location', 'value': 'North America'},
                {'property_label': 'currency', 'value': 'Dollar'},
                {'property_label': 'borders', 'value': 'United States'},
            ]},
            {'label': 'Italy', 'claims': [
                {'property_label': 'capital', 'value': 'Rome'},
                {'property_label': 'language', 'value': 'Italian'},
                {'property_label': 'population', 'value': '60 million'},
                {'property_label': 'location', 'value': 'Europe'},
                {'property_label': 'currency', 'value': 'Euro'},
            ]},
            {'label': 'Spain', 'claims': [
                {'property_label': 'capital', 'value': 'Madrid'},
                {'property_label': 'language', 'value': 'Spanish'},
                {'property_label': 'population', 'value': '47 million'},
                {'property_label': 'location', 'value': 'Europe'},
                {'property_label': 'currency', 'value': 'Euro'},
            ]},
            {'label': 'South Korea', 'claims': [
                {'property_label': 'capital', 'value': 'Seoul'},
                {'property_label': 'language', 'value': 'Korean'},
                {'property_label': 'population', 'value': '52 million'},
                {'property_label': 'location', 'value': 'Asia'},
                {'property_label': 'currency', 'value': 'Won'},
            ]},
            {'label': 'Mexico', 'claims': [
                {'property_label': 'capital', 'value': 'Mexico City'},
                {'property_label': 'language', 'value': 'Spanish'},
                {'property_label': 'population', 'value': '126 million'},
                {'property_label': 'location', 'value': 'North America'},
                {'property_label': 'currency', 'value': 'Peso'},
            ]},
            {'label': 'Norway', 'claims': [
                {'property_label': 'capital', 'value': 'Oslo'},
                {'property_label': 'language', 'value': 'Norwegian'},
                {'property_label': 'population', 'value': '5.4 million'},
                {'property_label': 'location', 'value': 'Europe'},
                {'property_label': 'currency', 'value': 'Krone'},
                {'property_label': 'borders', 'value': 'Sweden'},
                {'property_label': 'borders', 'value': 'Finland'},
                {'property_label': 'borders', 'value': 'Russia'},
            ]},
            {'label': 'Sweden', 'claims': [
                {'property_label': 'capital', 'value': 'Stockholm'},
                {'property_label': 'language', 'value': 'Swedish'},
                {'property_label': 'population', 'value': '10 million'},
                {'property_label': 'location', 'value': 'Europe'},
                {'property_label': 'currency', 'value': 'Krona'},
                {'property_label': 'borders', 'value': 'Norway'},
                {'property_label': 'borders', 'value': 'Finland'},
            ]},
            {'label': 'Finland', 'claims': [
                {'property_label': 'capital', 'value': 'Helsinki'},
                {'property_label': 'language', 'value': 'Finnish'},
                {'property_label': 'population', 'value': '5.5 million'},
                {'property_label': 'location', 'value': 'Europe'},
                {'property_label': 'currency', 'value': 'Euro'},
                {'property_label': 'borders', 'value': 'Sweden'},
                {'property_label': 'borders', 'value': 'Norway'},
                {'property_label': 'borders', 'value': 'Russia'},
            ]},
            {'label': 'Denmark', 'claims': [
                {'property_label': 'capital', 'value': 'Copenhagen'},
                {'property_label': 'language', 'value': 'Danish'},
                {'property_label': 'population', 'value': '5.8 million'},
                {'property_label': 'location', 'value': 'Europe'},
                {'property_label': 'currency', 'value': 'Krone'},
                {'property_label': 'borders', 'value': 'Germany'},
            ]},
            {'label': 'Poland', 'claims': [
                {'property_label': 'capital', 'value': 'Warsaw'},
                {'property_label': 'language', 'value': 'Polish'},
                {'property_label': 'population', 'value': '38 million'},
                {'property_label': 'location', 'value': 'Europe'},
                {'property_label': 'currency', 'value': 'Zloty'},
                {'property_label': 'borders', 'value': 'Germany'},
            ]},
        ]

        # Science facts
        science = [
            {'label': 'Water', 'claims': [
                {'property_label': 'formula', 'value': 'H2O'},
                {'property_label': 'type', 'value': 'chemical compound'},
            ]},
            {'label': 'Carbon Dioxide', 'claims': [
                {'property_label': 'formula', 'value': 'CO2'},
                {'property_label': 'type', 'value': 'chemical compound'},
            ]},
            {'label': 'Oxygen', 'claims': [
                {'property_label': 'symbol', 'value': 'O'},
                {'property_label': 'type', 'value': 'chemical element'},
            ]},
            {'label': 'Hydrogen', 'claims': [
                {'property_label': 'symbol', 'value': 'H'},
                {'property_label': 'type', 'value': 'chemical element'},
            ]},
            {'label': 'Gold', 'claims': [
                {'property_label': 'symbol', 'value': 'Au'},
                {'property_label': 'type', 'value': 'chemical element'},
            ]},
            {'label': 'Iron', 'claims': [
                {'property_label': 'symbol', 'value': 'Fe'},
                {'property_label': 'type', 'value': 'chemical element'},
            ]},
        ]

        # Famous people
        people = [
            {'label': 'Albert Einstein', 'claims': [
                {'property_label': 'birthplace', 'value': 'Ulm'},
                {'property_label': 'nationality', 'value': 'German'},
                {'property_label': 'occupation', 'value': 'physicist'},
                {'property_label': 'born', 'value': '1879'},
            ]},
            {'label': 'Marie Curie', 'claims': [
                {'property_label': 'birthplace', 'value': 'Warsaw'},
                {'property_label': 'nationality', 'value': 'Polish'},
                {'property_label': 'occupation', 'value': 'physicist'},
            ]},
            {'label': 'Radium', 'claims': [
                {'property_label': 'discoverer', 'value': 'Marie Curie'},
                {'property_label': 'type', 'value': 'chemical element'},
            ]},
            {'label': 'Isaac Newton', 'claims': [
                {'property_label': 'birthplace', 'value': 'Woolsthorpe'},
                {'property_label': 'nationality', 'value': 'English'},
                {'property_label': 'occupation', 'value': 'physicist'},
            ]},
            {'label': 'Leonardo da Vinci', 'claims': [
                {'property_label': 'birthplace', 'value': 'Vinci'},
                {'property_label': 'nationality', 'value': 'Italian'},
                {'property_label': 'occupation', 'value': 'polymath'},
            ]},
            {'label': 'Mona Lisa', 'claims': [
                {'property_label': 'creator', 'value': 'Leonardo da Vinci'},
                {'property_label': 'type', 'value': 'painting'},
            ]},
            {'label': 'William Shakespeare', 'claims': [
                {'property_label': 'birthplace', 'value': 'Stratford-upon-Avon'},
                {'property_label': 'nationality', 'value': 'English'},
                {'property_label': 'occupation', 'value': 'playwright'},
            ]},
            {'label': 'Hamlet', 'claims': [
                {'property_label': 'author', 'value': 'William Shakespeare'},
                {'property_label': 'type', 'value': 'play'},
            ]},
        ]

        # Additional countries (30 more)
        countries_extra = [
            {'label': 'Egypt', 'claims': [
                {'property_label': 'capital', 'value': 'Cairo'},
                {'property_label': 'language', 'value': 'Arabic'},
                {'property_label': 'population', 'value': '104 million'},
                {'property_label': 'location', 'value': 'Africa'},
                {'property_label': 'currency', 'value': 'Pound'},
            ]},
            {'label': 'South Africa', 'claims': [
                {'property_label': 'capital', 'value': 'Pretoria'},
                {'property_label': 'language', 'value': 'English'},
                {'property_label': 'population', 'value': '60 million'},
                {'property_label': 'location', 'value': 'Africa'},
                {'property_label': 'currency', 'value': 'Rand'},
            ]},
            {'label': 'Nigeria', 'claims': [
                {'property_label': 'capital', 'value': 'Abuja'},
                {'property_label': 'language', 'value': 'English'},
                {'property_label': 'population', 'value': '218 million'},
                {'property_label': 'location', 'value': 'Africa'},
                {'property_label': 'currency', 'value': 'Naira'},
            ]},
            {'label': 'Kenya', 'claims': [
                {'property_label': 'capital', 'value': 'Nairobi'},
                {'property_label': 'language', 'value': 'Swahili'},
                {'property_label': 'population', 'value': '54 million'},
                {'property_label': 'location', 'value': 'Africa'},
                {'property_label': 'currency', 'value': 'Shilling'},
            ]},
            {'label': 'Turkey', 'claims': [
                {'property_label': 'capital', 'value': 'Ankara'},
                {'property_label': 'language', 'value': 'Turkish'},
                {'property_label': 'population', 'value': '85 million'},
                {'property_label': 'location', 'value': 'Asia'},
                {'property_label': 'currency', 'value': 'Lira'},
            ]},
            {'label': 'Iran', 'claims': [
                {'property_label': 'capital', 'value': 'Tehran'},
                {'property_label': 'language', 'value': 'Persian'},
                {'property_label': 'population', 'value': '87 million'},
                {'property_label': 'location', 'value': 'Asia'},
                {'property_label': 'currency', 'value': 'Rial'},
            ]},
            {'label': 'Saudi Arabia', 'claims': [
                {'property_label': 'capital', 'value': 'Riyadh'},
                {'property_label': 'language', 'value': 'Arabic'},
                {'property_label': 'population', 'value': '36 million'},
                {'property_label': 'location', 'value': 'Asia'},
                {'property_label': 'currency', 'value': 'Riyal'},
            ]},
            {'label': 'Thailand', 'claims': [
                {'property_label': 'capital', 'value': 'Bangkok'},
                {'property_label': 'language', 'value': 'Thai'},
                {'property_label': 'population', 'value': '72 million'},
                {'property_label': 'location', 'value': 'Asia'},
                {'property_label': 'currency', 'value': 'Baht'},
            ]},
            {'label': 'Indonesia', 'claims': [
                {'property_label': 'capital', 'value': 'Jakarta'},
                {'property_label': 'language', 'value': 'Indonesian'},
                {'property_label': 'population', 'value': '275 million'},
                {'property_label': 'location', 'value': 'Asia'},
                {'property_label': 'currency', 'value': 'Rupiah'},
                {'property_label': 'type', 'value': 'island country'},
            ]},
            {'label': 'Argentina', 'claims': [
                {'property_label': 'capital', 'value': 'Buenos Aires'},
                {'property_label': 'language', 'value': 'Spanish'},
                {'property_label': 'population', 'value': '46 million'},
                {'property_label': 'location', 'value': 'South America'},
                {'property_label': 'currency', 'value': 'Peso'},
            ]},
            {'label': 'Colombia', 'claims': [
                {'property_label': 'capital', 'value': 'Bogota'},
                {'property_label': 'language', 'value': 'Spanish'},
                {'property_label': 'population', 'value': '51 million'},
                {'property_label': 'location', 'value': 'South America'},
                {'property_label': 'currency', 'value': 'Peso'},
            ]},
            {'label': 'Chile', 'claims': [
                {'property_label': 'capital', 'value': 'Santiago'},
                {'property_label': 'language', 'value': 'Spanish'},
                {'property_label': 'population', 'value': '19 million'},
                {'property_label': 'location', 'value': 'South America'},
                {'property_label': 'currency', 'value': 'Peso'},
            ]},
            {'label': 'Peru', 'claims': [
                {'property_label': 'capital', 'value': 'Lima'},
                {'property_label': 'language', 'value': 'Spanish'},
                {'property_label': 'population', 'value': '33 million'},
                {'property_label': 'location', 'value': 'South America'},
                {'property_label': 'currency', 'value': 'Sol'},
            ]},
            {'label': 'Greece', 'claims': [
                {'property_label': 'capital', 'value': 'Athens'},
                {'property_label': 'language', 'value': 'Greek'},
                {'property_label': 'population', 'value': '10.4 million'},
                {'property_label': 'location', 'value': 'Europe'},
                {'property_label': 'currency', 'value': 'Euro'},
            ]},
            {'label': 'Portugal', 'claims': [
                {'property_label': 'capital', 'value': 'Lisbon'},
                {'property_label': 'language', 'value': 'Portuguese'},
                {'property_label': 'population', 'value': '10.3 million'},
                {'property_label': 'location', 'value': 'Europe'},
                {'property_label': 'currency', 'value': 'Euro'},
                {'property_label': 'borders', 'value': 'Spain'},
            ]},
            {'label': 'Netherlands', 'claims': [
                {'property_label': 'capital', 'value': 'Amsterdam'},
                {'property_label': 'language', 'value': 'Dutch'},
                {'property_label': 'population', 'value': '17.5 million'},
                {'property_label': 'location', 'value': 'Europe'},
                {'property_label': 'currency', 'value': 'Euro'},
                {'property_label': 'borders', 'value': 'Germany'},
                {'property_label': 'borders', 'value': 'Belgium'},
            ]},
            {'label': 'Belgium', 'claims': [
                {'property_label': 'capital', 'value': 'Brussels'},
                {'property_label': 'language', 'value': 'Dutch'},
                {'property_label': 'population', 'value': '11.5 million'},
                {'property_label': 'location', 'value': 'Europe'},
                {'property_label': 'currency', 'value': 'Euro'},
                {'property_label': 'borders', 'value': 'France'},
                {'property_label': 'borders', 'value': 'Netherlands'},
                {'property_label': 'borders', 'value': 'Germany'},
            ]},
            {'label': 'Switzerland', 'claims': [
                {'property_label': 'capital', 'value': 'Bern'},
                {'property_label': 'language', 'value': 'German'},
                {'property_label': 'population', 'value': '8.7 million'},
                {'property_label': 'location', 'value': 'Europe'},
                {'property_label': 'currency', 'value': 'Franc'},
                {'property_label': 'borders', 'value': 'France'},
                {'property_label': 'borders', 'value': 'Germany'},
                {'property_label': 'borders', 'value': 'Italy'},
                {'property_label': 'borders', 'value': 'Austria'},
            ]},
            {'label': 'Austria', 'claims': [
                {'property_label': 'capital', 'value': 'Vienna'},
                {'property_label': 'language', 'value': 'German'},
                {'property_label': 'population', 'value': '9 million'},
                {'property_label': 'location', 'value': 'Europe'},
                {'property_label': 'currency', 'value': 'Euro'},
                {'property_label': 'borders', 'value': 'Germany'},
                {'property_label': 'borders', 'value': 'Switzerland'},
                {'property_label': 'borders', 'value': 'Italy'},
            ]},
            {'label': 'Ireland', 'claims': [
                {'property_label': 'capital', 'value': 'Dublin'},
                {'property_label': 'language', 'value': 'English'},
                {'property_label': 'population', 'value': '5 million'},
                {'property_label': 'location', 'value': 'Europe'},
                {'property_label': 'currency', 'value': 'Euro'},
            ]},
            {'label': 'Iceland', 'claims': [
                {'property_label': 'capital', 'value': 'Reykjavik'},
                {'property_label': 'language', 'value': 'Icelandic'},
                {'property_label': 'population', 'value': '370 thousand'},
                {'property_label': 'location', 'value': 'Europe'},
                {'property_label': 'currency', 'value': 'Krona'},
                {'property_label': 'type', 'value': 'island country'},
            ]},
            {'label': 'New Zealand', 'claims': [
                {'property_label': 'capital', 'value': 'Wellington'},
                {'property_label': 'language', 'value': 'English'},
                {'property_label': 'population', 'value': '5 million'},
                {'property_label': 'location', 'value': 'Oceania'},
                {'property_label': 'currency', 'value': 'Dollar'},
                {'property_label': 'type', 'value': 'island country'},
            ]},
            {'label': 'Pakistan', 'claims': [
                {'property_label': 'capital', 'value': 'Islamabad'},
                {'property_label': 'language', 'value': 'Urdu'},
                {'property_label': 'population', 'value': '230 million'},
                {'property_label': 'location', 'value': 'Asia'},
                {'property_label': 'currency', 'value': 'Rupee'},
                {'property_label': 'borders', 'value': 'India'},
                {'property_label': 'borders', 'value': 'Iran'},
                {'property_label': 'borders', 'value': 'China'},
            ]},
            {'label': 'Bangladesh', 'claims': [
                {'property_label': 'capital', 'value': 'Dhaka'},
                {'property_label': 'language', 'value': 'Bengali'},
                {'property_label': 'population', 'value': '170 million'},
                {'property_label': 'location', 'value': 'Asia'},
                {'property_label': 'currency', 'value': 'Taka'},
                {'property_label': 'borders', 'value': 'India'},
            ]},
            {'label': 'Vietnam', 'claims': [
                {'property_label': 'capital', 'value': 'Hanoi'},
                {'property_label': 'language', 'value': 'Vietnamese'},
                {'property_label': 'population', 'value': '98 million'},
                {'property_label': 'location', 'value': 'Asia'},
                {'property_label': 'currency', 'value': 'Dong'},
                {'property_label': 'borders', 'value': 'China'},
            ]},
            {'label': 'Philippines', 'claims': [
                {'property_label': 'capital', 'value': 'Manila'},
                {'property_label': 'language', 'value': 'Filipino'},
                {'property_label': 'population', 'value': '113 million'},
                {'property_label': 'location', 'value': 'Asia'},
                {'property_label': 'currency', 'value': 'Peso'},
                {'property_label': 'type', 'value': 'island country'},
            ]},
            {'label': 'Ukraine', 'claims': [
                {'property_label': 'capital', 'value': 'Kyiv'},
                {'property_label': 'language', 'value': 'Ukrainian'},
                {'property_label': 'population', 'value': '44 million'},
                {'property_label': 'location', 'value': 'Europe'},
                {'property_label': 'currency', 'value': 'Hryvnia'},
                {'property_label': 'borders', 'value': 'Russia'},
                {'property_label': 'borders', 'value': 'Poland'},
            ]},
            {'label': 'Czech Republic', 'claims': [
                {'property_label': 'capital', 'value': 'Prague'},
                {'property_label': 'language', 'value': 'Czech'},
                {'property_label': 'population', 'value': '10.7 million'},
                {'property_label': 'location', 'value': 'Europe'},
                {'property_label': 'currency', 'value': 'Koruna'},
                {'property_label': 'borders', 'value': 'Germany'},
                {'property_label': 'borders', 'value': 'Austria'},
                {'property_label': 'borders', 'value': 'Poland'},
            ]},
            {'label': 'Romania', 'claims': [
                {'property_label': 'capital', 'value': 'Bucharest'},
                {'property_label': 'language', 'value': 'Romanian'},
                {'property_label': 'population', 'value': '19 million'},
                {'property_label': 'location', 'value': 'Europe'},
                {'property_label': 'currency', 'value': 'Leu'},
                {'property_label': 'borders', 'value': 'Ukraine'},
            ]},
            {'label': 'Hungary', 'claims': [
                {'property_label': 'capital', 'value': 'Budapest'},
                {'property_label': 'language', 'value': 'Hungarian'},
                {'property_label': 'population', 'value': '9.7 million'},
                {'property_label': 'location', 'value': 'Europe'},
                {'property_label': 'currency', 'value': 'Forint'},
                {'property_label': 'borders', 'value': 'Austria'},
                {'property_label': 'borders', 'value': 'Romania'},
                {'property_label': 'borders', 'value': 'Ukraine'},
            ]},
        ]

        # Additional people (20 more)
        people_extra = [
            {'label': 'Charles Darwin', 'claims': [
                {'property_label': 'birthplace', 'value': 'Shrewsbury'},
                {'property_label': 'nationality', 'value': 'English'},
                {'property_label': 'occupation', 'value': 'naturalist'},
                {'property_label': 'born', 'value': '1809'},
                {'property_label': 'died', 'value': '1882'},
            ]},
            {'label': 'Nikola Tesla', 'claims': [
                {'property_label': 'birthplace', 'value': 'Smiljan'},
                {'property_label': 'nationality', 'value': 'Serbian-American'},
                {'property_label': 'occupation', 'value': 'inventor'},
                {'property_label': 'born', 'value': '1856'},
                {'property_label': 'died', 'value': '1943'},
            ]},
            {'label': 'Galileo Galilei', 'claims': [
                {'property_label': 'birthplace', 'value': 'Pisa'},
                {'property_label': 'nationality', 'value': 'Italian'},
                {'property_label': 'occupation', 'value': 'astronomer'},
                {'property_label': 'born', 'value': '1564'},
                {'property_label': 'died', 'value': '1642'},
            ]},
            {'label': 'Ada Lovelace', 'claims': [
                {'property_label': 'birthplace', 'value': 'London'},
                {'property_label': 'nationality', 'value': 'English'},
                {'property_label': 'occupation', 'value': 'mathematician'},
                {'property_label': 'born', 'value': '1815'},
                {'property_label': 'died', 'value': '1852'},
            ]},
            {'label': 'Alan Turing', 'claims': [
                {'property_label': 'birthplace', 'value': 'London'},
                {'property_label': 'nationality', 'value': 'British'},
                {'property_label': 'occupation', 'value': 'mathematician'},
                {'property_label': 'born', 'value': '1912'},
                {'property_label': 'died', 'value': '1954'},
            ]},
            {'label': 'Wolfgang Amadeus Mozart', 'claims': [
                {'property_label': 'birthplace', 'value': 'Salzburg'},
                {'property_label': 'nationality', 'value': 'Austrian'},
                {'property_label': 'occupation', 'value': 'composer'},
                {'property_label': 'born', 'value': '1756'},
                {'property_label': 'died', 'value': '1791'},
            ]},
            {'label': 'Ludwig van Beethoven', 'claims': [
                {'property_label': 'birthplace', 'value': 'Bonn'},
                {'property_label': 'nationality', 'value': 'German'},
                {'property_label': 'occupation', 'value': 'composer'},
                {'property_label': 'born', 'value': '1770'},
                {'property_label': 'died', 'value': '1827'},
            ]},
            {'label': 'Aristotle', 'claims': [
                {'property_label': 'birthplace', 'value': 'Stagira'},
                {'property_label': 'nationality', 'value': 'Greek'},
                {'property_label': 'occupation', 'value': 'philosopher'},
                {'property_label': 'born', 'value': '384 BC'},
            ]},
            {'label': 'Plato', 'claims': [
                {'property_label': 'birthplace', 'value': 'Athens'},
                {'property_label': 'nationality', 'value': 'Greek'},
                {'property_label': 'occupation', 'value': 'philosopher'},
                {'property_label': 'born', 'value': '428 BC'},
            ]},
            {'label': 'Cleopatra', 'claims': [
                {'property_label': 'birthplace', 'value': 'Alexandria'},
                {'property_label': 'nationality', 'value': 'Egyptian'},
                {'property_label': 'occupation', 'value': 'pharaoh'},
                {'property_label': 'born', 'value': '69 BC'},
            ]},
            {'label': 'Napoleon Bonaparte', 'claims': [
                {'property_label': 'birthplace', 'value': 'Ajaccio'},
                {'property_label': 'nationality', 'value': 'French'},
                {'property_label': 'occupation', 'value': 'emperor'},
                {'property_label': 'born', 'value': '1769'},
                {'property_label': 'died', 'value': '1821'},
            ]},
            {'label': 'Mahatma Gandhi', 'claims': [
                {'property_label': 'birthplace', 'value': 'Porbandar'},
                {'property_label': 'nationality', 'value': 'Indian'},
                {'property_label': 'occupation', 'value': 'political leader'},
                {'property_label': 'born', 'value': '1869'},
                {'property_label': 'died', 'value': '1948'},
            ]},
            {'label': 'Nelson Mandela', 'claims': [
                {'property_label': 'birthplace', 'value': 'Mvezo'},
                {'property_label': 'nationality', 'value': 'South African'},
                {'property_label': 'occupation', 'value': 'president'},
                {'property_label': 'born', 'value': '1918'},
                {'property_label': 'died', 'value': '2013'},
            ]},
            {'label': 'Frida Kahlo', 'claims': [
                {'property_label': 'birthplace', 'value': 'Mexico City'},
                {'property_label': 'nationality', 'value': 'Mexican'},
                {'property_label': 'occupation', 'value': 'painter'},
                {'property_label': 'born', 'value': '1907'},
                {'property_label': 'died', 'value': '1954'},
            ]},
            {'label': 'Pablo Picasso', 'claims': [
                {'property_label': 'birthplace', 'value': 'Malaga'},
                {'property_label': 'nationality', 'value': 'Spanish'},
                {'property_label': 'occupation', 'value': 'painter'},
                {'property_label': 'born', 'value': '1881'},
                {'property_label': 'died', 'value': '1973'},
            ]},
            {'label': 'Guernica', 'claims': [
                {'property_label': 'creator', 'value': 'Pablo Picasso'},
                {'property_label': 'type', 'value': 'painting'},
                {'property_label': 'founded', 'value': '1937'},
            ]},
            {'label': 'Confucius', 'claims': [
                {'property_label': 'birthplace', 'value': 'Qufu'},
                {'property_label': 'nationality', 'value': 'Chinese'},
                {'property_label': 'occupation', 'value': 'philosopher'},
                {'property_label': 'born', 'value': '551 BC'},
            ]},
            {'label': 'Alexander the Great', 'claims': [
                {'property_label': 'birthplace', 'value': 'Pella'},
                {'property_label': 'nationality', 'value': 'Macedonian'},
                {'property_label': 'occupation', 'value': 'king'},
                {'property_label': 'born', 'value': '356 BC'},
                {'property_label': 'died', 'value': '323 BC'},
            ]},
            {'label': 'Curie', 'claims': [
                {'property_label': 'known_as', 'value': 'Marie Curie'},
                {'property_label': 'type', 'value': 'surname'},
            ]},
        ]

        # Geography
        geography = [
            {'label': 'Amazon River', 'claims': [
                {'property_label': 'type', 'value': 'river'},
                {'property_label': 'country', 'value': 'Brazil'},
                {'property_label': 'description', 'value': 'the longest river in South America'},
            ]},
            {'label': 'Nile', 'claims': [
                {'property_label': 'type', 'value': 'river'},
                {'property_label': 'country', 'value': 'Egypt'},
                {'property_label': 'description', 'value': 'the longest river in Africa'},
            ]},
            {'label': 'Mount Everest', 'claims': [
                {'property_label': 'type', 'value': 'mountain'},
                {'property_label': 'country', 'value': 'Nepal'},
                {'property_label': 'description', 'value': 'the highest mountain on Earth'},
            ]},
            {'label': 'Sahara', 'claims': [
                {'property_label': 'type', 'value': 'desert'},
                {'property_label': 'location', 'value': 'Africa'},
                {'property_label': 'description', 'value': 'the largest hot desert in the world'},
            ]},
            {'label': 'Pacific Ocean', 'claims': [
                {'property_label': 'type', 'value': 'ocean'},
                {'property_label': 'description', 'value': 'the largest ocean on Earth'},
            ]},
            {'label': 'Atlantic Ocean', 'claims': [
                {'property_label': 'type', 'value': 'ocean'},
                {'property_label': 'description', 'value': 'the second largest ocean on Earth'},
            ]},
            {'label': 'Mars', 'claims': [
                {'property_label': 'type', 'value': 'planet'},
                {'property_label': 'known_as', 'value': 'the Red Planet'},
                {'property_label': 'description', 'value': 'the fourth planet from the Sun'},
            ]},
            {'label': 'Jupiter', 'claims': [
                {'property_label': 'type', 'value': 'planet'},
                {'property_label': 'description', 'value': 'the largest planet in the Solar System'},
            ]},
            {'label': 'Moon', 'claims': [
                {'property_label': 'type', 'value': 'natural satellite'},
                {'property_label': 'description', 'value': 'the only natural satellite of Earth'},
            ]},
            {'label': 'Sun', 'claims': [
                {'property_label': 'type', 'value': 'star'},
                {'property_label': 'description', 'value': 'the star at the center of the Solar System'},
            ]},
        ]

        # Additional science
        science_extra = [
            {'label': 'DNA', 'claims': [
                {'property_label': 'type', 'value': 'molecule'},
                {'property_label': 'known_as', 'value': 'deoxyribonucleic acid'},
                {'property_label': 'discoverer', 'value': 'James Watson and Francis Crick'},
            ]},
            {'label': 'Gravity', 'claims': [
                {'property_label': 'type', 'value': 'fundamental force'},
                {'property_label': 'description', 'value': 'the force of attraction between masses'},
            ]},
            {'label': 'Speed of Light', 'claims': [
                {'property_label': 'type', 'value': 'physical constant'},
                {'property_label': 'description', 'value': 'approximately 299,792 km per second'},
            ]},
            {'label': 'Nitrogen', 'claims': [
                {'property_label': 'symbol', 'value': 'N'},
                {'property_label': 'type', 'value': 'chemical element'},
            ]},
            {'label': 'Carbon', 'claims': [
                {'property_label': 'symbol', 'value': 'C'},
                {'property_label': 'type', 'value': 'chemical element'},
            ]},
            {'label': 'Silver', 'claims': [
                {'property_label': 'symbol', 'value': 'Ag'},
                {'property_label': 'type', 'value': 'chemical element'},
            ]},
            {'label': 'Copper', 'claims': [
                {'property_label': 'symbol', 'value': 'Cu'},
                {'property_label': 'type', 'value': 'chemical element'},
            ]},
            {'label': 'Helium', 'claims': [
                {'property_label': 'symbol', 'value': 'He'},
                {'property_label': 'type', 'value': 'chemical element'},
                {'property_label': 'discoverer', 'value': 'Pierre Janssen'},
            ]},
        ]

        # Technology and inventions
        tech = [
            {'label': 'Python', 'claims': [
                {'property_label': 'creator', 'value': 'Guido van Rossum'},
                {'property_label': 'type', 'value': 'programming language'},
                {'property_label': 'founded', 'value': '1991'},
            ]},
            {'label': 'Linux', 'claims': [
                {'property_label': 'creator', 'value': 'Linus Torvalds'},
                {'property_label': 'type', 'value': 'operating system'},
                {'property_label': 'founded', 'value': '1991'},
            ]},
            {'label': 'World Wide Web', 'claims': [
                {'property_label': 'inventor', 'value': 'Tim Berners-Lee'},
                {'property_label': 'founded', 'value': '1989'},
            ]},
            {'label': 'Telephone', 'claims': [
                {'property_label': 'inventor', 'value': 'Alexander Graham Bell'},
                {'property_label': 'founded', 'value': '1876'},
            ]},
            {'label': 'Electricity', 'claims': [
                {'property_label': 'discoverer', 'value': 'Benjamin Franklin'},
            ]},
            {'label': 'Penicillin', 'claims': [
                {'property_label': 'discoverer', 'value': 'Alexander Fleming'},
                {'property_label': 'type', 'value': 'antibiotic'},
                {'property_label': 'founded', 'value': '1928'},
            ]},
            {'label': 'Theory of Relativity', 'claims': [
                {'property_label': 'creator', 'value': 'Albert Einstein'},
                {'property_label': 'founded', 'value': '1905'},
            ]},
            {'label': 'C', 'claims': [
                {'property_label': 'creator', 'value': 'Dennis Ritchie'},
                {'property_label': 'type', 'value': 'programming language'},
                {'property_label': 'founded', 'value': '1972'},
            ]},
            {'label': 'Java', 'claims': [
                {'property_label': 'creator', 'value': 'James Gosling'},
                {'property_label': 'type', 'value': 'programming language'},
                {'property_label': 'founded', 'value': '1995'},
            ]},
            {'label': 'JavaScript', 'claims': [
                {'property_label': 'creator', 'value': 'Brendan Eich'},
                {'property_label': 'type', 'value': 'programming language'},
                {'property_label': 'founded', 'value': '1995'},
            ]},
            {'label': 'Wikipedia', 'claims': [
                {'property_label': 'founder', 'value': 'Jimmy Wales'},
                {'property_label': 'type', 'value': 'online encyclopedia'},
                {'property_label': 'founded', 'value': '2001'},
            ]},
            {'label': 'Google', 'claims': [
                {'property_label': 'founder', 'value': 'Larry Page'},
                {'property_label': 'type', 'value': 'technology company'},
                {'property_label': 'founded', 'value': '1998'},
                {'property_label': 'country', 'value': 'United States'},
            ]},
            {'label': 'Apple', 'claims': [
                {'property_label': 'founder', 'value': 'Steve Jobs'},
                {'property_label': 'type', 'value': 'technology company'},
                {'property_label': 'founded', 'value': '1976'},
                {'property_label': 'country', 'value': 'United States'},
            ]},
            {'label': 'Tesla', 'claims': [
                {'property_label': 'founder', 'value': 'Elon Musk'},
                {'property_label': 'type', 'value': 'electric vehicle company'},
                {'property_label': 'founded', 'value': '2003'},
                {'property_label': 'country', 'value': 'United States'},
            ]},
        ]

        # Literary works
        works = [
            {'label': 'Don Quixote', 'claims': [
                {'property_label': 'author', 'value': 'Miguel de Cervantes'},
                {'property_label': 'type', 'value': 'novel'},
                {'property_label': 'founded', 'value': '1605'},
            ]},
            {'label': 'War and Peace', 'claims': [
                {'property_label': 'author', 'value': 'Leo Tolstoy'},
                {'property_label': 'type', 'value': 'novel'},
                {'property_label': 'founded', 'value': '1869'},
            ]},
            {'label': 'The Republic', 'claims': [
                {'property_label': 'author', 'value': 'Plato'},
                {'property_label': 'type', 'value': 'philosophical work'},
            ]},
            {'label': 'Romeo and Juliet', 'claims': [
                {'property_label': 'author', 'value': 'William Shakespeare'},
                {'property_label': 'type', 'value': 'play'},
                {'property_label': 'founded', 'value': '1597'},
            ]},
            {'label': 'The Odyssey', 'claims': [
                {'property_label': 'author', 'value': 'Homer'},
                {'property_label': 'type', 'value': 'epic poem'},
            ]},
        ]

        all_entities = (countries + countries_extra + science + science_extra +
                        people + people_extra + geography + tech + works)
        return self.import_entities(all_entities)

    def report(self):
        """Summary of import statistics."""
        lines = [
            f"Wikidata Import Report",
            f"  Entities processed: {self.stats['entities_processed']}",
            f"  Facts added: {self.stats['facts_added']}",
            f"  Facts skipped: {self.stats['facts_skipped']}",
        ]
        if self.stats['unknown_properties']:
            lines.append(f"  Unknown properties: {len(self.stats['unknown_properties'])}")
        return "\n".join(lines)


# =============================================================================
# Live Wikidata SPARQL Importer
# =============================================================================

WIKIDATA_SPARQL_URL = 'https://query.wikidata.org/sparql'
WIKIDATA_USER_AGENT = 'FOSS-KI/1.0 (https://github.com/foss-ki; david@foss.com.de)'

# SPARQL queries for bulk knowledge import
SPARQL_QUERIES = {
    'countries': """
        SELECT ?countryLabel ?capitalLabel ?continentLabel ?currencyLabel
               ?languageLabel ?population
        WHERE {
          ?country wdt:P31 wd:Q6256 .
          OPTIONAL { ?country wdt:P36 ?capital . }
          OPTIONAL { ?country wdt:P30 ?continent . }
          OPTIONAL { ?country wdt:P38 ?currency . }
          OPTIONAL { ?country wdt:P37 ?language . }
          OPTIONAL { ?country wdt:P1082 ?population . }
          SERVICE wikibase:label { bd:serviceParam wikibase:language "en" . }
        }
    """,
    'country_borders': """
        SELECT ?countryLabel ?borderLabel
        WHERE {
          ?country wdt:P31 wd:Q6256 .
          ?country wdt:P47 ?border .
          ?border wdt:P31 wd:Q6256 .
          SERVICE wikibase:label { bd:serviceParam wikibase:language "en" . }
        }
    """,
    'famous_scientists': """
        SELECT ?personLabel ?occupationLabel ?nationalityLabel ?born
        WHERE {
          ?person wdt:P31 wd:Q5 .
          ?person wdt:P106 wd:Q901 .
          ?person wdt:P166 ?award .
          ?award wdt:P31/wdt:P279* wd:Q7191 .
          OPTIONAL { ?person wdt:P27 ?nationality . }
          OPTIONAL { ?person wdt:P569 ?born . }
          OPTIONAL { ?person wdt:P106 ?occupation . }
          SERVICE wikibase:label { bd:serviceParam wikibase:language "en" . }
        }
        LIMIT 200
    """,
    'famous_leaders': """
        SELECT ?personLabel ?nationalityLabel ?born
        WHERE {
          ?person wdt:P31 wd:Q5 .
          ?person wdt:P39 ?position .
          ?position wdt:P279* wd:Q48352 .
          OPTIONAL { ?person wdt:P27 ?nationality . }
          OPTIONAL { ?person wdt:P569 ?born . }
          SERVICE wikibase:label { bd:serviceParam wikibase:language "en" . }
        }
        LIMIT 200
    """,
    'cities': """
        SELECT ?cityLabel ?countryLabel ?population
        WHERE {
          ?city wdt:P31/wdt:P279* wd:Q515 .
          ?city wdt:P17 ?country .
          ?city wdt:P1082 ?population .
          FILTER(?population > 1000000)
          SERVICE wikibase:label { bd:serviceParam wikibase:language "en" . }
        }
        LIMIT 300
    """,
    'companies': """
        SELECT ?companyLabel ?founderLabel ?foundedLabel
               ?countryLabel ?industryLabel
        WHERE {
          ?company wdt:P31/wdt:P279* wd:Q4830453 .
          ?company wikibase:sitelinks ?sitelinks .
          FILTER(?sitelinks > 50)
          OPTIONAL { ?company wdt:P112 ?founder . }
          OPTIONAL { ?company wdt:P571 ?founded . }
          OPTIONAL { ?company wdt:P17 ?country . }
          OPTIONAL { ?company wdt:P452 ?industry . }
          SERVICE wikibase:label { bd:serviceParam wikibase:language "en" . }
        }
        LIMIT 200
    """,
    'inventions': """
        SELECT ?inventionLabel ?inventorLabel ?yearLabel
        WHERE {
          ?invention wdt:P31/wdt:P279* wd:Q39546 .
          ?invention wdt:P61 ?inventor .
          OPTIONAL { ?invention wdt:P571 ?year . }
          ?invention wikibase:sitelinks ?sitelinks .
          FILTER(?sitelinks > 20)
          SERVICE wikibase:label { bd:serviceParam wikibase:language "en" . }
        }
        LIMIT 200
    """,
    'languages': """
        SELECT ?langLabel ?familyLabel ?scriptLabel ?speakersLabel
        WHERE {
          ?lang wdt:P31 wd:Q34770 .
          ?lang wikibase:sitelinks ?sitelinks .
          FILTER(?sitelinks > 30)
          OPTIONAL { ?lang wdt:P279 ?family . }
          OPTIONAL { ?lang wdt:P282 ?script . }
          OPTIONAL { ?lang wdt:P1098 ?speakers . }
          SERVICE wikibase:label { bd:serviceParam wikibase:language "en" . }
        }
        LIMIT 200
    """,
}


def _sparql_query(query, timeout=30):
    """Execute a SPARQL query against Wikidata and return results."""
    url = WIKIDATA_SPARQL_URL + '?' + urllib.parse.urlencode({
        'query': query,
        'format': 'json',
    })
    req = urllib.request.Request(url)
    req.add_header('User-Agent', WIKIDATA_USER_AGENT)
    req.add_header('Accept', 'application/sparql-results+json')

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            return data.get('results', {}).get('bindings', [])
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
        print(f"  SPARQL query failed: {e}")
        return []


def _extract_value(binding, key):
    """Extract a clean value from SPARQL binding."""
    if key not in binding:
        return ''
    val = binding[key].get('value', '')
    # Strip Q-IDs that weren't resolved
    if val.startswith('http://www.wikidata.org/entity/Q'):
        return ''
    # Extract year from datetime
    if 'T' in val and val.startswith(('http', '+', '-')):
        m = re.match(r'.*?(\d{4})', val)
        return m.group(1) if m else ''
    return val.strip()


def _format_population(pop_str):
    """Format population number to human-readable string."""
    try:
        n = int(float(pop_str))
        if n >= 1_000_000_000:
            return f"{n / 1_000_000_000:.1f} billion"
        elif n >= 1_000_000:
            return f"{n / 1_000_000:.0f} million"
        elif n >= 1_000:
            return f"{n / 1_000:.0f} thousand"
        return str(n)
    except (ValueError, TypeError):
        return pop_str


def wikidata_live_import(knowledge_store, categories=None, verbose=True):
    """
    Pull structured knowledge from Wikidata SPARQL endpoint.

    Args:
        knowledge_store: KnowledgeStore instance to store facts into
        categories: list of category names to import, or None for all
        verbose: print progress

    Returns:
        dict with import stats
    """
    if categories is None:
        # Note: famous_scientists/leaders queries often timeout on public endpoint.
        # People are covered by import_countries_simple() hardcoded data.
        categories = ['countries', 'country_borders',
                       'cities', 'companies', 'inventions', 'languages']

    cache_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'data', 'wikidata_cache.json'
    )

    # Check cache (valid for 7 days)
    cache = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache = json.load(f)
            cache_age = time.time() - cache.get('timestamp', 0)
            if cache_age < 7 * 86400:
                if verbose:
                    print(f"  Using cached Wikidata data ({cache_age / 3600:.0f}h old)")
            else:
                cache = {}
        except (json.JSONDecodeError, OSError):
            cache = {}

    stats = {'total': 0, 'per_category': {}, 'errors': []}
    all_facts = []  # (subject, relation, object) tuples for cache

    for cat in categories:
        if cat not in SPARQL_QUERIES:
            stats['errors'].append(f"Unknown category: {cat}")
            continue

        # Use cache if available
        if cache and cat in cache.get('categories', {}):
            facts = cache['categories'][cat]
            for s, r, o in facts:
                knowledge_store.store_fact(s, r, o)
            stats['per_category'][cat] = len(facts)
            stats['total'] += len(facts)
            if verbose:
                print(f"  {cat}: {len(facts)} facts (cached)")
            continue

        if verbose:
            print(f"  Querying Wikidata: {cat}...")

        bindings = _sparql_query(SPARQL_QUERIES[cat], timeout=60)
        if not bindings:
            stats['errors'].append(f"No results for {cat}")
            continue

        facts = []

        if cat == 'countries':
            facts = _process_countries(bindings)
        elif cat == 'country_borders':
            facts = _process_borders(bindings)
        elif cat == 'famous_people':
            facts = _process_people(bindings)
        elif cat == 'cities':
            facts = _process_cities(bindings)
        elif cat == 'companies':
            facts = _process_companies(bindings)
        elif cat == 'inventions':
            facts = _process_inventions(bindings)
        elif cat == 'languages':
            facts = _process_languages(bindings)

        # Deduplicate
        seen = set()
        unique_facts = []
        for s, r, o in facts:
            key = (s.lower(), r.lower(), o.lower())
            if key not in seen and s and r and o:
                seen.add(key)
                unique_facts.append((s, r, o))

        # Store
        for s, r, o in unique_facts:
            knowledge_store.store_fact(s, r, o)

        all_facts.extend(unique_facts)
        stats['per_category'][cat] = len(unique_facts)
        stats['total'] += len(unique_facts)

        if verbose:
            print(f"  {cat}: {len(unique_facts)} facts")

        # Rate limit (Wikidata asks for 1 req/s)
        time.sleep(1.5)

    # Save cache
    if all_facts:
        cache_data = {
            'timestamp': time.time(),
            'categories': cache.get('categories', {}),
        }
        # Group by category for cache
        offset = 0
        for cat in categories:
            if cat in stats['per_category'] and cat not in cache_data['categories']:
                n = stats['per_category'][cat]
                cache_data['categories'][cat] = [
                    list(t) for t in all_facts[offset:offset + n]
                ]
                offset += n

        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=1)
            if verbose:
                size_kb = os.path.getsize(cache_path) / 1024
                print(f"  Cache saved: {cache_path} ({size_kb:.0f} KB)")
        except OSError as e:
            if verbose:
                print(f"  Cache save failed: {e}")

    return stats


def _process_countries(bindings):
    """Process country SPARQL results into (S, R, O) triplets."""
    facts = []
    seen_countries = {}

    for b in bindings:
        country = _extract_value(b, 'countryLabel')
        if not country:
            continue

        # Deduplicate: keep first non-empty values per country
        if country not in seen_countries:
            seen_countries[country] = {}

        capital = _extract_value(b, 'capitalLabel')
        continent = _extract_value(b, 'continentLabel')
        currency = _extract_value(b, 'currencyLabel')
        language = _extract_value(b, 'languageLabel')
        population = _extract_value(b, 'population')

        d = seen_countries[country]
        if capital and 'capital' not in d:
            d['capital'] = capital
        if continent and 'location' not in d:
            d['location'] = continent
        if currency and 'currency' not in d:
            d['currency'] = currency
        if language and 'language' not in d:
            d['language'] = language
        if population and 'population' not in d:
            d['population'] = _format_population(population)

    for country, rels in seen_countries.items():
        facts.append((country, 'type', 'country'))
        for rel, val in rels.items():
            facts.append((country, rel, val))

    return facts


def _process_borders(bindings):
    """Process border SPARQL results."""
    facts = []
    for b in bindings:
        country = _extract_value(b, 'countryLabel')
        border = _extract_value(b, 'borderLabel')
        if country and border:
            facts.append((country, 'borders', border))
    return facts


def _process_people(bindings):
    """Process famous people SPARQL results."""
    facts = []
    seen = {}

    for b in bindings:
        person = _extract_value(b, 'personLabel')
        if not person:
            continue

        if person not in seen:
            seen[person] = {}

        d = seen[person]
        birthplace = _extract_value(b, 'birthplaceLabel')
        nationality = _extract_value(b, 'nationalityLabel')
        occupation = _extract_value(b, 'occupationLabel')
        born = _extract_value(b, 'born')
        died = _extract_value(b, 'died')

        if birthplace and 'birthplace' not in d:
            d['birthplace'] = birthplace
        if nationality and 'nationality' not in d:
            d['nationality'] = nationality
        if occupation and 'occupation' not in d:
            d['occupation'] = occupation
        if born and 'born' not in d:
            d['born'] = born
        if died and 'died' not in d:
            d['died'] = died

    for person, rels in seen.items():
        for rel, val in rels.items():
            facts.append((person, rel, val))

    return facts


def _process_cities(bindings):
    """Process city SPARQL results."""
    facts = []
    seen = {}

    for b in bindings:
        city = _extract_value(b, 'cityLabel')
        if not city:
            continue

        if city not in seen:
            seen[city] = {}

        d = seen[city]
        country = _extract_value(b, 'countryLabel')
        population = _extract_value(b, 'population')

        if country and 'country' not in d:
            d['country'] = country
        if population and 'population' not in d:
            d['population'] = _format_population(population)

    for city, rels in seen.items():
        facts.append((city, 'type', 'city'))
        for rel, val in rels.items():
            facts.append((city, rel, val))

    return facts


def _process_companies(bindings):
    """Process company SPARQL results."""
    facts = []
    seen = {}

    for b in bindings:
        company = _extract_value(b, 'companyLabel')
        if not company:
            continue

        if company not in seen:
            seen[company] = {}

        d = seen[company]
        founder = _extract_value(b, 'founderLabel')
        founded = _extract_value(b, 'foundedLabel')
        country = _extract_value(b, 'countryLabel')
        industry = _extract_value(b, 'industryLabel')

        if founder and 'founder' not in d:
            d['founder'] = founder
        if founded and 'founded' not in d:
            d['founded'] = founded
        if country and 'country' not in d:
            d['country'] = country
        if industry and 'industry' not in d:
            d['industry'] = industry

    for company, rels in seen.items():
        facts.append((company, 'type', 'company'))
        for rel, val in rels.items():
            facts.append((company, rel, val))

    return facts


def _process_inventions(bindings):
    """Process invention SPARQL results."""
    facts = []
    seen = {}

    for b in bindings:
        invention = _extract_value(b, 'inventionLabel')
        if not invention:
            continue

        if invention not in seen:
            seen[invention] = {}

        d = seen[invention]
        inventor = _extract_value(b, 'inventorLabel')
        year = _extract_value(b, 'yearLabel')

        if inventor and 'inventor' not in d:
            d['inventor'] = inventor
        if year and 'founded' not in d:
            d['founded'] = year

    for invention, rels in seen.items():
        facts.append((invention, 'type', 'invention'))
        for rel, val in rels.items():
            facts.append((invention, rel, val))

    return facts


def _process_languages(bindings):
    """Process language SPARQL results."""
    facts = []
    seen = {}

    for b in bindings:
        lang = _extract_value(b, 'langLabel')
        if not lang:
            continue

        if lang not in seen:
            seen[lang] = {}

        d = seen[lang]
        family = _extract_value(b, 'familyLabel')
        script = _extract_value(b, 'scriptLabel')

        if family and 'language_family' not in d:
            d['language_family'] = family
        if script and 'writing_system' not in d:
            d['writing_system'] = script

    for lang, rels in seen.items():
        facts.append((lang, 'type', 'language'))
        for rel, val in rels.items():
            facts.append((lang, rel, val))

    return facts
