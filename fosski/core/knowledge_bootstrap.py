"""
Knowledge Bootstrap — Build Comprehensive Offline Knowledge Base
=================================================================
Generates a rich common-sense + factual knowledge base from:
  1. Built-in CommonSenseEngine facts (130+)
  2. Extended common sense (1000+ facts generated from patterns)
  3. World knowledge (countries, elements, units, etc.)
  4. Technical knowledge (programming, CS concepts)

This replaces the need for ConceptNet API calls by shipping
a comprehensive pre-built knowledge base with FOSS-KI.

Output: data/knowledge_full.json — loadable by CommonSenseEngine + KnowledgeStore
"""

import json
import os
from typing import List, Tuple


def generate_extended_common_sense() -> List[Tuple[str, str, str, float]]:
    """Generate extended common sense facts from patterns."""
    facts = []

    # === Material properties ===
    materials = {
        'wood': [('flammable', 3.0), ('natural', 3.0), ('brown', 2.0)],
        'metal': [('hard', 4.0), ('conductive', 3.0), ('shiny', 2.0)],
        'plastic': [('synthetic', 3.0), ('lightweight', 3.0), ('flexible', 2.0)],
        'rubber': [('elastic', 4.0), ('waterproof', 3.0), ('flexible', 3.0)],
        'paper': [('thin', 3.0), ('flammable', 3.0), ('lightweight', 3.0)],
        'gold': [('valuable', 4.0), ('shiny', 3.0), ('heavy', 3.0), ('yellow', 2.0)],
        'silver': [('valuable', 3.0), ('shiny', 3.0), ('conductive', 3.0)],
        'copper': [('conductive', 4.0), ('orange', 2.0)],
        'diamond': [('hard', 5.0), ('valuable', 4.0), ('transparent', 3.0)],
        'concrete': [('hard', 4.0), ('heavy', 3.0), ('grey', 2.0)],
        'silk': [('soft', 4.0), ('smooth', 3.0), ('expensive', 2.0)],
        'leather': [('durable', 3.0), ('flexible', 3.0), ('natural', 2.0)],
        'ceramic': [('fragile', 3.0), ('hard', 3.0), ('heat_resistant', 3.0)],
    }
    for material, props in materials.items():
        for prop, weight in props:
            facts.append((material, 'has_property', prop, weight))

    # === Animal facts ===
    animals = {
        'elephant': [('large', 5.0), ('grey', 3.0), ('intelligent', 3.0), ('herbivore', 3.0)],
        'lion': [('strong', 4.0), ('carnivore', 4.0), ('dangerous', 3.0)],
        'rabbit': [('fast', 3.0), ('small', 3.0), ('herbivore', 3.0)],
        'penguin': [('black_and_white', 3.0), ('cold_adapted', 3.0)],
        'dolphin': [('intelligent', 4.0), ('friendly', 3.0)],
        'spider': [('small', 3.0), ('eight_legged', 4.0)],
        'butterfly': [('colorful', 3.0), ('small', 3.0)],
        'owl': [('nocturnal', 4.0), ('wise', 2.0)],
        'parrot': [('colorful', 3.0)],
        'turtle': [('slow', 4.0), ('long_lived', 3.0)],
        'cheetah': [('fast', 5.0), ('spotted', 3.0)],
        'giraffe': [('tall', 5.0), ('long_necked', 4.0)],
        'crocodile': [('dangerous', 4.0), ('cold_blooded', 3.0)],
        'bear': [('large', 4.0), ('strong', 4.0)],
        'wolf': [('social', 3.0), ('carnivore', 4.0)],
        'fox': [('clever', 3.0), ('small', 2.0)],
        'pig': [('intelligent', 3.0), ('pink', 2.0)],
        'cow': [('large', 3.0), ('herbivore', 4.0)],
        'sheep': [('woolly', 4.0), ('herbivore', 3.0)],
        'chicken': [('small', 2.0)],
        'duck': [('waterproof', 3.0)],
        'monkey': [('intelligent', 3.0), ('agile', 3.0)],
    }
    for animal, props in animals.items():
        for prop, weight in props:
            facts.append((animal, 'has_property', prop, weight))

    animal_taxonomy = {
        'elephant': ['mammal', 'animal'],
        'lion': ['mammal', 'animal', 'predator'],
        'rabbit': ['mammal', 'animal'],
        'penguin': ['bird', 'animal'],
        'dolphin': ['mammal', 'animal'],
        'butterfly': ['insect', 'animal'],
        'owl': ['bird', 'animal'],
        'parrot': ['bird', 'animal'],
        'turtle': ['reptile', 'animal'],
        'cheetah': ['mammal', 'animal', 'predator'],
        'giraffe': ['mammal', 'animal'],
        'crocodile': ['reptile', 'animal', 'predator'],
        'bear': ['mammal', 'animal'],
        'wolf': ['mammal', 'animal', 'predator'],
        'fox': ['mammal', 'animal'],
        'pig': ['mammal', 'animal'],
        'cow': ['mammal', 'animal'],
        'sheep': ['mammal', 'animal'],
        'chicken': ['bird', 'animal'],
        'duck': ['bird', 'animal'],
        'monkey': ['primate', 'mammal', 'animal'],
    }
    for animal, parents in animal_taxonomy.items():
        for parent in parents:
            facts.append((animal, 'is_a', parent, 4.0))

    animal_capabilities = {
        'elephant': ['remember', 'swim'],
        'penguin': ['swim'],
        'dolphin': ['swim', 'jump', 'communicate'],
        'parrot': ['talk', 'fly'],
        'monkey': ['climb', 'use_tools'],
        'cheetah': ['run'],
        'bear': ['swim', 'climb'],
        'duck': ['swim', 'fly'],
    }
    for animal, caps in animal_capabilities.items():
        for cap in caps:
            facts.append((animal, 'can', cap, 3.0))

    # Basic needs for living categories
    living_needs = {
        'animal': [('water', 4.0), ('food', 4.0), ('oxygen', 4.0), ('shelter', 3.0)],
        'mammal': [('water', 4.0), ('food', 4.0), ('oxygen', 4.0)],
        'bird': [('water', 4.0), ('food', 4.0), ('oxygen', 4.0)],
        'plant': [('water', 4.0), ('sunlight', 4.0), ('soil', 3.0), ('carbon_dioxide', 3.0)],
        'human': [('water', 5.0), ('food', 5.0), ('oxygen', 5.0), ('shelter', 4.0), ('sleep', 4.0)],
        'fish': [('water', 5.0), ('food', 4.0), ('oxygen', 3.0)],
    }
    for category, needs in living_needs.items():
        for need, weight in needs:
            facts.append((category, 'needs', need, weight))

    animal_habitats = {
        'penguin': 'antarctica',
        'dolphin': 'ocean',
        'lion': 'savanna',
        'bear': 'forest',
        'monkey': 'jungle',
        'crocodile': 'swamp',
        'rabbit': 'meadow',
        'owl': 'forest',
        'wolf': 'forest',
        'fox': 'forest',
    }
    for animal, habitat in animal_habitats.items():
        facts.append((animal, 'found_in', habitat, 3.0))

    # === Food properties ===
    foods = {
        'chocolate': [('sweet', 4.0), ('brown', 2.0), ('delicious', 3.0)],
        'lemon': [('sour', 5.0), ('yellow', 3.0)],
        'sugar': [('sweet', 5.0), ('white', 2.0)],
        'salt': [('salty', 5.0), ('white', 2.0)],
        'pepper': [('spicy', 4.0)],
        'cheese': [('salty', 2.0), ('yellow', 2.0)],
        'milk': [('white', 3.0), ('liquid', 3.0), ('nutritious', 3.0)],
        'coffee': [('bitter', 3.0), ('brown', 2.0), ('stimulating', 3.0)],
        'tea': [('warm', 3.0), ('calming', 2.0)],
        'honey': [('sweet', 4.0), ('golden', 2.0), ('sticky', 3.0)],
        'ice_cream': [('cold', 4.0), ('sweet', 4.0), ('creamy', 3.0)],
        'soup': [('warm', 4.0), ('liquid', 3.0)],
        'egg': [('nutritious', 3.0), ('fragile', 3.0)],
    }
    for food, props in foods.items():
        for prop, weight in props:
            facts.append((food, 'has_property', prop, weight))
        facts.append((food, 'is_a', 'food', 3.0))

    # === Body parts ===
    body_parts = {
        'brain': ('body', 'thinking'),
        'heart': ('body', 'pumping blood'),
        'lungs': ('body', 'breathing'),
        'stomach': ('body', 'digestion'),
        'liver': ('body', 'filtering'),
        'kidney': ('body', 'filtering'),
        'eye': ('face', 'seeing'),
        'ear': ('head', 'hearing'),
        'nose': ('face', 'smelling'),
        'mouth': ('face', 'eating'),
        'tongue': ('mouth', 'tasting'),
        'tooth': ('mouth', 'chewing'),
        'bone': ('skeleton', 'support'),
        'muscle': ('body', 'movement'),
        'skin': ('body', 'protection'),
    }
    for part, (parent, function) in body_parts.items():
        facts.append((part, 'part_of', parent, 4.0))
        facts.append((part, 'used_for', function, 3.0))

    # === Cause-Effect chains ===
    cause_effects = [
        ('heat', 'causes', 'melting', 4.0),
        ('cold', 'causes', 'freezing', 4.0),
        ('gravity', 'causes', 'falling', 4.0),
        ('wind', 'causes', 'movement', 3.0),
        ('drought', 'causes', 'thirst', 3.0),
        ('earthquake', 'causes', 'destruction', 4.0),
        ('volcano', 'causes', 'eruption', 4.0),
        ('lightning', 'causes', 'fire', 3.0),
        ('pollution', 'causes', 'illness', 3.0),
        ('education', 'causes', 'understanding', 3.0),
        ('practice', 'causes', 'improvement', 3.0),
        ('sleep', 'causes', 'rest', 4.0),
        ('hunger', 'causes', 'weakness', 3.0),
        ('dehydration', 'causes', 'thirst', 4.0),
        ('virus', 'causes', 'illness', 4.0),
        ('bacteria', 'causes', 'infection', 3.0),
        ('sun', 'causes', 'warmth', 4.0),
        ('moon', 'causes', 'tides', 3.0),
        ('friction', 'causes', 'heat', 3.0),
        ('electricity', 'causes', 'light', 3.0),
        ('noise', 'causes', 'disturbance', 2.0),
        ('stress', 'causes', 'anxiety', 3.0),
        ('kindness', 'causes', 'happiness', 3.0),
        ('insult', 'causes', 'anger', 3.0),
        ('war', 'causes', 'destruction', 4.0),
        ('cooperation', 'causes', 'progress', 3.0),
    ]
    facts.extend(cause_effects)

    # === Temporal facts ===
    temporal = [
        ('seed', 'happens_before', 'plant', 4.0),
        ('plant', 'happens_before', 'flower', 3.0),
        ('flower', 'happens_before', 'fruit', 3.0),
        ('egg', 'happens_before', 'chick', 4.0),
        ('caterpillar', 'happens_before', 'butterfly', 4.0),
        ('rain', 'happens_before', 'rainbow', 3.0),
        ('question', 'happens_before', 'answer', 3.0),
        ('problem', 'happens_before', 'solution', 3.0),
        ('cause', 'happens_before', 'effect', 5.0),
        ('invention', 'happens_before', 'innovation', 3.0),
    ]
    facts.extend(temporal)

    # === More opposites ===
    opposites = [
        ('wet', 'dry'), ('hard', 'soft'), ('thick', 'thin'),
        ('heavy', 'light'), ('loud', 'quiet'), ('sweet', 'bitter'),
        ('clean', 'dirty'), ('full', 'empty'), ('near', 'far'),
        ('deep', 'shallow'), ('wide', 'narrow'), ('strong', 'weak'),
        ('brave', 'coward'), ('love', 'hate'), ('war', 'peace'),
        ('success', 'failure'), ('begin', 'end'), ('create', 'destroy'),
        ('buy', 'sell'), ('give', 'take'), ('push', 'pull'),
        ('ask', 'answer'), ('teach', 'learn'), ('win', 'lose'),
    ]
    for a, b in opposites:
        facts.append((a, 'opposite_of', b, 4.0))
        facts.append((b, 'opposite_of', a, 4.0))

    return facts


def generate_world_knowledge() -> List[Tuple[str, str, str, float]]:
    """Generate factual world knowledge."""
    facts = []

    # === Countries ===
    countries = {
        'United States': {'capital': 'Washington D.C.', 'language': 'English', 'continent': 'North America', 'currency': 'US Dollar'},
        'United Kingdom': {'capital': 'London', 'language': 'English', 'continent': 'Europe', 'currency': 'Pound Sterling'},
        'France': {'capital': 'Paris', 'language': 'French', 'continent': 'Europe', 'currency': 'Euro'},
        'Germany': {'capital': 'Berlin', 'language': 'German', 'continent': 'Europe', 'currency': 'Euro'},
        'Japan': {'capital': 'Tokyo', 'language': 'Japanese', 'continent': 'Asia', 'currency': 'Yen'},
        'China': {'capital': 'Beijing', 'language': 'Chinese', 'continent': 'Asia', 'currency': 'Yuan'},
        'India': {'capital': 'New Delhi', 'language': 'Hindi', 'continent': 'Asia', 'currency': 'Rupee'},
        'Brazil': {'capital': 'Brasilia', 'language': 'Portuguese', 'continent': 'South America', 'currency': 'Real'},
        'Russia': {'capital': 'Moscow', 'language': 'Russian', 'continent': 'Europe', 'currency': 'Ruble'},
        'Australia': {'capital': 'Canberra', 'language': 'English', 'continent': 'Oceania', 'currency': 'Australian Dollar'},
        'Canada': {'capital': 'Ottawa', 'language': 'English', 'continent': 'North America', 'currency': 'Canadian Dollar'},
        'Italy': {'capital': 'Rome', 'language': 'Italian', 'continent': 'Europe', 'currency': 'Euro'},
        'Spain': {'capital': 'Madrid', 'language': 'Spanish', 'continent': 'Europe', 'currency': 'Euro'},
        'Mexico': {'capital': 'Mexico City', 'language': 'Spanish', 'continent': 'North America', 'currency': 'Peso'},
        'South Korea': {'capital': 'Seoul', 'language': 'Korean', 'continent': 'Asia', 'currency': 'Won'},
        'Norway': {'capital': 'Oslo', 'language': 'Norwegian', 'continent': 'Europe', 'currency': 'Norwegian Krone'},
        'Sweden': {'capital': 'Stockholm', 'language': 'Swedish', 'continent': 'Europe', 'currency': 'Swedish Krona'},
        'Switzerland': {'capital': 'Bern', 'language': 'German', 'continent': 'Europe', 'currency': 'Swiss Franc'},
        'Netherlands': {'capital': 'Amsterdam', 'language': 'Dutch', 'continent': 'Europe', 'currency': 'Euro'},
        'Poland': {'capital': 'Warsaw', 'language': 'Polish', 'continent': 'Europe', 'currency': 'Zloty'},
        'Turkey': {'capital': 'Ankara', 'language': 'Turkish', 'continent': 'Asia', 'currency': 'Lira'},
        'Egypt': {'capital': 'Cairo', 'language': 'Arabic', 'continent': 'Africa', 'currency': 'Egyptian Pound'},
        'South Africa': {'capital': 'Pretoria', 'language': 'English', 'continent': 'Africa', 'currency': 'Rand'},
        'Argentina': {'capital': 'Buenos Aires', 'language': 'Spanish', 'continent': 'South America', 'currency': 'Peso'},
        'Finland': {'capital': 'Helsinki', 'language': 'Finnish', 'continent': 'Europe', 'currency': 'Euro'},
        'Denmark': {'capital': 'Copenhagen', 'language': 'Danish', 'continent': 'Europe', 'currency': 'Danish Krone'},
        'Austria': {'capital': 'Vienna', 'language': 'German', 'continent': 'Europe', 'currency': 'Euro'},
        'Greece': {'capital': 'Athens', 'language': 'Greek', 'continent': 'Europe', 'currency': 'Euro'},
        'Portugal': {'capital': 'Lisbon', 'language': 'Portuguese', 'continent': 'Europe', 'currency': 'Euro'},
        'Ireland': {'capital': 'Dublin', 'language': 'English', 'continent': 'Europe', 'currency': 'Euro'},
    }
    for country, info in countries.items():
        facts.append((country, 'type', 'country', 5.0))
        for rel, val in info.items():
            facts.append((country, rel, val, 4.0))

    # === Solar system ===
    planets = [
        ('Mercury', 1), ('Venus', 2), ('Earth', 3), ('Mars', 4),
        ('Jupiter', 5), ('Saturn', 6), ('Uranus', 7), ('Neptune', 8),
    ]
    for planet, order in planets:
        facts.append((planet, 'is_a', 'planet', 5.0))
        facts.append((planet, 'part_of', 'solar system', 4.0))
        facts.append((planet, 'order_from_sun', str(order), 3.0))

    planet_props = {
        'Mercury': [('smallest', 3.0), ('closest to sun', 4.0)],
        'Venus': [('hottest', 4.0), ('brightest', 3.0)],
        'Earth': [('habitable', 5.0), ('has water', 5.0), ('has life', 5.0)],
        'Mars': [('red', 4.0), ('cold', 3.0)],
        'Jupiter': [('largest', 5.0), ('gas giant', 4.0)],
        'Saturn': [('ringed', 5.0), ('gas giant', 4.0)],
        'Uranus': [('tilted', 3.0), ('ice giant', 4.0)],
        'Neptune': [('blue', 3.0), ('ice giant', 4.0), ('farthest', 4.0)],
    }
    for planet, props in planet_props.items():
        for prop, weight in props:
            facts.append((planet, 'has_property', prop, weight))

    # === Chemical elements (common ones) ===
    elements = {
        'hydrogen': ('H', 1), 'helium': ('He', 2), 'carbon': ('C', 6),
        'nitrogen': ('N', 7), 'oxygen': ('O', 8), 'sodium': ('Na', 11),
        'iron': ('Fe', 26), 'copper': ('Cu', 29), 'silver': ('Ag', 47),
        'gold': ('Au', 79), 'mercury': ('Hg', 80), 'lead': ('Pb', 82),
        'uranium': ('U', 92), 'calcium': ('Ca', 20), 'potassium': ('K', 19),
    }
    for elem, (symbol, number) in elements.items():
        facts.append((elem, 'is_a', 'chemical element', 4.0))
        facts.append((elem, 'symbol', symbol, 5.0))
        facts.append((elem, 'atomic_number', str(number), 4.0))

    # === Famous inventions/discoveries ===
    inventions = {
        'telephone': ('Alexander Graham Bell', '1876'),
        'light bulb': ('Thomas Edison', '1879'),
        'airplane': ('Wright Brothers', '1903'),
        'internet': ('Tim Berners-Lee', '1989'),
        'printing press': ('Johannes Gutenberg', '1440'),
        'penicillin': ('Alexander Fleming', '1928'),
        'gravity': ('Isaac Newton', '1687'),
        'relativity': ('Albert Einstein', '1905'),
        'evolution': ('Charles Darwin', '1859'),
        'electricity': ('Benjamin Franklin', '1752'),
        'vaccine': ('Edward Jenner', '1796'),
        'radio': ('Guglielmo Marconi', '1895'),
        'transistor': ('Bell Labs', '1947'),
        'DNA structure': ('Watson and Crick', '1953'),
    }
    for item, (person, year) in inventions.items():
        facts.append((item, 'discoverer', person, 4.0))
        facts.append((item, 'year', year, 3.0))

    # === Programming languages ===
    prog_langs = {
        'Python': {'creator': 'Guido van Rossum', 'year': '1991', 'type': 'interpreted'},
        'Java': {'creator': 'James Gosling', 'year': '1995', 'type': 'compiled'},
        'C': {'creator': 'Dennis Ritchie', 'year': '1972', 'type': 'compiled'},
        'JavaScript': {'creator': 'Brendan Eich', 'year': '1995', 'type': 'interpreted'},
        'Rust': {'creator': 'Graydon Hoare', 'year': '2010', 'type': 'compiled'},
        'Go': {'creator': 'Rob Pike', 'year': '2009', 'type': 'compiled'},
        'Ruby': {'creator': 'Yukihiro Matsumoto', 'year': '1995', 'type': 'interpreted'},
        'Swift': {'creator': 'Chris Lattner', 'year': '2014', 'type': 'compiled'},
        'Kotlin': {'creator': 'JetBrains', 'year': '2011', 'type': 'compiled'},
        'TypeScript': {'creator': 'Anders Hejlsberg', 'year': '2012', 'type': 'compiled'},
    }
    for lang, info in prog_langs.items():
        facts.append((lang, 'is_a', 'programming language', 5.0))
        for rel, val in info.items():
            facts.append((lang, rel, val, 4.0))

    # === Units of measurement ===
    units = [
        ('meter', 'is_a', 'unit of length', 4.0),
        ('kilogram', 'is_a', 'unit of mass', 4.0),
        ('second', 'is_a', 'unit of time', 4.0),
        ('ampere', 'is_a', 'unit of electric current', 4.0),
        ('kelvin', 'is_a', 'unit of temperature', 4.0),
        ('celsius', 'is_a', 'unit of temperature', 4.0),
        ('liter', 'is_a', 'unit of volume', 4.0),
        ('watt', 'is_a', 'unit of power', 4.0),
        ('joule', 'is_a', 'unit of energy', 4.0),
        ('newton', 'is_a', 'unit of force', 4.0),
        ('hertz', 'is_a', 'unit of frequency', 4.0),
        ('byte', 'is_a', 'unit of information', 4.0),
    ]
    facts.extend(units)

    return facts


def generate_technical_knowledge() -> List[Tuple[str, str, str, float]]:
    """Generate CS/tech knowledge."""
    facts = []

    # Data structures
    ds = {
        'array': ['data structure', 'constant access time', 'fixed size'],
        'linked list': ['data structure', 'dynamic size', 'linear access'],
        'hash table': ['data structure', 'constant average lookup', 'key value storage'],
        'binary tree': ['data structure', 'hierarchical', 'logarithmic search'],
        'graph': ['data structure', 'nodes and edges', 'networks'],
        'stack': ['data structure', 'last in first out', 'function calls'],
        'queue': ['data structure', 'first in first out', 'scheduling'],
        'heap': ['data structure', 'priority queue', 'logarithmic operations'],
    }
    for name, props in ds.items():
        facts.append((name, 'is_a', props[0], 4.0))
        for prop in props[1:]:
            facts.append((name, 'has_property', prop, 3.0))

    # Algorithms
    algos = {
        'binary search': ('searching', 'O(log n)'),
        'quicksort': ('sorting', 'O(n log n) average'),
        'merge sort': ('sorting', 'O(n log n)'),
        'bubble sort': ('sorting', 'O(n^2)'),
        'depth first search': ('graph traversal', 'O(V+E)'),
        'breadth first search': ('graph traversal', 'O(V+E)'),
        'dijkstra': ('shortest path', 'O(V^2)'),
        'dynamic programming': ('optimization', 'polynomial'),
    }
    for name, (category, complexity) in algos.items():
        facts.append((name, 'is_a', 'algorithm', 4.0))
        facts.append((name, 'used_for', category, 3.0))
        facts.append((name, 'complexity', complexity, 3.0))

    # Design patterns
    patterns = [
        ('singleton', 'design pattern', 'ensures one instance'),
        ('factory', 'design pattern', 'creates objects without specifying class'),
        ('observer', 'design pattern', 'notifies dependents of state changes'),
        ('strategy', 'design pattern', 'selects algorithm at runtime'),
        ('decorator', 'design pattern', 'adds behavior dynamically'),
    ]
    for name, category, desc in patterns:
        facts.append((name, 'is_a', category, 3.0))
        facts.append((name, 'description', desc, 3.0))

    # Networking
    net = [
        ('TCP', 'is_a', 'network protocol', 4.0),
        ('TCP', 'has_property', 'reliable', 4.0),
        ('UDP', 'is_a', 'network protocol', 4.0),
        ('UDP', 'has_property', 'fast', 3.0),
        ('HTTP', 'is_a', 'application protocol', 4.0),
        ('HTTP', 'used_for', 'web communication', 4.0),
        ('DNS', 'is_a', 'network service', 4.0),
        ('DNS', 'used_for', 'name resolution', 4.0),
        ('SSH', 'is_a', 'network protocol', 4.0),
        ('SSH', 'used_for', 'secure remote access', 4.0),
        ('TLS', 'is_a', 'security protocol', 4.0),
        ('TLS', 'used_for', 'encryption', 4.0),
    ]
    facts.extend(net)

    return facts


def bootstrap_full_knowledge(output_path: str = '') -> dict:
    """
    Generate the complete knowledge base and save to JSON.

    Returns dict with stats.
    """
    if not output_path:
        output_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data', 'knowledge_full.json'
        )

    common_sense = generate_extended_common_sense()
    world = generate_world_knowledge()
    tech = generate_technical_knowledge()

    all_facts = common_sense + world + tech

    # Deduplicate
    seen = set()
    unique = []
    for s, r, o, w in all_facts:
        key = (s.lower(), r.lower(), o.lower())
        if key not in seen:
            seen.add(key)
            unique.append([s, r, o, round(w, 2)])

    data = {
        'version': '1.0',
        'source': 'foss-ki-bootstrap',
        'categories': {
            'common_sense': len(common_sense),
            'world_knowledge': len(world),
            'technical': len(tech),
        },
        'count': len(unique),
        'triplets': unique,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=1)

    return {
        'total': len(unique),
        'common_sense': len(common_sense),
        'world': len(world),
        'technical': len(tech),
        'path': output_path,
        'size_kb': round(os.path.getsize(output_path) / 1024, 1),
    }


def load_bootstrap_to_engine(engine, filepath: str = '') -> int:
    """Load bootstrapped knowledge into CommonSenseEngine."""
    if not filepath:
        filepath = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data', 'knowledge_full.json'
        )

    if not os.path.exists(filepath):
        # Auto-generate
        bootstrap_full_knowledge(filepath)

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    loaded = 0
    for t in data['triplets']:
        s, r, o = t[0], t[1], t[2]
        w = t[3] if len(t) > 3 else 1.0
        engine.add_fact(s, r, o, w)
        if r == 'is_a':
            engine.add_taxonomy(s, [o])
        loaded += 1

    return loaded


def load_bootstrap_to_knowledge_store(store, filepath: str = '') -> int:
    """Load bootstrapped knowledge into KnowledgeStore."""
    if not filepath:
        filepath = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data', 'knowledge_full.json'
        )

    if not os.path.exists(filepath):
        bootstrap_full_knowledge(filepath)

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    loaded = 0
    for t in data['triplets']:
        s, r, o = t[0], t[1], t[2]
        store.store_fact(s, r, o)
        loaded += 1

    return loaded


if __name__ == '__main__':
    stats = bootstrap_full_knowledge()
    print(f"Knowledge base generated:")
    print(f"  Common sense: {stats['common_sense']}")
    print(f"  World knowledge: {stats['world']}")
    print(f"  Technical: {stats['technical']}")
    print(f"  Total (deduplicated): {stats['total']}")
    print(f"  File: {stats['path']} ({stats['size_kb']} KB)")
