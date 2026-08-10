"""
Auto Knowledge Extraction — Text → Triplets → Hopfield
========================================================
Regelbasierte Information Extraction. Kein Transformer.

Patterns:
1. "X is the Y of Z"          → (Z, Y, X)
2. "X is Y"                   → (X, type/identity, Y)
3. "X verb Y"                 → (X, verb, Y)
4. "X, the Y of Z"            → (Z, Y, X)
5. "X was born in Y"          → (X, birthplace, Y)
6. "X founded Y in Z"         → (Y, founder, X), (Y, founded, Z)

Pipeline: Text → Sentences → Pattern Match → Triplets → Hopfield
"""

import re
from typing import List, Tuple, Optional


class TripletExtractor:
    """
    Rule-based (Subject, Relation, Object) extraction from text.

    No ML, no transformer, no dependencies beyond stdlib.
    Uses regex patterns over normalized sentence structure.
    """

    # Flexible entity pattern: handles "France", "PS-Lifted", "FOSS-KI", "GPT-4",
    # "David Foss", "Guido van Rossum", multi-word with hyphens/digits
    # Must start with uppercase letter
    _E = r'[A-Z][\w-]+(?:\s+(?:[a-z]+\s+)*[A-Z][\w-]+)*'
    # Same but also allows lowercase-start objects (for descriptions)
    _EO = r'[A-Za-z][\w-]+(?:\s+[\w-]+)*'

    # Relation patterns: (regex, extraction_fn, description)
    # Group names: s=subject, r=relation, o=object
    PATTERNS = [
        # Wikipedia-style: "X's capital and largest city is Y" → (X, capital, Y)
        (r"(?P<s>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'s\s+(?P<r>capital)(?:\s+(?:and|,)\s+[^,]+?)?\s+is\s+(?P<o>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
         lambda m: (m.group('s'), m.group('r'), m.group('o')),
         "X's capital is Y"),

        # Wikipedia-style: "The nation's/country's capital ... is Y" — needs article context
        # "Its capital ... is Y" — pronoun, skip (handled by context in extract_from_text)

        # Wikipedia-style: "Y is the capital and ... of X" → (X, capital, Y)
        (r'(?P<o>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+is\s+the\s+(?P<r>capital)(?:\s+(?:and|,)\s+[^,]+?)?\s+of\s+(?P<s>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
         lambda m: (m.group('s'), m.group('r'), m.group('o')),
         "Y is the capital and ... of X"),

        # Wikipedia-style: "Y is the country's/nation's capital" — possessive subject
        (r"(?P<o>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+is\s+the\s+(?:country|nation|state|republic|kingdom)'?s?\s+(?P<r>capital)",
         lambda m: (None, m.group('r'), m.group('o')),  # subject needs context resolution
         "Y is the country's capital"),

        # Wikipedia-style: "Its capital is Y" or "Its capital, ... is Y"
        # or "Its capital and largest city is Y"
        # Can only extract object; subject must be resolved from context
        (r"[Ii]ts\s+(?P<r>capital)(?:\s+and\s+[^,]+?|\s*,\s*[^,]+?)?\s+is\s+(?P<o>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
         lambda m: (None, m.group('r'), m.group('o')),
         "Its capital is Y"),

        # "which hosts the capital, X" → (context, capital, X)
        (r'(?:hosts|houses)\s+the\s+(?P<r>capital),?\s+(?P<o>[A-Z][a-záéíóú]+(?:\s+[A-Z][a-z]+)*)',
         lambda m: (None, m.group('r'), m.group('o')),
         "hosts the capital, X"),

        # "X is the capital of Y" → (Y, capital, X)
        (r'(?P<o>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+is\s+the\s+(?P<r>\w+)\s+of\s+(?P<s>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
         lambda m: (m.group('s'), m.group('r'), m.group('o')),
         "X is the R of Y"),

        # "The capital of X is Y" → (X, capital, Y)
        (r'[Tt]he\s+(?P<r>\w+)\s+of\s+(?P<s>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+is\s+(?P<o>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
         lambda m: (m.group('s'), m.group('r'), m.group('o')),
         "The R of X is Y"),

        # "X, known as Y" → (X, known_as, Y)
        (r'(?P<s>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*),?\s+(?:also\s+)?known\s+as\s+(?P<o>[A-Z][\w\s]+?)(?:\.|,|$)',
         lambda m: (m.group('s'), 'known_as', m.group('o').strip()),
         "X known as Y"),

        # "X is a/an Y" → (X, type, Y) — also match at end of sentence
        (r'(?P<s>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+is\s+(?:a|an)\s+(?P<o>[a-z][\w\s]+?)(?:\s+(?:that|which|in|on|of|and)|[,.]|$)',
         lambda m: (m.group('s'), 'type', m.group('o').strip()),
         "X is a Y"),

        # "X was founded/created/invented/built by Y in Z" → (X, relation, Y) + (X, founded, Z)
        (r'(?P<s>(?:[Tt]he\s+)?[A-Z][\w-]+(?:\s+[\w-]+)*?)\s+was\s+(?P<v>founded|created|invented|built)\s+by\s+(?P<o>[A-Z][a-z]+(?:\s+[a-z]+)*(?:\s+[A-Z][a-z]+)*)\s+in\s+(?P<y>\d{4})',
         lambda m: [(m.group('s'), {'founded': 'founder', 'created': 'creator', 'invented': 'inventor', 'built': 'creator'}[m.group('v')], m.group('o')),
                     (m.group('s'), 'founded', m.group('y'))],
         "X was founded/created/invented/built by Y in Z"),

        # "X was founded in Y" → (X, founded, Y)
        (r'(?P<s>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+was\s+founded\s+in\s+(?P<o>\d{4}|[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
         lambda m: (m.group('s'), 'founded', m.group('o')),
         "X was founded in Y"),

        # "X was born in Y" → (X, birthplace, Y)
        (r'(?P<s>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+was\s+born\s+in\s+(?P<o>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
         lambda m: (m.group('s'), 'birthplace', m.group('o')),
         "X was born in Y"),

        # "X invented Y" → (Y, inventor, X)
        (r'(?P<s>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+invented\s+(?:the\s+)?(?P<o>[A-Za-z][\w\s]+?)(?:\s+in|\.|,|$)',
         lambda m: (m.group('o').strip(), 'inventor', m.group('s')),
         "X invented Y"),

        # "X discovered Y" → (Y, discoverer, X)
        (r'(?P<s>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+discovered\s+(?:the\s+)?(?P<o>[A-Za-z][\w\s]+?)(?:\s+in|\.|,|$)',
         lambda m: (m.group('o').strip(), 'discoverer', m.group('s')),
         "X discovered Y"),

        # "X created Y" → (Y, creator, X) — supports multi-word names like "Guido van Rossum"
        (r'(?P<s>[A-Z][a-z]+(?:\s+(?:[a-z]+\s+)*[A-Z][a-z]+)*)\s+created\s+(?:the\s+)?(?P<o>[A-Za-z][\w\s]+?)(?:\s+in|\.|,|$)',
         lambda m: (m.group('o').strip(), 'creator', m.group('s')),
         "X created Y"),

        # "X was created by Y" → (X, creator, Y)
        (r'(?P<s>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+was\s+created\s+by\s+(?P<o>[A-Z][a-z]+(?:\s+[a-z]+)*(?:\s+[A-Z][a-z]+)*)',
         lambda m: (m.group('s'), 'creator', m.group('o')),
         "X was created by Y"),

        # "X is located in Y" → (X, location, Y)
        (r'(?P<s>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+is\s+located\s+in\s+(?P<o>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
         lambda m: (m.group('s'), 'location', m.group('o')),
         "X is located in Y"),

        # "X has a population of Y" → (X, population, Y)
        (r'(?P<s>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+has\s+a\s+population\s+of\s+(?P<o>[\d,.\s]+\w*)',
         lambda m: (m.group('s'), 'population', m.group('o').strip()),
         "X has a population of Y"),

        # "X wrote Y" → (Y, author, X)
        (r'(?P<s>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+wrote\s+(?P<o>[A-Z][\w\s]+?)(?:\s+in|\.|,|$)',
         lambda m: (m.group('o').strip(), 'author', m.group('s')),
         "X wrote Y"),

        # "X is the Y" where Y is capitalized → (X, identity, the Y)
        (r'(?P<s>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+is\s+the\s+(?P<o>[A-Z][a-z]+(?:\s+[A-Za-z]+)*?)(?:\.|,|$)',
         lambda m: (m.group('s'), 'identity', m.group('o').strip()),
         "X is the Y (identity with article)"),

        # "X is Y" (capitalized Y = identity) → (X, identity, Y)
        (r'(?P<s>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+is\s+(?P<o>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
         lambda m: (m.group('s'), 'identity', m.group('o')),
         "X is Y (identity)"),

        # "X contains Y" → (Y, part_of, X)
        (r'(?P<s>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+contains\s+(?P<o>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
         lambda m: (m.group('o'), 'part_of', m.group('s')),
         "X contains Y"),

        # "X borders Y" → (X, borders, Y)
        (r'(?P<s>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+borders\s+(?P<o>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
         lambda m: (m.group('s'), 'borders', m.group('o')),
         "X borders Y"),

        # "X speaks Y" / "The official language of X is Y"
        (r'[Tt]he\s+official\s+language\s+of\s+(?P<s>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+is\s+(?P<o>[A-Z][a-z]+)',
         lambda m: (m.group('s'), 'language', m.group('o')),
         "official language of X is Y"),

        # "X, the R of Z, ..." → (Z, R, X)  (appositive pattern)
        (r'(?P<o>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*),\s+the\s+(?P<r>\w+)\s+of\s+(?P<s>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
         lambda m: (m.group('s'), m.group('r'), m.group('o')),
         "X, the R of Y (appositive)"),

        # "X has a/the Y of Z" → (X, Y, Z) — handles "formula of", "symbol of"
        (r'(?P<s>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+has\s+(?:a|the)\s+(?P<r>\w+)\s+of\s+(?P<o>\S+)',
         lambda m: (m.group('s'), m.group('r'), m.group('o').rstrip('.,;:')),
         "X has a R of Y"),

        # Multi-word entity support for "X is the R of Y" where Y has multiple words
        (r'(?P<o>[A-Z][\w]+(?:\s+[A-Z][\w]+)*)\s+is\s+the\s+(?P<r>\w+)\s+of\s+the\s+(?P<s>[A-Z][\w]+(?:\s+[A-Z][\w]+)*)',
         lambda m: (m.group('s'), m.group('r'), m.group('o')),
         "X is the R of the Y"),

        # "X is Y" where Y is lowercase (property, not identity)
        (r'(?P<s>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+is\s+the\s+(?P<o>[a-z][\w\s]+?)(?:\.|,|$)',
         lambda m: (m.group('s'), 'description', m.group('o').strip()),
         "X is the Y (description)"),

        # === Flexible entity patterns (handles PS-Lifted, FOSS-KI, GPT-4, etc.) ===

        # "X invented Y" with flexible entities
        (r'(?P<s>[A-Z][\w-]+(?:\s+(?:[a-z]+\s+)*[A-Z][\w-]+)*)\s+invented\s+(?:the\s+)?(?P<o>[A-Z][\w-]+(?:\s+[\w-]+)*?)(?:\s+in\s|\.|,|$)',
         lambda m: (m.group('o').strip(), 'inventor', m.group('s')),
         "X invented Y (flexible)"),

        # "X was invented by Y"
        (r'(?P<s>[A-Z][\w-]+(?:\s+[\w-]+)*)\s+was\s+invented\s+by\s+(?P<o>[A-Z][\w-]+(?:\s+(?:[a-z]+\s+)*[A-Z][\w-]+)*)',
         lambda m: (m.group('s'), 'inventor', m.group('o')),
         "X was invented by Y"),

        # "... invented by Y" mid-sentence (e.g., "X is a thing invented by Y")
        (r'invented\s+by\s+(?P<o>[A-Z][\w-]+(?:\s+(?:[a-z]+\s+)*[A-Z][\w-]+)*)',
         lambda m: (None, 'inventor', m.group('o')),
         "... invented by Y (mid-sentence, needs context)"),

        # "... created by Y" / "... developed by Y" / "... designed by Y" mid-sentence
        (r'(?:created|developed|designed|built)\s+by\s+(?P<o>[A-Z][\w-]+(?:\s+(?:[a-z]+\s+)*[A-Z][\w-]+)*)',
         lambda m: (None, 'creator', m.group('o')),
         "... created/developed by Y (mid-sentence, needs context)"),

        # "X is a/an Y" with flexible entities (FOSS-KI is a hybrid AI, PS-Lifted is a Markov chain)
        (r'(?P<s>[A-Z][\w-]+(?:\s+[A-Z][\w-]+)*)\s+is\s+(?:a|an)\s+(?P<o>[A-Za-z][\w\s-]+?)(?:\s+(?:that|which|in|on|of|and|built|using|based|invented|created|designed|developed)|[,.]|$)',
         lambda m: (m.group('s'), 'type', m.group('o').strip()),
         "X is a Y (flexible)"),

        # "X uses Y" / "X uses Y for Z" → (X, uses, Y)
        (r'(?P<s>[A-Z][\w-]+(?:\s+[A-Z][\w-]+)*)\s+uses\s+(?P<o>[A-Za-z][\w\s-]+?)(?:\s+(?:for|to|as|in|and)|[,.]|$)',
         lambda m: (m.group('s'), 'uses', m.group('o').strip()),
         "X uses Y"),

        # "X is based on Y" → (X, based_on, Y)
        (r'(?P<s>[A-Z][\w-]+(?:\s+[A-Z][\w-]+)*)\s+is\s+based\s+on\s+(?P<o>[A-Za-z][\w\s-]+?)(?:\s+(?:and|with|for)|[,.]|$)',
         lambda m: (m.group('s'), 'based_on', m.group('o').strip()),
         "X is based on Y"),

        # "X was built by Y" / "X was designed by Y" / "X was developed by Y"
        (r'(?P<s>[A-Z][\w-]+(?:\s+[\w-]+)*)\s+was\s+(?:built|designed|developed|made)\s+by\s+(?P<o>[A-Z][\w-]+(?:\s+(?:[a-z]+\s+)*[A-Z][\w-]+)*)',
         lambda m: (m.group('s'), 'creator', m.group('o')),
         "X was built/designed/developed by Y"),

        # "X belongs to Y" / "X is part of Y"
        (r'(?P<s>[A-Z][\w-]+(?:\s+[\w-]+)*)\s+(?:belongs\s+to|is\s+part\s+of)\s+(?P<o>[A-Z][\w-]+(?:\s+[\w-]+)*)',
         lambda m: (m.group('s'), 'part_of', m.group('o')),
         "X belongs to / is part of Y"),

        # "X runs on Y" / "X works on Y"
        (r'(?P<s>[A-Z][\w-]+(?:\s+[A-Z][\w-]+)*)\s+(?:runs|works)\s+on\s+(?P<o>[A-Za-z][\w\s-]+?)(?:\s+(?:and|with)|[,.]|$)',
         lambda m: (m.group('s'), 'platform', m.group('o').strip()),
         "X runs/works on Y"),

        # "X supports Y" → (X, supports, Y)
        (r'(?P<s>[A-Z][\w-]+(?:\s+[A-Z][\w-]+)*)\s+supports\s+(?P<o>[A-Za-z][\w\s-]+?)(?:\s+(?:and|with|for|through)|[,.]|$)',
         lambda m: (m.group('s'), 'supports', m.group('o').strip()),
         "X supports Y"),

        # "X provides Y" → (X, provides, Y)
        (r'(?P<s>[A-Z][\w-]+(?:\s+[A-Z][\w-]+)*)\s+provides\s+(?P<o>[A-Za-z][\w\s-]+?)(?:\s+(?:and|with|for|through|to)|[,.]|$)',
         lambda m: (m.group('s'), 'provides', m.group('o').strip()),
         "X provides Y"),

        # === Passive voice patterns ===

        # "X was discovered by Y" → (X, discoverer, Y)
        (r'(?P<s>[A-Z][\w-]+(?:\s+[A-Z][\w-]+)*)\s+was\s+discovered\s+by\s+(?P<o>[A-Z][\w-]+(?:\s+(?:[a-z]+\s+)*[A-Z][\w-]+)*)',
         lambda m: (m.group('s'), 'discoverer', m.group('o')),
         "X was discovered by Y"),

        # "X was founded by Y" → (X, founder, Y)
        (r'(?P<s>[A-Z][\w-]+(?:\s+[A-Z][\w-]+)*)\s+was\s+founded\s+by\s+(?P<o>[A-Z][\w-]+(?:\s+(?:[a-z]+\s+)*[A-Z][\w-]+)*)',
         lambda m: (m.group('s'), 'founder', m.group('o')),
         "X was founded by Y"),

        # "X was written by Y" → (X, author, Y)
        (r'(?P<s>[A-Z][\w-]+(?:\s+[A-Z][\w-]+)*)\s+was\s+written\s+by\s+(?P<o>[A-Z][\w-]+(?:\s+(?:[a-z]+\s+)*[A-Z][\w-]+)*)',
         lambda m: (m.group('s'), 'author', m.group('o')),
         "X was written by Y"),

        # === Verb patterns ===

        # "X consists of Y" → (X, consists_of, Y)
        (r'(?P<s>[A-Z][\w-]+(?:\s+[A-Z][\w-]+)*)\s+consists\s+of\s+(?P<o>[A-Za-z][\w\s-]+?)(?:[,.]|$)',
         lambda m: (m.group('s'), 'consists_of', m.group('o').strip()),
         "X consists of Y"),

        # "X produces Y" / "X generates Y" / "X requires Y"
        (r'(?P<s>[A-Z][\w-]+(?:\s+[A-Z][\w-]+)*)\s+(?:produces|generates|requires|needs|includes|features)\s+(?P<o>[A-Za-z][\w\s-]+?)(?:\s+(?:and|with|for|to)|[,.]|$)',
         lambda m: (m.group('s'), 'has', m.group('o').strip()),
         "X produces/generates/requires Y"),

        # "X connects to Y" / "X leads to Y" / "X flows through Y"
        (r'(?P<s>[A-Z][\w-]+(?:\s+[A-Z][\w-]+)*)\s+(?:connects?\s+to|leads?\s+to|flows?\s+through|flows?\s+into)\s+(?P<o>[A-Z][\w-]+(?:\s+[A-Z][\w-]+)*)',
         lambda m: (m.group('s'), 'connects_to', m.group('o')),
         "X connects/leads/flows to Y"),

        # "X is built on Y" → (X, based_on, Y)
        (r'(?P<s>[A-Z][\w-]+(?:\s+[A-Z][\w-]+)*)\s+is\s+built\s+on\s+(?:the\s+)?(?P<o>[A-Za-z][\w\s-]+?)(?:\s+(?:and|with|for)|[,.]|$)',
         lambda m: (m.group('s'), 'based_on', m.group('o').strip()),
         "X is built on Y"),

        # "X launched Y" / "X released Y" / "X introduced Y"
        (r'(?P<s>[A-Z][\w-]+(?:\s+[A-Z][\w-]+)*)\s+(?:launched|released|introduced|developed|published)\s+(?:the\s+)?(?P<o>[A-Z][\w-]+(?:\s+[A-Z][\w-]+)*)(?:\s+(?:in|from|at)|[,.]|$)',
         lambda m: (m.group('o').strip(), 'creator', m.group('s')),
         "X launched/released Y"),

        # === Temporal patterns ===

        # "In YEAR, X verb Y" → extract (Y, creator, X) and (Y, date, YEAR)
        (r'[Ii]n\s+(?P<year>\d{4}),?\s+(?P<s>[A-Z][\w-]+(?:\s+[A-Z][\w-]+)*)\s+(?:launched|created|discovered|invented|built|released|introduced)\s+(?:the\s+)?(?P<o>[A-Z][\w-]+(?:\s+[A-Z][\w-]+)*)',
         lambda m: (m.group('o').strip(), 'creator', m.group('s')),
         "In YEAR, X verb Y"),

        # === List patterns (X borders Y, Z, and W) ===

        # "X borders Y, Z, and W" → multiple (X, borders, Y), (X, borders, Z), etc.
        # Handled specially: only extract first item, context handles rest
        (r'(?P<s>[A-Z][\w-]+(?:\s+[A-Z][\w-]+)*)\s+borders\s+(?P<o>[A-Z][\w-]+)',
         lambda m: (m.group('s'), 'borders', m.group('o')),
         "X borders Y (first in list)"),

        # "X has a length/height/area of N" → (X, length/height/area, N)
        (r'(?P<s>[A-Z][\w-]+(?:\s+[A-Z][\w-]+)*)\s+has\s+(?:a\s+)?(?P<r>length|height|area|depth|width|diameter|radius|distance|speed|mass|weight|size|volume|capacity|span)\s+of\s+(?P<o>[\d,.\s]+\w*)',
         lambda m: (m.group('s'), m.group('r'), m.group('o').strip()),
         "X has a length of N"),

        # "X is the longest/tallest/largest Y in Z" → (X, known_for, "longest Y in Z")
        (r'(?P<s>[A-Z][\w-]+(?:\s+[A-Z][\w-]+)*)\s+is\s+the\s+(?P<o>(?:longest|tallest|largest|smallest|fastest|oldest|deepest|highest|hottest|coldest|biggest|most\s+\w+)\s+[\w\s]+?)(?:\.|,|$)',
         lambda m: (m.group('s'), 'known_for', m.group('o').strip()),
         "X is the longest/tallest Y"),

        # "X is produced by Y" → (X, creator, Y)
        (r'(?P<s>[A-Z][\w-]+(?:\s+[A-Z][\w-]+)*)\s+is\s+produced\s+by\s+(?:the\s+)?(?P<o>[A-Z][\w-]+(?:\s+[\w-]+)*)',
         lambda m: (m.group('s'), 'creator', m.group('o').strip()),
         "X is produced by Y"),
    ]

    # Common relation aliases → canonical form
    RELATION_NORMALIZE = {
        'capital': 'capital',
        'president': 'leader',
        'prime_minister': 'leader',
        'king': 'leader',
        'queen': 'leader',
        'ceo': 'leader',
        'founder': 'founder',
        'inventor': 'inventor',
        'discoverer': 'discoverer',
        'creator': 'creator',
        'created': 'creator',
        'author': 'author',
        'birthplace': 'birthplace',
        'born_in': 'birthplace',
        'location': 'location',
        'located_in': 'location',
        'identity': 'identity',
        'type': 'type',
        'known_as': 'known_as',
        'population': 'population',
        'language': 'language',
        'borders': 'borders',
        'part_of': 'part_of',
        'founded': 'founded',
        'currency': 'currency',
        'symbol': 'symbol',
        'formula': 'formula',
        'continent': 'location',
        'region': 'location',
        'area': 'location',
        'uses': 'uses',
        'based_on': 'based_on',
        'platform': 'platform',
        'supports': 'supports',
        'provides': 'provides',
    }

    def __init__(self):
        self._compiled = [(re.compile(pat), fn, desc)
                         for pat, fn, desc in self.PATTERNS]

    @staticmethod
    def _normalize_unicode(text: str) -> str:
        """Normalize accented characters to ASCII for regex matching."""
        import unicodedata
        # Decompose then drop combining marks
        nfkd = unicodedata.normalize('NFKD', text)
        return ''.join(c for c in nfkd if not unicodedata.combining(c))

    def extract_from_sentence(self, sentence: str, context_subject: str = None
                             ) -> List[Tuple[str, str, str]]:
        """Extract all triplets from a single sentence.

        Args:
            sentence: The sentence to extract from.
            context_subject: Subject to use when pattern returns None
                            (for pronoun resolution like "Its capital is X").
        """
        # Normalize unicode for regex matching but keep original for output
        sentence_norm = self._normalize_unicode(sentence)
        triplets = []
        seen = set()

        for pattern, extract_fn, desc in self._compiled:
            for match in pattern.finditer(sentence_norm):
                try:
                    result = extract_fn(match)
                    # Support patterns returning multiple triplets
                    if isinstance(result, list):
                        multi = result
                    else:
                        multi = [result]

                    for s, r, o in multi:
                        # Resolve None subjects from context
                        if s is None:
                            if context_subject:
                                s = context_subject
                            else:
                                continue

                        # Clean
                        s = s.strip().rstrip('.,;:')
                        r = r.strip().lower().replace(' ', '_')
                        o = o.strip().rstrip('.,;:')

                        # Normalize relation
                        r = self.RELATION_NORMALIZE.get(r, r)

                        # Skip empty or too short
                        if len(s) < 2 or len(o) < 2:
                            continue

                        # Skip garbage subjects (pronouns, prepositions, adverbs)
                        if s.lower() in ('it', 'its', 'he', 'she', 'they', 'this', 'that',
                                         'these', 'those', 'with', 'from', 'the', 'there',
                                         'here', 'where', 'when', 'while', 'which', 'what',
                                         'however', 'although', 'because', 'since', 'after',
                                         'before', 'during', 'between', 'within', 'under',
                                         'over', 'about', 'through', 'into', 'upon', 'at',
                                         'by', 'for', 'on', 'in', 'to', 'as', 'or', 'an',
                                         'no', 'not', 'but', 'if', 'so', 'up', 'out'):
                            continue

                        # Deduplicate
                        key = (s.lower(), r, o.lower())
                        if key not in seen:
                            seen.add(key)
                            triplets.append((s, r, o))
                except (IndexError, AttributeError):
                    continue

        return triplets

    def extract_from_text(self, text: str) -> List[Tuple[str, str, str]]:
        """Extract triplets from multi-sentence text.

        Tracks context subject for pronoun resolution:
        - Article headers (# Title) set context
        - First capitalized entity in a paragraph updates context
        - "Its/The country's" patterns use context subject
        """
        all_triplets = []
        seen = set()
        context_subject = None

        # First split by lines to catch headers, then by sentences
        lines = text.split('\n')
        sentences = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Article header sets context: "# France" → context = "France"
            header = re.match(r'^#\s+(.+)$', line)
            if header:
                # Insert a context marker
                sentences.append(('__CONTEXT__', header.group(1).strip()))
                continue
            # Split line into sentences
            for sent in re.split(r'(?<=[.!?])\s+', line):
                sent = sent.strip()
                if len(sent) >= 10:
                    sentences.append(('__SENT__', sent))

        for kind, value in sentences:
            if kind == '__CONTEXT__':
                context_subject = value
                continue

            sentence = value

            # Two-pass extraction:
            # Pass 1: extract with current context
            extracted = self.extract_from_sentence(sentence, context_subject)
            for s, r, o in extracted:
                key = (s.lower(), r, o.lower())
                if key not in seen:
                    seen.add(key)
                    all_triplets.append((s, r, o))

            # Pass 2: if we got subjects from pass 1, re-extract with them
            # as context to resolve any None-subject patterns
            subjects_found = [s for s, r, o in extracted]
            if subjects_found and not context_subject:
                re_extracted = self.extract_from_sentence(
                    sentence, subjects_found[0])
                for s, r, o in re_extracted:
                    key = (s.lower(), r, o.lower())
                    if key not in seen:
                        seen.add(key)
                        all_triplets.append((s, r, o))

            # Update context: Wikipedia pattern detection
            # "[Entity], officially the..." or "[Entity] is a/was a..."
            # = paragraph opener, sets context for all following sentences
            para_opener = re.match(
                r'^([A-Z][\w-]+(?:\s+[A-Z][\w-]+)*)'
                r'(?:\s*,\s*(?:officially|formerly|also)\s|\s+(?:is|was|are|were)\s+(?:a|an|the)\s)',
                sentence
            )
            if para_opener:
                context_subject = para_opener.group(1)
            else:
                # Fallback: first named entity updates context
                first_entity = re.match(r'^([A-Z][\w-]+(?:\s+[A-Z][\w-]+)*)', sentence)
                if first_entity:
                    entity = first_entity.group(1)
                    if entity.lower() not in ('the', 'it', 'its', 'this', 'that', 'these',
                                              'those', 'there', 'however', 'although',
                                              'while', 'since', 'after', 'before',
                                              'during', 'between', 'within', 'with',
                                              'from', 'under', 'over', 'about',
                                              'through', 'into', 'upon', 'among',
                                              'around', 'across', 'along', 'against',
                                              'toward', 'towards', 'beyond', 'near',
                                              'both', 'each', 'every', 'many',
                                              'most', 'some', 'several', 'much',
                                              'such', 'like', 'just', 'only',
                                              'also', 'even', 'still', 'already',
                                              'yet', 'now', 'then', 'here',
                                              'today', 'according', 'following',
                                              'originally', 'subsequently',
                                              'historically', 'approximately',
                                              'at', 'by', 'for', 'on', 'in',
                                              'to', 'as', 'or', 'an', 'no',
                                              'not', 'but', 'if', 'so', 'up',
                                              'out', 'until', 'despite'):
                        context_subject = entity

        return all_triplets

    def extract_and_store(self, text: str, knowledge_store) -> int:
        """
        Extract triplets from text and store directly in KnowledgeStore.

        The online learning loop:
        Text → Extract → Store → Immediately queryable.

        Returns: number of new facts stored.
        """
        triplets = self.extract_from_text(text)

        # Check for duplicates against existing facts
        existing = set()
        for s, r, o in knowledge_store.facts:
            existing.add((s.lower(), r.lower(), o.lower()))

        new_count = 0
        for s, r, o in triplets:
            key = (s.lower(), r.lower(), o.lower())
            if key not in existing:
                knowledge_store.store_fact(s, r, o)
                existing.add(key)
                new_count += 1

        return new_count
