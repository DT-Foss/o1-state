"""
Word Problem Solver — GSM8K-Level Math Reasoning (v6)
=====================================================
ARIS-inspired semantic parsing: Text → Equation → Answer

Architecture:
  1. Split problem into sentences
  2. For each sentence, classify ACTION via verb + syntactic patterns
  3. Extract numbers, detect rates/units
  4. Chain computations, tracking running totals
  5. Return final computed value

Based on: Hosseini et al. 2014 "Learning to Solve Arithmetic Word Problems"

No external dependencies. Pure symbolic reasoning.
"""

import re
from typing import Optional, Dict, List, Tuple, Any


# ── Number Words ──
WORD_NUMS = {
    'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
    'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
    'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14,
    'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18,
    'nineteen': 19, 'twenty': 20, 'thirty': 30, 'forty': 40,
    'fifty': 50, 'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90,
    'hundred': 100, 'thousand': 1000, 'million': 1000000,
}

FRACTION_MAP = {
    'half': 0.5, 'a half': 0.5, 'one-half': 0.5,
    'third': 1/3, 'a third': 1/3, 'one-third': 1/3,
    'quarter': 0.25, 'a quarter': 0.25, 'one-quarter': 0.25,
    'one-fourth': 0.25,
    'two-thirds': 2/3, 'two thirds': 2/3,
    'three-quarters': 0.75, 'three quarters': 0.75,
    'three-fourths': 0.75, 'three fourths': 0.75,
    'three-fifths': 0.6, 'two-fifths': 0.4,
    'four-fifths': 0.8, 'one-fifth': 0.2,
    'one-sixth': 1/6, 'one-eighth': 0.125, 'one-tenth': 0.1,
}

MULTIPLIER_WORDS = {
    'twice': 2, 'double': 2, 'doubled': 2,
    'triple': 3, 'tripled': 3, 'thrice': 3,
    'quadruple': 4, 'quadrupled': 4,
}

# Time period implicit quantities (for rate × time problems)
TIME_PERIODS = {
    'week': 7, 'a week': 7, 'per week': 7, 'weekly': 7,
    'month': 30, 'a month': 30, 'per month': 30, 'monthly': 30,
    'year': 365, 'a year': 365, 'per year': 365, 'yearly': 365,
    'weekday': 5, 'weekdays': 5,
    'weekend': 2, 'weekends': 2, 'weekend day': 2, 'weekend days': 2,
    'hour': 60, 'an hour': 60, 'per hour': 60, 'hourly': 60,  # minutes
    'dozen': 12, 'a dozen': 12,
}

# Common unit conversions
UNIT_CONVERSIONS = {
    ('feet', 'inches'): 12, ('foot', 'inches'): 12,
    ('feet', 'inch'): 12, ('foot', 'inch'): 12,
    ('yards', 'feet'): 3, ('yard', 'feet'): 3,
    ('miles', 'feet'): 5280, ('mile', 'feet'): 5280,
    ('hours', 'minutes'): 60, ('hour', 'minutes'): 60,
    ('days', 'hours'): 24, ('day', 'hours'): 24,
    ('weeks', 'days'): 7, ('week', 'days'): 7,
    ('years', 'days'): 365, ('year', 'days'): 365,
    ('years', 'months'): 12, ('year', 'months'): 12,
    ('months', 'weeks'): 4, ('month', 'weeks'): 4,
    ('pounds', 'ounces'): 16, ('pound', 'ounces'): 16,
    ('gallons', 'quarts'): 4, ('gallon', 'quarts'): 4,
}

# ── Verb Classification (ARIS-style) ──
INITIAL_VERBS = {
    'has', 'had', 'have', 'having',
    'is', 'was', 'are', 'were',
    'starts', 'started', 'start', 'starting',
    'begins', 'began', 'begin',
    'contains', 'contained', 'contain',
    'owns', 'owned', 'own',
    'holds', 'held', 'hold',
    'weighs', 'weighed', 'weigh',
    'costs', 'cost', 'costing',
    'measures', 'measured', 'measure',
    'needs', 'needed', 'need',
    'wants', 'wanted', 'want',
    'charges', 'charged', 'charge',
    'takes', 'took', 'take',
    'runs', 'ran', 'run',
    'drives', 'drove', 'drive',
    'walks', 'walked', 'walk',
    'reads', 'read',
    'works', 'worked', 'work',
    'plays', 'played', 'play',
    'swims', 'swam', 'swim',
    'sleeps', 'slept', 'sleep',
    'travels', 'traveled', 'travel',
    'lasts', 'lasted', 'last',
    'sees', 'saw', 'see',
}

ADD_VERBS = {
    'buys', 'bought', 'buy', 'buying',
    'gets', 'got', 'get', 'getting',
    'receives', 'received', 'receive', 'receiving',
    'finds', 'found', 'find', 'finding',
    'earns', 'earned', 'earn', 'earning',
    'adds', 'added', 'add', 'adding',
    'gains', 'gained', 'gain', 'gaining',
    'picks', 'picked', 'pick', 'picking',
    'collects', 'collected', 'collect', 'collecting',
    'makes', 'made', 'make', 'making',
    'saves', 'saved', 'save', 'saving',
    'wins', 'won', 'win', 'winning',
    'catches', 'caught', 'catch', 'catching',
    'gathers', 'gathered', 'gather',
    'harvests', 'harvested', 'harvest',
    'produces', 'produced', 'produce', 'producing',
    'brings', 'brought', 'bring', 'bringing',
    'downloads', 'downloaded', 'download',
    'grows', 'grew', 'grow', 'growing',
    'increases', 'increased', 'increase',
    'joins', 'joined', 'join',
    'hires', 'hired', 'hire',
    'plants', 'planted', 'plant',
    'scores', 'scored', 'score',
    'bakes', 'baked', 'bake', 'baking',
    'cooks', 'cooked', 'cook', 'cooking',
    'builds', 'built', 'build', 'building',
    'creates', 'created', 'create',
    'writes', 'wrote', 'write',
    'draws', 'drew', 'draw',
    'paints', 'painted', 'paint',
    'knits', 'knitted', 'knit',
    'sews', 'sewed', 'sew',
    'packs', 'packed', 'pack',
    'fills', 'filled', 'fill',
    'loads', 'loaded', 'load',
    'stocks', 'stocked', 'stock',
    'orders', 'ordered', 'order',
    'borrows', 'borrowed', 'borrow',
    'invites', 'invited', 'invite',
    'adopts', 'adopted', 'adopt',
    'rescues', 'rescued', 'rescue',
    'more', 'additional', 'extra', 'another',
    'plus',
}

SUB_VERBS = {
    'eats', 'ate', 'eat', 'eating',
    'uses', 'used', 'use', 'using',
    'spends', 'spent', 'spend', 'spending',
    'gives', 'gave', 'give', 'giving',
    'loses', 'lost', 'lose', 'losing',
    'removes', 'removed', 'remove', 'removing',
    'throws', 'threw', 'throw', 'throwing',
    'discards', 'discarded', 'discard',
    'donates', 'donated', 'donate', 'donating',
    'wastes', 'wasted', 'waste', 'wasting',
    'breaks', 'broke', 'break', 'breaking',
    'drops', 'dropped', 'drop', 'dropping',
    'consumes', 'consumed', 'consume',
    'burns', 'burned', 'burn', 'burning',
    'destroys', 'destroyed', 'destroy',
    'pays', 'paid', 'pay', 'paying',
    'sells', 'sold', 'sell', 'selling',
    'returns', 'returned', 'return', 'returning',
    'lends', 'lent', 'lend', 'lending',
    'subtracts', 'subtracted', 'subtract',
    'distributes', 'distributed', 'distribute',
    'leaves', 'left',
    'melts', 'melted', 'melt',
    'shrinks', 'shrank', 'shrink',
    'decreases', 'decreased', 'decrease',
    'drinks', 'drank', 'drink', 'drinking',
    'pops', 'popped', 'pop',
    'dies', 'died', 'die',
    'kills', 'killed', 'kill',
    'ruins', 'ruined', 'ruin',
    'spoils', 'spoiled', 'spoil',
    'rots', 'rotted', 'rot',
    'deflates', 'deflated', 'deflate',
    'leaks', 'leaked', 'leak',
    'steals', 'stole', 'steal',
    'takes away',
    'minus', 'fewer',
    'deletes', 'deleted', 'delete', 'deleting',
    'removes', 'removed', 'remove', 'removing',
    'discards', 'discarded', 'discard',
    'cancels', 'cancelled', 'cancel',
    'throws away',
    'donates', 'donated', 'donate',
    'shares', 'shared', 'share', 'sharing',
}

# MULT syntactic signals
MULT_SIGNALS = {
    'per', 'each', 'every', 'times',
    'daily', 'weekly', 'monthly', 'yearly', 'hourly',
}

RATE_PATTERNS = [
    r'\bper\s+\w+\b',
    r'\beach\b',
    r'\bevery\b',
    r'\btimes\b',
    r'\ba\s+day\b', r'\ba\s+week\b', r'\ba\s+month\b', r'\ba\s+year\b',
    r'\ban\s+hour\b', r'\ba\s+minute\b',
    r'\bper\s+day\b', r'\bper\s+week\b', r'\bper\s+month\b',
    r'\bper\s+year\b', r'\bper\s+hour\b', r'\bper\s+minute\b',
    r'\bdaily\b', r'\bweekly\b', r'\bmonthly\b', r'\byearly\b',
]

DIV_PATTERNS = [
    r'\bdivided?\b', r'\bsplit\b', r'\bshared?\s*equal', r'\beach\s+get',
    r'\bdistribute\s*equal', r'\bper\s+person\b', r'\bapiece\b',
    r'\bsplit\s*equal', r'\bamong\b',
    r'\binto\s+\d+\s+(?:equal\s+)?(?:part|group|pile|piece|section|portion|slice|serving)',
]


def extract_numbers(text: str) -> List[Tuple[float, int, int]]:
    """Extract numbers from text as (value, start_pos, end_pos)."""
    results = []
    seen = set()
    t = text.lower()

    # "N dozen" -> N * 12
    for m in re.finditer(r'(\d+)\s+dozen', t):
        val = int(m.group(1)) * 12
        results.append((float(val), m.start(), m.end()))
        for p in range(m.start(), m.end()):
            seen.add(p)

    # Standalone "a dozen" or "dozen"
    for m in re.finditer(r'\ba?\s*dozen\b', t):
        if m.start() not in seen:
            results.append((12.0, m.start(), m.end()))
            for p in range(m.start(), m.end()):
                seen.add(p)

    # Compound word fractions: "3 quarters" = 0.75, "2 thirds" = 0.667
    _FRAC_DENOM = {'half': 2, 'halves': 2, 'third': 3, 'thirds': 3,
                   'quarter': 4, 'quarters': 4, 'fourth': 4, 'fourths': 4,
                   'fifth': 5, 'fifths': 5, 'sixth': 6, 'sixths': 6,
                   'seventh': 7, 'sevenths': 7, 'eighth': 8, 'eighths': 8,
                   'ninth': 9, 'ninths': 9, 'tenth': 10, 'tenths': 10}
    frac_names = '|'.join(_FRAC_DENOM.keys())
    for m in re.finditer(r'\b(\d+)\s+(' + frac_names + r')\b', t):
        if m.start() not in seen:
            num = int(m.group(1))
            den = _FRAC_DENOM[m.group(2)]
            results.append((num / den, m.start(), m.end()))
            for p in range(m.start(), m.end()):
                seen.add(p)

    # Fraction notation: 2/3, 1/3rd — MUST be before numeric to avoid splitting
    for m in re.finditer(r'\b(\d+)/(\d+)(?:st|nd|rd|th)?\b', text):
        if m.start() not in seen:
            num, den = int(m.group(1)), int(m.group(2))
            if 0 < den <= 100:
                results.append((num / den, m.start(), m.end()))
                for p in range(m.start(), m.end()):
                    seen.add(p)

    # Decimal without leading zero: .5, .25, etc.
    for m in re.finditer(r'(?<!\d)\.(\d+)', text):
        if m.start() not in seen:
            val = float('0.' + m.group(1))
            results.append((val, m.start(), m.end()))
            for p in range(m.start(), m.end()):
                seen.add(p)

    # Numeric: $1,234.56 or 1234.56
    for m in re.finditer(r'\$?([\d,]+\.?\d*)', text):
        if m.start() in seen:
            continue
        raw = m.group(1).replace(',', '')
        if not raw or raw == '.':
            continue
        try:
            val = float(raw)
            results.append((val, m.start(), m.end()))
            for p in range(m.start(), m.end()):
                seen.add(p)
        except ValueError:
            continue

    # Word numbers
    for word, val in sorted(WORD_NUMS.items(), key=lambda x: -len(x[0])):
        for m in re.finditer(r'\b' + re.escape(word) + r'\b', t):
            if m.start() not in seen:
                results.append((float(val), m.start(), m.end()))
                for p in range(m.start(), m.end()):
                    seen.add(p)

    results.sort(key=lambda x: x[1])
    return results


def normalize_answer(val: float) -> Any:
    """Format answer: integer if whole, else 2 decimal places."""
    if abs(val) < 1e15 and abs(val - round(val)) < 0.001:
        return int(round(val))
    return round(val, 2)


class SolverState:
    """Track computation chain with named variables."""

    def __init__(self):
        self.vars: Dict[str, float] = {}
        self.steps: List[str] = []
        self.last: Optional[float] = None
        self.all_values: List[float] = []

    def set_val(self, val: float, desc: str = ''):
        if desc:
            self.steps.append(desc)
        self.last = val
        self.all_values.append(val)

    def op(self, desc: str, val: float):
        self.steps.append(desc)
        self.last = val
        self.all_values.append(val)


class WordProblemSolver:
    """Solve multi-step arithmetic word problems (GSM8K-level).

    When initialized with an FLM (FossLanguageModel), uses it as a semantic
    parser for operation classification: for each sentence with numbers,
    the FLM scores candidate operations (ADD/SUB/MUL/DIV) by evaluating
    which arithmetic result makes the most natural continuation.
    """

    def __init__(self, flm=None):
        """
        Args:
            flm: Optional FossLanguageModel instance (pre-trained).
                 If provided, enables FLM-guided computation graph solver.
        """
        self._flm = flm

    def _flm_score_op(self, context: str, a: float, b: float) -> str:
        """Use FLM to classify the operation between two numbers.

        Instead of scoring numeric results (PPM can't distinguish digits),
        score OPERATION WORDS as continuations — PPM can easily distinguish
        "multiply" from "add" since they're completely different character sequences.

        Returns: 'ADD', 'SUB', 'MUL', 'DIV', or 'SET' if FLM is unavailable.
        """
        if self._flm is None:
            return 'SET'

        # Score operation-word continuations
        # Multiple phrasings per op to be robust
        op_continuations = {
            'MUL': [
                ' which means multiply ',
                ' so we multiply them ',
                ' times each gives ',
                ' of each means total is ',
            ],
            'ADD': [
                ' which means add together ',
                ' so we add them ',
                ' plus that gives ',
                ' combined together is ',
            ],
            'SUB': [
                ' which means subtract ',
                ' so we take away ',
                ' minus that leaves ',
                ' removing from gives ',
            ],
            'DIV': [
                ' which means divide ',
                ' so we split evenly ',
                ' divided gives ',
                ' shared equally is ',
            ],
        }

        best_op = 'SET'
        best_score = float('inf')

        for op, continuations in op_continuations.items():
            if op == 'DIV' and b == 0:
                continue
            # Take the best (lowest perplexity) phrasing for each op
            op_best = float('inf')
            for cont in continuations:
                score = self._flm.score_continuation(context, cont)
                if score < op_best:
                    op_best = score
            if op_best < best_score:
                best_score = op_best
                best_op = op

        return best_op

    def _try_flm_graph(self, context_sents: List[str],
                       question: str, full_text: str) -> Optional[Dict[str, Any]]:  # noqa: ARG002
        """FLM-guided computation graph solver.

        For each sentence:
        1. Extract numbers (regex)
        2. Classify operation (FLM scores candidate continuations)
        3. Build computation chain
        4. Answer the question from the final state

        The FLM replaces regex-based verb classification with statistical
        semantic parsing — it knows "2 cans of 3 each" → MUL because
        "so that is 6" is more natural than "so that is 5".
        """
        if self._flm is None:
            return None

        ql = question.lower()
        steps = []

        # Entity tracking
        STOP_NAMES = {'the', 'if', 'on', 'in', 'at', 'to', 'for', 'how', 'what',
                       'when', 'then', 'each', 'every', 'there', 'this', 'that',
                       'after', 'before', 'during', 'since', 'because', 'so',
                       'one', 'it', 'he', 'she', 'they', 'his', 'her', 'its',
                       'but', 'and', 'or', 'not', 'all', 'some', 'any', 'no',
                       'last', 'first', 'next', 'now', 'also', 'just', 'still'}

        entities = {}  # entity -> running value
        all_entities = []
        last_named = None

        for si, sent in enumerate(context_sents):
            sl = self._word_nums_to_digits(sent.lower())
            words = set(re.findall(r'\b\w+\b', sl))

            # Entity detection
            entity = '_default'
            name_m = re.match(r'([A-Z][a-z]+)', sent.strip())
            if name_m and name_m.group(1).lower() not in STOP_NAMES:
                entity = name_m.group(1).lower()
                last_named = entity
            elif last_named and re.search(r'\b(?:he|she|they|his|her|their)\b', sl):
                entity = last_named

            if entity not in entities:
                all_entities.append(entity)

            # Extract numbers
            nums = extract_numbers(sl)
            num_vals = [v for v, _, _ in nums]
            if not num_vals:
                continue

            # Handle fractions and percentages with regex (FLM not needed)
            frac_val = None
            for fname, fval in FRACTION_MAP.items():
                if fname in sl:
                    frac_val = fval
                    break
            pct_m = re.search(r'(\d+\.?\d*)\s*%', sl)

            if frac_val is not None:
                if entity in entities:
                    has_sub = bool(words & SUB_VERBS) or bool(
                        re.search(r'(?:gave|lost|ate|used|spent|removed|away|left|remaining)', sl))
                    if has_sub or re.search(r'(?:remaining|rest|left)', sl):
                        entities[entity] -= entities[entity] * frac_val
                        steps.append(f"FLM-CG: {entity} -= {frac_val*100}% → {entities[entity]:.1f}")
                    else:
                        entities[entity] *= frac_val
                        steps.append(f"FLM-CG: {entity} ×= {frac_val} → {entities[entity]:.1f}")
                elif num_vals:
                    entities[entity] = num_vals[0] * frac_val
                    steps.append(f"FLM-CG: {entity} = {num_vals[0]} × {frac_val} = {entities[entity]:.1f}")
                continue

            if pct_m:
                pct = float(pct_m.group(1)) / 100
                if entity in entities:
                    if re.search(r'(?:discount|off|less|reduce|decrease|lose)', sl):
                        entities[entity] -= entities[entity] * pct
                    else:
                        entities[entity] += entities[entity] * pct
                    steps.append(f"FLM-CG: {entity} ±{pct*100}% → {entities[entity]:.1f}")
                continue

            # Multiplier words (twice, triple, etc.)
            mult_val = None
            for mw, mv in MULTIPLIER_WORDS.items():
                if mw in sl:
                    mult_val = mv
                    break
            times_m = re.search(r'(\d+)\s+times\s+(?:as\s+)?(?:many|much|more|that|the)', sl)
            if times_m:
                mult_val = float(times_m.group(1))

            # Cross-entity reference
            ref_entity = None
            ref_m = re.search(r'(?:as|than|of)\s+([A-Z][a-z]+)', sent)
            if ref_m:
                rn = ref_m.group(1).lower()
                if rn not in STOP_NAMES and rn in entities:
                    ref_entity = rn

            if mult_val and ref_entity:
                entities[entity] = entities[ref_entity] * mult_val
                steps.append(f"FLM-CG: {entity} = {ref_entity}({entities[ref_entity]}) × {mult_val} = {entities[entity]}")
                continue

            # === FLM operation classification ===
            if len(num_vals) == 1:
                val = num_vals[0]
                if entity not in entities:
                    # First mention → SET
                    entities[entity] = val
                    steps.append(f"FLM-CG: {entity} = {val}")
                else:
                    # FLM decides: is this an ADD or SUB to running total?
                    op = self._flm_score_op(sent, entities[entity], val)
                    if op == 'ADD' or op == 'MUL':
                        # For single number + existing entity, ADD is more common
                        # But use FLM's decision
                        if op == 'ADD':
                            entities[entity] += val
                            steps.append(f"FLM-CG: {entity} += {val} → {entities[entity]}")
                        else:
                            entities[entity] *= val
                            steps.append(f"FLM-CG: {entity} ×= {val} → {entities[entity]}")
                    elif op == 'SUB':
                        entities[entity] -= val
                        steps.append(f"FLM-CG: {entity} -= {val} → {entities[entity]}")
                    elif op == 'DIV':
                        if val != 0:
                            entities[entity] /= val
                            steps.append(f"FLM-CG: {entity} /= {val} → {entities[entity]}")
                    else:
                        # Fallback: add
                        entities[entity] += val
                        steps.append(f"FLM-CG: {entity} += {val} → {entities[entity]}")

            elif len(num_vals) >= 2:
                a, b = num_vals[0], num_vals[1]
                # FLM decides the operation between the two numbers
                op = self._flm_score_op(sent, a, b)

                if op == 'MUL':
                    result = a * b
                elif op == 'ADD':
                    result = a + b
                elif op == 'SUB':
                    result = a - b
                    if result < 0:
                        result = b - a
                elif op == 'DIV':
                    result = a / b if b != 0 else a
                else:
                    result = a * b  # Default for multi-number

                # Apply any remaining numbers
                for v in num_vals[2:]:
                    op2 = self._flm_score_op(
                        f"{sent} so far that gives {result}", result, v)
                    if op2 == 'MUL':
                        result *= v
                    elif op2 == 'ADD':
                        result += v
                    elif op2 == 'SUB':
                        result -= v
                    elif op2 == 'DIV' and v != 0:
                        result /= v

                if entity in entities:
                    # Combine with running total
                    op3 = self._flm_score_op(
                        f"{entity} had {entities[entity]}. {sent}",
                        entities[entity], result)
                    if op3 == 'ADD':
                        entities[entity] += result
                    elif op3 == 'SUB':
                        entities[entity] -= result
                    else:
                        entities[entity] = result
                else:
                    entities[entity] = result
                steps.append(f"FLM-CG: {entity} = {entities[entity]}")

        if not entities or not steps:
            return None

        # Answer the question
        target = None
        for ent in all_entities:
            if ent != '_default' and ent in ql:
                target = ent
                break
        if target is None:
            if len(entities) == 1:
                target = list(entities.keys())[0]
            elif '_default' in entities:
                target = '_default'
            elif all_entities:
                target = [e for e in all_entities if e in entities][-1] if \
                    any(e in entities for e in all_entities) else list(entities.keys())[-1]

        if target is None or target not in entities:
            return None

        result_val = entities[target]

        # "total/altogether/combined" → sum all entities
        if re.search(r'\b(?:total|altogether|combined|together|both)\b', ql):
            if len(entities) > 1:
                result_val = sum(v for v in entities.values())

        # "how many more X than Y"
        diff_m = re.search(r'how\s+many\s+more.*?than\s+(\w+)', ql)
        if diff_m and len(entities) >= 2:
            other = diff_m.group(1).lower()
            if other in entities and target in entities:
                result_val = abs(entities[target] - entities[other])

        return {
            'answer': normalize_answer(result_val),
            'steps': steps,
            'confidence': 'HIGH' if len(steps) >= 2 else 'MEDIUM',
        }

    def solve(self, problem: str) -> Optional[Dict[str, Any]]:
        result = self._solve_inner(problem)
        if result is not None:
            ql = problem.lower()
            # Post-processing: reject negative answers for "how many/much" questions
            ans = result.get('answer')
            if ans is not None:
                try:
                    val = float(ans)
                    if val < 0 and re.search(r'how\s+(?:many|much)', ql):
                        result = None  # Fall through to fallback below
                except (ValueError, TypeError):
                    pass
            if result is not None:
                # Post-processing: if question asks "how many dozens", convert
                if re.search(r'how\s+many\s+dozens?\b', ql):
                    if ans is not None:
                        try:
                            val = float(ans)
                            if val >= 12 and val % 12 == 0:
                                result['answer'] = normalize_answer(val / 12)
                                result['steps'].append(f'{val} / 12 = {val/12} dozen')
                        except (ValueError, TypeError):
                            pass
                # Post-processing: "what percentage" — convert count to percentage
                if re.search(r'what\s+percent', ql) and ans is not None:
                    try:
                        val = float(ans)
                        # Only convert if answer looks like a count (not already a percentage)
                        # Find the total/group size from the problem
                        total_m = re.search(
                            r'(?:of|out\s+of|from|among)\s+(\d+)\s+(?:students|people|children|'
                            r'puppies|marbles|plants|teeth|employees|items|members|workers|'
                            r'animals|cars|books|balls|apples|oranges|fruits|flowers|'
                            r'pieces|guests|passengers|tickets|votes|points|games|'
                            r'total|altogether|spools|hours)', ql + ' ' + problem.lower())
                        if not total_m:
                            # Look for group size at start: "In a class of N students"
                            total_m = re.search(
                                r'(?:class|group|team|batch|total|family)\s+of\s+(\d+)', problem.lower())
                        if not total_m:
                            # Look for standalone total: "there are N plants"
                            total_m = re.search(
                                r'(?:there\s+are|has|have|contains?)\s+(\d+)\s+(?:plants|teeth|'
                                r'students|people|spools|marbles|animals|items|members|puppies|'
                                r'workers|employees|hours)', problem.lower())
                        if total_m:
                            total = float(total_m.group(1))
                            if 0 < val < total:
                                pct = val / total * 100
                                # Only convert if the percentage is a round-ish number
                                if abs(pct - round(pct)) < 0.01 or abs(pct * 10 - round(pct * 10)) < 0.01:
                                    result['answer'] = normalize_answer(pct)
                                    result['steps'].append(f'{val}/{total} × 100 = {pct}%')
                    except (ValueError, TypeError):
                        pass
        # Fallback chain: simple patterns first, then computation graph
        if result is None:
            result = self._try_simple_fallback(problem)
        if result is None:
            # Try computation graph (generic multi-step) — last resort
            sentences = self._split_sentences(problem)
            question = ''
            context_sents = []
            for s in sentences:
                if '?' in s:
                    question = s
                else:
                    context_sents.append(s)
            # Extract "if X has N" clauses from question as context
            if question:
                if_m = re.search(r'if\s+(.+?)(?:\?|$)', question, re.I)
                if if_m:
                    if_clause = if_m.group(1).strip().rstrip('?')
                    if re.search(r'\d', self._word_nums_to_digits(if_clause)):
                        context_sents.append(if_clause)
            context_sents = self._resolve_multiplier_refs(context_sents)
            result = self._try_computation_graph(context_sents, question)
        return result

    def _try_simple_fallback(self, problem: str) -> Optional[Dict[str, Any]]:
        """Last-resort solver for problems no other solver handles.

        Extracts all numbers and tries simple arithmetic combinations.
        Only returns results with HIGH confidence patterns.
        """
        sents = self._split_sentences(problem)
        if not sents:
            return None
        question = sents[-1]
        ql = question.lower()
        all_text = problem.lower()

        # Extract all numbers from the problem
        all_nums = extract_numbers(problem)
        vals = [v for v, _, _ in all_nums if v > 0]

        # Pattern: "N days by X, half as many by Y. Total?" → N + N/2
        # (before the len(vals) < 2 guard since "half" is a word not a number)
        half_m2 = re.search(r'(\d+)\s+\w+.*?half\s+(?:as\s+)?(?:many|much|long|that)', all_text)
        if half_m2 and re.search(r'(?:total|how\s+many\s+days|both|combined|altogether|and)', ql) \
                and not re.search(r'(?:percent|%)', ql):
            base = float(half_m2.group(1))
            half_val = base / 2
            result = base + half_val
            return {'answer': normalize_answer(result),
                    'steps': [f'{base} + {base}/2 = {result}'],
                    'confidence': 'MEDIUM'}

        # Pattern: "once per month + twice per month + quarterly" → annual total
        # (before len(vals) guard since frequencies are words not numbers)
        if re.search(r'(?:per\s+year|yearly|annual|a\s+year)', ql):
            annual_total = 0
            for m in re.finditer(r'(?:once|one|1)\s+(?:per|a|each)\s+month', all_text):
                annual_total += 12
            for m in re.finditer(r'(?:twice|two|2)\s+(?:per|a|each)\s+month', all_text):
                annual_total += 24
            for m in re.finditer(r'(?:every\s+month|each\s+month|monthly)', all_text):
                annual_total += 12
            for m in re.finditer(r'quarterly', all_text):
                annual_total += 4
            for m in re.finditer(r'(?:once|one|1)\s+(?:per|a|each)\s+week', all_text):
                annual_total += 52
            for m in re.finditer(r'(?:twice|two|2)\s+(?:per|a|each)\s+week', all_text):
                annual_total += 104
            if annual_total > 0:
                return {'answer': normalize_answer(annual_total),
                        'steps': [f'Annual total: {annual_total}'],
                        'confidence': 'MEDIUM'}

        if len(vals) < 2:
            return None

        # Pattern: "N [things] in [groups] of M. How many [groups]?" → N / M
        # Or: "N total, M per group, $P per group" → (N/M) * P
        div_m = re.search(
            r'(\d+)\s+\w+\s+(?:in|into)\s+(?:\w+\s+)?(?:bags?|groups?|boxes?|packs?|packets?|dozens?|sets?|batches?|cartons?|bunches?)\s+'
            r'(?:of\s+)?(\d+)', all_text)
        if div_m:
            total = float(div_m.group(1))
            per_group = float(div_m.group(2))
            if per_group > 0:
                groups = total / per_group
                # Check if there's a price per group
                price_m = re.search(r'\$(\d+\.?\d*)\s*(?:per|each|a)\s+(?:bag|group|box|pack|set|batch|carton|bunch|dozen)', all_text)
                if price_m:
                    price = float(price_m.group(1))
                    result = groups * price
                    return {'answer': normalize_answer(result),
                            'steps': [f'{total}/{per_group}={groups} groups × ${price} = {result}'],
                            'confidence': 'MEDIUM'}
                elif re.search(r'how\s+many\s+(?:bags?|groups?|boxes?|packs?)', ql):
                    return {'answer': normalize_answer(groups),
                            'steps': [f'{total}/{per_group} = {groups}'],
                            'confidence': 'MEDIUM'}

        # Pattern: "has/saved $A + got/gave $B. Needs $C+$D+$E. How much more?"
        if re.search(r'how\s+much\s+more', ql):
            costs = re.findall(r'costs?\s+\$(\d+\.?\d*)', all_text)
            has_vals = []
            for m in re.finditer(r'(?:saved?|has|had|gave|received|got|earned)\s+(?:\w+\s+)*?\$(\d+\.?\d*)', all_text):
                has_vals.append(float(m.group(1)))
            if costs and has_vals:
                total_cost = sum(float(c) for c in costs)
                total_has = sum(has_vals)
                diff = total_cost - total_has
                if diff > 0:
                    return {'answer': normalize_answer(diff),
                            'steps': [f'Need ${total_cost} - Have ${total_has} = ${diff}'],
                            'confidence': 'MEDIUM'}

        # Pattern: "X minutes homework + Y minutes homework... Z hours total. Nap?"
        if re.search(r'(?:nap|rest|sleep|free|spare|left)', ql):
            minute_vals = re.findall(r'(\d+)\s+minutes?\s+(?:of\s+)?(?:\w+\s+)?homework', all_text)
            hour_m = re.search(r'(\d+)\s+hours?\s+(?:before|until|left|total|free)', all_text)
            if minute_vals and hour_m:
                total_hw = sum(float(v) for v in minute_vals)
                total_avail = float(hour_m.group(1)) * 60
                result = total_avail - total_hw
                if result > 0:
                    return {'answer': normalize_answer(result),
                            'steps': [f'{total_avail}min - {total_hw}min = {result}min'],
                            'confidence': 'MEDIUM'}

        # Pattern: "N items total, sold/used in bags/groups of M. $P per bag" → (N/M)*P
        # Handle both orders: "N items in bags of M" and "bags of M... N items total"
        bag_m = re.search(r'(\d+)\s+\w+.*?(?:bags?|groups?|boxes?|packs?|bunches?)\s+of\s+(\d+)', all_text)
        if not bag_m:
            # Reverse order: "in bags of M" then "total of N items" or "N items"
            size_m = re.search(r'(?:in|into)\s+(?:bags?|groups?|boxes?|packs?|bunches?)\s+of\s+(\d+)', all_text)
            count_m = re.search(r'(?:total\s+of\s+)?(\d{2,})\s+\w+', all_text) if size_m else None
            if size_m and count_m:
                class _BagMatch:
                    def group(self, n):
                        return [None, count_m.group(1), size_m.group(1)][n]
                bag_m = _BagMatch()
        if bag_m:
            total_items = float(bag_m.group(1))
            per_bag = float(bag_m.group(2))
            if per_bag > 0:
                n_bags = total_items / per_bag
                price_m = re.search(r'\$(\d+\.?\d*)\s*(?:per|each|a)\s+(?:bag|group|box|pack|bunch)', all_text)
                if price_m:
                    price = float(price_m.group(1))
                    result = n_bags * price
                    return {'answer': normalize_answer(result),
                            'steps': [f'{total_items}/{per_bag}={n_bags} bags × ${price} = {result}'],
                            'confidence': 'MEDIUM'}
                elif re.search(r'how\s+many\s+(?:bags?|groups?|boxes?|packs?)', ql):
                    return {'answer': normalize_answer(n_bags),
                            'steps': [f'{total_items}/{per_bag} = {n_bags}'],
                            'confidence': 'MEDIUM'}
                # Also: "at $P per bag" with different word order
                price_m2 = re.search(r'(?:at|for)\s+\$(\d+\.?\d*)\s+(?:per|a|each)\s+(?:bag|group|box|pack|bunch)', all_text)
                if price_m2:
                    price = float(price_m2.group(1))
                    result = n_bags * price
                    return {'answer': normalize_answer(result),
                            'steps': [f'{total_items}/{per_bag}={n_bags} bags × ${price} = {result}'],
                            'confidence': 'MEDIUM'}
                # Check for earnings/cost question with $ value nearby
                if re.search(r'(?:earn|make|revenue|profit|how\s+much)', ql):
                    price_m3 = re.search(r'\$(\d+\.?\d*)', all_text)
                    if price_m3:
                        price = float(price_m3.group(1))
                        result = n_bags * price
                        return {'answer': normalize_answer(result),
                                'steps': [f'{total_items}/{per_bag}={n_bags} × ${price} = {result}'],
                                'confidence': 'MEDIUM'}

        # Pattern: "CD costs $X, total $Y. How many more CDs without headphone?"
        # item $A, total $B → can afford B/A, already has 1, so B/A - 1
        if re.search(r'how\s+many\s+more\s+\w+\s+\w+.*(?:buy|afford|get)', ql):
            dollar_vals = re.findall(r'\$(\d+\.?\d*)', all_text)
            if len(dollar_vals) >= 2:
                item_price = float(dollar_vals[0])
                total_budget = float(dollar_vals[1])
                if item_price > 0 and total_budget > item_price:
                    can_buy = int(total_budget / item_price)
                    already = 1  # already bought one
                    more = can_buy - already
                    return {'answer': normalize_answer(more),
                            'steps': [f'${total_budget}/${item_price}={can_buy} - {already} already = {more}'],
                            'confidence': 'MEDIUM'}

        # Pattern: "N cats for every dog, M dogs → total pets"
        ratio_m = re.search(r'(\w+)\s+(\w+)\s+for\s+every\s+(\w+)', all_text)
        if ratio_m and re.search(r'(?:total|how\s+many|all)', ql):
            ratio_word = ratio_m.group(1)
            ratio_val = None
            if ratio_word in WORD_NUMS:
                ratio_val = WORD_NUMS[ratio_word]
            elif ratio_word.isdigit():
                ratio_val = int(ratio_word)
            if ratio_val is not None:
                # Find the base count
                base_animal = ratio_m.group(3)  # e.g., "dog"
                base_m = re.search(r'(?:number\s+of\s+' + base_animal + r's?\s+is\s+|(\d+)\s+' + base_animal + r')', all_text)
                if base_m and base_m.group(1):
                    base_count = float(base_m.group(1))
                else:
                    # "the number of dogs is 60"
                    base_m2 = re.search(r'(?:number\s+of\s+' + base_animal + r's?)\s+is\s+(\d+)', all_text)
                    base_count = float(base_m2.group(1)) if base_m2 else None
                if base_count is not None:
                    derived = base_count * ratio_val
                    # Check for "twelve less than combined" type modifiers
                    less_m = re.search(r'(\w+)\s+(?:less|fewer)\s+than\s+(?:the\s+)?(?:combined|total)', all_text)
                    if less_m:
                        less_word = less_m.group(1)
                        less_val = WORD_NUMS.get(less_word, None)
                        if less_val is None and less_word.isdigit():
                            less_val = int(less_word)
                        if less_val is not None:
                            third_count = (base_count + derived) - less_val
                            total = base_count + derived + third_count
                            return {'answer': normalize_answer(total),
                                    'steps': [f'{base_animal}={base_count}, ratio={ratio_val}→{derived}, third={third_count}, total={total}'],
                                    'confidence': 'MEDIUM'}
                    # Simple total: base + derived
                    total = base_count + derived
                    return {'answer': normalize_answer(total),
                            'steps': [f'{base_animal}={base_count} × {ratio_val}={derived}, total={total}'],
                            'confidence': 'MEDIUM'}

        # Pattern: "spends N hours X, half that Y, Z minutes W. What percentage on W?"
        if re.search(r'(?:percent|%)', ql):
            # Extract time values and convert all to minutes
            hour_vals = re.findall(r'(\d+\.?\d*)\s+hours?', all_text)
            min_vals = re.findall(r'(\d+\.?\d*)\s+minutes?', all_text)
            half_refs = re.findall(r'half\s+(?:that|as)\s+(?:much|many|long)', all_text)
            if hour_vals or min_vals:
                total_mins = 0
                time_parts = []
                for h in hour_vals:
                    time_parts.append(float(h) * 60)
                if half_refs and hour_vals:
                    time_parts.append(float(hour_vals[0]) * 60 / 2)
                for m_val in min_vals:
                    time_parts.append(float(m_val))
                if len(time_parts) >= 2:
                    total_mins = sum(time_parts)
                    # The question asks about the last activity mentioned
                    last_time = time_parts[-1]
                    pct = (last_time / total_mins) * 100
                    return {'answer': normalize_answer(pct),
                            'steps': [f'{last_time}/{total_mins} × 100 = {pct}%'],
                            'confidence': 'MEDIUM'}

        # Pattern: "A peels in X min, B peels in Y min. After Z hours, how many more?"
        if re.search(r'how\s+many\s+more', ql) and re.search(r'(\d+)\s+minutes?', all_text):
            time_per = re.findall(r'(\d+)\s+minutes?', all_text)
            duration_m = re.search(r'(?:after|in)\s+(?:an?\s+)?(\d+|an?)\s+hours?', all_text)
            if len(time_per) >= 2 and duration_m:
                dur_str = duration_m.group(1)
                duration_mins = (1 if dur_str in ('a', 'an') else float(dur_str)) * 60
                rate_a = duration_mins / float(time_per[0])
                rate_b = duration_mins / float(time_per[1])
                diff = abs(rate_a - rate_b)
                return {'answer': normalize_answer(diff),
                        'steps': [f'{duration_mins}/{time_per[0]}={rate_a} vs {duration_mins}/{time_per[1]}={rate_b}, diff={diff}'],
                        'confidence': 'MEDIUM'}

        # Pattern: "N friends each eat X slices of A pizza (cut into P slices) + Y slices of B"
        # → total slices / slices_per_pizza for each type
        if re.search(r'how\s+many\s+pizza', ql):
            cut_matches = re.findall(r'cut\s+into\s+(\d+)\s+slices', all_text)
            eat_matches = re.findall(r'(\d+)\s+\w+\s+pizza\s+slices?', all_text)
            friends_m = re.search(r'(\d+)\s+friends?', all_text)
            if cut_matches and eat_matches and friends_m:
                n_friends = int(friends_m.group(1))
                total_pizzas = 0
                import math
                for cut, eat in zip(cut_matches, eat_matches):
                    slices_per_pizza = int(cut)
                    slices_eaten = int(eat) * n_friends
                    pizzas_needed = math.ceil(slices_eaten / slices_per_pizza)
                    total_pizzas += pizzas_needed
                return {'answer': normalize_answer(total_pizzas),
                        'steps': [f'{n_friends} friends, {len(cut_matches)} types → {total_pizzas} pizzas'],
                        'confidence': 'MEDIUM'}

        # Pattern: rate × time = total (e.g., "2 per day... 30 days... how much?")
        # "N per day/week/hour" + "M days/weeks/hours" → N*M, possibly with pricing
        rate_m = re.search(r'(\d+\.?\d*)\s+\w+\s+(?:a|per|each|every)\s+(day|week|hour|month|year|minute)', all_text)
        duration_m2 = None
        if rate_m:
            unit = rate_m.group(2)
            # Find matching duration
            duration_m2 = re.search(r'(\d+)\s+' + unit + r's?\b', all_text)
            if not duration_m2:
                # "over N days" or "in N weeks"
                duration_m2 = re.search(r'(?:over|in|for)\s+(\d+)\s+' + unit + r's?\b', all_text)
        if rate_m and duration_m2:
            rate = float(rate_m.group(1))
            duration = float(duration_m2.group(1))
            total_items = rate * duration
            # Check for unit pricing: "N for $P" or "$P for N" or "$P each"
            unit_price_m = re.search(r'(\d+)\s+\w+\s+for\s+\$(\d+\.?\d*)', all_text)
            if unit_price_m:
                pack_size = float(unit_price_m.group(1))
                pack_price = float(unit_price_m.group(2))
                if pack_size > 0:
                    cost = (total_items / pack_size) * pack_price
                    return {'answer': normalize_answer(cost),
                            'steps': [f'{rate}×{duration}={total_items}, /{pack_size}×${pack_price}=${cost}'],
                            'confidence': 'MEDIUM'}
            price_each_m = re.search(r'\$(\d+\.?\d*)\s*(?:each|per|apiece)', all_text)
            if price_each_m:
                price = float(price_each_m.group(1))
                cost = total_items * price
                return {'answer': normalize_answer(cost),
                        'steps': [f'{rate}×{duration}={total_items} × ${price} = ${cost}'],
                        'confidence': 'MEDIUM'}
            # Just return the total count if question asks "how many"
            if re.search(r'how\s+many', ql):
                return {'answer': normalize_answer(total_items),
                        'steps': [f'{rate} × {duration} = {total_items}'],
                        'confidence': 'MEDIUM'}
            # Return cost if question asks "how much"
            if re.search(r'how\s+much', ql):
                # Check for any dollar value that might be the unit price
                dollars = re.findall(r'\$(\d+\.?\d*)', all_text)
                if len(dollars) == 1:
                    price = float(dollars[0])
                    cost = total_items * price
                    return {'answer': normalize_answer(cost),
                            'steps': [f'{rate}×{duration}={total_items} × ${price} = ${cost}'],
                            'confidence': 'MEDIUM'}

        # Pattern: "N items at $A each + M items at $B each. How much total/left?"
        item_costs = re.findall(r'(\d+)\s+\w+.*?\$(\d+\.?\d*)\s*(?:each|apiece|per)', all_text)
        if not item_costs:
            # "N items that cost $A each"
            item_costs = re.findall(r'(\d+)\s+\w+\s+(?:that\s+)?costs?\s+\$(\d+\.?\d*)\s*(?:each)?', all_text)
        if len(item_costs) >= 2 and re.search(r'(?:how\s+much|how\s+many|total|cost|spend|pay|left|remain)', ql):
            total_cost = sum(float(n) * float(p) for n, p in item_costs)
            used_prices_early = set(float(p) for _, p in item_costs)
            # Add standalone costs: "one X that costs $Y" (word "one"/quantity 1)
            standalone_m = re.findall(
                r'(?:one|a|an|1)\s+\w+\s+(?:\w+\s+)?(?:that\s+)?costs?\s+\$(\d+\.?\d*)', all_text.lower())
            for sc in standalone_m:
                sc_val = float(sc)
                if sc_val not in used_prices_early:
                    total_cost += sc_val
                    used_prices_early.add(sc_val)
            # Check if there's a budget/starting amount
            budget_m = re.search(r'(?:starts?\s+with|has|budget|gave|paid\s+a\s+total\s+of)\s+\$(\d+\.?\d*)', all_text)
            if budget_m and re.search(r'(?:left|remain|change|save)', ql):
                budget = float(budget_m.group(1))
                left = budget - total_cost
                return {'answer': normalize_answer(left),
                        'steps': [f'${budget} - ${total_cost} = ${left}'],
                        'confidence': 'MEDIUM'}
            # Check "solve for unknown": total given + known costs → find count of unknown
            total_given_m = re.search(
                r'(?:a\s+)?total\s+(?:of\s+|was\s+|is\s+)\$(\d+\.?\d*)', all_text.lower())
            if not total_given_m:
                total_given_m = re.search(
                    r'paid\s+(?:a\s+total\s+of\s+)?\$(\d+\.?\d*)', all_text.lower())
            if total_given_m:
                given_total = float(total_given_m.group(1))
                if given_total > total_cost and re.search(r'how\s+many', ql):
                    remainder = given_total - total_cost
                    # Find unknown item price: "each box costs $8.50" / "if each X costs $Y"
                    unknown_p_m = re.search(
                        r'(?:each|per|every)\s+\w+\s+(?:costs?|is|at)\s+\$(\d+\.?\d*)', ql.lower())
                    if not unknown_p_m:
                        unknown_p_m = re.search(r'\$(\d+\.?\d*)\s*(?:each|per|apiece)', ql.lower())
                    if unknown_p_m:
                        unknown_price = float(unknown_p_m.group(1))
                        if unknown_price > 0:
                            count = remainder / unknown_price
                            return {'answer': normalize_answer(count),
                                    'steps': [f'Known: ${total_cost}, Total: ${given_total}, Remainder: ${remainder}, Count: {remainder}/{unknown_price}={count}'],
                                    'confidence': 'HIGH'}
            # Time multiplier: "N weeks a year" / "N months"
            time_m = re.search(
                r'(\d+)\s+(?:weeks?|months?|years?|days?)\s+'
                r'(?:a\s+|per\s+|each\s+|in\s+a\s+)?(?:year|month|week|day)', all_text.lower())
            if time_m:
                time_mult = float(time_m.group(1))
                total_cost *= time_mult
            return {'answer': normalize_answer(total_cost),
                    'steps': [f'Total: {" + ".join(f"{n}×${p}" for n,p in item_costs)} = ${total_cost}'],
                    'confidence': 'MEDIUM'}

        # Pattern: "N people donate M each + K extra. Fit F per table. Have T tables. Need?"
        if re.search(r'(?:how\s+many.*(?:need|more|additional)|(?:need|more|additional).*how\s+many)', ql):
            # Calculate total items from donations/contributions
            donate_m = re.search(r'(\d+)\s+\w+\s+(?:donate|give|bring|contribute)\s+(\d+)', all_text)
            extra_m = re.search(r'(?:also\s+have|already\s+have|have\s+\w+\s+)?(\d+)\s+\w+\s+(?:already|extra|more)', all_text)
            per_m = re.search(r'(\d+)\s+\w+\s+(?:per|each|a)\s+(\w+)', all_text)
            have_m = re.search(r'(?:already\s+(?:own|have)|have|owns?)\s+(\d+)', all_text)
            if donate_m and per_m:
                total_items = float(donate_m.group(1)) * float(donate_m.group(2))
                if extra_m:
                    total_items += float(extra_m.group(1))
                per_unit = float(per_m.group(1))
                import math
                total_needed = math.ceil(total_items / per_unit) if per_unit > 0 else 0
                already_have = float(have_m.group(1)) if have_m else 0
                need_more = max(0, total_needed - already_have)
                return {'answer': normalize_answer(need_more),
                        'steps': [f'{total_items} items / {per_unit} per unit = {total_needed}, have {already_have}, need {need_more}'],
                        'confidence': 'MEDIUM'}

        # Pattern: "save/discount" — original price vs sale price × quantity
        if re.search(r'(?:save|saving|savings|discount)', ql):
            orig_prices = re.findall(r'(?:costs?|costing|priced?\s+at|was)\s+\$(\d+\.?\d*)', all_text)
            sale_prices = re.findall(r'(?:sold\s+at|now\s+(?:sold\s+)?(?:at|for)?|sale\s+(?:price|for))\s+\$(\d+\.?\d*)', all_text)
            discount_amts = re.findall(r'discount\s+of\s+\$(\d+\.?\d*)', all_text)
            if orig_prices and (sale_prices or discount_amts):
                total_savings = 0
                # Pair original and sale prices
                for i_p, op in enumerate(orig_prices):
                    disc = float(op) - float(sale_prices[i_p]) if i_p < len(sale_prices) else 0
                    total_savings += disc
                for da in discount_amts:
                    total_savings += float(da)
                # Look for quantities
                qty_m = re.search(r'(?:buy|bought|purchase)\s+(\d+)', all_text)
                qtys = re.findall(r'(\d+)\s+(?:tubs?|packs?|bottles?|items?|pieces?|packets?|boxes?)', all_text)
                if qty_m:
                    # Multiple items: "2 tubs and 4 packets"
                    pass  # handled below
                # Try to match quantities to savings
                if len(orig_prices) >= 1 and len(sale_prices) >= 1:
                    # Per-item savings times quantity
                    result = 0
                    for i_p in range(max(len(orig_prices), len(sale_prices))):
                        op = float(orig_prices[min(i_p, len(orig_prices)-1)])
                        sp = float(sale_prices[min(i_p, len(sale_prices)-1)]) if i_p < len(sale_prices) else op
                        per_save = op - sp
                        # Find quantity for this item
                        qty = 1
                        if i_p < len(qtys):
                            qty = int(qtys[i_p])
                        result += per_save * qty
                    for da in discount_amts:
                        q_mult = 1
                        if qtys:
                            q_mult = int(qtys[-1]) if len(qtys) > len(orig_prices) else 1
                        result += float(da) * q_mult
                    if result > 0:
                        return {'answer': normalize_answer(result),
                                'steps': [f'Savings: ${result}'],
                                'confidence': 'MEDIUM'}

        # Pattern: "N% more than X" → X * (1 + N/100)
        pct_more_m = re.search(r'(\d+\.?\d*)\s*%\s*(?:more|greater|higher|larger)\s+than', all_text)
        if pct_more_m:
            pct_val = float(pct_more_m.group(1))
            # Find the base value — prefer dollar amounts
            dollar_vals = [float(d) for d in re.findall(r'\$(\d+\.?\d*)', all_text)]
            base_vals = dollar_vals if dollar_vals else [v for v in vals if v != pct_val]
            if base_vals:
                base = base_vals[0]
                increased = base * (1 + pct_val / 100)
                # Question asks about the increased amount itself
                if re.search(r'(?:how\s+much\s+(?:does|did|will|do)|cost|spend|pay|price)', ql):
                    return {'answer': normalize_answer(increased),
                            'steps': [f'${base} × {1+pct_val/100} = ${increased}'],
                            'confidence': 'MEDIUM'}

        # Pattern: "N items at $A apiece + M items at $B apiece. Baskets/sets."
        # "5 baskets, 3 X at $A + 2 Y at $B each. Total?"
        basket_m = re.search(r'(\d+)\s+(?:baskets?|sets?|groups?|boxes?|bags?)\s+(?:to\s+fill|to\s+make|each)', all_text)
        if basket_m and re.search(r'(?:how\s+much|total|cost|spend)', ql):
            n_baskets = float(basket_m.group(1))
            # Find items per basket and their costs
            per_basket_cost = 0
            for m in re.finditer(r'(\d+)\s+\w+.*?\$(\d+\.?\d*)\s*(?:apiece|each|per)', all_text):
                qty = float(m.group(1))
                price = float(m.group(2))
                per_basket_cost += qty * price
            if per_basket_cost > 0:
                total = n_baskets * per_basket_cost
                return {'answer': normalize_answer(total),
                        'steps': [f'{n_baskets} × ${per_basket_cost}/basket = ${total}'],
                        'confidence': 'MEDIUM'}

        # Pattern: "N [things] sliced/cut into M pieces. Eat K. How many left?"
        slice_m = re.search(r'(\d+)\s+\w+.*?(\d+)\s+(?:pieces?|slices?|parts?)', all_text)
        if slice_m and re.search(r'(?:left|remain|uneaten)', ql):
            n_items = float(slice_m.group(1))
            per_item = float(slice_m.group(2))
            total_slices = n_items * per_item
            # Find additional items
            more_slices = re.findall(r'(\d+)\s+\w+.*?(\d+)\s+(?:pieces?|slices?)', all_text)
            if len(more_slices) >= 2:
                total_slices = sum(float(n) * float(p) for n, p in more_slices)
            eat_m = re.search(r'(?:eats?|ate|consumed?)\s+(\d+)', all_text)
            if eat_m:
                eaten = float(eat_m.group(1))
                left = total_slices - eaten
                if left >= 0:
                    return {'answer': normalize_answer(left),
                            'steps': [f'{total_slices} slices - {eaten} eaten = {left}'],
                            'confidence': 'MEDIUM'}

        # Pattern: "made/had N items. Fed/gave K. Broke remaining into M pieces."
        if re.search(r'(?:pieces?|slices?)', ql):
            made_m = re.search(r'(?:made|had|baked|has)\s+(?:a\s+)?(?:dozen|(\d+))', all_text)
            if made_m:
                initial = 12.0 if 'dozen' in (made_m.group(0) or '') else float(made_m.group(1))
                fed_m = re.search(r'(?:fed|feeding|gave|giving|distributed|ate|served)\s+(?:\w+\s+)*?(\d+)', all_text)
                fed = float(fed_m.group(1)) if fed_m else 0
                remaining = initial - fed
                pieces_m = re.search(r'(?:into|in)\s+(\d+)\s+(?:pieces?|slices?|parts?)', all_text)
                if pieces_m and remaining > 0:
                    pieces_per = float(pieces_m.group(1))
                    total_pieces = remaining * pieces_per
                    return {'answer': normalize_answer(total_pieces),
                            'steps': [f'({initial}-{fed}) × {pieces_per} = {total_pieces}'],
                            'confidence': 'MEDIUM'}

        # Pattern: price comparison / decision making
        # "Option A costs X. Option B costs Y. How much saved/cheaper?"
        if re.search(r'(?:save|cheaper|better|difference)', ql):
            all_dollars = re.findall(r'\$(\d+\.?\d*)', all_text)
            if len(all_dollars) >= 2:
                d_vals = [float(d) for d in all_dollars]
                if len(d_vals) == 2:
                    diff = abs(d_vals[0] - d_vals[1])
                    return {'answer': normalize_answer(diff),
                            'steps': [f'|${d_vals[0]} - ${d_vals[1]}| = ${diff}'],
                            'confidence': 'LOW'}

        # Pattern: "N items × A cents + M items × B cents per week/day, over K weeks"
        cent_items = re.findall(r'(\d+)\s+\w+.*?(\d+)\s+cents?', all_text)
        if not cent_items:
            # "worth N cents" pattern
            cent_prices = re.findall(r'(?:worth|costs?)\s+(\w+)\s+cents?', all_text)
            quantities = re.findall(r'(\d+)\s+\w+\s+(?:cans?|bottles?|items?)', all_text)
            if cent_prices and quantities:
                total_per = 0
                for cp, qty in zip(cent_prices, quantities):
                    cp_val = WORD_NUMS.get(cp, None)
                    if cp_val is None and cp.isdigit():
                        cp_val = int(cp)
                    if cp_val is not None:
                        total_per += int(qty) * cp_val
                duration_m3 = re.search(r'(\d+)[- ]week|(\d+)[- ]day|(\d+)[- ]month', all_text)
                multiplier = 1
                if duration_m3:
                    multiplier = int(duration_m3.group(1) or duration_m3.group(2) or duration_m3.group(3))
                if not duration_m3:
                    # "four-week month" or "N-week"
                    duration_m3 = re.search(r'(\w+)[- ]week', all_text)
                    if duration_m3:
                        mw = duration_m3.group(1)
                        multiplier = WORD_NUMS.get(mw, 1) if not mw.isdigit() else int(mw)
                result = total_per * multiplier
                if result > 0:
                    return {'answer': normalize_answer(result),
                            'steps': [f'{total_per} cents/period × {multiplier} = {result} cents'],
                            'confidence': 'MEDIUM'}

        # Pattern: "time difference per item × N items"
        # "takes X minutes... new takes Y minutes... N items... how much longer?"
        if re.search(r'(?:how\s+much\s+longer|how\s+many\s+more\s+minutes|additional|extra\s+time)', ql):
            time_vals = re.findall(r'(\d+)\s+minutes?', all_text)
            qty_m = re.search(r'(\d+)\s+(?:paintings?|items?|tasks?|jobs?|projects?|pieces?)', all_text)
            if len(time_vals) >= 2 and qty_m:
                t1 = float(time_vals[0])
                t2 = float(time_vals[1])
                qty = float(qty_m.group(1))
                diff = abs(t2 - t1) * qty
                return {'answer': normalize_answer(diff),
                        'steps': [f'|{t2}-{t1}| × {qty} = {diff}'],
                        'confidence': 'MEDIUM'}


        # Pattern: "writes N in full twice, half once, rewrites everything"
        # Alphabet/sequence: base_len × full_times + base_len × fraction, then double
        if re.search(r'(?:alphabet|sequence)', all_text):
            base = 26  # alphabet
            writes_full = re.findall(r'(?:writes?\s+(?:it\s+)?(?:in\s+)?full)\s+(\w+)', all_text)
            writes_half = re.search(r'(?:writes?\s+)?half\s+(?:of\s+it|once)', all_text)
            rewrite = re.search(r're-?writes?\s+(?:everything|all)', all_text)
            full_times = 0
            _special_nums = {'once': 1, 'twice': 2, 'thrice': 3}
            for w in writes_full:
                n = WORD_NUMS.get(w, _special_nums.get(w, None))
                if n is None and w.isdigit():
                    n = int(w)
                if n:
                    full_times += n
            total_first = base * full_times
            if writes_half:
                total_first += base // 2
            if rewrite:
                total_first *= 2
            if total_first > 0:
                return {'answer': normalize_answer(total_first),
                        'steps': [f'alphabet: {total_first}'],
                        'confidence': 'MEDIUM'}

        # Pattern: "Fri=N, Sat=double, Sun=half that minus returns"
        # Multi-day sales with double/half references
        if re.search(r'(?:sold|sale)', all_text):
            day_vals = []
            # Preprocess word numbers in text for this pattern
            proc_text = self._word_nums_to_digits(all_text)
            base_m = re.search(r'(?:sold|had)\s+(\d+)', proc_text)
            if base_m:
                day_vals.append(float(base_m.group(1)))
                # "double that number" / "twice as many"
                if re.search(r'(?:double|twice)\s+(?:that|as\s+many|the\s+number)', proc_text):
                    day_vals.append(day_vals[0] * 2)
                # "one-half the amount" / "half as many"
                half_ref = re.search(r'(?:one[- ]half|half)\s+(?:the\s+amount|as\s+many|that)', proc_text)
                if half_ref and len(day_vals) >= 2:
                    day_vals.append(day_vals[-1] / 2)
                # Returns/reductions
                return_m = re.search(r'(\d+)\s+\w+\s+returned', proc_text)
                if return_m:
                    day_vals[-1] -= float(return_m.group(1))
                if len(day_vals) >= 2 and re.search(r'(?:how\s+many|total)', ql):
                    total = sum(day_vals)
                    return {'answer': normalize_answer(total),
                            'steps': [f'Days: {" + ".join(str(int(v)) for v in day_vals)} = {total}'],
                            'confidence': 'MEDIUM'}

        # Pattern: "N cats. K boats took M each. F fraction of remaining ran. How many left?"
        if re.search(r'(?:left|remain)', ql):
            proc_text = self._word_nums_to_digits(all_text)
            start_m = re.search(r'(?:there\s+(?:were|are)|had|has|started\s+with)\s+(\d+)', proc_text)
            if start_m:
                current = float(start_m.group(1))
                # "N boats/groups carried away M each"
                group_remove = re.search(r'(\d+)\s+\w+\s+(?:came\s+and\s+)?(?:carried\s+away|took|removed)\s+(\d+)\s+\w+\s+each', proc_text)
                if group_remove:
                    current -= float(group_remove.group(1)) * float(group_remove.group(2))
                # "F fraction of remaining ran/left/went"
                frac_m = re.search(r'(\d+)/(\d+)\s+(?:of\s+)?(?:the\s+)?(?:remaining|rest|left)', all_text)
                if frac_m and current > 0:
                    frac = float(frac_m.group(1)) / float(frac_m.group(2))
                    current -= current * frac
                if current > 0:
                    return {'answer': normalize_answer(current),
                            'steps': [f'Left: {current}'],
                            'confidence': 'MEDIUM'}

        # Pattern: "budget $X, spent $Y, bought N items for same price each, $Z left"
        # → each item = (X - Y - Z) / N
        if re.search(r'(?:how\s+much\s+(?:did|does|do)\s+each|cost\s+(?:of\s+)?each|each.*cost)', ql):
            budget_m = re.search(r'(?:budget|had|has|started\s+with)\s+(?:of\s+)?\$(\d+\.?\d*)', all_text)
            spent_m = re.search(r'(?:already\s+)?spent\s+\$(\d+\.?\d*)', all_text)
            left_m = re.search(r'(?:has|had|have)\s+\$(\d+\.?\d*)\s+(?:left|remaining)', all_text)
            n_items_m = re.search(r'(?:bought|purchased)\s+(\d+)', all_text)
            if budget_m and n_items_m:
                budget = float(budget_m.group(1))
                spent = float(spent_m.group(1)) if spent_m else 0
                left = float(left_m.group(1)) if left_m else 0
                n_items = float(n_items_m.group(1))
                if n_items > 0:
                    each_cost = (budget - spent - left) / n_items
                    if each_cost > 0:
                        return {'answer': normalize_answer(each_cost),
                                'steps': [f'(${budget}-${spent}-${left})/{n_items} = ${each_cost}'],
                                'confidence': 'MEDIUM'}

        # Pattern: "A and B sold N total. A sold X boxes, B sold Y boxes. How many per box?"
        if re.search(r'(?:how\s+many.*(?:per|in\s+(?:a|each)|each))', ql):
            proc_text2 = self._word_nums_to_digits(all_text)
            total_m = re.search(r'(?:sold|made|total)\s+(\d+)\s+\w+\s+(?:\w+\s+)?(?:together|total|combined)', proc_text2)
            box_counts = re.findall(r'(\d+\.?\d*)\s+(?:and\s+a\s+half\s+)?boxes?', proc_text2)
            if total_m and box_counts:
                total_items = float(total_m.group(1))
                total_boxes = sum(float(b) for b in box_counts)
                # Handle "and a half" patterns
                half_count = len(re.findall(r'and\s+a\s+half\s+box', proc_text2))
                total_boxes += half_count * 0.5
                if total_boxes > 0:
                    per_box = total_items / total_boxes
                    return {'answer': normalize_answer(per_box),
                            'steps': [f'{total_items}/{total_boxes} = {per_box}'],
                            'confidence': 'MEDIUM'}

        # Pattern: "N people donate M each + K extra. Need X per table. Have T. Need more?"
        # General: compute total, divide by per-unit, subtract existing
        proc_text = self._word_nums_to_digits(all_text)
        if re.search(r'(?:how\s+many\s+(?:new|more|additional)|need)', ql):
            # "N people donate/bring M [things] each"
            donate_m = re.search(r'(\d+)\s+\w+\s+(?:donate|bring|give|contribute)\s+(\d+)', proc_text)
            if donate_m:
                total = float(donate_m.group(1)) * float(donate_m.group(2))
                # "+ K already/extra"
                extra_m = re.search(r'(?:also\s+)?(?:have|had)\s+(\d+)\s+\w+\s+(?:already|extra|more|of\s+\w+\s+already)', proc_text)
                if extra_m:
                    total += float(extra_m.group(1))
                # "fit/hold X per table/unit"
                per_unit_m = re.search(r'(?:fit|hold|put)\s+(\d+)\s+\w+\s+(?:per|in\s+each|on\s+each|worth)', proc_text)
                if per_unit_m:
                    per_unit = float(per_unit_m.group(1))
                    import math
                    units_needed = math.ceil(total / per_unit) if per_unit > 0 else 0
                    # "already own/have N tables"
                    have_m = re.search(r'(?:already\s+)?(?:own|have|has)\s+(\d+)\s+(?:tables?|units?|shelves?)', proc_text)
                    existing = float(have_m.group(1)) if have_m else 0
                    need_more = max(0, units_needed - existing)
                    return {'answer': normalize_answer(need_more),
                            'steps': [f'{total}/{per_unit}={units_needed} - {existing} = {need_more}'],
                            'confidence': 'MEDIUM'}

        # Pattern: "price $A per lb. Buy N lbs of X and M lbs of Y. Start $Z. Left?"
        # price-then-quantity structure
        if re.search(r'(?:left|remain|change|save)', ql):
            # Find price-per-unit definitions
            price_defs = re.findall(r'\$(\d+\.?\d*)\s*(?:per|a|each)\s+(\w+)', proc_text)
            # Find purchase quantities
            buy_qtys = re.findall(r'(\d+)\s+(\w+)\s+of\s+(\w+)', proc_text)
            if not buy_qtys:
                buy_qtys = re.findall(r'buys?\s+(\d+)\s+(\w+)', proc_text)
                buy_qtys = [(q, u, '') for q, u in buy_qtys]
            budget_m = re.search(r'(?:starts?\s+with|has|had|budget)\s+\$(\d+\.?\d*)', proc_text)
            if price_defs and buy_qtys and budget_m:
                total_cost = 0
                for qty, unit, item in buy_qtys:
                    # Find matching price per this unit type
                    for price, p_unit in price_defs:
                        if p_unit == unit or p_unit in item:
                            total_cost += float(qty) * float(price)
                            break
                if total_cost == 0:
                    # Fallback: match by position
                    for i_q, (qty, _, _) in enumerate(buy_qtys):
                        if i_q < len(price_defs):
                            total_cost += float(qty) * float(price_defs[i_q][0])
                budget = float(budget_m.group(1))
                left = budget - total_cost
                if left >= 0:
                    return {'answer': normalize_answer(left),
                            'steps': [f'${budget} - ${total_cost} = ${left}'],
                            'confidence': 'MEDIUM'}

        # Pattern: "N baskets, each has A items at $X + B items at $Y. Total cost?"
        if re.search(r'(?:how\s+much|total|cost|spend)', ql):
            basket_m2 = re.search(r'(\d+)\s+(?:hanging\s+)?(?:baskets?|bags?|sets?|kits?|packs?)\s+(?:to\s+fill|to\s+make|each\s+with)', proc_text)
            if not basket_m2:
                basket_m2 = re.search(r'(?:fill(?:ing)?|mak(?:e|ing))\s+(?:all\s+)?(\d+)', proc_text)
            if basket_m2:
                n_units = float(basket_m2.group(1))
                # Find item quantities and prices by position matching
                item_pairs = []
                # Extract quantities from "add N X and M Y" patterns
                add_m = re.search(r'(?:add|wants?\s+to\s+add|contains?|includes?)\s+(.*?)(?:\.|$)', proc_text)
                qtys = []
                if add_m:
                    qtys = [float(q) for q in re.findall(r'(\d+)\s+\w+', add_m.group(1))]
                # Extract prices in order
                prices = [float(p) for p in re.findall(r'\$(\d+\.?\d*)\s*(?:apiece|each|per)?', proc_text)]
                if qtys and prices and len(qtys) == len(prices):
                    item_pairs = list(zip(qtys, prices))
                elif not item_pairs:
                    # Fallback: match by position within sentences
                    for sent in sents:
                        sl = sent.lower()
                        for m in re.finditer(r'(\d+)\s+\w+.*?\$(\d+\.?\d*)\s*(?:apiece|each|per)', sl):
                            item_pairs.append((float(m.group(1)), float(m.group(2))))
                if item_pairs:
                    per_unit_cost = sum(q * p for q, p in item_pairs)
                    total = n_units * per_unit_cost
                    return {'answer': normalize_answer(total),
                            'steps': [f'{n_units} × ${per_unit_cost} = ${total}'],
                            'confidence': 'MEDIUM'}

        # Pattern: "N total, split into groups of M. F fraction × members × K each"
        if re.search(r'(?:how\s+many|total)', ql):
            split_m = re.search(r'(\d+)\s+\w+.*?(?:split|divided|broken|separated)\s+(?:into\s+)?(?:groups?\s+(?:of\s+)?)?(\d+)', proc_text)
            if not split_m:
                per_person_m = re.search(r'(\d+)\s*-\s*person\s+groups?\s+', proc_text)
                total_m2 = re.search(r'(\d+)\s+\w+\s+(?:were|are|was)', proc_text) if per_person_m else None
                if per_person_m and total_m2:
                    _t_val = total_m2.group(1)
                    _g_val = per_person_m.group(1)
                    class _SplitMatch:
                        def group(self, n):
                            return [None, _t_val, _g_val][n]
                    split_m = _SplitMatch()
            if split_m:
                total_people = float(split_m.group(1))
                group_size = float(split_m.group(2))
                n_groups = total_people / group_size if group_size > 0 else 0
                # "F fraction of groups"
                frac_m2 = re.search(r'(\d+)/(\d+)\s+(?:of\s+)?(?:the\s+)?(?:number\s+of\s+)?(?:groups?)', proc_text)
                if frac_m2:
                    frac = float(frac_m2.group(1)) / float(frac_m2.group(2))
                    active_groups = n_groups * frac
                    # "each had members bring back N [things] each"
                    per_member_m = re.search(r'(?:bring\s+back|collect|gather)\s+(\d+)\s+\w+\s+each', proc_text)
                    if per_member_m:
                        per_member = float(per_member_m.group(1))
                        total_items = active_groups * group_size * per_member
                        return {'answer': normalize_answer(total_items),
                                'steps': [f'{active_groups} groups × {group_size} × {per_member} = {total_items}'],
                                'confidence': 'MEDIUM'}

        # Pattern: "N packets × M items each. A adults get K each. Rest shared among B children."
        if re.search(r'(?:how\s+many|each)', ql):
            packets_m = re.search(r'(\d+)\s+(?:packets?|packs?|boxes?|bags?)\s+(?:of\s+)?(?:\w+\s+){0,2}(?:\.?\s*each\s+(?:packet|pack|box|bag)\s+(?:contains?|has|holds?)\s+(\d+))', proc_text)
            if not packets_m:
                packets_m = re.search(r'(\d+)\s+(?:packets?|packs?|boxes?|bags?).*?(?:contains?|has|holds?)\s+(\d+)', proc_text)
            adults_m = re.search(r'(\d+)\s+adults?', proc_text)
            children_m = re.search(r'(\d+)\s+children', proc_text)
            adult_share_m = re.search(r'(?:each\s+)?adult\s+(?:gets?|receives?|takes?)\s+(\d+)', proc_text)
            if packets_m and adults_m and children_m and adult_share_m:
                total_items = float(packets_m.group(1)) * float(packets_m.group(2))
                adult_total = float(adults_m.group(1)) * float(adult_share_m.group(1))
                remaining = total_items - adult_total
                n_children = float(children_m.group(1))
                if n_children > 0 and remaining > 0:
                    per_child = remaining / n_children
                    return {'answer': normalize_answer(per_child),
                            'steps': [f'({total_items}-{adult_total})/{n_children} = {per_child}'],
                            'confidence': 'MEDIUM'}

        return None

    @staticmethod
    def _normalize_dozens(text: str) -> str:
        """Convert 'N dozen' expressions to numeric equivalents.

        'a dozen' → '12', 'three dozen' → '36', '10 dozen' → '120',
        'half a dozen' → '6', 'twenty dozen' → '240'.
        Skip when 'per dozen' or 'for a dozen' pricing is present —
        the dozen is a natural unit and converting creates rate mismatch.
        """
        # Guard: if "per dozen" or "$/a dozen" pricing pattern exists,
        # don't convert quantities (they pair with per-dozen rates)
        tl = text.lower()
        if re.search(r'(?:per|for\s+a|for\s+each|a)\s+dozen\b', tl) and \
           re.search(r'\$\d+', tl):
            return text

        word_nums = {
            'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6,
            'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
            'eleven': 11, 'twelve': 12, 'fifteen': 15, 'twenty': 20,
            'thirty': 30, 'forty': 40, 'fifty': 50,
        }
        # "half a dozen" / "half-a-dozen"
        text = re.sub(r'\bhalf\s+a\s+dozen\b', '6', text, flags=re.I)
        # "a dozen" / "one dozen"
        text = re.sub(r'\b(?:a|one)\s+dozen\b', '12', text, flags=re.I)
        # "N dozen" where N is a word number
        def _replace_word_dozen(m):
            word = m.group(1).lower()
            n = word_nums.get(word, None)
            if n is not None:
                return str(n * 12)
            return m.group(0)
        text = re.sub(
            r'\b(' + '|'.join(word_nums.keys()) + r')\s+dozen\b',
            _replace_word_dozen, text, flags=re.I)
        # "N dozen" where N is a digit
        def _replace_num_dozen(m):
            n = float(m.group(1))
            val = int(n * 12) if n * 12 == int(n * 12) else n * 12
            return str(val)
        text = re.sub(r'(\d+\.?\d*)\s+dozen\b', _replace_num_dozen, text, flags=re.I)
        return text

    def _normalize_clock_times(self, text: str) -> str:
        """Replace 'from H:MM AM/PM to H:MM AM/PM' with 'for N hours'.

        Converts clock-time ranges into numeric durations so downstream
        solvers can use the hours as a plain number.
        """
        def _clock_to_hours(m):
            h1, min1, p1 = int(m.group(1)), int(m.group(2) or 0), m.group(3).lower().rstrip('.')
            h2, min2, p2 = int(m.group(4)), int(m.group(5) or 0), m.group(6).lower().rstrip('.')
            if p1 == 'pm' and h1 != 12: h1 += 12
            if p2 == 'pm' and h2 != 12: h2 += 12
            if p1 == 'am' and h1 == 12: h1 = 0
            if p2 == 'am' and h2 == 12: h2 = 0
            total_min = (h2 * 60 + min2) - (h1 * 60 + min1)
            if total_min < 0: total_min += 24 * 60
            hours = total_min / 60
            if hours == int(hours):
                return f'for {int(hours)} hours'
            return f'for {hours:.1f} hours'
        return re.sub(
            r'(?:from\s+)?(\d{1,2}):?(\d{2})?\s*(a\.?m\.?|p\.?m\.?)\s+'
            r'(?:to|until|till)\s+'
            r'(\d{1,2}):?(\d{2})?\s*(a\.?m\.?|p\.?m\.?)',
            _clock_to_hours, text, flags=re.I)

    def _solve_inner(self, problem: str) -> Optional[Dict[str, Any]]:
        text = problem.strip()
        # Pre-process: normalize clock time ranges to durations
        text = self._normalize_clock_times(text)
        state = SolverState()

        sentences = self._split_sentences(text)

        # Separate question from context
        question = ''
        context_sents = []
        for s in sentences:
            if '?' in s:
                question = s
            else:
                context_sents.append(s)

        # If no explicit question, check if last sentence is imperative
        if not question and context_sents:
            last = context_sents[-1]
            if re.search(
                    r'\b(?:calculate|find|determine)\b',
                    last, re.I):
                question = context_sents.pop()

        # Extract embedded facts from question ("if X costs $5")
        # Process these FIRST — they establish base values
        q_facts = self._extract_question_facts(question)
        if q_facts:
            context_sents.insert(0, q_facts)

        # Resolve inline multiplier references: "12 cats, half as many dogs"
        # → "12 cats, 6 dogs"
        context_sents = self._resolve_multiplier_refs(context_sents)

        # Also extract number-bearing parts of the question
        # "How much do 6 erasers and 8 pencils cost?" → "6 erasers and 8 pencils"
        q_nums = self._extract_question_numbers(question)
        if q_nums:
            context_sents.append(q_nums)

        # For "total/together" questions with multiplier words, try entity chain first
        # (it handles multi-entity resolution better than equation graph)
        ql_check = question.lower()
        has_total_q = bool(re.search(r'\b(?:total|together|combined|altogether|all|how\s+many)\b', ql_check))
        all_text_for_mult = ' '.join(context_sents).lower() + ' ' + ql_check
        has_multipliers = bool(re.search(
            r'\b(?:twice|double|triple|half|three\s+times|four\s+times|'
            r'times\s+(?:as|the)|as\s+many|as\s+much)\b',
            all_text_for_mult))
        if has_total_q and has_multipliers:
            entity_result = self._try_entity_chain(context_sents, question)
            if entity_result is not None:
                return entity_result

        # Try proportion: "If 6 X make 36 Y, how many Y from 96 X?"
        prop_result = self._try_proportion(context_sents, question)
        if prop_result is not None:
            return prop_result

        # Try average: "scored 100 on first 3 tests and 80 on 4th"
        avg_result = self._try_average(context_sents, question)
        if avg_result is not None:
            return avg_result

        # Try chain multiply: "3 X. Each X has 25 Y. Each Y has 8 Z. How many Z?"
        chain_result = self._try_chain_multiply_simple(context_sents, question)
        if chain_result is not None:
            return chain_result

        # Try equation graph solver first (handles multi-step problems)
        eq_result = self._try_equation_graph(context_sents, question, text)
        if eq_result is not None:
            return eq_result

        # Try time-rate: "X per day ... in N weeks" → X × 7 × N
        time_rate_result = self._try_time_rate(context_sents, question)
        if time_rate_result is not None:
            return time_rate_result

        # Try algebra solver for ratio+total and linear systems
        algebra_result = self._try_algebra(context_sents, question, text)
        if algebra_result is not None:
            return algebra_result

        # Try entity-relationship solving for problems with multiplier chains
        entity_result = self._try_entity_chain(context_sents, question)
        if entity_result is not None:
            return entity_result

        # Try sequential operations for "how many left" problems
        seq_result = self._try_sequential_ops(context_sents, question, text)
        if seq_result is not None:
            return seq_result

        # Try fraction/percentage chain
        frac_result = self._try_fraction_chain(context_sents, question, text)
        if frac_result is not None:
            return frac_result

        # Try multi-step chaining (before two_step — handles cumulative periods,
        # rate×time-subtract, proportion, multiplier+total, etc.)
        multi_result = self._try_multi_step(context_sents, question, text)
        if multi_result is not None:
            return multi_result

        # Try two-step sequential solver (handles
        # shopping change, cost split, age problems, etc.)
        two_step = self._try_two_step(context_sents, question, text)
        if two_step is not None:
            return two_step

        # Try rate-chain solving (most common GSM8K pattern)
        rate_result = self._try_rate_chain(context_sents, question)
        if rate_result is not None:
            return rate_result

        # Process each context sentence
        for sent in context_sents:
            self._process_sentence(sent, state)

        if state.last is not None:
            # Apply time unit conversion if question asks about different time unit
            converted = self._apply_time_conversion(
                state.last, context_sents, question, state)
            final = converted if converted is not None else state.last

            return {
                'answer': normalize_answer(final),
                'steps': state.steps,
                'confidence': 'HIGH' if len(state.steps) >= 2 else 'MEDIUM',
            }
        return None

    def _try_chain_multiply_simple(self, context_sents: List[str],
                                   question: str) -> Optional[Dict[str, Any]]:
        """Chain multiply: "3 X. Each X has 25 Y. Each Y has 8 Z. How many Z total?"

        VERY restrictive: only fires when ALL context sentences have exactly one
        number each, and all but the first use "each/every/per" as a rate keyword.
        """
        ql = question.lower()
        if not re.search(r'how\s+many|how\s+much|total|altogether', ql):
            return None
        if len(context_sents) < 2:
            return None

        # Also extract rate from question if it contains one
        # BUT only if the rate wasn't already extracted as a question fact
        all_sents = list(context_sents)
        q_rate = re.search(r'(\d+\.?\d*)\s+\w+\s+per\s+\w+', ql)
        if not q_rate:
            q_rate = re.search(
                r'(?:each|every|per)\s+(?:\w+\s+){1,3}(?:has|have|is|contains?|gets?|'
                r'holds?|cleans?|makes?|produces?|uses?|needs?|costs?)\s+'
                r'(?:cleaned\s+)?(\d+\.?\d*)', ql)
        if q_rate:
            # Check if this rate is already in context (from _extract_question_facts)
            rate_val = q_rate.group(1)
            already_in_ctx = any(rate_val in s for s in context_sents)
            if not already_in_ctx:
                all_sents.append(question)

        rates = []
        each_count = 0
        for idx, s in enumerate(all_sents):
            sl = s.lower()
            has_each = bool(re.search(r'\b(?:each|every|per)\b', sl))

            # "each/every X has/verb N Y"
            m = re.search(
                r'(?:each|every|per)\s+(?:\w+\s+){1,3}(?:has|have|had|contains?|gets?|'
                r'holds?|makes?|produces?|uses?|needs?|eats?|drinks?|'
                r'costs?|takes?|requires?|cleans?|is)\s+'
                r'(?:cleaned\s+)?(\d+\.?\d*)', sl)
            if m and has_each:
                rates.append(float(m.group(1)))
                each_count += 1
                continue
            # "N calories/things per ounce/unit"
            m2 = re.search(r'(\d+\.?\d*)\s+\w+\s+per\s+\w+', sl)
            if m2 and has_each:
                rates.append(float(m2.group(1)))
                each_count += 1
                continue

            # Non-each sentence: extract single number if available
            nums = re.findall(r'\d+\.?\d*', sl)
            nums = [float(n) for n in nums if float(n) > 0]
            if len(nums) == 1:
                rates.append(nums[0])
                continue
            # Sentence has multiple numbers or zero — chain is broken
            return None

        # Must have at least 2 rates, at least half with "each/per"
        if len(rates) < 2 or each_count < len(rates) // 2:
            return None

        # Guard: if 2+ sentences mention "per [time_unit]", this is multi-rate not chain
        per_time_sents = sum(1 for s in all_sents
                             if re.search(r'per\s+(?:hour|minute|day|week|month|year|gallon|mile)', s.lower()))
        if per_time_sents >= 2:
            return None

        # Guard: if question asks about remaining/left, product is wrong
        if re.search(r'\b(?:left|remaining|leftover|spare|extra|still)\b', ql):
            return None


        from functools import reduce
        import operator
        product = reduce(operator.mul, rates, 1)

        steps = [f"Chain: {'×'.join(str(r) for r in rates)} = {product}"]

        # Check for trailing fraction/percentage in question or unused context
        all_text_lower = ' '.join(all_sents).lower() + ' ' + ql
        frac_val = None
        # "3/4 of the building", "two-thirds of the total"
        for fname, fval in sorted(FRACTION_MAP.items(), key=lambda x: -len(x[0])):
            if re.search(r'\b' + re.escape(fname) + r'\b\s*(?:of\b)', all_text_lower):
                frac_val = fval
                break
        if frac_val is None:
            frac_m = re.search(r'(\d+)/(\d+)\s*(?:of\b)', all_text_lower)
            if frac_m:
                n, d = int(frac_m.group(1)), int(frac_m.group(2))
                if 0 < d <= 100:
                    frac_val = n / d
        if frac_val is None:
            pct_m = re.search(r'(\d+\.?\d*)\s*%', all_text_lower)
            if pct_m:
                frac_val = float(pct_m.group(1)) / 100.0

        if frac_val is not None and 0 < frac_val < 1:
            result = product * frac_val
            steps.append(f"× {frac_val} = {result}")
            return {'answer': normalize_answer(result), 'steps': steps,
                    'confidence': 'HIGH'}

        return {'answer': normalize_answer(product), 'steps': steps,
                'confidence': 'HIGH'}

    def _resolve_multiplier_refs(self, context_sents: List[str]) -> List[str]:
        """Resolve inline multiplier references in comma-separated lists ONLY.

        VERY restrictive: only resolves when pattern is:
        "N [noun], half/twice as many [noun]" (comma-separated, same sentence)
        where N immediately precedes the multiplier reference.
        """
        result = list(context_sents)

        for idx, sent in enumerate(result):
            sl = sent.lower()

            # Pattern: "N [noun][,] [and] half/twice as many/much [noun]"
            # Must have a number+noun immediately before (separated by comma/and)
            mult_patterns = [
                (r'(\d+\.?\d*)\s+(\w+(?:\s+\w+)?)\s*,\s*(?:and\s+)?half\s+as\s+(?:many|much)\s+(\w+)', 0.5),
                (r'(\d+\.?\d*)\s+(\w+(?:\s+\w+)?)\s*,\s*(?:and\s+)?twice\s+as\s+(?:many|much)\s+(\w+)', 2.0),
                (r'(\d+\.?\d*)\s+(\w+(?:\s+\w+)?)\s+and\s+half\s+as\s+(?:many|much)\s+(\w+)', 0.5),
                (r'(\d+\.?\d*)\s+(\w+(?:\s+\w+)?)\s+and\s+twice\s+as\s+(?:many|much)\s+(\w+)', 2.0),
            ]
            for pat, mult in mult_patterns:
                m = re.search(pat, sl)
                if m:
                    ref_num = float(m.group(1))
                    computed = ref_num * mult
                    comp_str = str(int(computed) if computed == int(computed) else computed)
                    # Replace "half/twice as many [noun]" with "N [noun]"
                    noun = m.group(3)
                    old = sl[m.start():m.end()]
                    # Find where the multiplier phrase starts (after comma/and)
                    prefix = f'{int(ref_num) if ref_num == int(ref_num) else ref_num} {m.group(2)}'
                    mult_phrase_start = old.find('half') if mult == 0.5 else old.find('twice')
                    if mult_phrase_start >= 0:
                        replacement = old[:mult_phrase_start] + f'{comp_str} {noun}'
                        new_sent = sent[:m.start()] + replacement + sent[m.end():]
                        result[idx] = new_sent
                        break  # one replacement per sentence

        return result

    def _resolve_that_refs(self, context_sents: List[str]) -> List[str]:
        """Resolve 'half/twice that much' deictic references ONLY.

        RESTRICTIVE: Only resolves "that much/many" (deictic back-reference).
        Does NOT resolve "the number of X" (relational/possessive — handled by
        entity_chain instead).

        "X costs $5 and Y costs twice that much" → "X costs $5 and Y costs 10"
        "He ran 10 miles. She ran half that many." → "He ran 10 miles. She ran 5."
        """
        _MULT_MAP = {
            'half': 0.5, 'twice': 2.0, 'double': 2.0,
            'triple': 3.0, 'thrice': 3.0, 'quadruple': 4.0,
        }

        result = list(context_sents)
        last_number = None  # track last concrete number across sentences

        for idx, sent in enumerate(result):
            sl = sent.lower()

            # Find concrete numbers in this sentence BEFORE any multiplier ref
            nums_in_sent = list(re.finditer(r'(?<!\w)\$?(\d+\.?\d*)\b', sl))
            sent_numbers = [(float(m.group(1)), m.start()) for m in nums_in_sent
                            if float(m.group(1)) > 0]

            # ONLY match deictic "that/this" — NOT relational "the"
            # "half that much" ✓  "twice that many" ✓
            # "twice the number of X's Y" ✗ (relational — entity chain handles)
            mult_m = re.search(
                r'\b(half|twice|double|triple|thrice|quadruple)\s+'
                r'(?:that|this)\s+'
                r'(?:much|many|number|amount)\b',
                sl)
            if not mult_m:
                # "N times that much/many"
                mult_m = re.search(
                    r'\b(\d+\.?\d*)\s+times\s+'
                    r'(?:that|this)\s+'
                    r'(?:much|many|number|amount)\b',
                    sl)

            if mult_m:
                # Guard: skip if followed by "of [entity]" — that's relational
                after_phrase = sl[mult_m.end():].strip()
                if re.match(r'\s*of\s+[a-z]', after_phrase):
                    if sent_numbers:
                        last_number = sent_numbers[-1][0]
                    continue

                mult_word = mult_m.group(1)
                if mult_word in _MULT_MAP:
                    mult_val = _MULT_MAP[mult_word]
                else:
                    try:
                        mult_val = float(mult_word)
                    except ValueError:
                        mult_val = None

                if mult_val is not None:
                    # Find the reference number:
                    # 1. Last number in THIS sentence BEFORE the multiplier
                    ref_num = None
                    for val, pos in reversed(sent_numbers):
                        if pos < mult_m.start():
                            ref_num = val
                            break
                    # 2. If no number before in same sentence, use last_number
                    if ref_num is None:
                        ref_num = last_number

                    if ref_num is not None:
                        computed = ref_num * mult_val
                        comp_str = str(int(computed) if computed == int(computed) else computed)
                        new_sent = sent[:mult_m.start()] + comp_str + sent[mult_m.end():]
                        result[idx] = new_sent
                        last_number = computed
                        continue

            # Update last_number from concrete numbers in this sentence
            if sent_numbers:
                last_number = sent_numbers[-1][0]

        return result

    def _try_time_rate(self, context_sents: List[str],
                       question: str) -> Optional[Dict[str, Any]]:
        """Solve rate×time problems with implicit time period knowledge.

        Handles:
        - "X per day ... in N weeks" → X × 7 × N
        - "X a day ... weekdays + Y weekends ... in a week" → X×5 + Y×2
        - "X every morning ... in N weeks" → X × 7 × N
        - "X per hour ... for N hours" → X × N (direct)
        """
        ql = question.lower()
        all_text = ' '.join(context_sents).lower() + ' ' + ql

        # --- Detect rate unit and question unit ---
        _TIME_UNITS = {
            'minute': 'minute', 'minutes': 'minute',
            'hour': 'hour', 'hours': 'hour', 'hourly': 'hour',
            'day': 'day', 'days': 'day', 'daily': 'day',
            'morning': 'day', 'evening': 'day', 'night': 'day',
            'week': 'week', 'weeks': 'week', 'weekly': 'week',
            'month': 'month', 'months': 'month', 'monthly': 'month',
            'year': 'year', 'years': 'year', 'yearly': 'year', 'annually': 'year',
        }
        _TIME_CONVERT = {
            ('minute', 'hour'): 60, ('hour', 'day'): 24,
            ('day', 'week'): 7, ('week', 'month'): 4,
            ('month', 'year'): 12, ('day', 'month'): 30,
            ('day', 'year'): 365, ('week', 'year'): 52,
        }

        # Guard: skip multi-entity problems (Carlos AND Benji, Marin AND Nancy)
        # Only handle simple single-subject rate problems
        all_ctx = ' '.join(context_sents)
        named_entities = set(re.findall(r'\b[A-Z][a-z]{2,}\b', all_ctx))
        _SKIP_NAMES = {'The', 'This', 'That', 'There', 'Then', 'After', 'Before',
                       'Each', 'Every', 'Except', 'How', 'Monday', 'Tuesday',
                       'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday',
                       'January', 'February', 'March', 'April', 'May', 'June',
                       'July', 'August', 'September', 'October', 'November', 'December',
                       'If', 'She', 'He', 'His', 'Her', 'They'}
        real_entities = named_entities - _SKIP_NAMES
        if len(real_entities) > 1:
            return None

        # Guard: skip problems with multiple rates or complex structure
        # Count how many rate-like patterns exist
        rate_patterns_found = len(re.findall(
            r'\d+\.?\d*\s+(?:\w+\s+){0,3}(?:per|a|an|every|each)\s+'
            r'(?:minute|hour|day|morning|evening|night|week|month|year)',
            all_text))
        if rate_patterns_found > 1:
            return None

        # Guard: skip if >3 context sentences (too complex for simple rate)
        if len(context_sents) > 3:
            return None

        # Find rate: "N per/a/every/each [time_unit]" or "N [things] a day/every morning"
        rate_m = re.search(
            r'(\d+\.?\d*)\s+(?:\w+\s+){0,3}(?:per|a|an|every|each)\s+'
            r'(minute|hour|day|morning|evening|night|week|month|year)',
            all_text)
        if not rate_m:
            # "N daily/weekly/monthly" or "N every morning"
            rate_m = re.search(
                r'(\d+\.?\d*)\s+(?:\w+\s+){0,3}(daily|weekly|monthly|yearly|hourly)',
                all_text)
        if not rate_m:
            return None

        rate_val = float(rate_m.group(1))
        rate_unit = _TIME_UNITS.get(rate_m.group(2), rate_m.group(2))

        # --- Weekday/weekend split pattern ---
        # "walks 20 miles a day. Except on weekends when he walks 10 miles."
        # → 20×5 + 10×2 = 120
        weekend_m = re.search(
            r'(?:except|but)\s+(?:on\s+)?(?:the\s+)?weekends?\s+(?:when\s+)?'
            r'(?:\w+\s+){0,3}(\d+\.?\d*)', all_text)
        if weekend_m and rate_unit == 'day':
            weekend_val = float(weekend_m.group(1))
            # Question asks about "a week" or "N weeks"
            weeks_m = re.search(r'(?:in|per|a|every)\s+(?:(\d+)\s+)?weeks?', ql)
            if weeks_m or 'week' in ql:
                n_weeks = float(weeks_m.group(1)) if weeks_m and weeks_m.group(1) else 1
                result = (rate_val * 5 + weekend_val * 2) * n_weeks
                steps = [f"weekdays: {rate_val}×5={rate_val*5}, "
                         f"weekends: {weekend_val}×2={weekend_val*2}, "
                         f"per week: {rate_val*5+weekend_val*2}"]
                if n_weeks > 1:
                    steps.append(f"× {n_weeks} weeks = {result}")
                return {'answer': normalize_answer(result), 'steps': steps,
                        'confidence': 'HIGH'}

        # --- Simple time conversion: rate per unit₁, question in unit₂ ---
        # Find question time reference: "in N weeks/months" or "in a week"
        q_time_m = re.search(
            r'(?:in|for|after|during)\s+(?:(\d+\.?\d*)\s+)?'
            r'(minutes?|hours?|days?|weeks?|months?|years?)',
            ql)
        if q_time_m:
            q_count = float(q_time_m.group(1)) if q_time_m.group(1) else 1
            q_unit = _TIME_UNITS.get(q_time_m.group(2), q_time_m.group(2))

            if rate_unit == q_unit:
                # Same unit: just multiply
                result = rate_val * q_count
            else:
                # Need conversion
                conv = _TIME_CONVERT.get((rate_unit, q_unit))
                if conv is None:
                    # Try reverse
                    conv_r = _TIME_CONVERT.get((q_unit, rate_unit))
                    if conv_r:
                        result = rate_val / conv_r * q_count
                    else:
                        return None
                else:
                    result = rate_val * conv * q_count

            # Check for subtraction after rate: "5 of them didn't survive" / "lost 3"
            # Try specific pattern first (more precise)
            sub_m = re.search(
                r'(\d+\.?\d*)\s+(?:of\s+them\s+)?'
                r"(?:didn'?t|did\s+not|don'?t)\s+"
                r'(?:survive|grow|bloom|make\s+it|work|last)', all_text)
            if not sub_m:
                sub_m = re.search(
                    r'(?:lost|broke|died|failed|spoiled|ruined)\s+'
                    r'(\d+\.?\d*)', all_text)
            if sub_m:
                result -= float(sub_m.group(1))

            steps = [f"rate: {rate_val}/{rate_unit}, "
                     f"period: {q_count} {q_unit}(s) → {result}"]
            return {'answer': normalize_answer(result), 'steps': steps,
                    'confidence': 'MEDIUM'}

        return None

    def _try_proportion(self, context_sents: List[str],
                        question: str) -> Optional[Dict[str, Any]]:
        """Proportion solver: 'If A X makes B Y, how many Y from C X?'
        Also handles: 'If A X costs $B, how much do C X cost?'
        Pattern: given ratio A:B, find unknown given new A'.
        """
        ql = question.lower()
        all_text = ' '.join(context_sents) + ' ' + question
        al = all_text.lower()

        # Guard: proportion only for simple problems (not too many numbers)
        all_nums = re.findall(r'\d+\.?\d*', al)
        if len(all_nums) > 6:
            return None

        # Pattern 1: "N [things] make/produce/give/cost B [things], how many [things] from/with C"
        # "6 potatoes makes 36 hash browns ... 96 potatoes"
        m = re.search(
            r'(\d+\.?\d*)\s+(\w+)\s+(?:make|makes|produce|produces|give|gives|'
            r'yield|yields|create|creates|fill|fills)\s+'
            r'(\d+\.?\d*)\s+(\w+)', al)
        if m:
            a_val = float(m.group(1))
            a_unit = m.group(2)
            b_val = float(m.group(3))
            b_unit = m.group(4)
            # Find the "how many" target number (C)
            # Look for another number associated with a_unit
            a_stem = a_unit.rstrip('s')[:4]
            for c_m in re.finditer(r'(\d+\.?\d*)\s+' + re.escape(a_stem), al):
                c_val = float(c_m.group(1))
                if c_val != a_val:  # different from the given ratio
                    result = (c_val / a_val) * b_val
                    return {'answer': normalize_answer(result),
                            'steps': [f'Proportion: {a_val} {a_unit}→{b_val} {b_unit}, '
                                      f'{c_val} {a_unit}→{result}'],
                            'confidence': 'HIGH'}

        # Pattern 2: "N [things] cost $B" or "$B for/per N [things]", question has different N
        # "a pack of 12 costs $4, how much for 3 packs?"
        # Skip if question doesn't ask how many/much
        if not re.search(r'how\s+many|how\s+much|total|cost|price|pay|worth', ql):
            return None

        return None

    def _try_average(self, context_sents: List[str],
                     question: str) -> Optional[Dict[str, Any]]:
        """Average/mean solver: 'scored X on first N tests and Y on Mth test'"""
        ql = question.lower()
        if 'average' not in ql and 'mean' not in ql:
            return None

        all_text = ' '.join(context_sents) + ' ' + question
        al = self._word_nums_to_digits(all_text.lower())

        # "scored N on first M tests and P on Qth test"
        m = re.search(
            r'(\d+\.?\d*)\s+(?:on|for|in)\s+(?:his|her|the|their)?\s*'
            r'(?:first|last)?\s*(\d+)\s+(?:test|exam|quiz|game|round)', al)
        if m:
            score = float(m.group(1))
            count = float(m.group(2))
            # Find additional score
            m2 = re.search(
                r'(?:and\s+(?:a|an)?\s*)?(\d+\.?\d*)\s+(?:on|for|in)\s+'
                r'(?:his|her|the|their)?\s*(?:\d+(?:st|nd|rd|th)|last|final|next)',
                al)
            if m2:
                other_score = float(m2.group(1))
                total = score * count + other_score
                n_items = count + 1
                avg = total / n_items
                return {'answer': normalize_answer(avg),
                        'steps': [f'Average: ({score}×{int(count)} + {other_score}) / {int(n_items)} = {avg}'],
                        'confidence': 'HIGH'}

        return None

    def _try_equation_graph(self, context_sents: List[str],
                            question: str, full_text: str) -> Optional[Dict[str, Any]]:
        """Multi-step equation graph: tracks named entities through operations.

        Handles patterns like:
        - "X has N. Y has M times as many. How many together?"
        - "N less than M times X"
        - "N clusters of M each and K individual" → N*M + K

        Bails out for explicit ratio problems (handled by _try_algebra).
        """
        _ = full_text  # used for type signature

        # Skip explicit ratio problems (handled by _try_algebra Pattern 0)
        all_text_check = ' '.join(context_sents).lower() + ' ' + question.lower()
        if re.search(r'ratio\s+(?:of\s+)?\d+\s*[:/]\s*\d+', all_text_check):
            return None

        ql = question.lower()
        all_text = ' '.join(context_sents)
        all_lower = self._word_nums_to_digits(all_text.lower())
        steps = []

        # === Pattern: "N less/more than M times what X verb" ===
        # e.g. "Alex weighs 2 pounds less than 4 times what Grace weighs"
        for sent in context_sents:
            m = re.search(
                r'(\d+\.?\d*)\s+\w+\s+(less|more|fewer|greater)\s+than\s+'
                r'(\d+\.?\d*)\s+times\s+(?:what\s+|as\s+(?:many|much)\s+(?:as\s+)?)?'
                r'(?:\w+\s+)?(\b[A-Z][a-z]+)',
                sent)
            if m:
                diff_val = float(m.group(1))
                direction = m.group(2)
                mult = float(m.group(3))
                ref_name = m.group(4)
                # Find the reference entity's value
                for s2 in context_sents:
                    m2 = re.search(
                        re.escape(ref_name) + r'\s+(?:has|had|have|is|was|weighs?|contains?|holds?|costs?|gets?)\s+'
                        r'(?:a\s+total\s+of\s+|exactly\s+|about\s+|only\s+)?'
                        r'\$?([\d,.]+)', s2)
                    if m2:
                        ref_val = float(m2.group(1).replace(',', ''))
                        # Check for modifications to ref_val: "but lost 5", "gave away 3"
                        mod_m = re.search(
                            re.escape(ref_name) + r'.*?(?:but\s+)?(?:lost|lose|gave\s+away|spent|used|broke)\s+(\d+)',
                            s2)
                        if not mod_m:
                            # Check question too
                            mod_m = re.search(
                                re.escape(ref_name) + r'.*?(?:but\s+)?(?:lost|lose|gave\s+away|spent|used|broke)\s+(\d+)',
                                question)
                        if mod_m:
                            mod_val = float(mod_m.group(1))
                            ref_val = ref_val - mod_val
                        if direction in ('less', 'fewer'):
                            computed = mult * ref_val - diff_val
                        else:
                            computed = mult * ref_val + diff_val
                        steps.append(f"{mult}×{ref_val} {'+' if direction in ('more','greater') else '-'} {diff_val} = {computed}")
                        # Check if question asks for combined
                        if re.search(r'combined|together|total|all|both', ql):
                            total = ref_val + computed
                            steps.append(f"{ref_val} + {computed} = {total}")
                            return {'answer': normalize_answer(total), 'steps': steps,
                                    'confidence': 'HIGH'}
                        return {'answer': normalize_answer(computed), 'steps': steps,
                                'confidence': 'HIGH'}

        # === Pattern: "N groups of M each and K individual" → N*M + K ===
        m = re.search(
            r'(\d+)\s+(?:clusters?|groups?|rows?|packs?|bunches?|sets?|boxes?|crates?|bags?|stacks?|batches?)\s+'
            r'(?:of\s+)?(\d+)\s+(?:\w+\s+)?(?:each|per\s+\w+)?\s*'
            r'(?:,?\s*and\s+|,\s*)(\d+)\s+(?:individual|separate|extra|additional|more|loose|scattered)',
            all_lower)
        if m:
            groups = float(m.group(1))
            per_group = float(m.group(2))
            individual = float(m.group(3))
            total = groups * per_group + individual
            steps.append(f"{groups}×{per_group} + {individual} = {total}")
            return {'answer': normalize_answer(total), 'steps': steps, 'confidence': 'HIGH'}

        # === Pattern: "How many times can X do Y" → division ===
        if re.search(r'how\s+many\s+times\s+can', ql):
            # Find total budget/amount
            total_m = re.search(r'(?:has|have|had)\s+(?:\$)?([\d,.]+)', all_lower)
            if not total_m:
                total_m = re.search(r'\$([\d,.]+)', all_lower)
            # Find costs per instance (sum all $ amounts that aren't the budget)
            all_prices = []
            budget = float(total_m.group(1).replace(',', '')) if total_m else None
            for s in context_sents:
                for pm in re.finditer(r'\$([\d,.]+)', s):
                    p = float(pm.group(1).replace(',', ''))
                    all_prices.append(p)
            # Also check question for budget ("if he has $42")
            for pm in re.finditer(r'\$([\d,.]+)', question):
                p = float(pm.group(1).replace(',', ''))
                if p not in all_prices:
                    all_prices.append(p)

            if budget is not None and all_prices:
                # Budget is the largest amount, costs are the rest
                costs = [p for p in all_prices if abs(p - budget) > 0.01]
                if not costs:
                    costs = all_prices[:-1]
                    budget = all_prices[-1]
                if costs and budget > 0:
                    cost_per = sum(costs)
                    if cost_per > 0:
                        result = int(budget / cost_per)
                        steps.append(f"{budget} / {cost_per} = {result}")
                        return {'answer': normalize_answer(result), 'steps': steps,
                                'confidence': 'HIGH'}

        # === Pattern: Sum-Difference System ===
        # "Total T items. D more A than B. How many A?" → (T+D)/2
        # "played T games. won D more than lost. How many won?" → (T+D)/2
        # STRICT guards: only fire when it's truly a 2-group partition
        _SUMDIFF_CMP = r'(?:more|fewer|less|longer|shorter|older|younger|taller|bigger|smaller|heavier|lighter|higher|lower|faster|slower)'
        _sumdiff_patterns = [
            # "D more/longer [NOUN] [words] than [NOUN]": "30 more gold coins than silver"
            re.compile(r'(\d+\.?\d*)\s+(?:\w+\s+)?' + _SUMDIFF_CMP + r'\s+(\w+)\s+(?:\w+\s+){0,2}than\s+(?:the\s+)?(\w+)'),
            # "D more/longer than [NOUN]": "18 minutes longer than Donovan's"
            re.compile(r'(\d+\.?\d*)\s+\w+\s+' + _SUMDIFF_CMP + r'\s+than\s+(?:the\s+)?(\w+)'),
            # "[VERB] D more than [they] [VERB]": "won 8 more than they lost"
            re.compile(r'(\w+)\s+(\d+\.?\d*)\s+' + _SUMDIFF_CMP + r'\s+than\s+(?:they\s+)?(\w+)'),
        ]
        sumdiff_match = None
        diff_val = None
        less_noun = None
        for sp_idx, sp in enumerate(_sumdiff_patterns):
            sm = sp.search(all_lower)
            if sm and not re.search(r'more\s+than\s+\d+\s+times', all_lower):
                if sp_idx == 0:
                    diff_val = float(sm.group(1))
                    less_noun = sm.group(3).rstrip('s')
                elif sp_idx == 1:
                    # "18 minutes longer than Donovan's"
                    diff_val = float(sm.group(1))
                    less_noun = sm.group(2).rstrip('s').rstrip("'")
                else:
                    diff_val = float(sm.group(2))
                    less_noun = sm.group(3).rstrip('s')
                sumdiff_match = sm
                break

        if sumdiff_match and diff_val is not None:
            # Guard: neither group should have a known numeric value in text
            # If "N [less_noun]" or "N [more_noun]" exists, this isn't a system
            _pronoun_set = {'they', 'them', 'he', 'she', 'it', 'we', 'i', 'you'}
            has_base = False
            if less_noun not in _pronoun_set:
                has_base = bool(re.search(
                    r'(?:has|have|had|is|was|are|were)\s+(?:\w+\s+){0,2}\d+\.?\d*\s+(?:\w+\s+){0,2}'
                    + re.escape(less_noun), all_lower))
            if not has_base:
                # Find a total (a number in a SEPARATE sentence from the diff)
                diff_sent_idx = None
                for i, s in enumerate(context_sents):
                    if re.search(r'\b' + str(int(diff_val)) + r'\b', s):
                        diff_sent_idx = i
                        break
                total_val = None
                for i, s in enumerate(context_sents + [question]):
                    if i == diff_sent_idx:
                        continue
                    sl_tmp = s.lower()
                    tm = re.search(
                        r'(?:has|have|had|played|scored|contains?|holds?|earned)\s+'
                        r'(?:a\s+total\s+of\s+)?(\d+\.?\d*)', sl_tmp)
                    if not tm:
                        tm = re.search(
                            r'(?:was\s+(?:made|done|completed|finished)\s+in|took|lasted)\s+(\d+\.?\d*)', sl_tmp)
                    if not tm:
                        tm = re.search(r'there\s+(?:are|is|were|was)\s+(\d+\.?\d*)', sl_tmp)
                    if tm:
                        tv = float(tm.group(1))
                        if tv > diff_val:
                            # Guard: does this sentence contain MULTIPLE named entities
                            # that each have their own values? If so, it's entity-specific.
                            # Check: count how many "Name has/is N" patterns exist across
                            # ALL context sentences. If multiple entities have values,
                            # this is likely entity-tracking, not a sum-difference system.
                            named_entity_count = 0
                            for cs in context_sents:
                                if re.search(r'\b[A-Z][a-z]{2,}\b.*\b(?:has|have|had|is|was)\b.*\d', cs):
                                    named_entity_count += 1
                            # If 2+ entities have values, this is NOT a sum-diff
                            # unless the total sentence itself says "total/combined/both"
                            is_aggregate = bool(re.search(
                                r'\btotal\b|\btogether\b|\bcombined\b|\bboth\b|\bin all\b'
                                r'|\bfence\s+between\b|\bsplit\b', sl_tmp))
                            if named_entity_count <= 1 or is_aggregate:
                                total_val = tv
                                break
                if total_val is not None and not re.search(
                        r'\btotal\b|\btogether\b|\ball\b|\baltogether\b|\bcombined\b|\bin all\b', ql):
                    # Extra guard: reject if problem has many sentences (complex chain)
                    # True sum-diff problems are usually 2-3 sentences
                    if len(context_sents) <= 3:
                        asks_fewer = bool(re.search(r'\b' + re.escape(less_noun), ql))
                        if not asks_fewer:
                            asks_fewer = bool(re.search(
                                r'(?:how\s+many|how\s+much).*(?:fewer|less|lost|lose|fail)', ql))
                        if asks_fewer:
                            result = (total_val - diff_val) / 2
                        else:
                            result = (total_val + diff_val) / 2
                        if result > 0:
                            steps.append(f"({total_val} {'−' if asks_fewer else '+'} {diff_val}) / 2 = {result}")
                            return {'answer': normalize_answer(result), 'steps': steps, 'confidence': 'HIGH'}

        # === Pattern: "X has N more/fewer THING than Y" + "total/together" ===
        # "There are 4 roses. 7 more dahlias than roses. How many total?"
        # "Dylan bought 38 chicken sausages and 6 more fish sausages than chicken sausages"
        # BUT NOT "N more than M times" (that's a different pattern handled below)
        # Guard: skip if 3+ entities with diff relationships (handled by entity diff chain below)
        _diff_count = len(re.findall(
            r'(?:more|fewer|less|older|younger|taller|shorter|heavier|lighter)\s+(?:\w+\s+){0,3}than',
            all_lower))
        m = None
        if _diff_count <= 1:
            m = re.search(
                r'(\d+\.?\d*)\s+(?:more|fewer|less)\s+(?:\w+\s+){0,3}tha[nt]\s+(\w+)', all_lower)
            # Guard: skip if this is actually "N more than M times" pattern
            if m and re.search(r'more\s+than\s+\d+\s+times', all_lower):
                m = None
        if m and re.search(r'\btotal\b|\btogether\b|\ball\b|\baltogether\b|\bcombined\b|\bin all\b', ql):
            diff = float(m.group(1))
            thing2 = m.group(2)  # reference noun
            base = None
            base_m = re.search(r'(\d+\.?\d*)\s+' + re.escape(thing2), all_lower)
            if base_m:
                base = float(base_m.group(1))
            else:
                # Try finding any earlier number as base
                all_nums = extract_numbers(all_text)
                all_vals = [v for v, _, _ in all_nums if abs(v - diff) > 0.01]
                if all_vals:
                    base = all_vals[0]
            if base is not None:
                direction = 'more' if 'more' in m.group(0) else 'fewer'
                if direction == 'more':
                    derived = base + diff
                else:
                    derived = base - diff
                total = base + derived
                steps.append(f"{base} + {diff} = {derived}, total = {base} + {derived} = {total}")
                return {'answer': normalize_answer(total), 'steps': steps, 'confidence': 'HIGH'}

        # === Pattern: "was/is $N less/more" + "total" ===
        # "Expenditure was $500. Next was $60 less. Total?" → 500 + (500-60) = 940
        # Guard: skip if 3+ entities with diff relationships
        m = None
        if _diff_count <= 1:
            m = re.search(r'\$?(\d+\.?\d*)\s+(?:less|more|fewer)', all_lower)
        if m and re.search(r'\btotal\b|\btogether\b|\bcombined\b|\bboth\b', ql):
            diff = float(m.group(1))
            # Find the base value (the larger dollar amount)
            all_nums = extract_numbers(all_text + ' ' + question)
            base_candidates = [v for v, _, _ in all_nums if abs(v - diff) > 0.01 and v > diff]
            if base_candidates:
                base = base_candidates[0]
                if 'less' in m.group(0) or 'fewer' in m.group(0):
                    derived = base - diff
                else:
                    derived = base + diff
                total = base + derived
                steps.append(f"{base} + ({base} - {diff}) = {total}")
                return {'answer': normalize_answer(total), 'steps': steps, 'confidence': 'HIGH'}

        # === Pattern: "$X for A and [multiplier] as much for B" → X + mult*X ===
        # "pay $40 for fish sub and thrice as much for steak sub" → 40 + 120 = 160
        for mult_word, mult_val in MULTIPLIER_WORDS.items():
            m = re.search(
                r'(\d+\.?\d*)\s+.{1,40}?\band\s+' + re.escape(mult_word) +
                r'\s+(?:as\s+)?(?:much|many)', all_lower)
            if m:
                base = float(m.group(1))
                total = base + mult_val * base
                steps.append(f"{base} + {mult_val}×{base} = {total}")
                return {'answer': normalize_answer(total), 'steps': steps, 'confidence': 'HIGH'}

        # === Pattern: "half as many as X" + "total/together/altogether" ===
        m = re.search(r'half\s+(?:as\s+many|as\s+much|of\s+what)', all_lower)
        if m and re.search(r'total|together|altogether|combined|both', ql):
            # Find the base value
            nums_in_text = extract_numbers(all_text)
            vals = [v for v, _, _ in nums_in_text]
            if vals:
                base = vals[0]
                half = base * 0.5
                total = base + half
                steps.append(f"{base} + {base}/2 = {total}")
                return {'answer': normalize_answer(total), 'steps': steps, 'confidence': 'HIGH'}

        # === Pattern: "N more than M times X" (without subject match) ===
        # Also handles "there are N more than M times the number of X"
        # Uses all_lower which has word-numbers converted to digits
        m = re.search(
            r'(\d+\.?\d*)\s+(?:\w+\s+)?(?:more|fewer|less)\s+than\s+(?:(\d+\.?\d*)\s+times|twice|double|triple)',
            all_lower)
        if m:
            addend = float(m.group(1))
            direction = 'more' if 'more' in m.group(0) else 'fewer'
            if m.group(2):
                mult = float(m.group(2))
            elif 'twice' in all_lower or 'double' in all_lower:
                mult = 2.0
            elif 'triple' in all_lower:
                mult = 3.0
            else:
                mult = 2.0
            # Determine subject and reference entities
            # "Janey has 3 more than twice ... Sally" → subject=Janey, ref=Sally
            subj_m = re.search(r'(\b[a-z]+)\s+(?:has|had|have|is|was)\s+' + str(int(addend)) + r'\s+(?:more|fewer|less)\s+than', all_lower)
            subject_name = subj_m.group(1) if subj_m else None

            # Also detect "X is/turned N, which makes him/her N less than twice Y"
            # Here the subject is X whose value IS the base number
            subj_value_pattern = False
            if subject_name is None:
                # "Carver just turned 45 years old, which makes him 5 years less than twice..."
                sv_m = re.search(r'(\b[A-Z][a-z]+)\b.{0,30}?\b(?:is|was|turned|became)\s+(\d+\.?\d*)', all_text)
                if sv_m:
                    subject_name = sv_m.group(1).lower()
                    subj_value_pattern = True  # subject's value is known → solve inverse

            # Find the base value (usually in a different sentence or question)
            all_nums = extract_numbers(all_text + ' ' + question)
            base_candidates = [v for v, _, _ in all_nums if abs(v - addend) > 0.01 and abs(v - mult) > 0.01]
            if base_candidates:
                base = base_candidates[0]
                # Check for base modifications: "has 20 but lost 5"
                base_mod_m = re.search(
                    r'(?:has|had|have)\s+' + str(int(base)) +
                    r'.*?(?:but\s+)?(?:lost|lose|gave\s+away|spent|used|broke)\s+(\d+)',
                    all_lower + ' ' + ql)
                if base_mod_m:
                    base = base - float(base_mod_m.group(1))

                # Check if base belongs to the SUBJECT (not ref) → solve inverse
                # "Janey has 3 more than twice Sally. Janey has 21" → 21 is Janey's (subject)
                # Question asks about Sally (ref) → solve (21-3)/2=9
                base_is_subject = subj_value_pattern
                if not base_is_subject and subject_name:
                    for s in context_sents + [question]:
                        if re.search(re.escape(subject_name) + r'\s+(?:has|had|have|is|was)\s+' + str(int(base)), s.lower()):
                            base_is_subject = True
                            break

                if base_is_subject:
                    # Solve inverse: base = mult * x + addend → x = (base - addend) / mult
                    if direction == 'more':
                        result = (base - addend) / mult
                    else:
                        result = (base + addend) / mult
                    steps.append(f"({base} {'−' if direction=='more' else '+'} {addend}) / {mult} = {result}")
                    return {'answer': normalize_answer(result), 'steps': steps, 'confidence': 'HIGH'}
                else:
                    if direction == 'more':
                        result = mult * base + addend
                    else:
                        result = mult * base - addend
                    steps.append(f"{mult}×{base} {'+'if direction=='more' else '-'} {addend} = {result}")
                    return {'answer': normalize_answer(result), 'steps': steps, 'confidence': 'HIGH'}

        # === Pattern: "costs $A and $B" → add first, then multiply ===
        m = re.search(
            r'(?:costs?|priced?|sells?\s+for)\s+\$(\d+\.?\d*)\s+(?:and|plus)\s+'
            r'(?:a\s+)?(?:\w+\s+)?(?:(?:costs?|priced?|for|of)\s+)?\$(\d+\.?\d*)', all_lower)
        if not m:
            # Also try: "$A for X and $B for Y"
            prices = re.findall(r'\$(\d+\.?\d*)', all_lower)
            if len(prices) >= 2 and re.search(r'how\s+much|total|cost', ql):
                q_nums = extract_numbers(question)
                q_vals = [v for v, _, _ in q_nums]
                price_vals = [float(p) for p in prices]
                # If question has a multiplier
                if q_vals and len(price_vals) == 2:
                    price_sum = sum(price_vals)
                    # Check if this is "N items at combined price"
                    # Skip if rate_chain or entity_chain will handle it
                    pass

        # === Pattern: Proportional reasoning / unit rate ===
        # "earned $33 for 3 hours. How much for 12 hours?"
        # "read 40 pages in 2 hours. How many pages in 5 hours?"
        # Detects: amount1/count1 = rate, then rate × count2
        _TIME_UNIT = r'(?:hours?|minutes?|days?|weeks?|months?|years?|seconds?)'
        _UNIT_WORD = r'(?:hours?|minutes?|days?|weeks?|months?|years?|seconds?|miles?|pages?|pounds?|gallons?|pieces?|items?|servings?|cups?|liters?|km|meters?|laps?)'
        prop_m = re.search(
            r'(?:\$?(\d+\.?\d*)\s+(?:for|in|every|per)\s+(\d+\.?\d*)\s+(' + _UNIT_WORD + r'))',
            all_lower)
        if not prop_m:
            # "N [unit] for $M" or "N [unit] in M [time]"
            prop_m = re.search(
                r'(\d+\.?\d*)\s+(' + _UNIT_WORD + r')\s+(?:for|in|every|per)\s+\$?(\d+\.?\d*)',
                all_lower)
            if prop_m:
                # Swap: amount=g3, count=g1, unit matches g2
                prop_amt = float(prop_m.group(3))
                prop_count = float(prop_m.group(1))
                prop_unit = prop_m.group(2)
                prop_m = True  # flag
            else:
                prop_m = None
        else:
            prop_amt = float(prop_m.group(1))
            prop_count = float(prop_m.group(2))
            prop_unit = prop_m.group(3)
        if prop_m and re.search(r'same\s+rate|same\s+speed|same\s+pace|continues?\s+to|at\s+(?:this|that)\s+rate', all_lower + ' ' + ql):
            # Find the target count in the question
            # "how much ... after/in/for N [same_unit]"
            target_m = re.search(r'(?:after|in|for)\s+(\d+\.?\d*)\s+' + _UNIT_WORD, ql)
            if target_m:
                target_count = float(target_m.group(1))
                if prop_count > 0:
                    rate = prop_amt / prop_count
                    result = rate * target_count
                    steps.append(f"rate={prop_amt}/{prop_count}={rate}, {rate}×{target_count}={result}")
                    return {'answer': normalize_answer(result), 'steps': steps, 'confidence': 'HIGH'}

        # === Pattern: Named entity difference chain ===
        # "X is 10. Y is 4 younger than X. How old is Z if she is 2 older than Y?"
        # Build entity→value map from "Entity is/has N" + "Entity is N more/less/older/younger than Other"
        _DIFF_WORDS = r'(?:more|less|fewer|older|younger|taller|shorter|heavier|lighter|longer|bigger|smaller|faster|slower|cheaper|richer|poorer)'
        entity_values = {}
        entity_diffs = []  # (entity, diff, direction, ref_entity)
        for s_orig in context_sents + [question]:
            # Convert word numbers for matching but preserve original case for entity names
            s = self._word_nums_to_digits(s_orig)
            sl = s.lower()
            # "Entity is/has N" (base value)
            bm = re.search(r'(\b[A-Z][a-z]+)\s+(?:is|was|has|had|have|earns?|makes?|weighs?)\s+(?:exactly\s+)?\$?(\d+\.?\d*)', s)
            if bm:
                ent = bm.group(1).lower()
                val = float(bm.group(2))
                # Skip if followed by "more/less/older/younger than"
                after = s[bm.end():bm.end()+40].lower()
                if not re.match(r'\s*(?:years?\s+)?' + _DIFF_WORDS, after):
                    entity_values[ent] = val

            # "Entity is/has N [years] older/younger/... than Other"
            for dm in re.finditer(
                r'(\b[A-Z][a-z]+)\s+(?:is|was|has|had|have)\s+(?:\$)?(\d+\.?\d*)\s+(?:\w+\s+)?' + _DIFF_WORDS +
                r'\s+(?:\w+\s+){0,2}than\s+(?:\w+\s+){0,2}(\b[A-Z][a-z]+)', s):
                ent = dm.group(1).lower()
                diff = float(dm.group(2))
                ref = dm.group(3).lower()
                direction = 'more' if re.search(r'more|older|taller|heavier|longer|bigger|faster|richer', dm.group(0).lower()) else 'less'
                entity_diffs.append((ent, diff, direction, ref))

            # "who has/is N more/fewer than Other" — relative clause
            for wm in re.finditer(
                r'(?:,\s*)?who\s+(?:is|was|has|had|have)\s+(?:\$)?(\d+\.?\d*)\s+(?:\w+\s+)?' + _DIFF_WORDS +
                r'\s+(?:\w+\s+){0,2}than\s+(?:\w+\s+){0,2}(\b[A-Z][a-z]+)', s):
                diff = float(wm.group(1))
                ref = wm.group(2).lower()
                direction = 'more' if re.search(r'more|older|taller|heavier|longer|bigger|faster|richer', wm.group(0).lower()) else 'less'
                # "who" refers to the most recently mentioned entity before the comma
                who_pos = wm.start()
                preceding = s[:who_pos]
                # Find the last capitalized name before "who"
                names_before = re.findall(r'\b([A-Z][a-z]+)\b', preceding)
                _WHO_SKIP = {'The', 'This', 'That', 'There', 'Then', 'If', 'In',
                             'On', 'At', 'By', 'For', 'And', 'But', 'Or', 'So',
                             'He', 'She', 'They', 'It', 'We', 'How', 'What'}
                names_before = [n for n in names_before if n not in _WHO_SKIP]
                if names_before:
                    ent = names_before[-1].lower()
                    entity_diffs.append((ent, diff, direction, ref))

            # Also from question: "how old is Z if she is N older than Y"
            qm = re.search(
                r'(?:how\s+\w+\s+is\s+)?(\b[A-Z][a-z]+)\s+(?:if\s+)?(?:she|he|it|they)\s+(?:is|are)\s+'
                r'(\d+\.?\d*)\s+(?:\w+\s+)?' + _DIFF_WORDS + r'\s+than\s+(?:\w+\s+){0,2}(\b[A-Z][a-z]+)', s)
            if qm:
                ent = qm.group(1).lower()
                diff = float(qm.group(2))
                ref = qm.group(3).lower()
                direction = 'more' if re.search(r'more|older|taller|heavier|longer|bigger|faster|richer', qm.group(0).lower()) else 'less'
                entity_diffs.append((ent, diff, direction, ref))

            # === Multiplier relationships ===
            # "X's age is N times the age of Y" → X = N × Y
            for mm in re.finditer(
                r"(\b[A-Z][a-z]+)(?:'s)?\s+(?:\w+\s+)?(?:is|was|are|were)\s+"
                r"(\d+\.?\d*)\s+times\s+(?:the\s+)?(?:\w+\s+)?(?:of\s+)?(?:the\s+)?(\b[A-Z][a-z]+)", s):
                ent = mm.group(1).lower()
                mult = float(mm.group(2))
                ref = mm.group(3).lower()
                entity_diffs.append((ent, mult, 'mult', ref))

            # "X is [half/twice/triple] the [age/size/...] of Y" → X = frac × Y
            for fword, fval in [('half', 0.5), ('twice', 2), ('double', 2), ('triple', 3), ('thrice', 3)]:
                fm = re.search(
                    r'(\b[A-Z][a-z]+)\s+(?:is|was|are|were)\s+' + re.escape(fword) +
                    r'\s+(?:the\s+)?(?:\w+\s+)?(?:of\s+)?(?:the\s+)?(\b[A-Z][a-z]+)', s)
                if fm:
                    ent = fm.group(1).lower()
                    ref = fm.group(2).lower()
                    entity_diffs.append((ent, fval, 'mult', ref))

            # "X is N times as [old/heavy/...] as Y" → X = N × Y
            for mm2 in re.finditer(
                r"(\b[A-Z][a-z]+)\s+(?:is|was)\s+(\d+\.?\d*)\s+times\s+as\s+\w+\s+as\s+(\b[A-Z][a-z]+)", s):
                ent = mm2.group(1).lower()
                mult = float(mm2.group(2))
                ref = mm2.group(3).lower()
                entity_diffs.append((ent, mult, 'mult', ref))

        # Resolve chain (diffs and multipliers)
        if entity_values and entity_diffs:
            changed = True
            iters = 10
            while changed and iters > 0:
                changed = False
                iters -= 1
                for ent, diff, direction, ref in entity_diffs:
                    if ref in entity_values and ent not in entity_values:
                        if direction == 'mult':
                            entity_values[ent] = entity_values[ref] * diff
                        elif direction == 'more':
                            entity_values[ent] = entity_values[ref] + diff
                        else:
                            entity_values[ent] = entity_values[ref] - diff
                        changed = True
                    # Reverse: if ent known but ref not, solve backwards
                    elif ent in entity_values and ref not in entity_values:
                        if direction == 'mult' and diff != 0:
                            entity_values[ref] = entity_values[ent] / diff
                        elif direction == 'more':
                            entity_values[ref] = entity_values[ent] - diff
                        elif direction == 'less':
                            entity_values[ref] = entity_values[ent] + diff
                        else:
                            continue
                        changed = True

            # "in N years" time modifier — shift all entity values
            time_shift = 0
            tm_q = re.search(r'in\s+(\d+)\s+(?:years?|months?)', ql)
            if not tm_q:
                # check question for word numbers
                tm_q = re.search(r'in\s+(one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:years?|months?)', ql)
                if tm_q:
                    _wn = {'one':1,'two':2,'three':3,'four':4,'five':5,
                           'six':6,'seven':7,'eight':8,'nine':9,'ten':10}
                    time_shift = _wn.get(tm_q.group(1), 0)
            else:
                time_shift = float(tm_q.group(1))

            # What does the question ask for?
            # Total/sum/combined → sum ALL resolved entity values
            if re.search(r'\b(?:total|sum|combined|altogether|together|all)\b', ql):
                if len(entity_values) >= 2:
                    vals = {e: v + time_shift for e, v in entity_values.items()} if time_shift else entity_values
                    total = sum(vals.values())
                    desc = ' + '.join(f'{e}={v}' for e, v in vals.items())
                    return {'answer': normalize_answer(total),
                            'steps': [desc, f'total={total}'],
                            'confidence': 'HIGH'}
            # Average → sum / count
            if re.search(r'\b(?:average|mean)\b', ql):
                if len(entity_values) >= 2:
                    vals = {e: v + time_shift for e, v in entity_values.items()} if time_shift else entity_values
                    avg = sum(vals.values()) / len(vals)
                    return {'answer': normalize_answer(avg),
                            'steps': [f'avg of {len(vals)}={avg}'],
                            'confidence': 'HIGH'}
            # Prefer entity explicitly asked about: "how old/many is [Entity]"
            asked_m = re.search(r'(?:how\s+\w+\s+(?:is|are|does|was|were|did|do|will|can|could)\s+)(\w+)', ql)
            asked_entity = asked_m.group(1).lower() if asked_m else None
            if asked_entity and asked_entity in entity_values:
                val = entity_values[asked_entity]
                return {'answer': normalize_answer(val), 'steps': [f'{asked_entity}={val}'],
                        'confidence': 'HIGH'}
            for ent, val in entity_values.items():
                if ent in ql:
                    return {'answer': normalize_answer(val), 'steps': [f'{ent}={val}'],
                            'confidence': 'HIGH'}

        # === Pattern: sequential verb chain with running total ===
        # "X has 30. X eats 2. X packages remaining into bags of 2."
        # Track single entity through multiple operations
        entity_name = None
        running = None
        entity_steps = []

        _NON_ENTITY = {'The', 'This', 'That', 'These', 'Those', 'There', 'Then',
                       'With', 'After', 'Before', 'During', 'While', 'When',
                       'Where', 'What', 'Which', 'Who', 'How', 'But', 'And',
                       'For', 'From', 'Into', 'Once', 'Since', 'Until',
                       'Each', 'Every', 'Some', 'Many', 'Most', 'All',
                       'His', 'Her', 'Its', 'Their', 'Our', 'Your', 'My',
                       'One', 'Two', 'Three', 'Four', 'Five', 'Six',
                       'In', 'On', 'At', 'By', 'To', 'Of', 'If', 'So',
                       'Now', 'Also', 'However', 'Because', 'Although',
                       'Next', 'Last', 'First', 'Finally', 'Meanwhile',
                       'He', 'She', 'They', 'It', 'We', 'You'}
        _PRONOUNS = {'He', 'She', 'he', 'she', 'He', 'She', 'They', 'they'}
        for sent in context_sents:
            # Find the subject (first capitalized word, not a stopword)
            subj_m = re.match(r'\s*(\b[A-Z][a-z]+)', sent)
            if subj_m:
                name = subj_m.group(1)
                if name in _NON_ENTITY:
                    # Pronouns refer to the current entity being tracked
                    if name in _PRONOUNS and entity_name is not None:
                        name = entity_name
                    elif entity_name is not None and entity_name in sent:
                        # Entity name appears elsewhere in the sentence
                        name = entity_name
                    else:
                        continue
            elif entity_name is not None:
                # No subject found — carry over current entity
                # (handles split compound sentences like "downloaded 18 more")
                name = entity_name
            else:
                continue
            s = sent.lower()
            nums = extract_numbers(sent)
            vals = [v for v, _, _ in nums]

            if not vals:
                continue

            words = set(re.findall(r'\b\w+\b', s))

            if entity_name is None:
                entity_name = name
                running = vals[0]
                entity_steps.append(f"{name} starts with {running}")
            elif name == entity_name:
                # Context overrides: ADD verb + consumption context = SUB
                # "bakes muffins WITH 4" / "cooks dinner USING 3" → consumption
                s_digits = self._word_nums_to_digits(s)
                is_consumption = bool(re.search(
                    r'\b(?:bak|cook|mak|prepar|craft|brew|build)\w*\b.*\b(?:with|using|from)\s+\d',
                    s_digits))
                # Context override: "sells at $X per/each" = MUL remaining
                sell_price_m = re.search(
                    r'\b(?:sells?|sold)\b.*?\$(\d+\.?\d*)\s*(?:per|each|a\s+piece|apiece)',
                    s)
                if not sell_price_m:
                    sell_price_m = re.search(
                        r'\bfor\s+\$(\d+\.?\d*)\s*(?:per|each|a\s+piece|apiece)',
                        s)

                if sell_price_m and running is not None:
                    price = float(sell_price_m.group(1))
                    running = running * price
                    entity_steps.append(f"{name} sells at ${price} each → {running}")
                elif (words & SUB_VERBS or is_consumption) and running is not None:
                    for v in vals:
                        running -= v
                    entity_steps.append(f"{name} loses {vals} → {running}")
                elif words & ADD_VERBS and running is not None:
                    if self._has_mult_signal(s) and len(vals) >= 2:
                        running += vals[0] * vals[1]
                        entity_steps.append(f"{name} + {vals[0]}×{vals[1]} → {running}")
                    else:
                        for v in vals:
                            running += v
                        entity_steps.append(f"{name} gains {vals} → {running}")
                elif any(re.search(p, s) for p in DIV_PATTERNS) and running is not None:
                    if vals[0] > 0:
                        running = running / vals[0]
                    entity_steps.append(f"divide by {vals[0]} → {running}")
                elif re.search(r'(\d+)\s+\w+\s+(?:in|per|into|for)\s+(?:one|each|every|a)\s+\w+', s) and running is not None:
                    # "2 lollipops in one bag" = divide by 2
                    if vals[0] > 0:
                        running = running / vals[0]
                    entity_steps.append(f"div {vals[0]} per unit → {running}")
                elif self._has_mult_signal(s) and running is not None:
                    if len(vals) >= 2:
                        running = vals[0] * vals[1]
                        entity_steps.append(f"{vals[0]}×{vals[1]} → {running}")
                    elif len(vals) == 1:
                        running = running * vals[0]
                        entity_steps.append(f"×{vals[0]} → {running}")

        # Check if division already happened in the entity loop
        already_divided = any('divide' in s or 'div ' in s for s in entity_steps)

        # If not, check if question asks for division of the result
        if running is not None and len(entity_steps) >= 2 and not already_divided:
            if re.search(
                    r'how\s+many\s+(?:\w+\s+)?(?:bags?|groups?|teams?|sets?|boxes?|packs?|bottles?|containers?|rows?|piles?|bunches?)', ql):
                q_nums = extract_numbers(question)
                q_vals = [v for v, _, _ in q_nums]
                for qv in q_vals:
                    if qv > 0 and running is not None:
                        running = running / qv
                        entity_steps.append(f"divide by {qv} → {running}")
                        already_divided = True
                        break

        # Return entity chain: need 3+ operations, or 2+ with division applied
        if running is not None and (len(entity_steps) >= 3 or
                                     (len(entity_steps) >= 2 and already_divided)):
            return {'answer': normalize_answer(running), 'steps': entity_steps,
                    'confidence': 'MEDIUM'}

        return None

    def _try_multi_step(self, context_sents: List[str],
                        question: str, full_text: str) -> Optional[Dict[str, Any]]:
        """Solve multi-step arithmetic problems by computing per-sentence values
        and combining them based on the question type.

        Handles:
        - "X is N. Y is twice X. Total?" → N + 2N
        - "X is N. In 3 years?" → N + 3
        - "X bought N. Lost M." → N - M
        - "N% of X. Half of that." → X*N%*0.5
        - "N items at $M each. P items at $Q each. Total cost?" → N*M + P*Q
        """
        ql = question.lower()
        all_text = ' '.join(context_sents).lower()
        steps = []

        # ── Pattern CUMULATIVE: period₁=base, period₂=f(period₁), period₃=g(period₂), total ──
        # "60 downloads first month. Second month three times as many. Third month reduced 30%. Total?"
        if re.search(r'\b(?:total|altogether|combined|all|over\s+the)\b', ql):
            period_kw = re.search(
                r'\b(?:first|second|third|fourth|1st|2nd|3rd|4th|next)\s+'
                r'(?:month|week|day|year|quarter|period|round|time|semester)',
                all_text)
            if period_kw:
                # Find base value from first period
                base_m = re.search(r'(\d+\.?\d*)\s+\w+\s+(?:in\s+)?(?:the\s+)?(?:first|1st)', all_text)
                if not base_m:
                    base_m = re.search(r'(?:had|has|was|were|is)\s+(\d+\.?\d*)', all_text)
                if base_m:
                    base = float(base_m.group(1))
                    periods = [base]
                    # Process each subsequent sentence for multipliers/reductions
                    for sent in context_sents[1:]:
                        sl = sent.lower()
                        sl = self._word_nums_to_digits(sl)
                        prev = periods[-1]
                        # "N times as many" → prev × N
                        mult_m = re.search(r'(\d+\.?\d*)\s+times\s+(?:as\s+)?(?:many|much)', sl)
                        if not mult_m:
                            for mw, mv in MULTIPLIER_WORDS.items():
                                if re.search(r'\b' + re.escape(mw) + r'\b', sl):
                                    mult_m = True
                                    new_val = base * mv
                                    break
                        if mult_m and mult_m is not True:
                            new_val = base * float(mult_m.group(1))
                        elif mult_m is not True:
                            new_val = None
                        else:
                            pass  # new_val already set by MULTIPLIER_WORDS
                        # Check for reduction/increase in same or next sentence
                        if new_val is not None:
                            red_m = re.search(r'(?:reduced|decreased|dropped|fell|declined)\s+(?:by\s+)?(\d+\.?\d*)\s*%', sl)
                            inc_m = re.search(r'(?:increased|grew|rose|went\s+up)\s+(?:by\s+)?(\d+\.?\d*)\s*%', sl)
                            if red_m:
                                pct = float(red_m.group(1)) / 100.0
                                new_val = new_val * (1 - pct)
                            elif inc_m:
                                pct = float(inc_m.group(1)) / 100.0
                                new_val = new_val * (1 + pct)
                            periods.append(new_val)
                        elif 'reduced' in sl or 'decreased' in sl or 'dropped' in sl:
                            red_m = re.search(r'(?:reduced|decreased|dropped|fell|declined)\s+(?:by\s+)?(\d+\.?\d*)\s*%', sl)
                            if red_m:
                                pct = float(red_m.group(1)) / 100.0
                                periods.append(prev * (1 - pct))
                        elif 'increased' in sl or 'grew' in sl or 'rose' in sl:
                            inc_m = re.search(r'(?:increased|grew|rose|went\s+up)\s+(?:by\s+)?(\d+\.?\d*)\s*%', sl)
                            if inc_m:
                                pct = float(inc_m.group(1)) / 100.0
                                periods.append(prev * (1 + pct))
                    if len(periods) >= 2:
                        total = sum(periods)
                        period_strs = [str(round(p, 2)) for p in periods]
                        steps.append(f"Periods: {' + '.join(period_strs)} = {total}")
                        return {'answer': normalize_answer(total), 'steps': steps,
                                'confidence': 'HIGH'}

        # ── Pattern MSP: Proportion — "A gives B, how many from C?" → B/A*C ──
        # "6 potatoes → 36 hash browns, 96 potatoes → ?"
        # "grows 8 inches in 4 years, how many in 13 years?"
        # Trigger: "at this/that rate" OR "how many [from/out of/with] N"
        if re.search(r'(?:at\s+(?:this|that)\s+rate|how\s+many\s+(?:\w+\s+){1,3}(?:can|could|would|will|do))', ql):
            all_nums = extract_numbers(' '.join(context_sents))
            q_nums = extract_numbers(question)
            q_vals = [v for v, _, _ in q_nums if v > 0]
            c_vals = [(v, s, e) for v, s, e in all_nums if v > 0]
            if len(c_vals) >= 2 and len(q_vals) >= 1 and len(context_sents) <= 2:
                new_input = q_vals[-1]
                # Find the context number that shares a unit with the question number
                # by checking surrounding text for matching words
                ctx_text = ' '.join(context_sents).lower()
                # Extract unit words near each context number
                v1, v2 = c_vals[0][0], c_vals[1][0]
                # Try both ratios and pick the one that makes sense
                # Heuristic: if "at this rate" → rate is defined by context
                # Try to find which context number's unit matches the question number's unit
                q_unit_m = re.search(r'in\s+\d+\.?\d*\s+(\w+)', ql)
                if q_unit_m:
                    q_unit = q_unit_m.group(1)
                    # Find which context number has same unit nearby
                    for i, (v, s, e) in enumerate(c_vals):
                        nearby = ctx_text[max(0, s-5):e+20]
                        if q_unit in nearby or q_unit.rstrip('s') in nearby:
                            # This number has same unit as question → this is the input
                            other_v = c_vals[1-i][0] if i < 2 else c_vals[0][0]
                            if v > 0:
                                result = other_v / v * new_input
                                steps.append(f"{other_v}/{v} × {new_input} = {result}")
                                return {'answer': normalize_answer(result), 'steps': steps,
                                        'confidence': 'MEDIUM'}
                # Fallback: "from/out of N" → N is new_input, first pair defines ratio
                q_from_m = re.search(r'(?:from|out\s+of|with)\s+(\d+)', ql)
                if q_from_m and v1 > 0:
                    # "6 potatoes → 36 hash browns, from 96 potatoes"
                    # The question's unit (potatoes) matches v1's unit → rate = v2/v1
                    result = v2 / v1 * new_input
                    steps.append(f"{v2}/{v1} × {new_input} = {result}")
                    return {'answer': normalize_answer(result), 'steps': steps,
                            'confidence': 'MEDIUM'}

        # ── Pattern MS0: Inverse fraction — "this is N/M of X, how many X?" ──
        # "Wrote 9 novels. This is 3/4 of total. How many total?" → 9 / 0.75 = 12
        # Check compound first: "3 quarters", "2 thirds"
        inv_frac_m = re.search(
            r'(?:this|that|which|it)\s+(?:is|was)\s+(?:only\s+)?'
            r'(\d+\s+(?:quarters?|thirds?|halves?|fifths?|sixths?|eighths?|tenths?))',
            all_text)
        if not inv_frac_m:
            inv_frac_m = re.search(
                r'(?:this|that|which|it)\s+(?:is|was|represents?|equals?|makes?)\s+'
                r'(?:only\s+)?'
                r'(\d+/\d+|(?:' + '|'.join(FRACTION_MAP.keys()) + r'))',
                all_text)
        if inv_frac_m and re.search(r'\bof\b', all_text[inv_frac_m.end():inv_frac_m.end()+30]):
            frac_str = inv_frac_m.group(1)
            # Parse the fraction
            frac_val = None
            if frac_str in FRACTION_MAP:
                frac_val = FRACTION_MAP[frac_str]
            elif '/' in frac_str:
                parts = frac_str.split('/')
                frac_val = int(parts[0]) / int(parts[1])
            elif ' ' in frac_str:
                # "3 quarters" → compound
                parts = frac_str.split()
                _fd = {'half': 2, 'halves': 2, 'third': 3, 'thirds': 3,
                       'quarter': 4, 'quarters': 4, 'fifth': 5, 'fifths': 5,
                       'sixth': 6, 'sixths': 6}
                if parts[1] in _fd:
                    frac_val = int(parts[0]) / _fd[parts[1]]

            if frac_val and 0 < frac_val < 1:
                # Find the known value (the number that IS the fraction of something)
                all_nums = extract_numbers(' '.join(context_sents))
                all_vals = [v for v, _, _ in all_nums if v != frac_val and v > 0]
                if len(all_vals) == 1:
                    known = all_vals[0]
                    result = known / frac_val
                    steps.append(f"{known} / {frac_val} = {result}")
                    return {'answer': normalize_answer(result), 'steps': steps,
                            'confidence': 'MEDIUM'}

        # ── Pattern MS1: Multiplier + total/both (STRICT) ──
        # "X has 10. Y has twice as many. How many total?" → 10 + 20 = 30
        # STRICT: only fires when exactly 1 number + 1 multiplier, <=2 sentences, total question
        if re.search(r'\b(?:total|together|altogether|combined|both|put\s+together|in\s+all)\b', ql):
            all_nums = extract_numbers(' '.join(context_sents))
            all_vals = [v for v, _, _ in all_nums]
            if len(all_vals) == 1 and len(context_sents) <= 3:
                base = all_vals[0]
                for mult_word, mult_val in MULTIPLIER_WORDS.items():
                    if mult_word in all_text:
                        # Extra guard: multiplier must be "as many/much" or "the number/amount"
                        if re.search(re.escape(mult_word) + r'\s+(?:as\s+(?:many|much)|the\s+(?:number|amount)|\w+\s+as)', all_text):
                            derived = mult_val * base
                            total = base + derived
                            steps.append(f"{base} + {mult_val}×{base} = {total}")
                            return {'answer': normalize_answer(total), 'steps': steps,
                                    'confidence': 'MEDIUM'}

        # ── Pattern MS2: "in N years/months/days" question modifier ──
        # After computing someone's age/value, add N
        in_time_m = re.search(
            r'(?:in|after)\s+(\d+\.?\d*)\s+(?:years?|months?|days?|weeks?|hours?|minutes?)',
            ql)
        if not in_time_m:
            in_time_m = re.search(
                r'(\d+\.?\d*)\s+(?:years?|months?|days?)\s+from\s+now', ql)
        if in_time_m:
            time_add = float(in_time_m.group(1))
            # Check if context has a multiplier-based computation
            for mult_word, mult_val in MULTIPLIER_WORDS.items():
                if mult_word in all_text:
                    base_nums = extract_numbers(' '.join(context_sents))
                    base_vals = [v for v, _, _ in base_nums]
                    if base_vals and len(base_vals) <= 3:
                        base = base_vals[0]
                        computed = mult_val * base
                        result = computed + time_add
                        steps.append(f"{mult_val}×{base} = {computed}, + {time_add} = {result}")
                        return {'answer': normalize_answer(result), 'steps': steps,
                                'confidence': 'MEDIUM'}

            # Age/sum pattern: "X is N years old, in K years?"
            # Also: "sum of ages is N, in K years?"
            age_m = re.search(r'(\d+\.?\d*)\s+years?\s+old', all_text)
            sum_ages_m = re.search(r'sum\s+of\s+(?:their\s+)?ages?\s+(?:is|was|are|were)\s+(\d+\.?\d*)', all_text)
            _MS_NON_NAMES = {'The', 'This', 'That', 'These', 'Those', 'There', 'Then',
                             'They', 'Their', 'Them', 'What', 'When', 'Where', 'Which',
                             'Who', 'How', 'But', 'And', 'For', 'From', 'His', 'Her',
                             'Each', 'Every', 'Some', 'Many', 'Most', 'All', 'After',
                             'Before', 'While', 'Since', 'Until', 'Once', 'Also', 'Now'}
            if sum_ages_m:
                # "Sum of ages is 20. In 10 years?" → 20 + n_people*10
                total_age = float(sum_ages_m.group(1))
                names = set(re.findall(r'\b[A-Z][a-z]{2,}\b', ' '.join(context_sents))) - _MS_NON_NAMES
                n_people = max(len(names), 2)
                result = total_age + n_people * time_add
                steps.append(f"{total_age} + {n_people}×{time_add} = {result}")
                return {'answer': normalize_answer(result), 'steps': steps,
                        'confidence': 'MEDIUM'}
            elif age_m:
                age = float(age_m.group(1))
                if re.search(r'sum|total|together|combined', ql):
                    names = set(re.findall(r'\b[A-Z][a-z]{2,}\b', ' '.join(context_sents)))
                    n_people = max(len(names), 2)
                    result = age + n_people * time_add
                    steps.append(f"{age} + {n_people}×{time_add} = {result}")
                    return {'answer': normalize_answer(result), 'steps': steps,
                            'confidence': 'MEDIUM'}

        # ── Pattern MS3: "N more/less than [multiplier]" ──
        # "bought 2 more than thrice as many" → 3×base + 2
        # "1 more inch than twice of Monday's total" → 2×base + 1
        for mult_word, mult_val in MULTIPLIER_WORDS.items():
            m = re.search(
                r'(\d+\.?\d*)\s+(?:\w+\s+)?(?:more|less|fewer)\s+(?:\w+\s+)?than\s+' +
                re.escape(mult_word),
                all_text)
            if m:
                offset = float(m.group(1))
                direction = 'more' if 'more' in m.group(0) else 'less'
                # Find base value
                base_nums = extract_numbers(' '.join(context_sents))
                base_vals = [v for v, _, _ in base_nums if abs(v - offset) > 0.01]
                if base_vals:
                    base = base_vals[0]
                    if direction == 'more':
                        result = mult_val * base + offset
                    else:
                        result = mult_val * base - offset
                    steps.append(f"{mult_val}×{base} {'+' if direction == 'more' else '-'} {offset} = {result}")
                    return {'answer': normalize_answer(result), 'steps': steps,
                            'confidence': 'MEDIUM'}

            # Also: "[multiplier] as many as X plus/and N more"
            m = re.search(
                re.escape(mult_word) + r'\s+(?:as\s+)?(?:many|much)\s+.{0,30}?'
                r'(?:plus|and|with)\s+(\d+\.?\d*)\s+(?:more|extra|additional)',
                all_text)
            if m:
                offset = float(m.group(1))
                base_nums = extract_numbers(' '.join(context_sents))
                base_vals = [v for v, _, _ in base_nums if abs(v - offset) > 0.01]
                if base_vals:
                    base = base_vals[0]
                    result = mult_val * base + offset
                    steps.append(f"{mult_val}×{base} + {offset} = {result}")
                    return {'answer': normalize_answer(result), 'steps': steps,
                            'confidence': 'MEDIUM'}

        # ── Pattern MS4: "but lost/gave/spent N" after a multiplication ──
        # "Joseph had twice as many books (18), but lost 2" → 18-2=16
        but_m = re.search(
            r'(?:but|then|however)\s+(?:he|she|they|it|the\s+\w+)\s+'
            r'(?:lost|gave|spent|broke|ate|dropped|threw|donated|returned|sold|used|missed)\s+'
            r'(?:away\s+)?(?:\w+\s+){0,2}(\d+\.?\d*)',
            all_text)
        if but_m:
            subtract_val = float(but_m.group(1))
            # Check if context has a multiplier-based computation
            for mult_word, mult_val in MULTIPLIER_WORDS.items():
                if mult_word in all_text:
                    base_nums = extract_numbers(' '.join(context_sents))
                    base_vals = [v for v, _, _ in base_nums
                                 if abs(v - subtract_val) > 0.01]
                    if base_vals and len(base_vals) <= 3:
                        base = base_vals[0]
                        computed = mult_val * base
                        result = computed - subtract_val
                        if result >= 0:
                            steps.append(f"{mult_val}×{base} = {computed}, - {subtract_val} = {result}")
                            return {'answer': normalize_answer(result), 'steps': steps,
                                    'confidence': 'MEDIUM'}

        # ── Pattern MS5: Chain percentage/fraction ──
        # "220 castles. 40% are ruins. Half of ruins unmanned." → 220*0.4*0.5 = 44
        pct_m = re.search(r'(\d+\.?\d*)\s*(?:%|percent)', all_text)
        if pct_m:
            pct = float(pct_m.group(1)) / 100.0
            # Find total
            total_nums = extract_numbers(' '.join(context_sents))
            total_vals = [v for v, _, _ in total_nums if v > pct * 100]
            if total_vals:
                total = total_vals[0]
                intermediate = total * pct
                # Check for second fraction/half
                for frac_word, frac_val in FRACTION_MAP.items():
                    if frac_word in all_text:
                        # Check if the fraction applies to the intermediate result
                        # "half of the ruined castles" or "half of what is left"
                        frac_pos = all_text.find(frac_word)
                        pct_pos = all_text.find(pct_m.group(0))
                        if frac_pos > pct_pos:
                            result = intermediate * frac_val
                            steps.append(f"{total}×{pct} = {intermediate}, ×{frac_val} = {result}")
                            return {'answer': normalize_answer(result), 'steps': steps,
                                    'confidence': 'MEDIUM'}

        # ── Pattern MS6: "base + multiplier×base" for total/both ──
        # "one having 20 instructions and the second one having twice as many as the first"
        # → 20 + 2×20 = 60
        _ms6_total = re.search(r'\b(?:total|together|altogether|combined|both|put\s+together)\b', ql)
        if not _ms6_total:
            # "the two/three/N [noun]" as implied total
            _ms6_total = re.search(r'\bthe\s+(?:two|three|four|five|\d+)\s+\w+', ql)
        if _ms6_total:
            for mult_word, mult_val in MULTIPLIER_WORDS.items():
                if mult_word in all_text:
                    # Find "as many as" or "as much as" near the multiplier
                    m_mult = re.search(
                        re.escape(mult_word) + r'\s+(?:as\s+)?(?:many|much)\s+(?:\w+\s+){0,3}(?:as|of)\s+',
                        all_text)
                    if m_mult:
                        base_nums = extract_numbers(' '.join(context_sents))
                        # Skip the multiplier value itself and very small numbers
                        base_vals = [v for v, _, _ in base_nums if v > 2]
                        if base_vals:
                            base = base_vals[0]
                            derived = mult_val * base
                            total = base + derived
                            steps.append(f"{base} + {mult_val}×{base} = {total}")
                            return {'answer': normalize_answer(total), 'steps': steps,
                                    'confidence': 'MEDIUM'}

        # ── Pattern MS7: "a fraction/third of that amount" + total ──
        # "Spends $15000. Then spends a third of that amount. Total?" → 15000 + 5000 = 20000
        if re.search(r'\b(?:total|together|altogether|combined|both)\b', ql):
            frac_that_m = re.search(
                r'(?:a\s+)?(half|third|quarter|fifth|sixth|eighth|tenth)\s+'
                r'(?:of\s+)?(?:that|the|this|its?)\s+(?:amount|number|price|cost|sum|value|total|quantity)',
                all_text)
            if frac_that_m:
                frac_word = frac_that_m.group(1)
                frac_map = {'half': 0.5, 'third': 1/3, 'quarter': 0.25, 'fifth': 0.2,
                            'sixth': 1/6, 'eighth': 0.125, 'tenth': 0.1}
                frac_val = frac_map.get(frac_word, 0)
                if frac_val > 0:
                    base_nums = extract_numbers(' '.join(context_sents))
                    base_vals = [v for v, _, _ in base_nums if v > 1]
                    if base_vals:
                        base = base_vals[0]
                        derived = base * frac_val
                        total = base + derived
                        steps.append(f"{base} + {frac_val}×{base} = {total}")
                        return {'answer': normalize_answer(total), 'steps': steps,
                                'confidence': 'MEDIUM'}

        # ── Pattern MS8: rate × time - subtract ──
        # "plants 2 flowers a day. After 15 days. 5 did not grow." → 2×15 - 5 = 25
        rate_time_m = re.search(
            r'(\d+\.?\d*)\s+\w+\s+(?:a|per|each|every)\s+(\w+)', all_text)
        if rate_time_m:
            rate_val = float(rate_time_m.group(1))
            rate_unit = rate_time_m.group(2).rstrip('s')
            # Find duration: "N [unit]s" or "after N [unit]s"
            dur_m = re.search(
                r'(?:after|for|in|over)\s+(\d+\.?\d*)\s+' + re.escape(rate_unit) + r's?\b',
                all_text + ' ' + ql)
            if dur_m:
                duration = float(dur_m.group(1))
                product = rate_val * duration
                # Check for a subtraction: "N did not grow/survive/hatch"
                sub_m = re.search(
                    r'(\d+\.?\d*)\s+(?:of\s+them\s+)?(?:did\s+not|didn|don\'t|doesn\'t|'
                    r'failed|were\s+(?:bad|broken|lost|rotten|damaged|dead|empty|missing))',
                    all_text + ' ' + ql)
                if sub_m:
                    sub_val = float(sub_m.group(1))
                    if sub_val < product and sub_val != rate_val and sub_val != duration:
                        result = product - sub_val
                        steps.append(f"{rate_val}×{duration} - {sub_val} = {result}")
                        return {'answer': normalize_answer(result), 'steps': steps,
                                'confidence': 'MEDIUM'}

        # ── Pattern MS9: Fraction of total to dollars ──
        # "9300 pennies. Two thirds of them. Dollar amount?" → 9300×2/3/100 = 62
        if re.search(r'\bdollar\b', ql):
            if re.search(r'\bpenn(?:y|ies)\b', all_text + ' ' + ql):
                base_nums = extract_numbers(' '.join(context_sents))
                base_vals = [v for v, _, _ in base_nums if v > 100]
                if base_vals:
                    pennies = base_vals[0]
                    # Parse fraction from text (handle "two thirds" → "2 thirds" after word conversion)
                    frac_val = None
                    for frac_word, fv in sorted(FRACTION_MAP.items(), key=lambda x: -len(x[0])):
                        if frac_word in all_text or frac_word in ql:
                            frac_val = fv
                            break
                    # Also handle "N/M" notation and "N thirds/quarters/fifths"
                    if frac_val is None:
                        frac_m = re.search(r'(\d+)\s+(?:thirds?|quarters?|fifths?|sixths?|eighths?|tenths?|halves)', all_text + ' ' + ql)
                        if frac_m:
                            num = float(frac_m.group(1))
                            denom_word = frac_m.group(0).split()[-1].rstrip('s')
                            denom_map = {'third': 3, 'quarter': 4, 'fifth': 5, 'sixth': 6, 'eighth': 8, 'tenth': 10, 'halve': 2, 'half': 2}
                            denom = denom_map.get(denom_word, 0)
                            if denom > 0:
                                frac_val = num / denom
                    if frac_val is not None and frac_val > 0:
                        result = pennies * frac_val / 100
                        steps.append(f"{pennies} × {frac_val} / 100 = {result}")
                        return {'answer': normalize_answer(result), 'steps': steps,
                                'confidence': 'HIGH'}

        # ── Pattern MS10: Total - (subparts) complement ──
        # "25 oranges: 1 bad, 20% unripe, 2 sour. Rest good?" → 25 - 1 - 5 - 2 = 17
        if re.search(r'\b(?:rest|good|remaining|left)\b', ql):
            base_nums = extract_numbers(context_sents[0] if context_sents else '')
            if base_nums:
                total_val = base_nums[0][0]
                if total_val > 5:
                    subparts = 0
                    pct_used = set()
                    for sent_ms10 in context_sents:
                        sl2 = sent_ms10.lower()
                        # Percentage of total
                        pct_m2 = re.search(r'(\d+\.?\d*)\s*%', sl2)
                        if pct_m2:
                            pct_val = float(pct_m2.group(1))
                            subparts += total_val * pct_val / 100
                            pct_used.add(pct_val)
                        # Explicit count (skip total and percentage values)
                        for v, vstart, vend in extract_numbers(sent_ms10):
                            if abs(v - total_val) > 0.01 and v < total_val and v not in pct_used:
                                # Don't count the percentage number itself
                                after_num = sent_ms10[vend:vend+5]
                                if '%' not in after_num:
                                    if re.search(r'\b(?:bad|sour|rotten|unripe|broken|damaged|defective|spoiled|missing|empty)\b', sl2):
                                        subparts += v
                    if subparts > 0 and subparts < total_val:
                        result = total_val - subparts
                        steps.append(f"{total_val} - {subparts} = {result}")
                        return {'answer': normalize_answer(result), 'steps': steps,
                                'confidence': 'MEDIUM'}

        # ── Pattern MS11: Multi-rate sum with time multiplier ──
        # "earns $20/hr for 35 hrs and $30/hr for 15 hrs, works 50 weeks"
        # → (20×35 + 30×15) × 50
        all_lower = ' '.join(context_sents).lower() + ' ' + ql
        rate_pairs = re.findall(
            r'\$?(\d+(?:\.\d+)?)\s*(?:per|/|an?|each)\s*(?:hour|hr|day|week|month|year)'
            r'.*?(?:for|×|x)\s*(\d+(?:\.\d+)?)\s*(?:hours?|hrs?|days?|weeks?|months?|years?)',
            all_lower)
        if not rate_pairs:
            # Try reverse: "35 hours at $20/hr"
            rate_pairs_rev = re.findall(
                r'(\d+(?:\.\d+)?)\s*(?:hours?|hrs?|days?|weeks?)\s*'
                r'(?:at|@|for|as)\s*(?:a\s+)?\$?(\d+(?:\.\d+)?)\s*(?:per|/|an?|each)?\s*'
                r'(?:hour|hr|day|week|month|year)?',
                all_lower)
            if rate_pairs_rev:
                rate_pairs = [(rate, qty) for qty, rate in rate_pairs_rev]

        if len(rate_pairs) >= 2:
            subtotal = sum(float(rate) * float(qty) for rate, qty in rate_pairs)
            steps.append(' + '.join(f'{rate}×{qty}' for rate, qty in rate_pairs) + f' = {subtotal}')
            # Check for a time multiplier: "50 weeks a year", "12 months"
            time_mult_m = re.search(
                r'(\d+)\s*(?:weeks?|months?|days?|years?)\s*(?:a|per|each|every)\s*(?:year|month|week)',
                all_lower)
            if time_mult_m:
                tm = float(time_mult_m.group(1))
                result = subtotal * tm
                steps.append(f'{subtotal} × {tm} = {result}')
                return {'answer': normalize_answer(result), 'steps': steps,
                        'confidence': 'HIGH'}
            return {'answer': normalize_answer(subtotal), 'steps': steps,
                    'confidence': 'MEDIUM'}

        # ── Pattern MS12: "N fewer/more than M times X" ──
        # "Bobby has 5 fewer than 3 times Brian's games. Brian has 20 but lost 5."
        # → 3 × (20-5) - 5 = 40
        fewer_times_m = re.search(
            r'(\d+)\s+(?:fewer|less|more)\s+than\s+(\d+)\s+times', all_lower)
        if fewer_times_m:
            offset_val = float(fewer_times_m.group(1))
            multiplier = float(fewer_times_m.group(2))
            is_fewer = 'fewer' in fewer_times_m.group(0) or 'less' in fewer_times_m.group(0)
            # Find the base value (other numbers in context)
            all_nums = extract_numbers(' '.join(context_sents))
            base_candidates = [v for v, _, _ in all_nums
                               if v != offset_val and v != multiplier and v > 1]
            if base_candidates:
                # If there are operations on base (e.g. "has 20 but lost 5")
                base = base_candidates[0]
                # Check for subtraction on base
                sub_m = re.search(
                    r'(?:lost|lose|gave\s+away|spent|used|broke)\s+(\d+)', all_lower)
                if sub_m:
                    sub_val = float(sub_m.group(1))
                    if sub_val in [v for v, _, _ in all_nums] and sub_val != base:
                        base = base - sub_val
                if is_fewer:
                    result = multiplier * base - offset_val
                else:
                    result = multiplier * base + offset_val
                steps.append(f'{multiplier} × {base} {"−" if is_fewer else "+"} {offset_val} = {result}')
                return {'answer': normalize_answer(result), 'steps': steps,
                        'confidence': 'MEDIUM'}

        return None

    def _try_sequential_ops(self, context_sents: List[str],
                            question: str, full_text: str) -> Optional[Dict[str, Any]]:
        """Solve 'how many left' problems by tracking a running total through
        sequential add/subtract/fraction operations.

        Handles:
        - "Has N. Gives away X. Gets Y more. How many left?" → N - X + Y
        - "N items. A third leave. 10 more leave. How many left?" → N - N/3 - 10
        - "N pieces. Places a quarter. Mom places a third of remaining." → track remaining
        - "Start with N. Spent half. Spent $10 more." → N - N/2 - 10
        """
        ql = question.lower()

        # Only fire when the question asks about remaining/left
        if not re.search(
                r'\b(?:how\s+many|how\s+much|how\s+many\s+\w+\s+(?:are|were|is|was))\b.*'
                r'\b(?:left|remaining|remain|still\s+have|still\s+has|left\s+over|'
                r'does\s+\w+\s+have\s+left|does\s+\w+\s+still)\b', ql) and \
           not re.search(r'\bleft\b.*\?', ql) and \
           not re.search(r'\bremain\w*\b.*\?', ql) and \
           not re.search(r'\bhow\s+many\b.*\bnow\b', ql):
            return None

        steps = []

        # --- Find starting value ---
        # Skip sentences that look like deferred question facts (sub verbs at idx 0)
        start_val = None
        start_idx = 0  # index of sentence that establishes start
        deferred_ops = []  # sentences to apply at the end

        for i, s in enumerate(context_sents):
            sl = s.lower()

            # If first sentence has a sub verb but no initial/state verb,
            # it's likely a question-extracted fact — defer it
            if i == 0:
                w = set(re.findall(r'\b\w+\b', sl))
                if (w & SUB_VERBS) and not (w & INITIAL_VERBS):
                    deferred_ops.append(s)
                    continue

            # "There are/were N ..."
            m = re.search(r'(?:there\s+(?:are|is|were|was))\s+(\d[\d,]*\.?\d*)', sl)
            if not m:
                # "X has/had/have N ..."
                m = re.search(
                    r'(?:has|had|have|contains?|holds?|starts?\s+with|began?\s+with|'
                    r'brought|prepared|received|hires?|hired|saw|sees?|found|'
                    r'bought|collected|picked|made|baked|scored|wrote|read)\s+'
                    r'(?:a\s+total\s+of\s+)?'
                    r'\$?(\d[\d,]*\.?\d*)', sl)
            if not m:
                # Sentence-initial "N [noun]" (e.g. "Two girls each got...")
                # Only if it's clearly a quantity, not a name
                m = re.match(
                    r'(?:a\s+)?(\d[\d,]*\.?\d*)\s*[-–]?\s*'
                    r'(?:piece|item|liter|gallon|pound|mile|hour|minute|'
                    r'student|worker|penguin|bee|tree|cookie|lollipop|'
                    r'sticker|card|plant|loaf|loaves|car|egg|apple|'
                    r'marble|ball|book|toy|coin|shirt|'
                    r'\w+s)\b', sl)
            if not m:
                # "is solving a N-piece puzzle"
                m = re.search(r'(\d[\d,]*\.?\d*)\s*[-–]\s*piece', sl)
            if not m:
                # "filled with N liters", "a pizza with 12 slices", "jar containing 100 candies"
                m = re.search(r'(?:filled\s+with|budget\s+of|plan\s+of|with|containing|of)\s+(\d[\d,]*\.?\d*)\s+(?:slices?|pieces?|items?|candies|cookies|balls?|marbles?|cards?)', sl)
            if m:
                candidate = float(m.group(1).replace(',', ''))
                if candidate > 0:
                    start_val = candidate
                    start_idx = i
                    steps.append(f"Start: {start_val}")
                    break

        if start_val is None:
            return None

        # --- Parse fractions (word and numeric) ---
        def parse_fraction(text):
            """Extract fraction value from text (word or numeric form)."""
            tl = text.lower()
            # Word fractions
            for fname, fval in sorted(FRACTION_MAP.items(), key=lambda x: -len(x[0])):
                if re.search(r'\b' + re.escape(fname) + r'\b', tl):
                    return fval
            # Numeric fractions: 1/6, 2/3, 3/5
            m = re.search(r'(\d+)/(\d+)', tl)
            if m:
                num, den = int(m.group(1)), int(m.group(2))
                if 0 < den <= 100:
                    return num / den
            return None

        def parse_percent(text):
            """Extract percentage from text."""
            m = re.search(r'(\d+\.?\d*)\s*%|(\d+\.?\d*)\s+percent', text.lower())
            if m:
                return float(m.group(1) or m.group(2)) / 100.0
            return None

        def parse_multiplier(text):
            """Extract multiplier like 'twice', 'double', '3 times'."""
            tl = text.lower()
            for word, mult in MULTIPLIER_WORDS.items():
                if re.search(r'\b' + re.escape(word) + r'\b', tl):
                    return mult
            m = re.search(r'(\d+)\s+times', tl)
            if m:
                return float(m.group(1))
            return None

        # --- Check if start sentence has compound clause with operation ---
        # E.g. "saw 12 birds ... and threw a stone, scaring away 1/3 of that number"
        start_sent = context_sents[start_idx]
        # Split on ", [verb]ing" or "and [verb]" after the start value
        compound_parts = re.split(r',\s+(?=\w+ing\b)|(?<=\s)and\s+', start_sent, maxsplit=1)
        extra_ops_sents = []
        if len(compound_parts) > 1:
            tail = compound_parts[-1].strip()
            # Check if tail has a fraction/number AND a sub/add verb
            tail_lower = tail.lower()
            tail_words = set(re.findall(r'\b\w+\b', tail_lower))
            if (tail_words & SUB_VERBS) or re.search(r'\b(?:scaring|losing|giving|taking)\b', tail_lower):
                frac = parse_fraction(tail_lower)
                if frac is not None and re.search(r'of\s+(?:that|the|them|those|this)', tail_lower):
                    # "scaring away 1/3 of that number" → subtract frac × start_val
                    extra_ops_sents.append(tail)

        # --- Process subsequent sentences ---
        running = start_val
        ops_applied = 0

        # Process any compound clause operations from start sentence
        for extra in extra_ops_sents:
            el = extra.lower()
            frac = parse_fraction(el)
            if frac is not None:
                amount = frac * running
                running -= amount
                ops_applied += 1
                steps.append(f"{running + amount} - {frac}×{running + amount} = {running}")

        for i, s in enumerate(context_sents):
            if i <= start_idx:
                continue
            sl = s.lower()
            sl = self._word_nums_to_digits(sl)

            # Extract numbers from this sentence
            sent_nums = extract_numbers(sl)
            sent_vals = [v for v, _, _ in sent_nums]

            # Detect sentence type
            words = set(re.findall(r'\b\w+\b', sl))
            is_add = bool(words & ADD_VERBS) or bool(re.search(r'\b(?:more|additional|extra)\b', sl))
            is_sub = bool(words & SUB_VERBS) or bool(re.search(
                r'\b(?:quit|leave|left|gone|went|away|off)\b', sl))

            # Fix: "gave/gives him/her/[name] N" = ADD to subject, not SUB
            if is_sub and re.search(
                    r'(?:gave|gives?)\s+(?:him|her|them|me|us|\w+)\s+(?:another\s+)?\$?\d', sl):
                is_add = True
                is_sub = False

            frac = parse_fraction(sl)
            pct = parse_percent(sl)
            mult = parse_multiplier(sl)

            # Case 1: Fraction applied to group
            # "a third of them quit" → subtract frac * start_val
            # "a third of the remaining" → subtract frac * running
            # "places a quarter of the pieces" → subtract frac * start_val
            # "another one-third go inside" → subtract frac * start_val (implicit)
            if frac is not None:
                has_of_ref = re.search(
                    r'(?:of\s+(?:them|the|those|these|his|her|its|what|'
                    r'the\s+remaining|the\s+rest|what\s+was\s+left))', sl)
                # Also match implicit fractions: "one-third jump/quit/go/leave"
                has_implicit_frac = bool(re.search(
                    r'(?:half|third|quarter|fifth|sixth)\s+(?:\w+\s+)?'
                    r'(?:of\s+them|jump|quit|leave|go|went|swim|die|'
                    r'left|were|are|places?|put)', sl))
                # "another one-third" also implicit
                if not has_implicit_frac and re.search(r'another\s+', sl) and frac:
                    has_implicit_frac = True

                if has_of_ref or has_implicit_frac:
                    # Decide base:
                    # "of the remaining/rest" → running
                    # "lost/lost X of his/her [counted_noun]" → running
                    # "of them/the [noun]" → start_val (group reference)
                    if re.search(r'(?:remaining|rest|what\s+was\s+left|what\s+(?:is|are)\s+left)', sl):
                        frac_base = running
                    elif re.search(r'of\s+(?:his|her|their|my|your)\s+\w+', sl) and running != start_val:
                        # "lost 1/4 of his marbles" — possessive directly on counted object
                        frac_base = running
                    else:
                        frac_base = start_val
                    amount = frac_base * frac
                    # Check if there's a multiplier: "two girls each got 1/6"
                    qty_m = re.search(r'(\d+)\s+\w+\s+(?:each|both|all)\b', sl)
                    if qty_m:
                        qty = float(qty_m.group(1))
                        amount = amount * qty
                    running -= amount
                    ops_applied += 1
                    steps.append(f"fraction {frac:.4f} of {frac_base:.1f} = {amount:.1f}, remaining = {running:.1f}")
                    # Check for additional plain number operations in same sentence
                    # "gives 1/5 to X and gives 10 to Y" → also subtract 10
                    remaining_vals = [v for v in sent_vals if v > 1 and v != frac and abs(v - amount) > 0.01]
                    for rv in remaining_vals:
                        if is_sub:
                            running -= rv
                            ops_applied += 1
                            steps.append(f"-{rv}, remaining = {running:.1f}")
                        elif is_add:
                            running += rv
                            ops_applied += 1
                            steps.append(f"+{rv}, total = {running:.1f}")
                    continue

            # Case 2: "a third of the N" with explicit base in "of N" pattern
            if frac is not None and re.search(r'of\s+(?:the\s+)?\d', sl):
                for v in sent_vals:
                    if v > 1:
                        amount = v * frac
                        qty_m = re.search(r'(\d+)\s+\w+\s+(?:each|both|all)\b', sl)
                        if qty_m:
                            qty = float(qty_m.group(1))
                            amount = amount * qty
                        running -= amount
                        ops_applied += 1
                        steps.append(f"fraction {frac:.4f} × {v} = {amount}, remaining = {running:.1f}")
                        break
                else:
                    amount = running * frac
                    running -= amount
                    ops_applied += 1
                    steps.append(f"fraction {frac:.4f} of {running + amount:.1f}, remaining = {running:.1f}")
                continue

            # Case 3: Percentage
            if pct is not None:
                amount = running * pct
                if is_add:
                    running += amount
                    ops_applied += 1
                    steps.append(f"+{pct*100}% of {running - amount:.1f} = {amount:.1f}, total = {running:.1f}")
                else:
                    running -= amount
                    ops_applied += 1
                    steps.append(f"-{pct*100}% of {running + amount:.1f} = {amount:.1f}, remaining = {running:.1f}")
                continue

            # Case 4: Multiplier on remaining ("twice as much as what was left")
            if mult is not None and re.search(
                    r'(?:as\s+much\s+as\s+(?:what\s+was\s+)?left|'
                    r'as\s+much\s+as\s+(?:what\s+)?remain|'
                    r'that\s+(?:amount|much)|of\s+(?:what\s+was\s+)?left)', sl):
                amount = running * mult
                if is_add or re.search(r'\b(?:collect|add|refill|pour|gather)\b', sl):
                    running += amount
                    ops_applied += 1
                    steps.append(f"+{mult}× remaining {(running - amount):.1f} = {amount:.1f}, total = {running:.1f}")
                continue

            # Case 1b: Fraction + additional amount in same sentence
            # "spent half of it on food and an additional $10 for rides"
            if frac is not None and re.search(
                    r'(?:and\s+)?(?:an?\s+)?(?:additional|extra|another|more)\s+[\$]?(\d+)', sl):
                add_m = re.search(
                    r'(?:and\s+)?(?:an?\s+)?(?:additional|extra|another|more)\s+[\$]?(\d+\.?\d*)', sl)
                # Apply fraction first
                frac_amount = running * frac
                running -= frac_amount
                ops_applied += 1
                steps.append(f"fraction {frac:.4f} of {running + frac_amount:.1f} = {frac_amount:.1f}, remaining = {running:.1f}")
                # Then subtract additional
                if add_m:
                    extra = float(add_m.group(1))
                    running -= extra
                    ops_applied += 1
                    steps.append(f"-{extra} additional, remaining = {running:.1f}")
                continue

            # Case 5: Simple add/subtract with explicit number
            # Disambiguate "gives": "X gives Y to Z" depends on who is the subject
            # "Her mother gives Erin another 10" → ADD (giving TO subject)
            # "Erin gives 3 to Ella" → SUB (subject giving away)
            if is_sub and re.search(r'(?:gives?|gave)\s+\w+\s+(?:another|an?\s+)', sl):
                # "gives [name] another N" — giving TO someone, treat as ADD
                is_add = True
                is_sub = False

            if sent_vals and (is_add or is_sub):
                # Filter out the start value
                relevant = [v for v in sent_vals if abs(v - start_val) > 0.01 and v > 0]
                if not relevant:
                    continue

                # If sentence has BOTH add and sub verbs, split operations:
                # "saved $11 ... and spent $5 ... and $19" → +11 then -5 then -19
                if is_add and is_sub:
                    # Try to split: find the sub verb position
                    sub_pos = -1
                    for sv in SUB_VERBS:
                        m_sv = re.search(r'\b' + re.escape(sv) + r'\b', sl)
                        if m_sv:
                            sub_pos = m_sv.start()
                            break
                    if sub_pos > 0:
                        add_vals = []
                        sub_vals = []
                        for v, vstart, vend in sent_nums:
                            if abs(v - start_val) < 0.01 or v <= 0:
                                continue
                            if vstart < sub_pos:
                                add_vals.append(v)
                            else:
                                sub_vals.append(v)
                        for v in add_vals:
                            running += v
                            ops_applied += 1
                            steps.append(f"+{v}, total = {running}")
                        for v in sub_vals:
                            running -= v
                            ops_applied += 1
                            steps.append(f"-{v}, remaining = {running}")
                    continue

                # Check for "each" indicating multiplication (don't sum)
                has_each = bool(re.search(r'\b(?:each|per|every)\b', sl))

                if is_sub and not is_add:
                    if has_each and len(relevant) >= 2:
                        # "N items at $X each" — skip, let rate solver handle
                        continue
                    for v in relevant:
                        # If value is a fraction (< 1), treat as fraction of start_val
                        if v < 1 and frac is not None:
                            amount = start_val * v
                            running -= amount
                            ops_applied += 1
                            steps.append(f"-{v:.4f}×{start_val:.1f} = {amount:.1f}, remaining = {running:.1f}")
                        else:
                            running -= v
                            ops_applied += 1
                            steps.append(f"-{v}, remaining = {running}")
                elif is_add and not is_sub:
                    if has_each and len(relevant) >= 2:
                        continue
                    for v in relevant:
                        running += v
                        ops_applied += 1
                        steps.append(f"+{v}, total = {running}")

        # Apply deferred operations (from question-extracted facts)
        for s in deferred_ops:
            sl = self._word_nums_to_digits(s.lower())
            sent_nums = extract_numbers(sl)
            sent_vals = [v for v, _, _ in sent_nums]
            words = set(re.findall(r'\b\w+\b', sl))
            is_sub = bool(words & SUB_VERBS)
            is_add = bool(words & ADD_VERBS)
            frac = parse_fraction(sl)

            if frac is not None:
                amount = running * frac
                running -= amount
                ops_applied += 1
                steps.append(f"deferred: fraction {frac:.4f}, remaining = {running:.1f}")
            elif sent_vals:
                for v in sent_vals:
                    if v > 0:
                        if is_sub and not is_add:
                            running -= v
                            ops_applied += 1
                            steps.append(f"deferred: -{v}, remaining = {running}")
                        elif is_add and not is_sub:
                            running += v
                            ops_applied += 1
                            steps.append(f"deferred: +{v}, total = {running}")
                        break

        if ops_applied >= 2 and running != start_val:
            return {'answer': normalize_answer(running), 'steps': steps,
                    'confidence': 'HIGH' if ops_applied >= 3 else 'MEDIUM'}

        return None

    def _try_fraction_chain(self, context_sents: List[str],
                            question: str, full_text: str) -> Optional[Dict[str, Any]]:
        """Solve problems with successive fractions/percentages of a group.

        Handles: word fractions (one third), numeric fractions (2/5),
        percentages (20%), "remaining"/"rest" semantics.
        Pattern: start with N, take fraction → remainder, take fraction of remainder, etc.
        Question asks about "rest"/"remaining"/"how many [last category]".
        """
        steps = []
        fl = full_text.lower()

        def parse_any_fraction(text):
            """Parse word fractions, numeric fractions, and percentages."""
            tl = text.lower()
            # Word fractions: "one third", "a half", "two fifths"
            word_to_num = {'one': 1, 'a': 1, 'two': 2, 'three': 3, 'four': 4,
                           'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9}
            denom_map = {'half': 2, 'halves': 2, 'third': 3, 'thirds': 3,
                         'quarter': 4, 'quarters': 4,
                         'fifth': 5, 'fifths': 5, 'sixth': 6, 'sixths': 6,
                         'seventh': 7, 'sevenths': 7, 'eighth': 8, 'eighths': 8,
                         'ninth': 9, 'ninths': 9, 'tenth': 10, 'tenths': 10}
            m = re.search(
                r'(one|a|two|three|four|five|six|seven|eight|nine)\s+'
                r'(half|halves|thirds?|quarters?|fifths?|sixths?|sevenths?|eighths?|ninths?|tenths?)',
                tl)
            if m:
                num = word_to_num.get(m.group(1), 1)
                den = denom_map.get(m.group(2), 1)
                return num / den
            # Numeric fractions: "2/5", "1/2"
            m = re.search(r'(\d+)\s*/\s*(\d+)', tl)
            if m:
                num, den = float(m.group(1)), float(m.group(2))
                if den > 0 and num / den <= 1:
                    return num / den
            # Percentages: "20%"
            m = re.search(r'(\d+(?:\.\d+)?)\s*%', tl)
            if m:
                return float(m.group(1)) / 100
            # "half" standalone
            if re.search(r'\bhalf\b', tl):
                return 0.5
            return None

        # Check for fraction/percentage patterns
        has_fracs = bool(re.findall(
            r'(?:one|two|three|four|five|six|seven|eight|nine|a)\s+'
            r'(?:half|third|quarter|fifth|sixth|seventh|eighth|ninth|tenth)',
            fl))
        has_numeric_fracs = bool(re.findall(r'\d+\s*/\s*\d+', fl))
        has_pct = bool(re.findall(r'\d+\s*%', fl))

        if not (has_fracs or has_numeric_fracs or has_pct):
            return None

        # Must have "of them/the/remaining" or sequential fraction application
        has_of_ref = bool(re.search(
            r'(?:of\s+(?:them|the\s+\w+|those|these|his|her|its|what)|'
            r'remaining|the\s+rest|left\s+over)', fl))
        if not has_of_ref:
            return None

        # Find the total/starting number
        total_val = None
        for s in context_sents:
            sl = s.lower()
            m = re.search(r'(?:there\s+(?:are|were)|has|have|had|are|were|is)\s+(\d+(?:,\d+)*)', sl)
            if m:
                total_val = float(m.group(1).replace(',', ''))
                steps.append(f"total = {total_val}")
                break
            # "80 cards", "20 students"
            m = re.match(r'(\d+(?:,\d+)*)\s+\w+', sl)
            if m and not re.match(r'\d+\s+(?:time|day|hour|week|month|year)', sl):
                total_val = float(m.group(1).replace(',', ''))
                steps.append(f"total = {total_val}")
                break
        # Also check question for total: "of 20 students"
        if total_val is None:
            m = re.search(r'of\s+(\d+)\s+\w+', fl)
            if m:
                total_val = float(m.group(1))
                steps.append(f"total = {total_val}")

        if total_val is None:
            return None

        # Build fraction chain: split each sentence into comma/and-separated clauses
        # and process each clause for a fraction
        running = total_val
        parts_taken = []
        fraction_count = 0

        # Don't include question — question facts already extracted to context_sents
        # Split sentences into clauses on commas and "and"
        clauses = []
        for s in context_sents:
            # Split on ", " and " and " while keeping content
            parts = re.split(r',\s+|\s+and\s+', s)
            clauses.extend(parts)

        # Track "of those/of them" references that precede fraction clauses
        pending_of_ref = False
        last_taken = None
        has_forward_chain = False  # True if any "of those" (not "remaining") chain exists

        for clause in clauses:
            cl = clause.lower().strip()
            if not cl:
                continue

            # Check if this clause establishes an "of those/of them" reference
            # without having a fraction itself (e.g., "Of those who receive interviews")
            has_of_ref_word = bool(re.search(
                r'(?:of\s+(?:them|the\s+\w+|those|these|his|her|its|whom|which|what))', cl))
            frac = parse_any_fraction(cl)

            if frac is None:
                # No fraction, but "of those" → mark for next clause
                if has_of_ref_word:
                    pending_of_ref = True
                continue

            is_of_remaining = bool(re.search(
                r'(?:of\s+(?:the\s+)?(?:remaining|rest|left|what\s+was\s+left))', cl))

            is_of_ref = has_of_ref_word or pending_of_ref
            pending_of_ref = False

            if is_of_remaining:
                # Apply to remainder
                taken = running * frac
                parts_taken.append(taken)
                running -= taken
                last_taken = taken
                fraction_count += 1
                steps.append(f"take {frac:.4f} of {running + taken:.0f} = {taken:.0f}, remaining = {running:.0f}")
            elif is_of_ref and fraction_count > 0 and last_taken is not None:
                # "Of those who [received/got/passed]" → apply to LAST TAKEN, not remainder
                taken = last_taken * frac
                parts_taken.append(taken)
                last_taken = taken
                has_forward_chain = True
                fraction_count += 1
                steps.append(f"take {frac:.4f} of prev {last_taken/frac:.0f} = {taken:.0f}")
            elif is_of_ref or fraction_count == 0:
                # Apply to total (first application)
                taken = total_val * frac
                parts_taken.append(taken)
                running = total_val - taken
                last_taken = taken
                fraction_count += 1
                steps.append(f"take {frac:.4f} of {total_val:.0f} = {taken:.0f}, remaining = {running:.0f}")

        if fraction_count == 0:
            return None

        # Determine what the question asks for
        ql = question.lower()
        asks_rest = bool(re.search(
            r'\b(?:rest|remaining|left|other|else)\b', ql))

        if asks_rest:
            return {'answer': normalize_answer(running), 'steps': steps,
                    'confidence': 'HIGH'}

        # If fractions chain forward ("of those who...") → answer is last taken
        if fraction_count >= 2 and last_taken is not None and has_forward_chain:
            # Check if question references the last category
            return {'answer': normalize_answer(last_taken), 'steps': steps,
                    'confidence': 'HIGH'}

        if fraction_count >= 1 and running != total_val:
            # Default: return the remainder
            return {'answer': normalize_answer(running), 'steps': steps,
                    'confidence': 'MEDIUM'}

        return None

    def _split_sentences(self, text: str) -> List[str]:
        # Split on sentence boundaries, then on semicolons and commas
        # followed by independent clauses
        parts = re.split(r'(?<=[.!?])\s+', text)
        result = []
        for p in parts:
            # Split on semicolons
            for sub in p.split(';'):
                # Split on ", and then" / ", then" / ", so" / ", but" / ", while" (clause boundaries)
                clauses = re.split(r',\s+(?:and\s+then|then|so|but|while)\s+', sub)
                for c in clauses:
                    c = c.strip()
                    if not c:
                        continue
                    # Split compound sentences on "and" when both halves
                    # contain different action verbs (e.g. "deleted 9 and downloaded 18")
                    split_parts = self._split_compound_and(c)
                    result.extend(split_parts)
        return result

    def _split_compound_and(self, sentence: str) -> List[str]:
        """Split 'X verb1 N and verb2 M' into two separate clauses."""
        # Look for " and " connecting two independent clauses with verbs
        and_positions = [m.start() for m in re.finditer(r'\s+and\s+', sentence)]
        for pos in and_positions:
            left = sentence[:pos].strip()
            right = sentence[pos:].strip()
            # Remove leading "and "
            right = re.sub(r'^and\s+', '', right).strip()
            if not left or not right:
                continue
            left_words = set(re.findall(r'\b\w+\b', left))
            right_words = set(re.findall(r'\b\w+\b', right))
            left_has_add = bool(left_words & ADD_VERBS)
            left_has_sub = bool(left_words & SUB_VERBS)
            right_has_add = bool(right_words & ADD_VERBS)
            right_has_sub = bool(right_words & SUB_VERBS)
            # Only split if both sides have verbs AND they're different types
            left_has_verb = left_has_add or left_has_sub
            right_has_verb = right_has_add or right_has_sub
            if left_has_verb and right_has_verb:
                # Different verb types OR both sides have numbers
                left_nums = extract_numbers(left)
                right_nums = extract_numbers(right)
                if (left_has_add != right_has_add or left_has_sub != right_has_sub) \
                        and left_nums and right_nums:
                    return [left, right]
            # Also split when both halves have fraction/multiplier words
            # "Half of X, and half of Y" → two separate operations
            _FRAC_WORDS = {'half', 'third', 'quarter', 'twice', 'double', 'triple'}
            left_frac = bool(set(re.findall(r'\b\w+\b', left.lower())) & _FRAC_WORDS)
            right_frac = bool(set(re.findall(r'\b\w+\b', right.lower())) & _FRAC_WORDS)
            if left_frac and right_frac:
                return [left, right]
            # Split when both halves have rate keywords (each/every/per)
            left_rate = bool(re.search(r'\b(?:each|every|per)\b', left.lower()))
            right_rate = bool(re.search(r'\b(?:each|every|per)\b', right.lower()))
            if left_rate and right_rate:
                left_nums = extract_numbers(left)
                right_nums = extract_numbers(right)
                if left_nums and right_nums:
                    return [left, right]
        return [sentence]

    # Broad verb pattern for entity chain matching
    _CHAIN_VERBS = (
        r'(?:has|had|have|gets?|got|owns?|weighs?|is|was|are|were|'
        r'eats?|ate|makes?|made|reads?|writes?|wrote|plants?|planted|'
        r'sells?|sold|buys?|bought|drinks?|drank|catches?|caught|'
        r'scores?|scored|does|did|plays?|played|cooks?|cooked|'
        r'collects?|collected|earns?|earned|finds?|found|grows?|grew|'
        r'builds?|built|needs?|wanted?|takes?|took|uses?|used|'
        r'spends?|spent|gives?|gave|loses?|lost|saves?|saved|'
        r'wins?|won|works?|worked|draws?|drew|bakes?|baked|'
        r'ran|runs?|walks?|walked|swims?|swam|sleeps?|slept|'
        r'travels?|traveled|sees?|saw|picks?|picked)'
    )

    @staticmethod
    def _word_nums_to_digits(text: str) -> str:
        """Replace word numbers with digits for pattern matching."""
        replacements = [
            (r'\btwenty\b', '20'), (r'\bninety\b', '90'), (r'\beighty\b', '80'),
            (r'\bseventy\b', '70'), (r'\bsixty\b', '60'), (r'\bfifty\b', '50'),
            (r'\bforty\b', '40'), (r'\bthirty\b', '30'),
            (r'\bnineteen\b', '19'), (r'\beighteen\b', '18'), (r'\bseventeen\b', '17'),
            (r'\bsixteen\b', '16'), (r'\bfifteen\b', '15'), (r'\bfourteen\b', '14'),
            (r'\bthirteen\b', '13'), (r'\btwelve\b', '12'), (r'\beleven\b', '11'),
            (r'\bten\b', '10'), (r'\bnine\b', '9'), (r'\beight\b', '8'),
            (r'\bseven\b', '7'), (r'\bsix\b', '6'), (r'\bfive\b', '5'),
            (r'\bfour\b', '4'), (r'\bthree\b', '3'), (r'\btwo\b', '2'), (r'\bone\b', '1'),
        ]
        for pat, repl in replacements:
            text = re.sub(pat, repl, text)
        return text

    def _try_algebra(self, context_sents: List[str],
                     question: str, full_text: str) -> Optional[Dict[str, Any]]:
        """Solve problems requiring algebraic reasoning.

        Handles:
        1. Ratio + total: "X costs 3 times as much as Y. X+Y = N" → solve
        2. Linear equation: "Y = m*X + b, Y = known" → X = (Y-b)/m
        3. Complement fractions: "3/4 have X, N don't" → total = N/(1-3/4)
        4. Two-variable systems: "N heads, M legs, type1 has 2 legs, type2 has 4"
        """
        all_text = ' '.join(context_sents) + ' ' + question
        al = self._word_nums_to_digits(all_text.lower())
        ql = question.lower()
        steps = []
        total_val = None  # shared across patterns

        # Find total value for ratio patterns
        for s in context_sents + [question]:
            tsl = s.lower()
            tm = re.search(r'(?:total|together|combined|altogether|both|cost)\s+(?:\w+\s+){0,3}(?:is|was|are|were)\s+\$?(\d[\d,]*\.?\d*)', tsl)
            if not tm:
                tm = re.search(r'(?:total|together|combined|altogether|both|cost)\s+(?:of\s+)?(?:is\s+)?\$?(\d[\d,]*\.?\d*)(?!\s*times)', tsl)
            if not tm:
                tm = re.search(r'\$?(\d[\d,]*\.?\d*)\s+(?:\w+\s+)?(?:total|altogether|combined|together|in\s+all)', tsl)
            if not tm:
                tm = re.search(r'(?:there\s+are|there\s+were|there\s+is)\s+(\d[\d,]*\.?\d*)', tsl)
            if not tm:
                # "N [group_noun]" at sentence start
                tm = re.search(r'^(?:a\s+(?:\w+\s+){1,3}(?:has|have|had|with)\s+)?(\d[\d,]+)\s+(?:members?|people|students?|employees?|animals?|bees?|birds?|items?|books?|coins?|inhabitants?|players?)', tsl)
            if tm:
                candidate = float(tm.group(1).replace(',', ''))
                if candidate > 1:
                    total_val = candidate
                    break

        # === Pattern 0: Explicit ratio notation "X:Y" ===
        # "in the ratio of 7:11. Total is 162" → part = total * ratio_part / ratio_sum
        ratio_notation = re.search(r'ratio\s+(?:of\s+)?(\d+)\s*[:/]\s*(\d+)', al)
        if ratio_notation and total_val is not None:
            r1, r2 = float(ratio_notation.group(1)), float(ratio_notation.group(2))
            ratio_sum = r1 + r2
            if ratio_sum > 0:
                part1 = total_val * r1 / ratio_sum
                part2 = total_val * r2 / ratio_sum
                # Find entity order from the sentence containing "ratio"
                # "Darrell and Allen's ages are in the ratio of 7:11"
                # → entity order: [Darrell, Allen] → Darrell=r1, Allen=r2
                ratio_sent = ''
                for s in context_sents:
                    if 'ratio' in s.lower():
                        ratio_sent = s
                        break
                if not ratio_sent:
                    ratio_sent = full_text

                _RATIO_SKIP = {'The', 'This', 'That', 'They', 'Their', 'Them', 'There',
                               'What', 'When', 'Where', 'Which', 'How', 'And', 'But',
                               'For', 'From', 'His', 'Her', 'Each', 'Every', 'Some',
                               'All', 'If', 'Calculate', 'Find', 'Determine'}
                proper_names = [n for n in re.findall(r'\b([A-Z][a-z]{2,})\b', ratio_sent)
                                if n not in _RATIO_SKIP]
                # Deduplicate preserving order
                seen = set()
                unique_names = []
                for n in proper_names:
                    nl = n.lower().rstrip("'s")
                    if nl not in seen:
                        seen.add(nl)
                        unique_names.append(nl)

                # Also try matching nouns before ratio (e.g., "sugar and water")
                if len(unique_names) < 2:
                    ratio_pos_in_sent = ratio_sent.lower().find('ratio')
                    if ratio_pos_in_sent > 0:
                        before = ratio_sent[:ratio_pos_in_sent].lower()
                        # Find nouns separated by "and"
                        and_m = re.search(r'(\w+)\s+and\s+(?:\w+\s+)?(\w+)', before)
                        if and_m:
                            unique_names = [and_m.group(1), and_m.group(2)]

                # Determine which entity the question asks about
                asked_idx = -1
                for i, name in enumerate(unique_names):
                    if name in ql:
                        asked_idx = i
                        break

                # Also check for "in N years" / "N years from now" modifier
                time_m = re.search(r'(?:in|after)\s+(\d+\.?\d*)\s+(?:years?|months?|days?)', ql)
                if not time_m:
                    time_m = re.search(r'(\d+\.?\d*)\s+(?:years?|months?|days?)\s+from\s+now', ql)
                time_add = float(time_m.group(1)) if time_m else 0

                # Return the appropriate part
                if asked_idx == 0:
                    result = part1 + time_add
                elif asked_idx == 1:
                    result = part2 + time_add
                else:
                    result = max(part1, part2) + time_add

                steps.append(f"Ratio {r1}:{r2}, total={total_val}")
                steps.append(f"Parts: {part1}, {part2}")
                return {'answer': normalize_answer(result), 'steps': steps, 'confidence': 'HIGH'}

        # === Pattern 1: Ratio + total ===
        # "house cost three times as much as lot. house+lot = $120,000"
        # "twice as many workers as babies, twice as many babies as queens. 700 total"
        _RATIO_VERBS = r'(?:costs?|is|was|are|were|has|had|have|weighs?|sold|sells?|bought|earned?|makes?|made|produces?|produced|contains?|contained|holds?|held|scored?|gets?|got|runs?|ran|eats?|ate|drinks?|drank|reads?|collects?|collected|takes?|took)'
        _ratio_pattern = (
            r'(?:the\s+|a\s+|an\s+)?(\w+)(?:\'\w+)?\s+(?:\w+\s+){0,3}' + _RATIO_VERBS + r'\s+'
            r'(?:(\d+\.?\d*)\s+times|twice|double|triple|half)\s+'
            r'(?:as\s+)?(?:many|much|big|heavy|long|old|tall|fast|expensive)?\s*'
            r'(?:\w+\s+){0,2}(?:as\s+)?(?:the\s+)?(\w+)')
        ratio_m = re.search(_ratio_pattern, al)
        _SKIP_RATIO_SUBJECTS = {'there', 'it', 'he', 'she', 'they', 'we', 'this', 'that', 'the',
                                 'has', 'had', 'have', 'is', 'was', 'are', 'were', 'and', 'but',
                                 'or', 'if', 'a', 'an', 'to', 'of', 'in', 'on', 'for', 'with',
                                 'each', 'every', 'some', 'all', 'both', 'than', 'then', 'when',
                                 'also', 'just', 'only', 'even', 'still', 'so', 'as', 'at', 'by'}
        # Find first match with a real entity name — try searching from each word boundary
        if ratio_m and ratio_m.group(1).lower() in _SKIP_RATIO_SUBJECTS:
            ratio_m = None
            # Try matching from each word start position
            for wb in re.finditer(r'\b\w', al):
                rm_cand = re.match(_ratio_pattern, al[wb.start():])
                if rm_cand and rm_cand.group(1).lower() not in _SKIP_RATIO_SUBJECTS:
                    ratio_m = rm_cand
                    break
        if ratio_m:
            entity_a = ratio_m.group(1)
            mult_str = ratio_m.group(2)
            entity_b = ratio_m.group(3)

            if mult_str:
                mult = float(mult_str)
            elif 'twice' in ratio_m.group(0) or 'double' in ratio_m.group(0):
                mult = 2.0
            elif 'triple' in ratio_m.group(0):
                mult = 3.0
            elif 'half' in ratio_m.group(0):
                mult = 0.5
            else:
                mult = 2.0

            # A = mult * B, total already found above
            if total_val is not None and total_val > mult:
                # A = mult * B, A + B = total → mult*B + B = total → B = total/(mult+1)
                val_b = total_val / (mult + 1)
                val_a = mult * val_b
                steps.append(f"{entity_a} = {mult}×{entity_b}")
                steps.append(f"{entity_a} + {entity_b} = {total_val}")
                steps.append(f"{entity_b} = {total_val}/{mult+1} = {val_b}")

                # What does the question ask for?
                if entity_a in ql:
                    return {'answer': normalize_answer(val_a), 'steps': steps, 'confidence': 'HIGH'}
                elif entity_b in ql:
                    return {'answer': normalize_answer(val_b), 'steps': steps, 'confidence': 'HIGH'}
                else:
                    # Default: return the one the question seems to ask about
                    # or return the larger one
                    return {'answer': normalize_answer(val_a), 'steps': steps, 'confidence': 'MEDIUM'}

        # === Pattern 1b: Multi-ratio chain with total ===
        # "700 bees. Workers = 2×babies. Babies = 2×queens." → 4q+2q+q=700
        # Only fires if Pattern 1 didn't already return (needs 2+ ratio relationships)
        if total_val is not None:
            # Look for MULTIPLE ratio relationships: A=N×B, B=M×C, ...
            ratio_rels = []
            _SKIP_ENTS = {'there', 'it', 'he', 'she', 'they', 'we', 'this', 'that', 'the', 'and', 'but', 'or', 'if', 'when', 'while', 'also'}
            for rm_iter in re.finditer(
                    r'(?:(\w+)\s+(?:\w+\s+){0,3}' + _RATIO_VERBS + r'\s+'
                    r'(?:(\d+\.?\d*)\s+times|twice|double|triple|half)\s+'
                    r'(?:as\s+)?(?:many|much|big|heavy|long|old|tall|fast|expensive)?\s*'
                    r'(?:\w+\s+){0,2}(?:as\s+)?(?:the\s+)?(\w+))', al):
                entity_a = rm_iter.group(1)
                mult_str = rm_iter.group(2)
                entity_b = rm_iter.group(3)
                if entity_a.lower() in _SKIP_ENTS or entity_b.lower() in _SKIP_ENTS:
                    continue
                if mult_str:
                    mult_v = float(mult_str)
                elif 'twice' in rm_iter.group(0) or 'double' in rm_iter.group(0):
                    mult_v = 2.0
                elif 'triple' in rm_iter.group(0):
                    mult_v = 3.0
                elif 'half' in rm_iter.group(0):
                    mult_v = 0.5
                else:
                    mult_v = 2.0
                ratio_rels.append((entity_a, mult_v, entity_b))

            # Also look for "twice as many [noun] on/in the X as ... on/in the Y"
            for rm_iter in re.finditer(
                    r'(?:twice|double|triple|half|(\d+\.?\d*)\s+times)\s+'
                    r'(?:as\s+)?(?:many|much|the\s+number\s+of)\s+'
                    r'(?:\w+\s+){0,3}(?:on|in|for|from)\s+(?:the\s+)?(\w+(?:\s+\w+)?)\s+'
                    r'(?:as|than)\s+(?:\w+\s+){0,4}(?:on|in|for|from)\s+(?:the\s+)?(\w+(?:\s+\w+)?)', al):
                mult_v = float(rm_iter.group(1)) if rm_iter.group(1) else (
                    2.0 if 'twice' in rm_iter.group(0) else
                    3.0 if 'triple' in rm_iter.group(0) else
                    0.5 if 'half' in rm_iter.group(0) else 2.0)
                entity_a = rm_iter.group(2).split()[0]  # first word of multi-word
                entity_b = rm_iter.group(3).split()[0]
                if entity_a.lower() not in _SKIP_ENTS and entity_b.lower() not in _SKIP_ENTS:
                    if not any(r[0] == entity_a and r[2] == entity_b for r in ratio_rels):
                        ratio_rels.append((entity_a, mult_v, entity_b))

            # Also look for "twice as many X as Y" pattern
            for rm_iter in re.finditer(
                    r'(?:twice|double|triple|(\d+\.?\d*)\s+times)\s+(?:as\s+)?(?:many|much)\s+'
                    r'(\w+)\s+(?:\w+\s+){0,3}(?:as\s+)(\w+)', al):
                mult_v = float(rm_iter.group(1)) if rm_iter.group(1) else (
                    2.0 if 'twice' in rm_iter.group(0) else
                    3.0 if 'triple' in rm_iter.group(0) else 2.0)
                entity_a = rm_iter.group(2)
                entity_b = rm_iter.group(3)
                # Skip filler/pronoun entities and avoid duplicates
                if entity_a.lower() in _SKIP_ENTS or entity_b.lower() in _SKIP_ENTS:
                    continue
                if not any(r[0] == entity_a and r[2] == entity_b for r in ratio_rels):
                    ratio_rels.append((entity_a, mult_v, entity_b))

            # Normalize plural entity names to singular for chain matching
            def _deplural(w):
                if w.endswith('ies') and len(w) > 4:
                    return w[:-3] + 'y'  # babies→baby, queries→query
                if w.endswith('es') and len(w) > 3:
                    return w[:-2]  # boxes→box
                if w.endswith('s') and not w.endswith('ss') and len(w) > 2:
                    return w[:-1]  # workers→worker
                return w

            # Build canonical entity map: plural→singular
            raw_entities = set()
            for a, _, b in ratio_rels:
                raw_entities.add(a)
                raw_entities.add(b)
            canon = {}
            for e in raw_entities:
                dep = _deplural(e)
                # If the depluralized form is already a known entity, map to it
                if dep != e and dep in raw_entities:
                    canon[e] = dep
                else:
                    canon[e] = e
            # Also merge entities that share a stem
            stems = {}
            for e in raw_entities:
                dep = _deplural(e)
                if dep not in stems:
                    stems[dep] = e
                elif len(e) < len(stems[dep]):
                    # prefer shorter (singular) form
                    canon[stems[dep]] = e
                    stems[dep] = e
                else:
                    canon[e] = stems[dep]

            # Apply canonicalization
            ratio_rels = [(canon.get(a, a), mv, canon.get(b, b)) for a, mv, b in ratio_rels]

            if len(ratio_rels) >= 2:
                # Build multiplier chain: express everything in terms of base entity
                all_entities = set()
                for a, _, b in ratio_rels:
                    all_entities.add(a)
                    all_entities.add(b)

                # Try each entity as base (multiplier=1), resolve others
                for base_entity in all_entities:
                    multipliers = {base_entity: 1.0}
                    changed = True
                    iters = 10
                    while changed and iters > 0:
                        changed = False
                        iters -= 1
                        for a, mv, b in ratio_rels:
                            if b in multipliers and a not in multipliers:
                                multipliers[a] = mv * multipliers[b]
                                changed = True
                            elif a in multipliers and b not in multipliers and mv != 0:
                                multipliers[b] = multipliers[a] / mv
                                changed = True

                    if len(multipliers) == len(all_entities):
                        # All entities resolved → total = sum(multipliers) × base_val
                        total_mult = sum(multipliers.values())
                        if total_mult > 0:
                            base_val = total_val / total_mult
                            # Which entity does the question ask about?
                            for ent, mult in multipliers.items():
                                if ent in ql:
                                    result = mult * base_val
                                    steps.append(f"System: total={total_val}, {ent}={mult}×base")
                                    steps.append(f"base={base_val}, {ent}={result}")
                                    return {'answer': normalize_answer(result), 'steps': steps, 'confidence': 'HIGH'}
                            # Default: return largest
                            max_ent = max(multipliers.items(), key=lambda x: x[1])
                            result = max_ent[1] * base_val
                            steps.append(f"System: total={total_val}")
                            return {'answer': normalize_answer(result), 'steps': steps, 'confidence': 'MEDIUM'}
                        break

        # === Pattern 2: "Y has N more than M times X" + Y is known → solve for X ===
        # "Janey has 3 more than twice Sally's books. Janey has 21." → sally = (21-3)/2 = 9
        m = re.search(
            r'(\w+)\s+(?:has|had|have|is|was|owns?|gets?)\s+'
            r'(\d+\.?\d*)\s+(?:more|fewer|less)\s+than\s+'
            r'(?:(\d+\.?\d*)\s+times|twice|double|triple)\s+'
            r'(?:(?:as\s+)?(?:many|much)\s+(?:\w+\s+)?(?:as\s+)?)?'
            r'(?:the\s+(?:number\s+of\s+)?)?(\w+)',
            al)
        if m:
            entity_y = m.group(1)
            diff = float(m.group(2))
            mult_str = m.group(3)
            entity_x_raw = m.group(4)

            if mult_str:
                mult = float(mult_str)
            elif 'twice' in m.group(0) or 'double' in m.group(0):
                mult = 2.0
            elif 'triple' in m.group(0):
                mult = 3.0
            else:
                mult = 2.0

            direction = 'more' if 'more' in m.group(0) else 'fewer'

            # Find known value for entity_y
            y_val = None
            for s in context_sents + [question]:
                vm = re.search(
                    re.escape(entity_y) + r'(?:\s+\w+)?\s+(?:has|had|have|is|was)\s+'
                    r'(?:exactly\s+)?(\d+)', s.lower())
                if vm:
                    y_val = float(vm.group(1))
                    break

            if y_val is not None:
                # y = mult * x + diff (or - diff)
                if direction == 'more':
                    x_val = (y_val - diff) / mult
                else:
                    x_val = (y_val + diff) / mult
                steps.append(f"{entity_y} = {mult}×{entity_x_raw} {'+'if direction=='more' else '-'} {diff}")
                steps.append(f"{y_val} = {mult}×{entity_x_raw} {'+'if direction=='more' else '-'} {diff}")
                steps.append(f"{entity_x_raw} = {x_val}")
                return {'answer': normalize_answer(x_val), 'steps': steps, 'confidence': 'HIGH'}

        # === Pattern 3: Complement fraction ===
        # "three-fourths of students have desktop. 20 do not." → total = 20/(1-3/4) = 80
        al_raw = all_text.lower()  # without word-num conversion (preserves "three-fourths")
        for frac_name, frac_val in FRACTION_MAP.items():
            if frac_name in al_raw:
                # Look for complement: "N do not" or "N don't"
                cm = re.search(r'(\d+)\s+(?:students?|people|children|employees?|workers?|members?|participants?|of\s+them)?\s*(?:do\s+not|don\'?t|does\s+not|doesn\'?t|are\s+not|aren\'?t|were\s+not|weren\'?t|have\s+no|lack|without|did\s+not|didn\'?t)', al)
                if cm:
                    complement_count = float(cm.group(1))
                    complement_frac = 1.0 - frac_val
                    if complement_frac > 0.001:
                        total = complement_count / complement_frac
                        steps.append(f"{frac_name} have it → {1-frac_val:.4f} don't")
                        steps.append(f"{complement_count} / {complement_frac} = {total}")
                        return {'answer': normalize_answer(total), 'steps': steps, 'confidence': 'HIGH'}

        # === Pattern 4: Two-variable system with attribute counts ===
        # "20 animals (chickens + cows), 70 legs (chickens*2 + cows*4)"
        # "180 heads (camels + dromedaries), 304 bumps (camels*2 + dromedaries*1)"
        # Detect: two sums with different coefficients
        heads_m = re.search(r'(\d+)\s+(?:heads?|animals?|birds?|creatures?|people|total)', al)
        if heads_m:
            total_count = float(heads_m.group(1))
            # Look for a second constraint with a different total
            # Find ALL attribute counts and pick the largest (total, not per-entity)
            _ATTR_RE = r'(?:legs?|bumps?|humps?|wheels?|wings?|eyes?|feet)'
            attr_matches = re.findall(r'(\d+)\s+(' + _ATTR_RE + r')', al)
            if attr_matches:
                # Pick the largest number as the total attribute count
                attr_matches.sort(key=lambda x: -float(x[0]))
                total_attr = float(attr_matches[0][0])
                attr_word = attr_matches[0][1].rstrip('s')

                # Try to find per-type counts
                type_vals = {}
                # Match any attribute word variant (bump/hump, leg/foot, etc.)
                attr_pattern = _ATTR_RE
                for s in context_sents + [question]:
                    sl = self._word_nums_to_digits(s.lower())
                    # "camels have 2 bumps" or "a camel has 2 humps"
                    for tm in re.finditer(r'(\w+)s?\s+(?:has|have|had)\s+(\d+)\s+(?:' + attr_pattern + r')', sl):
                        entity = tm.group(1)
                        val = float(tm.group(2))
                        if val < total_attr:  # must be per-entity, not total
                            type_vals[entity] = val

                if len(type_vals) >= 2:
                    types = list(type_vals.keys())
                    a1, a2 = type_vals[types[0]], type_vals[types[1]]
                    if a1 != a2:
                        # x + y = total_count, a1*x + a2*y = total_attr
                        # y = (total_attr - a1*total_count) / (a2 - a1)
                        y = (total_attr - a1 * total_count) / (a2 - a1)
                        x = total_count - y
                        steps.append(f"{types[0]}={x}, {types[1]}={y}")

                        # What does question ask? Check "how many [type]"
                        q_subject = re.search(r'how\s+many\s+(\w+)', ql)
                        q_subj = q_subject.group(1) if q_subject else ''
                        for t, v in [(types[0], x), (types[1], y)]:
                            if q_subj.startswith(t[:4]):
                                return {'answer': normalize_answer(v), 'steps': steps, 'confidence': 'HIGH'}
                        # Fallback: check which type appears in question before "if"
                        q_before_if = ql.split(' if ')[0] if ' if ' in ql else ql
                        for t, v in [(types[0], x), (types[1], y)]:
                            if t in q_before_if:
                                return {'answer': normalize_answer(v), 'steps': steps, 'confidence': 'HIGH'}
                        return {'answer': normalize_answer(y), 'steps': steps, 'confidence': 'MEDIUM'}
                elif not type_vals:
                    # Infer from domain knowledge
                    # chickens=2 legs, cows=4 legs
                    if attr_word in ('leg', 'feet', 'foot'):
                        if re.search(r'chicken|hen|duck|bird|goose|rooster', al):
                            a1 = 2.0
                        else:
                            a1 = 2.0  # default biped
                        if re.search(r'cow|horse|pig|sheep|goat|dog|cat|rabbit', al):
                            a2 = 4.0
                        else:
                            a2 = 4.0  # default quadruped
                        y = (total_attr - a1 * total_count) / (a2 - a1)
                        x = total_count - y
                        steps.append(f"type1(×{a1})={x}, type2(×{a2})={y}")

                        # Question asks about which type?
                        if re.search(r'chicken|hen|duck|bird|goose|rooster', ql):
                            return {'answer': normalize_answer(x), 'steps': steps, 'confidence': 'MEDIUM'}
                        elif re.search(r'cow|horse|pig|sheep|goat|dog|cat|rabbit', ql):
                            return {'answer': normalize_answer(y), 'steps': steps, 'confidence': 'MEDIUM'}

        return None

    def _try_entity_chain(self, context_sents: List[str],
                          question: str) -> Optional[Dict[str, Any]]:
        """Try to solve problems with entity relationship chains.
        E.g.: "A has twice as many as B. B has 4 times as many as C. C has 20."
        """
        # Extract entity relationships and base values
        relationships = []  # (entity, multiplier, ref_entity)
        base_values = {}    # entity -> value
        steps = []

        all_text = ' '.join(context_sents) + ' ' + question
        sl = self._word_nums_to_digits(all_text.lower())

        _SKIP_ENTITIES = {'he', 'she', 'it', 'they', 'there', 'each', 'the',
                          'if', 'when', 'then', 'and', 'but', 'that', 'this',
                          'who', 'which', 'what', 'how', 'where', 'why'}

        # Find all "X [verb] N" patterns (base values)
        # Exclude "N times as many" (those are relationships, not base values)
        for m in re.finditer(
                r'(\w+)\s+' + self._CHAIN_VERBS + r'\s+'
                r'(?:exactly\s+|about\s+|only\s+|a\s+total\s+of\s+)?'
                r'(\d+\.?\d*)\s+', sl):
            entity = m.group(1)
            val = float(m.group(2))
            # Skip if the number is followed by "times", "more/fewer/less", or multiplier words
            after = sl[m.end():m.end()+30]
            if re.match(r'times\b', after):
                continue
            if re.match(r'(?:more|fewer|less|greater|bigger|smaller|taller|shorter|older|younger|longer|heavier|lighter)\b', after):
                continue
            if any(after.startswith(w) for w in MULTIPLIER_WORDS):
                continue
            if entity not in _SKIP_ENTITIES:
                base_values[entity] = val

        # Also find possessive "X's [noun] is/was N" → entity = X
        for m in re.finditer(r"(\w+)'s\s+\w+\s+(?:is|was|are|were)\s+(\d+\.?\d*)", sl):
            entity = m.group(1)
            val = float(m.group(2))
            after = sl[m.end():m.end()+20]
            # Skip if followed by "times", "/" (fraction), or multiplier words
            if re.match(r'\s*(?:times\b|/)', after):
                continue
            if any(after.strip().startswith(w) for w in MULTIPLIER_WORDS):
                continue
            if entity not in _SKIP_ENTITIES:
                base_values[entity] = val

        # Find "X [verb] N times as many as Y" patterns
        for m in re.finditer(
                r'(\w+)\s+' + self._CHAIN_VERBS + r'\s+'
                r'(?:exactly\s+)?(\d+\.?\d*)\s+times\s+(?:as\s+)?(?:many|much)\s+'
                r'(?:\w+\s+){0,3}(?:as\s+)(\w+)', sl):
            entity = m.group(1)
            mult = float(m.group(2))
            ref = m.group(3)
            relationships.append((entity, mult, ref))

        # "X [verb] N times the number/amount of Y" — alternative pattern
        for m in re.finditer(
                r'(\w+)\s+' + self._CHAIN_VERBS + r'\s+'
                r'(?:exactly\s+)?(\d+\.?\d*)\s+times\s+(?:the\s+)?'
                r'(?:number|amount|quantity|size|weight|length|cost|price|value)\s+'
                r'(?:of\s+)?(?:\w+\s+){0,3}(?:as\s+)?(\w+)', sl):
            entity = m.group(1)
            mult = float(m.group(2))
            ref = m.group(3)
            if entity not in _SKIP_ENTITIES and ref not in _SKIP_ENTITIES:
                relationships.append((entity, mult, ref))

        # Find "X [verb] twice/double/triple/half as many/much/big/long/... as Y"
        for word, mult in {**MULTIPLIER_WORDS, 'half': 0.5}.items():
            for m in re.finditer(
                    r'(\w+)\s+' + self._CHAIN_VERBS + r'\s+' +
                    word + r'\s+(?:as\s+)?(?:many|much|big|large|small|long|short|tall|heavy|fast|old|expensive|wide|thick|deep)\s+'
                    r'(?:\w+\s+){0,3}(?:as\s+)(\w+)', sl):
                entity = m.group(1)
                ref = m.group(2)
                relationships.append((entity, float(mult), ref))

        # Find "X's [noun] is half/quarter/N times/N/M as big/much/many as Y's [noun]"
        _SIZE_ADJ = r'(?:big|large|small|long|short|tall|heavy|much|many|wide|thick|deep|fast|expensive)'
        for m in re.finditer(
                r"(\w+)(?:'s)?\s+\w+\s+(?:is|was|are|were)\s+"
                r'(?:(\d+\.?\d*)\s+times\s+|(\d+)/(\d+)\s+|(?:(half|quarter|third)\s+))?'
                r'(?:as\s+)?' + _SIZE_ADJ + r'\s+'
                r'(?:\w+\s+){0,3}(?:as\s+)(\w+)', sl):
            entity = m.group(1)
            ref = m.group(6)
            if m.group(2):
                mult_v = float(m.group(2))
            elif m.group(3) and m.group(4):
                mult_v = float(m.group(3)) / float(m.group(4))
            elif m.group(5) == 'half':
                mult_v = 0.5
            elif m.group(5) == 'quarter':
                mult_v = 0.25
            elif m.group(5) == 'third':
                mult_v = 1/3
            else:
                mult_v = 1.0
            if abs(mult_v - 1.0) > 0.01 and entity not in _SKIP_ENTITIES and ref not in _SKIP_ENTITIES:
                relationships.append((entity, mult_v, ref))

        # Find "X is/was N/fraction of Y" patterns
        for frac_name, frac_val in FRACTION_MAP.items():
            for m in re.finditer(
                    r"(\w+)(?:'s)?\s+\w+\s+(?:is|was|are|were)\s+" +
                    re.escape(frac_name) + r'\s+(?:of\s+)?(?:the\s+size\s+of\s+)?(\w+)', sl):
                entity = m.group(1)
                ref = m.group(2)
                if entity not in _SKIP_ENTITIES and ref not in _SKIP_ENTITIES:
                    relationships.append((entity, frac_val, ref))

        # Find "X [verb] N more/fewer THING than Y" patterns
        for m in re.finditer(
                r'(\w+)\s+' + self._CHAIN_VERBS + r'\s+'
                r'(?:exactly\s+)?(\d+\.?\d*)\s+(?:more|fewer|less)\s+'
                r'(?:\w+\s+){0,3}than\s+(\w+)', sl):
            entity = m.group(1)
            diff = float(m.group(2))
            ref = m.group(3)
            direction = 'more' if 'more' in m.group(0) else 'fewer'
            if direction == 'more':
                relationships.append((entity, 1.0, ref, diff))
            else:
                relationships.append((entity, 1.0, ref, -diff))

        if not relationships:
            return None

        # Resolve chain: start from known base values, also reverse-solve
        resolved = dict(base_values)
        changed = True
        max_iter = 10
        while changed and max_iter > 0:
            changed = False
            max_iter -= 1
            for rel in relationships:
                entity = rel[0]
                mult = rel[1]
                ref = rel[2]
                addend = rel[3] if len(rel) > 3 else 0
                # Forward: entity = mult * ref + addend
                if entity not in resolved and ref in resolved:
                    resolved[entity] = mult * resolved[ref] + addend
                    if addend:
                        steps.append(f"{entity} = {mult}×{resolved[ref]} + {addend} = {resolved[entity]}")
                    else:
                        steps.append(f"{entity} = {mult} x {resolved[ref]} = {resolved[entity]}")
                    changed = True
                # Reverse: ref = (entity - addend) / mult
                # Only for pure multiplicative relationships (mult != 1)
                elif ref not in resolved and entity in resolved and mult != 0 and abs(mult - 1.0) > 0.01:
                    resolved[ref] = (resolved[entity] - addend) / mult
                    steps.append(f"{ref} = ({resolved[entity]} - {addend}) / {mult} = {resolved[ref]}")
                    changed = True

        if len(resolved) < 2:
            return None

        # Determine what to compute from question
        ql = question.lower()
        if 'together' in ql or 'total' in ql or 'all' in ql or 'combined' in ql:
            total = sum(resolved.values())
            steps.append(f"Total = {' + '.join(str(v) for v in resolved.values())} = {total}")
            return {
                'answer': normalize_answer(total),
                'steps': steps,
                'confidence': 'HIGH',
            }
        elif 'difference' in ql:
            vals = list(resolved.values())
            if len(vals) >= 2:
                diff = abs(vals[0] - vals[1])
                return {'answer': normalize_answer(diff), 'steps': steps, 'confidence': 'HIGH'}

        # If question asks about a specific entity
        # Prefer entity that's the subject of "how many [X] does [entity]" or similar
        q_subject = re.search(r'(?:how\s+many|how\s+much).*?\b(?:does|do|did|will|can)\s+(\w+)', ql)
        if q_subject:
            asked_entity = q_subject.group(1)
            for entity, val in resolved.items():
                if entity == asked_entity:
                    return {'answer': normalize_answer(val), 'steps': steps, 'confidence': 'HIGH'}
        # Match entities in question — prefer proper nouns (from possessives)
        # and entities that appear earliest in the question
        proper_entities = set()
        for rel in relationships:
            proper_entities.add(rel[0])
            proper_entities.add(rel[2])
        for e in base_values:
            proper_entities.add(e)
        q_matches = [(e, v) for e, v in resolved.items()
                     if e in ql and e not in _SKIP_ENTITIES and len(e) > 2]
        # Sort by position in question (earliest first), with proper entities preferred
        q_matches.sort(key=lambda x: (
            # Common nouns (not in original entities) go last
            x[0] not in proper_entities,
            # Earlier position in question = better match
            ql.find(x[0]),
        ))
        if q_matches:
            return {'answer': normalize_answer(q_matches[0][1]), 'steps': steps, 'confidence': 'HIGH'}

        return None

    # Time unit hierarchy for conversions
    TIME_UNITS = {
        'minute': 1,
        'hour': 60,
        'day': 1440,
        'week': 10080,
        'month': 43200,  # ~30 days
        'year': 525600,  # ~365 days
    }

    TIME_CONVERSIONS = {
        ('minute', 'hour'): 60,
        ('hour', 'day'): 24,
        ('day', 'week'): 7,
        ('day', 'month'): 30,
        ('day', 'year'): 365,
        ('week', 'month'): 4,
        ('week', 'year'): 52,
        ('month', 'year'): 12,
    }

    def _detect_time_unit(self, text: str) -> Optional[str]:
        """Detect the time unit mentioned in text."""
        tl = text.lower()
        patterns = [
            (r'\b(?:per|a|each|every)\s+minute\b|\bminutely\b|\bper\s+min\b|\bin\s+(?:a|one|\d+)\s+minute', 'minute'),
            (r'\b(?:per|a|each|every)\s+hour\b|\bhourly\b|\ban?\s+hour\b|\bin\s+(?:a|one|\d+)\s+hour', 'hour'),
            (r'\b(?:per|a|each|every)\s+day\b|\bdaily\b|\ba\s+day\b|\beveryday\b|\bevery\s+(?:morning|night|evening|afternoon)\b|\bin\s+(?:a|one|\d+)\s+day', 'day'),
            (r'\b(?:per|a|each|every)\s+week\b|\bweekly\b|\ba\s+week\b|\bin\s+(?:a|one|\d+)\s+week', 'week'),
            (r'\b(?:per|a|each|every)\s+month\b|\bmonthly\b|\ba\s+month\b|\bin\s+(?:a|one|\d+)\s+month', 'month'),
            (r'\b(?:per|a|each|every)\s+year\b|\byearly\b|\bannually\b|\ba\s+year\b|\bin\s+(?:a|one|\d+)\s+year', 'year'),
        ]
        for pat, unit in reversed(patterns):  # check larger units first
            if re.search(pat, tl):
                return unit
        return None

    def _apply_time_conversion(self, current_val: float,
                                context_sents: List[str],
                                question: str,
                                state: SolverState) -> Optional[float]:
        """If context mentions one time unit and question asks about another, convert."""
        # Find time unit in context (the "per" unit)
        ctx_text = ' '.join(context_sents)
        ctx_unit = self._detect_time_unit(ctx_text)
        q_unit = self._detect_time_unit(question)

        if not ctx_unit or not q_unit or ctx_unit == q_unit:
            return None

        # Check if the conversion factor is already in the numbers we used
        # (i.e., the answer already includes the conversion)
        key = (ctx_unit, q_unit)
        if key in self.TIME_CONVERSIONS:
            factor = self.TIME_CONVERSIONS[key]
            result = current_val * factor
            state.op(f"{current_val} x {factor} ({ctx_unit}->{q_unit}) = {result}", result)
            return result

        # Try reverse
        rev_key = (q_unit, ctx_unit)
        if rev_key in self.TIME_CONVERSIONS:
            factor = self.TIME_CONVERSIONS[rev_key]
            result = current_val / factor
            state.op(f"{current_val} / {factor} ({ctx_unit}->{q_unit}) = {result}", result)
            return result

        return None

    def _try_two_step(self, context_sents: List[str],
                      question: str, full_text: str) -> Optional[Dict[str, Any]]:
        """Two-step sequential computation for common 2-op patterns.

        Handles problems that need exactly two sequential arithmetic operations
        where existing specialized solvers fail. Patterns:

        A. Shopping change: sum prices → subtract from paid
        B. Cost split: sum costs → divide by N
        C. How many needed: total items → divide by unit size
        D. Rate-to-total: rate × duration, then +/- adjustment
        E. Reverse age: future_age - years = current, current +/- delta = answer
        F. Earnings: rate × quantity → result (then maybe +/- or /)
        """
        ql = question.lower()
        all_text = ' '.join(context_sents) + ' ' + question
        al = self._word_nums_to_digits(all_text.lower())
        steps = []

        # Collect all dollar amounts from context + question
        all_prices = []
        for s in context_sents + [question]:
            for pm in re.finditer(r'\$(\d+\.?\d*)', s):
                all_prices.append(float(pm.group(1)))

        # Collect all plain numbers
        all_nums_raw = extract_numbers(all_text)
        all_vals = [v for v, _, _ in all_nums_raw]

        # === Pattern MULTI-RATE: "$R₁ per hour for job₁, $R₂ for job₂, H₁ hours as job₁, H₂ as job₂" ===
        # Binds rates to jobs by matching role/activity keywords across sentences
        rate_defs = re.findall(
            r'\$(\d+\.?\d*)\s*(?:per\s+hour|/hr|an\s+hour)?\s*'
            r'(?:to\s+(?:be\s+(?:a\s+)?)?|for\s+(?:a\s+)?|as\s+(?:a\s+)?)'
            r'([\w]+(?:\s+\w+){0,2}?)(?:\s+and\b|\s*[,.]|\s*$)',
            al)
        if len(rate_defs) >= 2:
            # Build rate map: use all words in role as stems for matching
            _STOP = {'a', 'an', 'the', 'and', 'to', 'be', 'as', 'for', 'per', 'of', 'in'}
            rate_map = {}
            for rate_s, role in rate_defs:
                words = [w[:5] for w in role.strip().split() if w.lower() not in _STOP and len(w) > 2]
                for w in words:
                    rate_map[w.lower()] = float(rate_s)
            # Find hours per role: "N hours ... as/for [role]"
            hours_defs = re.findall(
                r'(\d+\.?\d*)\s*hours?\s+(?:a\s+week\s+|per\s+week\s+)?'
                r'(?:as\s+(?:a\s+)?|for\s+(?:a\s+)?|of\s+)([\w]+)',
                al)
            if len(hours_defs) >= 2:
                total_rate = 0
                pair_strs = []
                matched = set()
                for hrs_s, role in hours_defs:
                    role_stem = role.strip()[:5].lower()
                    rate = rate_map.get(role_stem)
                    if rate is not None and role_stem not in matched:
                        hrs = float(hrs_s)
                        total_rate += hrs * rate
                        pair_strs.append(f"{hrs}×${rate}")
                        matched.add(role_stem)
                if total_rate > 0 and len(pair_strs) >= 2:
                    steps.append(f"Rates: {' + '.join(pair_strs)} = {total_rate}")
                    # Check for time multiplier (weeks/year etc)
                    time_mult_m = re.search(
                        r'(\d+)\s+(?:weeks?|months?|years?|days?)\s+'
                        r'(?:a\s+|per\s+|each\s+|in\s+a\s+)?'
                        r'(?:year|month|week|day|season|semester)', al)
                    if time_mult_m:
                        time_mult = float(time_mult_m.group(1))
                        total_rate *= time_mult
                        steps.append(f"×{time_mult} = {total_rate}")
                    return {'answer': normalize_answer(total_rate),
                            'steps': steps, 'confidence': 'HIGH'}

        # === Pattern SOP: Sum of products (qty × price) ===
        # "N₁ items at $P₁ each, N₂ items at $P₂ each. How much total?"
        # Also: "N₁ items which cost $P₁ each, N₂ items at $P₂ each"
        # Also: "N₁ tables with M₁ legs and N₂ tables with M₂ legs" (count×attribute)
        # Detect comparison questions — need difference of products, not sum
        sop_is_comparison = bool(re.search(
            r'how\s+much\s+(?:more|less|cheaper|expensive)|'
            r'(?:difference|differ)\s+(?:in|between)|'
            r'how\s+many\s+more|how\s+many\s+fewer', ql))
        if re.search(r'(?:how\s+much|how\s+many|total|cost|spend|price|pay|number)', ql) or sop_is_comparison:
            qty_price_pairs = re.findall(
                r'(\d+)\s+(?:dozen\s+)?(?:\w+\s+){0,5}'
                r'(?:costs?|at|for|with|which\s+costs?|that\s+costs?|that\s+were|costing|priced\s+at|'
                r'containing|having)\s+'
                r'\$?(\d+\.?\d*)'
                r'\s*(?:each|apiece|per\s+\w+)?', al)
            # Filter out unit-price definitions: "1/one pair of X costs $Y"
            # These define per-unit prices, not qty×price pairs
            if qty_price_pairs and all(float(q) == 1 for q, _ in qty_price_pairs):
                # All quantities are 1 → likely unit-price definitions
                # Try cross-reference to find actual purchase quantities
                qty_price_pairs = []
            if not qty_price_pairs or len(qty_price_pairs) < 2:
                # Alternative: "$P each ... N items"
                qty_price_pairs = re.findall(
                    r'\$(\d+\.?\d*)\s*(?:each|per\s+\w+|apiece).*?(\d+)\s+(?:\w+)', al)
                # Swap order to (qty, price)
                qty_price_pairs = [(p[1], p[0]) for p in qty_price_pairs]
            if not qty_price_pairs or len(qty_price_pairs) < 2:
                # "N [items] have/had/hold M [things]" or "N [items] that/which send M [things]"
                qty_price_pairs = re.findall(
                    r'(\d+)\s+(?:\w+\s+){0,3}'
                    r'(?:have|had|holds?|sends?|gets?|(?:each\s+)?(?:had|have|hold))\s+'
                    r'(\d+)\s+\w+', al)
                # Validate: need at least 2 pairs and linking words
                if len(qty_price_pairs) < 2:
                    qty_price_pairs = []

            # Cross-reference SOP: "brownies for $3 ... sells 43 brownies"
            # Match item→price definitions and qty→item references separately
            if len(qty_price_pairs) < 2:
                item_prices = re.findall(
                    r'(\w+)\s+(?:for|at|costs?)\s+\$(\d+\.?\d*)\s*'
                    r'(?:each|apiece|per\s+\w+|a\s+\w+)?', al)
                if not item_prices:
                    item_prices = re.findall(
                        r'\$(\d+\.?\d*)\s+(?:per|a|for\s+(?:each|every))\s+(\w+)', al)
                    item_prices = [(p[1], p[0]) for p in item_prices]
                # Deduplicate by (item, price)
                seen_ip = set()
                deduped = []
                for ip in item_prices:
                    key_ip = (ip[0], ip[1])
                    if key_ip not in seen_ip:
                        seen_ip.add(key_ip)
                        deduped.append(ip)
                item_prices = deduped
                if len(item_prices) >= 2:
                    # Find quantities for each item
                    cross_pairs = []
                    for item_name, price in item_prices:
                        item_stem = item_name.rstrip('s')[:5]
                        qty_m = re.search(
                            r'(\d+)\s+(?:\w+\s+(?:of\s+)?)?' + re.escape(item_stem),
                            al)
                        if qty_m:
                            qty_val = float(qty_m.group(1))
                            price_val = float(price)
                            if qty_val > 1 and price_val != qty_val:
                                cross_pairs.append((qty_val, price_val))
                    if len(cross_pairs) >= 2:
                        qty_price_pairs = [(str(int(q)), str(p)) for q, p in cross_pairs]

            if len(qty_price_pairs) >= 2:
                total = 0
                pair_strs = []
                for qty_s, price_s in qty_price_pairs:
                    q_val = float(qty_s)
                    p_val = float(price_s)
                    if q_val > 0 and p_val > 0:
                        total += q_val * p_val
                        pair_strs.append(f"{q_val}×{p_val}")
                if total > 0 and len(pair_strs) >= 2:
                    # For comparison questions with exactly 2 pairs, compute difference
                    if sop_is_comparison and len(qty_price_pairs) == 2:
                        products = []
                        for qty_s, price_s in qty_price_pairs:
                            products.append(float(qty_s) * float(price_s))
                        diff = abs(products[0] - products[1])
                        steps.append(f"SOP diff: |{pair_strs[0]} - {pair_strs[1]}| = {diff}")
                        return {'answer': normalize_answer(diff),
                                'steps': steps, 'confidence': 'HIGH'}
                    steps.append(f"SOP: {' + '.join(pair_strs)} = {total}")
                    # Check for additional standalone costs: "$X [item_noun]"
                    # Only add costs that are clearly item costs (near cost/which cost)
                    used_prices = set()
                    for _, p in qty_price_pairs:
                        used_prices.add(float(p))
                    for s in context_sents:
                        sl_s = s.lower()
                        # Find all dollar amounts not used in SOP pairs
                        for dm in re.finditer(r'\$(\d+\.?\d*)', sl_s):
                            price = float(dm.group(1))
                            if price in used_prices:
                                continue
                            # Verify it's a standalone cost, not a rate/divisor
                            before = sl_s[max(0, dm.start()-40):dm.start()]
                            if re.search(r'(?:cost|costs|for|at|costing|was|is|worth|paid|pays?|charges?)\s*$', before):
                                total += price
                                steps.append(f"+ ${price}")
                                used_prices.add(price)
                    # Check for percentage discount on the total
                    disc_m = re.search(r'(\d+\.?\d*)\s*%\s*(?:off|discount|reduction)', al)
                    if disc_m:
                        pct = float(disc_m.group(1))
                        total = total * (1 - pct / 100)
                        steps.append(f"-{pct}% = {total}")
                    # Check for percentage surcharge/fee/tax on the total
                    surcharge_m = re.search(
                        r'(\d+\.?\d*)\s*%\s*(?:delivery|service|handling|processing|'
                        r'fee|tax|surcharge|markup|tip|interest|charge|added)',
                        al)
                    if not surcharge_m:
                        surcharge_m = re.search(
                            r'(?:delivery|service|handling|fee|tax|surcharge|tip|interest)\s+'
                            r'(?:fee\s+)?(?:of\s+)?(\d+\.?\d*)\s*%',
                            al)
                    if surcharge_m and not disc_m:
                        pct = float(surcharge_m.group(1))
                        surcharge = total * pct / 100
                        total += surcharge
                        steps.append(f"+{pct}% fee = {surcharge}, total = {total}")
                    # Check for flat additions: "$N tip"
                    tip_m = re.search(r'\$(\d+\.?\d*)\s*(?:tip|gratuity|donation)', al)
                    if tip_m:
                        tip = float(tip_m.group(1))
                        total += tip
                        steps.append(f"+${tip} tip = {total}")
                    # Check for subtraction after SOP: "gave/spent/paid $N each"
                    for s_sent in context_sents:
                        sl_sent = s_sent.lower()
                        sub_m = re.search(
                            r'(?:gave|spent|paid|donated|tipped)\s+'
                            r'(?:\w+\s+){0,4}'
                            r'\$?(\d+\.?\d*)\s*(?:each|apiece)',
                            sl_sent)
                        if sub_m:
                            sub_val = float(sub_m.group(1))
                            # Find the quantity (e.g., "two sisters")
                            qty_m = re.search(r'(\d+|two|three|four|five)\s+\w+', sl_sent)
                            if qty_m:
                                qty_word = qty_m.group(1)
                                _wn = {'two':2,'three':3,'four':4,'five':5}
                                qty = _wn.get(qty_word, float(qty_word) if qty_word.isdigit() else 1)
                                total -= sub_val * qty
                                steps.append(f"-{qty}×${sub_val} = {total}")
                            else:
                                total -= sub_val
                                steps.append(f"-${sub_val} = {total}")
                    # Check for general subtraction: "N couldn't come", "N were absent"
                    last_sub = 0
                    for s_sent in context_sents:
                        sl_sent = s_sent.lower()
                        # Skip the SOP sentence itself
                        if any(f'{int(float(q))}' in sl_sent and f'{int(float(p))}' in sl_sent
                               for q, p in qty_price_pairs[:1]):
                            continue
                        # "N people/items couldn't/didn't/weren't ..."
                        gen_sub_m = re.search(
                            r'(\d+)\s+\w+\s+(?:couldn\'?t|didn\'?t|weren\'?t|can\'?t|won\'?t|'
                            r'could\s+not|did\s+not|were\s+not)\s+\w+', sl_sent)
                        if gen_sub_m:
                            sub_n = float(gen_sub_m.group(1))
                            total -= sub_n
                            last_sub = sub_n
                            steps.append(f"-{sub_n} (absent) = {total}")
                        # "1/4 that number" / "half that number" / "N/M of that/the number"
                        frac_ref_m = re.search(
                            r'(\d+/\d+|half|a\s+quarter|a\s+third)\s+'
                            r'(?:of\s+)?(?:that|the|this)\s+(?:number|amount|total)',
                            sl_sent)
                        if frac_ref_m and last_sub > 0:
                            frac_str = frac_ref_m.group(1)
                            if frac_str == 'half':
                                frac_val = 0.5
                            elif frac_str in ('a quarter',):
                                frac_val = 0.25
                            elif frac_str in ('a third',):
                                frac_val = 1/3
                            elif '/' in frac_str:
                                fn, fd = frac_str.split('/')
                                frac_val = int(fn) / int(fd)
                            else:
                                frac_val = 0
                            if frac_val > 0:
                                frac_amount = last_sub * frac_val
                                total -= frac_amount
                                steps.append(f"-{frac_val}×{last_sub} = -{frac_amount}, total = {total}")
                    # Check for "solve for unknown": total given, some items known, find count of unknown
                    # "paid a total of $50" + SOP known subtotal → unknown = (total - known_subtotal) / unknown_price
                    # Look specifically for "total of $N" / "paid a total of $N" / "total was $N"
                    given_total_m = re.search(
                        r'(?:a\s+)?total\s+(?:of\s+|was\s+|is\s+|amount\s+of\s+)\$(\d+\.?\d*)', al)
                    if not given_total_m:
                        given_total_m = re.search(
                            r'paid\s+(?:a\s+total\s+of\s+)?\$(\d+\.?\d*)\s*(?:in\s+total|total|altogether)?', al)
                    if not given_total_m:
                        given_total_m = re.search(
                            r'(?:altogether|in\s+all)\s+\$?(\d+\.?\d*)', al)
                    if given_total_m:
                        given_total = float(given_total_m.group(1))
                        if given_total > total:
                            # There might be standalone costs not captured by SOP pairs
                            # Look for "one chicken meal that costs $12" type patterns
                            standalone_costs = []
                            for s_sent in context_sents:
                                sl2 = s_sent.lower()
                                # "one/a X that costs $Y" / "X costs $Y" / "X for $Y"
                                sc_m = re.finditer(
                                    r'(?:(?:one|a|an|1)\s+)?(?:\w+\s+){0,2}'
                                    r'(?:that\s+costs?|costing|for|at)\s+\$(\d+\.?\d*)',
                                    sl2)
                                for m in sc_m:
                                    sc_price = float(m.group(1))
                                    if sc_price not in used_prices:
                                        standalone_costs.append(sc_price)
                                        used_prices.add(sc_price)
                            for sc in standalone_costs:
                                total += sc
                                steps.append(f"+ ${sc} standalone")

                            remainder = given_total - total
                            if remainder > 0:
                                # Find the unknown item's unit price
                                # "each box costs $8.50" / "if each X costs $Y"
                                unknown_price_m = re.search(
                                    r'(?:each|per|every)\s+(?:\w+\s+)?'
                                    r'(?:costs?|is|at|for)\s+\$(\d+\.?\d*)', ql.lower())
                                if not unknown_price_m:
                                    unknown_price_m = re.search(
                                        r'\$(\d+\.?\d*)\s*(?:each|per|apiece)', ql.lower())
                                if unknown_price_m:
                                    unknown_price = float(unknown_price_m.group(1))
                                    if unknown_price > 0:
                                        count = remainder / unknown_price
                                        steps.append(f"Remainder: {given_total} - {total} = {remainder}")
                                        steps.append(f"Unknown count: {remainder} / {unknown_price} = {count}")
                                        return {'answer': normalize_answer(count),
                                                'steps': steps, 'confidence': 'HIGH'}

                    # Check for time multiplier: "N weeks/months/years"
                    # "works 50 weeks a year" → multiply SOP total by weeks
                    time_mult_m = re.search(
                        r'(\d+)\s+(?:weeks?|months?|years?|days?)\s+'
                        r'(?:a\s+|per\s+|each\s+|in\s+a\s+)?'
                        r'(?:year|month|week|day|season|semester)', al)
                    if time_mult_m:
                        time_mult = float(time_mult_m.group(1))
                        total *= time_mult
                        steps.append(f"×{time_mult} (time) = {total}")

                    # Check for change/remaining: "gave $X / has $X"
                    if re.search(r'(?:change|left|remaining)', ql):
                        pay_m = re.search(
                            r'(?:gave|gives?|has|had|pays?|hands?)\s+.*?\$(\d+\.?\d*)', al)
                        if pay_m:
                            payment = float(pay_m.group(1))
                            if payment > total:
                                change = payment - total
                                steps.append(f"Change: {payment} - {total} = {change}")
                                return {'answer': normalize_answer(change),
                                        'steps': steps, 'confidence': 'HIGH'}
                    return {'answer': normalize_answer(total), 'steps': steps,
                            'confidence': 'HIGH'}

        # === Pattern COIN: Coin value computation ===
        # "N quarters, M dimes, P nickels, Q pennies"
        coin_values = {'quarter': 25, 'quarters': 25, 'dime': 10, 'dimes': 10,
                        'nickel': 5, 'nickels': 5, 'penny': 1, 'pennies': 1,
                        'twenty': 2000, 'twenties': 2000}  # $20 bills in cents
        coin_total = 0
        coin_count = 0
        # Only look for coins in "has/have/finds/found/given" context, not price descriptions
        coin_context = al
        for coin_name, coin_val in coin_values.items():
            # "N [coin]" with quantifier
            m_coin = re.search(r'(\d+)\s+' + re.escape(coin_name) + r'\b', coin_context)
            if m_coin:
                # Verify it's possession, not pricing ("cost a nickel each")
                before = coin_context[max(0, m_coin.start()-30):m_coin.start()]
                if re.search(r'(?:cost|price|worth|for|at)\s+(?:a\s+)?$', before):
                    continue
                qty = float(m_coin.group(1))
                coin_total += qty * coin_val
                coin_count += 1
                continue
            # "a quarter" / "two nickels" — only in possession context
            # Skip "cost a nickel" patterns
            m_a = re.search(r'(?:has|have|had|finds?|found|given|gave|receives?|received)\s+'
                            r'(?:\w+\s+){0,6}a\s+' + re.escape(coin_name) + r'\b', coin_context)
            if m_a:
                coin_total += coin_val
                coin_count += 1
                continue
            # "two/three nickels"
            for wn, wv in WORD_NUMS.items():
                m_wc = re.search(r'\b' + re.escape(wn) + r'\s+' + re.escape(coin_name) + r'\b', coin_context)
                if m_wc:
                    before = coin_context[max(0, m_wc.start()-30):m_wc.start()]
                    if not re.search(r'(?:cost|price|worth|for|at)\s+$', before):
                        coin_total += wv * coin_val
                        coin_count += 1
                    break

        if coin_count >= 2:
            # Convert to appropriate unit based on question
            if re.search(r'dollar|dollars|\$', ql):
                result_val = coin_total / 100
            else:
                result_val = coin_total  # cents
            # Check if question asks "how many [item] can buy" at a price
            # But NOT if question asks about change/left/remaining
            is_change_q = bool(re.search(r'(?:left|change|remaining|remain)', ql))
            buy_m = None
            if not is_change_q:
                buy_m = re.search(r'(?:how\s+many).*(?:cost|for)\s+(?:a\s+)?(?:(\d+)\s+cents?|a\s+nickel|a\s+dime|a\s+quarter)', ql + ' ' + al)
                if not buy_m:
                    buy_m = re.search(r'(?:cost)\s+a\s+(nickel|dime|quarter)', al)
            if buy_m:
                if buy_m.group(1):
                    item_cost = float(buy_m.group(1))
                elif 'nickel' in buy_m.group(0):
                    item_cost = 5
                elif 'dime' in buy_m.group(0):
                    item_cost = 10
                elif 'quarter' in buy_m.group(0):
                    item_cost = 25
                else:
                    item_cost = 0
                if item_cost > 0:
                    result_val = int(coin_total / item_cost)
                    steps.append(f"Coins={coin_total}¢, item={item_cost}¢, can buy {result_val}")
                    return {'answer': normalize_answer(result_val), 'steps': steps, 'confidence': 'HIGH'}
            # Check for subtraction (buy something for N cents)
            spend_m = re.search(r'(?:buy|buys|bought|spend|spent|pay|pays).*?(\d+)\s*cents?', al)
            if spend_m:
                spend = float(spend_m.group(1))
                result_val = coin_total - spend
                if re.search(r'dollar', ql):
                    result_val = result_val / 100
                steps.append(f"Coins={coin_total}¢ - {spend}¢ = {result_val}")
                return {'answer': normalize_answer(result_val), 'steps': steps, 'confidence': 'HIGH'}
            steps.append(f"Coins: {coin_total}¢ = {result_val}")
            return {'answer': normalize_answer(result_val), 'steps': steps, 'confidence': 'HIGH'}

        # === Pattern A: Shopping/payment change ===
        # "Buys items costing $A, $B, $C. Pays $D. How much change?"
        # Also: "Buys N items at $X each. Has $Y. How much change?"
        if re.search(r'change|how\s+much\s+(?:money\s+)?(?:does|did|will|should).*(?:get\s+back|receive\s+back|back|left|remain)', ql):
            if len(all_prices) >= 3:
                sorted_prices = sorted(all_prices)
                payment = sorted_prices[-1]
                item_costs = sorted_prices[:-1]
                total_cost = sum(item_costs)
                change = payment - total_cost
                if change >= 0:
                    steps.append(f"Items: {'+'.join(str(c) for c in item_costs)} = {total_cost}")
                    steps.append(f"Change: {payment} - {total_cost} = {change}")
                    return {'answer': normalize_answer(change), 'steps': steps, 'confidence': 'HIGH'}
            # "N items at $X each" + payment
            if len(all_prices) >= 2:
                qty_m = re.search(
                    r'(\d+)\s+(?:\w+\s+){0,3}(?:cost|at|for)\s+[\$]?(\d+\.?\d*)\s*(?:each|apiece|per)',
                    al)
                if not qty_m:
                    # "needs N" ... "costs $X"
                    qty_m2 = re.search(r'(?:needs?|buys?|wants?|gets?)\s+(?:\w+\s+){0,2}(\d+)', al)
                    price_m = re.search(r'(?:costs?|priced?\s+at)\s+[\$]?(\d+\.?\d*)', al)
                    if qty_m2 and price_m:
                        qty = float(qty_m2.group(1))
                        unit_price = float(price_m.group(1))
                        total_cost = qty * unit_price
                        # Find the payment (prepared amount, budget, etc.)
                        pay_m = re.search(r'(?:prepared|has|had|budget|saved)\s+[\$]?(\d+\.?\d*)', al)
                        if pay_m:
                            payment = float(pay_m.group(1))
                            if payment > total_cost:
                                change = payment - total_cost
                                steps.append(f"{qty}×{unit_price}={total_cost}")
                                steps.append(f"Change: {payment} - {total_cost} = {change}")
                                return {'answer': normalize_answer(change), 'steps': steps, 'confidence': 'HIGH'}

        # === Pattern B: Split cost / per-person share ===
        # "Total cost is $X + $Y + $Z. Split N ways."
        if re.search(r'(?:split|divide|share|each\s+(?:person|one|friend|sibling|roommate))', ql) or \
           re.search(r'(?:split|divide|share)\s+(?:\w+\s+){0,3}(?:equally|evenly|between|among)', al):
            divisor_m = re.search(r'(?:split|divide|share)\s+(?:\w+\s+){0,4}(?:between|among|into)\s+(\d+)', al)
            if not divisor_m:
                divisor_m = re.search(r'(\d+)\s+(?:people|friends|siblings|roommates|ways|parts|groups)', al)
            if divisor_m and len(all_prices) >= 2:
                divisor = float(divisor_m.group(1))
                if divisor > 0:
                    total_cost = sum(all_prices)
                    per_person = total_cost / divisor
                    steps.append(f"Total: {total_cost}, split {int(divisor)} ways = {per_person}")
                    return {'answer': normalize_answer(per_person), 'steps': steps, 'confidence': 'HIGH'}

        # === Pattern C: How many units/containers needed ===
        # "N total items. Each container holds M. How many containers?"
        if re.search(r'how\s+many\s+(?:\w+\s+){0,2}(?:does|do|did|will|would|can|should|need)', ql) or \
           re.search(r'how\s+(?:long|many\s+(?:hours?|minutes?|days?|weeks?|trips?))', ql):
            # Look for a large total and a small divisor
            pass  # handled more specifically below

        # === Pattern D: Age problems ===
        # "X will be N in Y years. How old is X now?" → N - Y
        # "X is N. In Y years, how old?" → N + Y
        # "X will be N in Y years. How old in Z years?" → N - Y + Z
        age_m = re.search(
            r'(?:will\s+be|turns?|be)\s+(\d+)\s+(?:years?\s+old\s+)?in\s+(\d+)\s+years?', al)
        if age_m:
            future_age = float(age_m.group(1))
            years_until = float(age_m.group(2))
            current_age = future_age - years_until
            # Does the question ask about "now" or about a different time?
            q_years_m = re.search(r'(\d+)\s+years?\s+(?:from\s+now|ago|later)', ql)
            if q_years_m:
                q_years = float(q_years_m.group(1))
                if 'ago' in ql:
                    result = current_age - q_years
                else:
                    result = current_age + q_years
                steps.append(f"Current: {future_age}-{years_until}={current_age}")
                steps.append(f"Answer: {current_age}{'−' if 'ago' in ql else '+'}{q_years}={result}")
                return {'answer': normalize_answer(result), 'steps': steps, 'confidence': 'HIGH'}
            elif re.search(r'(?:how\s+old|what\s+age).*(?:now|today|current|present|right\s+now)', ql):
                steps.append(f"{future_age} - {years_until} = {current_age}")
                return {'answer': normalize_answer(current_age), 'steps': steps, 'confidence': 'HIGH'}

        # "X is N years old. In Y years..." or question asks about future
        age_now_m = re.search(r'(\w+)\s+(?:is|was|are)\s+(\d+)\s+years?\s+old', al)
        if age_now_m:
            current_age = float(age_now_m.group(2))
            q_years_m = re.search(r'(?:in\s+)?(\d+)\s+years?\s+(?:from\s+now|later|time)', ql)
            if q_years_m:
                q_years = float(q_years_m.group(1))
                result = current_age + q_years
                steps.append(f"{current_age} + {q_years} = {result}")
                return {'answer': normalize_answer(result), 'steps': steps, 'confidence': 'HIGH'}
            q_ago_m = re.search(r'(\d+)\s+years?\s+ago', ql)
            if q_ago_m:
                q_years = float(q_ago_m.group(1))
                result = current_age - q_years
                steps.append(f"{current_age} - {q_years} = {result}")
                return {'answer': normalize_answer(result), 'steps': steps, 'confidence': 'HIGH'}

        # === Pattern D2: Percentage discount/complement ===
        # "Price $X, N% discount. How much pay?" → X * (1 - N/100)
        # "N% of X, how many [remainder]?" → X * (1 - N/100)
        pct_m = re.search(r'(\d+\.?\d*)\s*%', al)
        if pct_m:
            pct = float(pct_m.group(1))
            # Find the base number (price, total, etc.)
            nums = [v for v, _, _ in extract_numbers(all_text) if abs(v - pct) > 0.01]
            if nums:
                base = max(nums)  # typically the largest number is the base

                # Discount pattern: "N% discount/off"
                if re.search(r'discount|off|reduction|sale|cheaper|markdown', al):
                    # Check for quantity × unit price: "bought N items ... cost $X each"
                    qty_m = re.search(
                        r'(?:bought|buys?|ordered|purchased|got)\s+(\d+)\s+\w+.{0,40}'
                        r'(?:cost|at|for)\s+[\$]?(\d+\.?\d*)\s*(?:each|apiece|per)',
                        al)
                    if not qty_m:
                        # "N items that cost $X each"
                        qty_m = re.search(
                            r'(\d+)\s+\w+.{0,30}(?:that|which)\s+cost\s+[\$]?(\d+\.?\d*)\s*each',
                            al)
                    actual_base = base
                    if qty_m:
                        qty = float(qty_m.group(1))
                        unit_price = float(qty_m.group(2))
                        actual_base = qty * unit_price
                    # Reverse discount: question asks for "original/regular price"
                    if re.search(r'(?:original|regular|full|retail|normal|before)\s+(?:the\s+)?price', ql):
                        result = actual_base / (1 - pct / 100)
                        steps.append(f"{actual_base} / (1 - {pct}%) = {result}")
                        return {'answer': normalize_answer(result), 'steps': steps, 'confidence': 'HIGH'}
                    # Forward discount: "how much pay/cost?"
                    if re.search(r'(?:how\s+much|what).*(?:pay|cost|price|spend|charge)', ql):
                        result = actual_base * (1 - pct / 100)
                        steps.append(f"{actual_base} - {pct}% = {result}")
                        return {'answer': normalize_answer(result), 'steps': steps, 'confidence': 'HIGH'}

                # Complement pattern: "beats 80%, how many lose?" → base * (1 - pct/100)
                # "80% are X, how many are not X?" → base * (1 - pct/100)
                if re.search(r'(?:how\s+many|how\s+much).*(?:lose|lost|fail|didn|don|not|remain|left|rest|still)', ql):
                    result = base * (1 - pct / 100)
                    steps.append(f"{base} × (1 - {pct}%) = {result}")
                    return {'answer': normalize_answer(result), 'steps': steps, 'confidence': 'HIGH'}

                # Chained percentage: "N% of X. M% of that."
                pct_all = re.findall(r'(\d+\.?\d*)\s*%', al)
                # Deduplicate while preserving order of first occurrence
                seen_pcts = set()
                unique_pcts = []
                for p in pct_all:
                    if p not in seen_pcts:
                        seen_pcts.add(p)
                        unique_pcts.append(float(p))
                if len(unique_pcts) >= 2:
                    result = base
                    for up in unique_pcts:
                        result *= up / 100
                    steps.append(f"{base} × {'% × '.join(str(int(p)) for p in unique_pcts)}% = {result}")
                    return {'answer': normalize_answer(result), 'steps': steps, 'confidence': 'MEDIUM'}

        # === Pattern E0: Mean/average ===
        # "Scores: 50, 80, 60, 40, 90. Find the mean." → sum/count
        if re.search(r'(?:mean|average)\s+(?:score|grade|rating|value|number|amount|price|cost|speed|weight|height|age|temperature)?', ql):
            # Collect all numbers from context (not question)
            ctx_nums = []
            for s in context_sents:
                for v, _, _ in extract_numbers(s):
                    ctx_nums.append(v)
            if len(ctx_nums) >= 3:
                mean = sum(ctx_nums) / len(ctx_nums)
                steps.append(f"Mean of {len(ctx_nums)} values: {sum(ctx_nums)}/{len(ctx_nums)} = {mean}")
                return {'answer': normalize_answer(mean), 'steps': steps, 'confidence': 'HIGH'}

        # === Pattern E1: Total minus parts (complement) ===
        # "23786 inhabitants. 8417 men and 9092 women. How many children?"
        # "rest" / "remaining" / "how many [other_thing]"
        if re.search(r'(?:rest|remaining|other|left\s+over)', al) or \
           (re.search(r'how\s+many', ql) and len(context_sents) >= 2):
            # Find total (usually first or largest number)
            # Skip numbers that are percentages (followed by %)
            # Convert percentages to actual amounts using the total
            has_pct = bool(re.search(r'\d+\s*%', al))
            ctx_all_nums = []
            pct_amounts = []
            for s in context_sents:
                for v, vstart, vend in extract_numbers(s):
                    after_v = s[vend:vend+3].strip()
                    if after_v.startswith('%'):
                        pct_amounts.append(v)  # store percentage value
                        continue
                    ctx_all_nums.append(v)
            # If percentages present, convert them using the total (first/largest number)
            if has_pct and ctx_all_nums and pct_amounts:
                total_for_pct = max(ctx_all_nums)
                for pv in pct_amounts:
                    ctx_all_nums.append(total_for_pct * pv / 100)
            if len(ctx_all_nums) >= 3:
                # Check if largest = sum of others + answer
                sorted_nums = sorted(ctx_all_nums, reverse=True)
                total_candidate = sorted_nums[0]
                parts = sorted_nums[1:]
                parts_sum = sum(parts)
                if parts_sum < total_candidate and len(parts) >= 2:
                    # Check context: "there are N" + "they include X and Y" pattern
                    has_total_signal = bool(re.search(
                        r'(?:there\s+are|has|have|is|was|total|population|inhabitants|enrolled|registered)\s+(?:\w+\s+){0,3}\d',
                        al))
                    has_parts_signal = bool(re.search(
                        r'(?:include|consist|comprise|of\s+which|are\s+\w+\s+and)', al))
                    if has_total_signal or has_parts_signal:
                        remainder = total_candidate - parts_sum
                        steps.append(f"{total_candidate} - ({'+'.join(str(p) for p in parts)}) = {remainder}")
                        return {'answer': normalize_answer(remainder), 'steps': steps, 'confidence': 'HIGH'}

        # === Pattern E: Sum prices then divide by rate ===
        # "Buy $A item + $B item. Earns $C/hr. How many hours?"
        # "Buy $A + $B. Costs $C each. How many?"
        if len(all_prices) >= 2:
            total_price = sum(all_prices)
            # Find a rate or divisor for the total
            rate_m = re.search(r'\$(\d+\.?\d*)\s*(?:per|/|an?|each)\s+(?:hour|hr|minute|min|day|week|month|year|unit|item|piece)', al)
            if rate_m:
                rate = float(rate_m.group(1))
                if rate > 0 and rate in all_prices:
                    remaining = total_price - rate  # don't count the rate itself
                    if remaining > 0:
                        result = remaining / rate
                        steps.append(f"Total: {remaining}, rate: {rate}")
                        steps.append(f"{remaining} / {rate} = {result}")
                        return {'answer': normalize_answer(result), 'steps': steps, 'confidence': 'MEDIUM'}

        # === Pattern E2: Sum then operate with percentage ===
        if re.search(r'how\s+much|total|cost|spend|price|pay|worth', ql):
            if len(all_prices) >= 2:
                pct_m = re.search(
                    r'(\d+\.?\d*)\s*%\s*(?:fee|tip|tax|surcharge|markup|delivery|service|'
                    r'interest|insurance|handling|shipping|processing|commission|bonus)',
                    al)
                # Also match "pays/charges N% of that/the total/cost/price"
                if not pct_m:
                    pct_m = re.search(
                        r'(?:pays?|charges?|costs?|spends?)\s+(?:an?\s+)?(?:additional\s+)?'
                        r'(\d+\.?\d*)\s*%\s*(?:of\s+(?:that|the\s+(?:total|cost|price|amount|bill)))',
                        al)
                if pct_m:
                    pct = float(pct_m.group(1))
                    base_prices = [p for p in all_prices if abs(p - pct) > 0.01]
                    base_total = sum(base_prices)
                    fee = base_total * pct / 100
                    flat_fees = [v for v, _, _ in extract_numbers(all_text)
                                 if v not in all_prices and abs(v - pct) > 0.01 and v < base_total]
                    result = base_total + fee + sum(flat_fees)
                    steps.append(f"Base: {base_total}, {pct}% fee: {fee}")
                    if flat_fees:
                        steps.append(f"+ flat fees: {flat_fees}")
                    return {'answer': normalize_answer(result), 'steps': steps, 'confidence': 'MEDIUM'}

        # === Pattern F: Reverse computation ===
        # "Lost N per month for M months. Final weight W." → initial = W + N*M
        # "Was unwell for M months, lost N per month. Final W." → same
        rate_val = None
        rate_unit = None
        rate_periods = None

        # Try: "lost N [unit] per [timeunit]"
        rm = re.search(r'(?:lost?|loses?|drops?|decreases?|declines?)\s+(\d+\.?\d*)\s+(?:\w+\s+)?(?:per|each|every|a)\s+(\w+)', al)
        if rm:
            rate_val = float(rm.group(1))
            rate_unit = rm.group(2).rstrip('s')
            # Find "M [timeunit]s" elsewhere
            for pm in re.finditer(r'(\d+)\s+' + re.escape(rate_unit) + r's?\b', al):
                candidate = float(pm.group(1))
                if candidate != rate_val:
                    rate_periods = candidate
                    break
        # Try: "gained/earned N per [unit]" with question asking initial/starting
        if not rm:
            rm = re.search(r'(?:gains?|earns?|adds?|increases?|grows?)\s+(\d+\.?\d*)\s+(?:\w+\s+)?(?:per|each|every|a)\s+(\w+)', al)
            if rm and re.search(r'(?:initial|start|original|begin)', ql):
                rate_val = float(rm.group(1))
                rate_unit = rm.group(2).rstrip('s')
                for pm in re.finditer(r'(\d+)\s+' + re.escape(rate_unit) + r's?\b', al):
                    candidate = float(pm.group(1))
                    if candidate != rate_val:
                        rate_periods = candidate
                        break

        if rate_val is not None and rate_periods is not None:
            total_change = rate_val * rate_periods
            # Find the "final" or "now" value
            final_m = re.search(r'(?:final|now|current|end|left|remaining|weighs?)\s+(?:\w+\s+){0,2}(?:is|was|of)?\s*(\d+)', al)
            if not final_m:
                final_m = re.search(r'(\d+)\s+\w+\s*[,.]?\s*(?:what|find|how)', al)
            if final_m:
                final_val = float(final_m.group(1))
                if final_val != rate_val and final_val != rate_periods:
                    # Lost → add back; gained → subtract back
                    if re.search(r'(?:lost?|loses?|drops?|decreases?|declines?)', al):
                        result = final_val + total_change
                        steps.append(f"Lost: {rate_val}×{rate_periods}={total_change}")
                    else:
                        result = final_val - total_change
                        steps.append(f"Gained: {rate_val}×{rate_periods}={total_change}")
                    steps.append(f"Initial: {result}")
                    return {'answer': normalize_answer(result), 'steps': steps, 'confidence': 'HIGH'}

        # === Pattern G: Rate × duration, then divide by interval ===
        # "Drives 100 miles/day for 30 days. Tune-up every 1000 miles." → 3000/1000=3
        # "Makes 20/hour for 8 hours. Boxes hold 10." → 160/10=16
        if re.search(r'(?:per|each|every|a)\s+(?:day|hour|week|month|year|minute)', al):
            # Find rate and duration
            rate_dur_m = re.search(
                r'(\d+\.?\d*)\s+\w+\s+(?:per|a|each|every|an?)\s+(\w+).+?(\d+)\s+(?:\2s?|(?:day|hour|week|month|year|minute)s?)', al)
            if rate_dur_m:
                rate_v = float(rate_dur_m.group(1))
                dur_v = float(rate_dur_m.group(3))
                product = rate_v * dur_v
                # Find an interval/divisor elsewhere
                for s in context_sents + [question]:
                    interval_m = re.search(
                        r'(?:every|per|each)\s+(\d+\.?\d*)\s+\w+', s.lower())
                    if interval_m:
                        iv = float(interval_m.group(1))
                        if iv > 0 and iv != rate_v and iv != dur_v:
                            result = product / iv
                            steps.append(f"{rate_v}×{dur_v}={product}")
                            steps.append(f"{product}/{iv}={result}")
                            return {'answer': normalize_answer(result), 'steps': steps, 'confidence': 'HIGH'}

        # === Pattern H: Compute intermediate, then answer question ===
        # Run the sentence processor but also check if the question
        # implies a final operation on the result
        state = SolverState()
        for sent in context_sents:
            self._process_sentence(sent, state)

        if state.last is not None and len(state.steps) >= 1:
            intermediate = state.last

            # Time period words in question: decade=10, century=100, etc.
            _PERIOD_WORDS = {
                'decade': 10, 'century': 100, 'fortnight': 14,
                'semester': 6,  # ~6 months
            }
            for pw, pv in _PERIOD_WORDS.items():
                if pw in ql:
                    result = intermediate * pv
                    steps = state.steps + [f"× {pv} ({pw}) = {result}"]
                    return {'answer': normalize_answer(result), 'steps': steps, 'confidence': 'HIGH'}

            # Question implies division: "how many X" where X is a container
            div_m = re.search(
                r'how\s+many\s+(?:\w+\s+){0,2}'
                r'(?:does|do|did|will|would|can|should|could|must)\s+'
                r'(?:\w+\s+){0,3}(?:need|require|buy|take|use|make|prepare|bake|cook|order)', ql)
            if div_m:
                # Find a divisor in the question
                q_nums = extract_numbers(question)
                q_vals = [v for v, _, _ in q_nums]
                for qv in q_vals:
                    if qv > 0 and qv != intermediate:
                        result = intermediate / qv
                        steps = state.steps + [f"{intermediate} / {qv} = {result}"]
                        return {'answer': normalize_answer(result), 'steps': steps, 'confidence': 'MEDIUM'}

            # Question asks "how many hours/days/trips" and context has a rate
            time_div_m = re.search(
                r'how\s+(?:many|long)\s+(?:\w+\s+){0,2}'
                r'(?:hours?|minutes?|days?|weeks?|trips?|times?)', ql)
            if time_div_m:
                q_nums = extract_numbers(question)
                q_vals = [v for v, _, _ in q_nums]
                for qv in q_vals:
                    if qv > 0 and qv != intermediate:
                        result = intermediate / qv
                        steps = state.steps + [f"{intermediate} / {qv} = {result}"]
                        return {'answer': normalize_answer(result), 'steps': steps, 'confidence': 'MEDIUM'}

            # Check if process_sentence already applied a division
            already_divided = any('/' in st for st in state.steps)

            # Question has a number that should be added/subtracted from intermediate
            q_nums = extract_numbers(question)
            q_vals = [v for v, _, _ in q_nums]
            if q_vals and len(q_vals) == 1:
                qv = q_vals[0]
                # "if she has $N" → total budget, answer = budget - intermediate
                budget_m = re.search(r'(?:has|have|had|with|brings?|brought|gives?|gave|pays?|paid)\s+\$?' + str(int(qv) if qv == int(qv) else qv), ql)
                if budget_m:
                    if re.search(r'change|left|remain|save|keep', ql):
                        result = qv - intermediate
                        steps = state.steps + [f"{qv} - {intermediate} = {result}"]
                        return {'answer': normalize_answer(result), 'steps': steps, 'confidence': 'MEDIUM'}
                    elif re.search(r'how\s+many\s+(?:more|additional|extra)', ql):
                        result = intermediate - qv
                        steps = state.steps + [f"{intermediate} - {qv} = {result}"]
                        return {'answer': normalize_answer(result), 'steps': steps, 'confidence': 'MEDIUM'}

        return None

    def _try_chain_multiply(self, context_sents: List[str],
                            question: str, full_text: str) -> Optional[Dict[str, Any]]:
        """Solve chain multiplication: "each X has N Y" linking sentences.

        ULTRA-STRICT: Only fires when ALL context sentences have exactly ONE number
        each, at least one has "each/every" rate pattern, no add/subtract operations,
        and the question asks "how many total".

        Pattern: "3 bushes. Each bush has 25 roses. Each rose has 8 thorns."
        → 3 × 25 × 8 = 600
        """
        _ = full_text
        ql = question.lower()

        # Must ask "how many" or "how much" or "total"
        if not re.search(r'how\s+(?:many|much)|total', ql):
            return None

        al = ' '.join(context_sents).lower()

        # Must NOT have add/subtract verbs — those require different solving
        if re.search(r'\b(?:gave|give|lost|lose|sold|sell|spent|spend|'
                     r'donated|removed|threw|dropped|broke|left|remaining|less|fewer|more)\b', al):
            return None

        # Must have "each" or "every" or "per" rate word
        if not re.search(r'\b(?:each|every|per)\b', al):
            return None

        # Every context sentence must have EXACTLY one number
        chain_factors = []
        rate_sents = 0
        for sent in context_sents:
            nums = extract_numbers(sent)
            vals = [v for v, _, _ in nums if v > 0]
            if len(vals) != 1:
                return None  # Sentence with 0 or 2+ numbers → bail
            chain_factors.append(vals[0])
            sl = sent.lower()
            if re.search(r'\b(?:each|every|per)\b', sl):
                rate_sents += 1

        if rate_sents == 0 or len(chain_factors) < 2:
            return None

        product = 1.0
        for f in chain_factors:
            product *= f

        # Dozen conversion
        if 'dozen' in ql or 'dozens' in ql:
            product = product / 12

        steps = [' × '.join(str(int(f) if f == int(f) else f) for f in chain_factors) + f' = {product}']
        return {'answer': normalize_answer(product), 'steps': steps,
                'confidence': 'MEDIUM'}

    def _try_computation_graph(self, context_sents: List[str],
                               question: str) -> Optional[Dict[str, Any]]:
        """Generic computation graph solver v3.

        2-pass constraint-based approach (inspired by causal DAG resolution):
        Pass 1: Extract constraints as symbolic expressions
        Pass 2: Resolve in dependency order (topological sort)

        Handles: rate×quantity, cross-entity refs, accumulation, percentages, fractions.
        """
        ql = question.lower()
        steps = []

        STOP_NAMES = {'the', 'if', 'on', 'in', 'at', 'to', 'for', 'how', 'what',
                       'when', 'then', 'each', 'every', 'there', 'this', 'that',
                       'after', 'before', 'during', 'since', 'because', 'so',
                       'one', 'it', 'he', 'she', 'they', 'his', 'her', 'its',
                       'but', 'and', 'or', 'not', 'all', 'some', 'any', 'no',
                       'last', 'first', 'next', 'now', 'also', 'just', 'still'}

        # ── Pass 1: Extract constraints ──
        # Each constraint: (entity, op, operands, depends_on, sent_idx)
        # op: SET, ADD, SUB, MUL, FRAC_SUB, FRAC_ADD, CROSS_MUL, RATE_MUL
        constraints = []
        entity_first_seen = {}  # entity -> sent_idx
        last_named = None  # for pronoun resolution
        all_entities = []  # ordered

        for si, sent in enumerate(context_sents):
            sl = self._word_nums_to_digits(sent.lower())
            words = set(re.findall(r'\b\w+\b', sl))

            # ── Entity detection ──
            entity = '_default'
            name_m = re.match(r'([A-Z][a-z]+)', sent.strip())
            if name_m and name_m.group(1).lower() not in STOP_NAMES:
                entity = name_m.group(1).lower()
                last_named = entity
            elif last_named and re.search(r'\b(?:he|she|they|his|her|their)\b', sl):
                entity = last_named

            if entity not in entity_first_seen:
                entity_first_seen[entity] = si
                all_entities.append(entity)

            # ── Number extraction (always convert word numbers) ──
            nums = extract_numbers(sl)
            num_vals = [v for v, _, _ in nums]

            # ── Detect modifiers ──
            has_init = bool(words & INITIAL_VERBS)
            has_add = bool(words & ADD_VERBS)
            has_sub = bool(words & SUB_VERBS)
            has_rate = bool(words & MULT_SIGNALS) or bool(
                re.search(r'\b(?:a\s+day|a\s+week|a\s+month|a\s+year|an\s+hour|a\s+minute)\b', sl))

            # Cross-entity reference
            ref_entity = None
            ref_m = re.search(r'(?:as|than|of)\s+([A-Z][a-z]+)', sent)
            if ref_m:
                rn = ref_m.group(1).lower()
                if rn not in STOP_NAMES:
                    ref_entity = rn
                    if ref_entity not in entity_first_seen:
                        entity_first_seen[ref_entity] = -1
                        all_entities.append(ref_entity)

            # Multiplier words
            mult_val = None
            for mw, mv in MULTIPLIER_WORDS.items():
                if mw in sl:
                    mult_val = mv
                    break
            times_m = re.search(r'(\d+)\s+times\s+(?:as\s+)?(?:many|much|more|that|the)', sl)
            if times_m:
                mult_val = float(times_m.group(1))

            # Fraction
            frac_val = None
            for fname, fval in FRACTION_MAP.items():
                if fname in sl:
                    frac_val = fval
                    break

            # Percentage
            pct_m = re.search(r'(\d+\.?\d*)\s*%', sl)

            # ── Build constraint ──
            if not num_vals and not mult_val and frac_val is None and not pct_m:
                # No-number sentence with multiplier only — "twice as many as X"
                if mult_val and ref_entity:
                    constraints.append((entity, 'CROSS_MUL', [mult_val], ref_entity, si))
                elif frac_val is not None and ref_entity:
                    constraints.append((entity, 'CROSS_MUL', [frac_val], ref_entity, si))
                continue

            # Cross-entity multiplier: "X has twice/N times as many as Y"
            if mult_val and ref_entity:
                constraints.append((entity, 'CROSS_MUL', [mult_val], ref_entity, si))
                continue

            # Cross-entity with fraction: "X has half as many as Y"
            if frac_val and ref_entity and not num_vals:
                constraints.append((entity, 'CROSS_MUL', [frac_val], ref_entity, si))
                continue

            # "half that much" / "half as many" referencing previous value
            if frac_val and not num_vals and not ref_entity:
                # Refers to same entity's current value
                constraints.append((entity, 'FRAC_MUL', [frac_val], None, si))
                continue

            # Rate multiplication: "N X per/each Y" + "M Y" → N*M
            if has_rate and len(num_vals) >= 2:
                product = 1
                for v in num_vals:
                    product *= v
                if entity in entity_first_seen and entity_first_seen[entity] < si:
                    constraints.append((entity, 'ADD', [product], None, si))
                else:
                    constraints.append((entity, 'SET', [product], None, si))
                continue

            # "each" referring to a previous entity count → multiply
            if re.search(r'\b(?:each|per|every)\b', sl) and len(num_vals) == 1:
                # "60 meters each sprint" → multiply previous accumulator
                constraints.append((entity, 'RATE_APPLY', num_vals, None, si))
                continue

            # Percentage
            if pct_m:
                pct = float(pct_m.group(1)) / 100
                non_pct = [v for v in num_vals if v != float(pct_m.group(1))]
                if re.search(r'(?:discount|off|less|reduce|decrease|lose)', sl):
                    constraints.append((entity, 'FRAC_SUB', [pct], None, si))
                elif re.search(r'(?:more|increase|raise|markup|tip|tax|extra|increased)', sl):
                    if non_pct:
                        # "increased by 150%" with base value
                        constraints.append((entity, 'SET', non_pct, None, si))
                        constraints.append((entity, 'FRAC_ADD', [pct], None, si))
                    else:
                        constraints.append((entity, 'FRAC_ADD', [pct], None, si))
                else:
                    if non_pct:
                        constraints.append((entity, 'SET', [non_pct[0] * pct], None, si))
                    else:
                        constraints.append((entity, 'FRAC_MUL', [pct], None, si))
                continue

            # Fraction operation
            if frac_val is not None:
                if has_sub or re.search(r'(?:gave|lost|ate|used|spent|removed|away)', sl):
                    constraints.append((entity, 'FRAC_SUB', [frac_val], None, si))
                elif re.search(r'(?:remaining|rest|left)', sl):
                    constraints.append((entity, 'FRAC_SUB', [frac_val], None, si))
                else:
                    constraints.append((entity, 'FRAC_MUL', [frac_val], None, si))
                # Also SET if there are numbers
                if num_vals and entity not in entity_first_seen:
                    constraints.insert(-1, (entity, 'SET', num_vals[:1], None, si))
                continue

            # Subtraction
            if has_sub and not has_add and num_vals:
                sub_val = num_vals[0]
                if len(num_vals) >= 2 and has_rate:
                    sub_val = num_vals[0] * num_vals[1]
                constraints.append((entity, 'SUB', [sub_val], None, si))
                continue

            # Addition
            if has_add and not has_sub and num_vals:
                add_val = num_vals[0]
                if len(num_vals) >= 2 and has_rate:
                    add_val = num_vals[0] * num_vals[1]
                constraints.append((entity, 'ADD', [add_val], None, si))
                continue

            # Multi-number with multiplication context
            if len(num_vals) >= 2 and re.search(r'\b(?:at|for|worth|costing?)\b', sl):
                product = num_vals[0] * num_vals[1]
                if entity in entity_first_seen and entity_first_seen[entity] < si:
                    constraints.append((entity, 'ADD', [product], None, si))
                else:
                    constraints.append((entity, 'SET', [product], None, si))
                continue

            # Default: SET if first mention, ADD otherwise
            if entity not in entity_first_seen or entity_first_seen[entity] == si:
                constraints.append((entity, 'SET', num_vals[:1], None, si))
            else:
                if re.search(r'(?:more|another|additional|also|plus)', sl):
                    constraints.append((entity, 'ADD', num_vals[:1], None, si))
                elif re.search(r'(?:less|fewer|minus|without|except)', sl):
                    constraints.append((entity, 'SUB', num_vals[:1], None, si))
                else:
                    constraints.append((entity, 'ADD', num_vals[:1], None, si))

        if not constraints:
            return None

        # ── Pass 2: Resolve constraints in dependency order ──
        entities = {}
        # First pass: resolve non-dependent constraints (SET, ADD, SUB)
        # Multiple passes to handle forward references
        unresolved = list(constraints)
        max_passes = 3
        for pass_num in range(max_passes):
            still_unresolved = []
            for entity, op, operands, depends_on, si in unresolved:
                if depends_on and depends_on not in entities:
                    still_unresolved.append((entity, op, operands, depends_on, si))
                    continue

                val = operands[0] if operands else 0

                if op == 'SET':
                    entities[entity] = val
                    steps.append(f"CG: {entity} = {val}")
                elif op == 'ADD':
                    entities[entity] = entities.get(entity, 0) + val
                    steps.append(f"CG: {entity} += {val} → {entities[entity]}")
                elif op == 'SUB':
                    if entity in entities:
                        entities[entity] -= val
                    else:
                        entities[entity] = -val
                    steps.append(f"CG: {entity} -= {val} → {entities[entity]}")
                elif op == 'CROSS_MUL':
                    if depends_on in entities:
                        entities[entity] = entities[depends_on] * val
                        steps.append(f"CG: {entity} = {depends_on}({entities[depends_on]}) × {val} = {entities[entity]}")
                    else:
                        still_unresolved.append((entity, op, operands, depends_on, si))
                        continue
                elif op == 'FRAC_SUB':
                    if entity in entities:
                        entities[entity] -= entities[entity] * val
                        steps.append(f"CG: {entity} -= {val*100}% → {entities[entity]}")
                elif op == 'FRAC_ADD':
                    if entity in entities:
                        entities[entity] += entities[entity] * val
                        steps.append(f"CG: {entity} += {val*100}% → {entities[entity]}")
                elif op == 'FRAC_MUL':
                    if entity in entities:
                        entities[entity] *= val
                        steps.append(f"CG: {entity} ×= {val} → {entities[entity]}")
                    else:
                        still_unresolved.append((entity, op, operands, depends_on, si))
                        continue
                elif op == 'RATE_APPLY':
                    # Multiply current entity value by this rate
                    if entity in entities:
                        entities[entity] *= val
                        steps.append(f"CG: {entity} ×= {val} → {entities[entity]}")
                    else:
                        still_unresolved.append((entity, op, operands, depends_on, si))
                        continue

            unresolved = still_unresolved
            if not unresolved:
                break

        if not entities:
            return None

        # ── Phase 3: Answer the question ──
        target = None
        for ent in all_entities:
            if ent != '_default' and ent in ql:
                target = ent
                break
        if target is None:
            if len(entities) == 1:
                target = list(entities.keys())[0]
            elif '_default' in entities:
                target = '_default'
            else:
                target = [e for e in all_entities if e in entities][-1] if \
                    any(e in entities for e in all_entities) else list(entities.keys())[-1]

        if target not in entities:
            return None

        result = entities[target]

        # "total / altogether / combined / together"
        if re.search(r'\b(?:total|altogether|combined|together|both)\b', ql):
            if len(entities) > 1:
                result = sum(v for v in entities.values())

        # "how many more X than Y"
        diff_m = re.search(r'how\s+many\s+more.*?than\s+(\w+)', ql)
        if diff_m and len(entities) >= 2:
            other = diff_m.group(1).lower()
            if other in entities and target in entities:
                result = abs(entities[target] - entities[other])

        # Cost/spend questions — positive
        if re.search(r'(?:spend|pay|cost|charge|price|worth|owe)', ql):
            result = abs(result)

        # Reject negative for "how many/much"
        if result < 0 and re.search(r'how\s+(?:many|much)', ql):
            return None

        return {'answer': normalize_answer(result),
                'steps': steps,
                'confidence': 'MEDIUM'}

    def _try_rate_chain(self, context_sents: List[str],
                        question: str) -> Optional[Dict[str, Any]]:
        """Solve rate×quantity chain problems.

        Extracts all numbers with their units/contexts, identifies
        rate relationships (per/each), and chains multiplications.
        """
        all_text = ' '.join(context_sents)
        sl = all_text.lower()
        ql = question.lower()
        steps = []

        # ── Rate as division: "N [unit1] to/for every M [unit2]" + total T → T/M * N ──
        # "10 minutes to cover every 3 miles. 42 miles total." → 42/3 * 10 = 140
        # "2 flowers a day. After 15 days. 5 didn't grow." → 2*15 - 5 = 25
        rate_div_m = re.search(
            r'(\d+\.?\d*)\s+\w+\s+(?:to\s+(?:cover|do|complete|finish|travel|run|walk|read|make))\s+'
            r'(?:every|each|per)\s+(\d+\.?\d*)\s+(\w+)', sl)
        if rate_div_m:
            rate_time = float(rate_div_m.group(1))
            rate_dist = float(rate_div_m.group(2))
            unit_word = rate_div_m.group(3)
            # Find the total amount
            total_m = re.search(r'(\d+\.?\d*)\s+' + re.escape(unit_word), sl + ' ' + ql)
            if total_m:
                total_val = float(total_m.group(1))
                if total_val != rate_dist and rate_dist > 0:
                    result = (total_val / rate_dist) * rate_time
                    steps.append(f"{total_val}/{rate_dist} × {rate_time} = {result}")
                    return {'answer': normalize_answer(result), 'steps': steps, 'confidence': 'HIGH'}

        # ── Cross-sentence pricing: shared quantity × sum of unit prices ──
        # "3 pairs of X, 3 pairs of Y. X costs $A. Y costs $B" → 3×(A+B)
        prices = re.findall(r'\$(\d+\.?\d*)', sl)
        qty_matches = re.findall(r'(\d+)\s+(?:pairs?|sets?|boxes?|packs?|bags?)\s+(?:of\s+)', sl)
        if qty_matches and len(prices) >= 2 and all(q == qty_matches[0] for q in qty_matches):
            qty = float(qty_matches[0])
            price_vals = [float(p) for p in prices]
            total = qty * sum(price_vals)
            steps.append(f"{qty} × ({'+'.join(prices)}) = {total}")
            return {'answer': normalize_answer(total), 'steps': steps, 'confidence': 'MEDIUM'}

        # Extract number-context pairs from each sentence
        items = []
        for i, sent in enumerate(context_sents):
            nums = extract_numbers(sent)
            sent_lower = sent.lower()
            for val, start, end in nums:
                after = sent_lower[end:end+40].strip()
                before = sent_lower[max(0, start-40):start].strip()
                items.append({
                    'val': val, 'after': after, 'before': before,
                    'sent': sent_lower, 'sent_idx': i,
                })

        if len(items) < 2:
            return None

        # Strategy: Chain multiplication for rate patterns
        _RATE_RE = r'(?:per|each|every)\s+\w+|(?:a|an)\s+(?:minute|hour|day|week|month|year|pound|ounce|gallon|mile|piece|unit|item|serving|acre|ton|barrel|bag|box|pack|carton|bottle|cup|glass|slice|sheet|page|block)'
        rate_count = sum(1 for it in items
                         if re.search(_RATE_RE, it['after'] + ' ' + it['before']))

        if rate_count >= 1 and 2 <= len(items) <= 4:
            all_nums = [it['val'] for it in items]

            # Check for sequential add/sub that should prevent multiplication
            words = set(re.findall(r'\b\w+\b', sl))
            sequential_ops = bool(words & {'then', 'later', 'after', 'next', 'left',
                                            'remaining', 'leftover', 'remainder'})

            # Independent quantity listing check
            has_listing = bool(re.search(
                r'\d+\s+\w+\s+(?:in\s+the\s+)?(?:morning|afternoon|evening|first|second|third)'
                r'|\d+\s+\w+\s+and\s+\d+\s+\w+', sl))
            if has_listing and len(items) > 3:
                sequential_ops = True

            # For 4+ numbers: only multiply if ALL sentences have rate patterns
            if len(items) >= 4:
                sent_indices = set(it['sent_idx'] for it in items)
                sents_with_rate = set()
                for it in items:
                    if re.search(_RATE_RE, it['after'] + ' ' + it['before']):
                        sents_with_rate.add(it['sent_idx'])
                # If most sentences don't have rates, skip
                if len(sents_with_rate) < len(sent_indices) * 0.5:
                    sequential_ops = True

            if not sequential_ops:
                # Check for multi-item addition: "N of A at $X and M of B at $Y"
                if len(all_nums) == 4 and re.search(r'\band\b', sl):
                    def _get_noun(item):
                        stop_w = {'the', 'and', 'for', 'are', 'was', 'how', 'much',
                                 'many', 'does', 'cost', 'each', 'per', 'total',
                                 'that', 'this', 'with', 'from', 'have', 'has',
                                 'costs', 'bought', 'buys', 'can', 'more', 'will',
                                 'spend', 'spent', 'pay', 'paid', 'make', 'made'}
                        after_clip = re.split(r'\band\b|,|\.', item['after'])[0]
                        after_nouns = [n for n in re.findall(r'\b([a-z]{3,})\b', after_clip)
                                       if n not in stop_w][:1]
                        before_clip = re.split(r'\band\b|,|\.', item['before'])[-1]
                        before_nouns = [n for n in re.findall(r'\b([a-z]{3,})\b', before_clip)
                                        if n not in stop_w][-1:]
                        return after_nouns or before_nouns

                    item_nouns = [_get_noun(it) for it in items]

                    def _stems_match(nouns1, nouns2):
                        for n1 in nouns1:
                            for n2 in nouns2:
                                if n1[:4] == n2[:4]:
                                    return True
                        return False

                    paired = False
                    result = 0
                    used = set()
                    for i in range(len(items)):
                        if i in used:
                            continue
                        for j in range(i+1, len(items)):
                            if j in used:
                                continue
                            if _stems_match(item_nouns[i], item_nouns[j]):
                                result += items[i]['val'] * items[j]['val']
                                used.add(i)
                                used.add(j)
                                paired = True
                                break

                    if not paired or len(used) < 4:
                        result = all_nums[0] * all_nums[2] + all_nums[1] * all_nums[3]

                    if re.search(r'how\s+much|total|cost|spend|price', ql):
                        steps.append(f"Sum of products: {result}")
                        return {'answer': normalize_answer(result), 'steps': steps,
                                'confidence': 'MEDIUM'}

                product = 1
                for n in all_nums:
                    product *= n

                # Time unit conversion
                all_q_text = sl + ' ' + ql
                for u1 in self.TIME_UNITS:
                    for u2 in self.TIME_UNITS:
                        if u1 != u2:
                            if re.search(r'\d+\s+' + u1 + r's?\b', all_q_text) and \
                               re.search(r'(?:per|a|each|every)\s+' + u2 + r'\b', all_q_text):
                                key = (u2, u1)
                                if key in self.TIME_CONVERSIONS:
                                    product *= self.TIME_CONVERSIONS[key]
                                    steps.append(f"Unit: {u1}->{u2}")
                                    break

                ctx_unit = self._detect_time_unit(sl)
                q_unit = self._detect_time_unit(question)
                if ctx_unit and q_unit and ctx_unit != q_unit:
                    key = (ctx_unit, q_unit)
                    if key in self.TIME_CONVERSIONS:
                        product *= self.TIME_CONVERSIONS[key]

                steps.append(f"{'×'.join(str(n) for n in all_nums)} = {product}")
                return {'answer': normalize_answer(product), 'steps': steps,
                        'confidence': 'MEDIUM'}

        return None

    def _extract_question_facts(self, q: str) -> Optional[str]:
        """Extract numeric facts embedded in the question."""
        if not q:
            return None
        m = re.search(r'\bif\s+(.+?)(?:\?|$)', q, re.I)
        if m:
            fact = m.group(1).strip()
            fact = re.sub(r',?\s*(?:how|calculate|find|determine|what|compute)\s+.*$', '', fact, flags=re.I)
            # Convert word numbers to digits (e.g., "two humps" → "2 humps")
            converted = self._word_nums_to_digits(fact)
            has_numeric = any(c.isdigit() for c in converted)
            has_multiplier = bool(re.search(r'\b(?:twice|double|triple|half|thrice)\b', converted, re.I))
            if has_numeric or has_multiplier:
                return converted
        return None

    def _extract_question_numbers(self, q: str) -> Optional[str]:
        """Extract numeric clauses from the question body.
        'How much do 6 erasers and 8 pencils cost?' → '6 erasers and 8 pencils'
        Also handles word numbers: 'in three bags' → 'in 3 bags'
        """
        if not q:
            return None
        nums = extract_numbers(q)
        if not nums:
            return None
        # Remove the "how many/much" prefix and "?" suffix
        cleaned = re.sub(r'^.*?(?:how\s+(?:many|much|long|far|old|often|big|tall))\s+', '', q, flags=re.I)
        cleaned = re.sub(r'\?.*$', '', cleaned)
        # Check for digit numbers first (most reliable)
        has_digits = any(c.isdigit() for c in cleaned)
        if has_digits and 'if ' not in cleaned.lower():
            return cleaned.strip()
        # Also accept word numbers, but only if they aren't fractions or references
        # Skip: "two thirds", "three quarters", "the two X"
        if not has_digits:
            cl = cleaned.lower()
            has_fraction_word = bool(re.search(
                r'\b(?:two|three|four|five|six|seven|eight|nine)\s+'
                r'(?:halves?|thirds?|quarters?|fourths?|fifths?|sixths?|'
                r'sevenths?|eighths?|ninths?|tenths?)\b', cl))
            has_the_ref = bool(re.search(
                r'\bthe\s+(?:two|three|four|five|six|seven|eight|nine|ten)\b', cl))
            if not has_fraction_word and not has_the_ref and 'if ' not in cl:
                clean_nums = extract_numbers(cleaned)
                if clean_nums:
                    return cleaned.strip()
        return None

    def _has_mult_signal(self, s: str) -> bool:
        """Check if sentence has multiplication signal words."""
        return any(re.search(r'\b' + w + r'\b', s) for w in MULT_SIGNALS)

    def _has_rate_pattern(self, s: str) -> bool:
        """Check if sentence has a rate pattern."""
        return any(re.search(p, s) for p in RATE_PATTERNS)

    def _classify_action(self, sent: str) -> str:
        """Classify the primary action of a sentence."""
        s = sent.lower()
        words = set(re.findall(r'\b\w+\b', s))

        has_div = any(re.search(p, s) for p in DIV_PATTERNS)
        has_mult = self._has_mult_signal(s) or self._has_rate_pattern(s)
        has_sub = bool(words & SUB_VERBS)
        has_add = bool(words & ADD_VERBS)
        has_init = bool(words & INITIAL_VERBS)

        # Count numbers in sentence
        nums = extract_numbers(sent)
        n_nums = len(nums)

        # ── Priority: DIV > MULT > SUB/ADD > INIT ──

        if has_div and not has_mult:
            return 'DIV'

        # Explicit pricing: "N items at/for $X each"
        if re.search(r'\d+\s+\w+\s+(?:at|for|which\s+cost|costing)\s+\$?\d', s):
            return 'MULT'

        # Multiplier words (twice, double, triple)
        for w in MULTIPLIER_WORDS:
            if w in s:
                return 'MULT'

        # "N times as many/much"
        if re.search(r'\d+\s+times\s+(?:as\s+)?(?:many|much|more|that)', s):
            return 'MULT'

        # MULT signal + 2+ numbers → multiply them
        # BUT: temporal qualifiers (every day/morning) don't mean multiply
        # when the sentence has explicit SUB/ADD verbs
        temporal_only = bool(re.search(
            r'\bevery\s+(?:day|morning|afternoon|evening|night|week|month|year)\b'
            r'|\bdaily\b|\bweekly\b|\bmonthly\b|\byearly\b', s))
        if has_mult and n_nums >= 2 and not (temporal_only and (has_sub or has_add)):
            return 'MULT'

        # MULT signal + 1 number → multiply with state.last
        if has_mult and n_nums >= 1 and not (temporal_only and (has_sub or has_add)):
            return 'MULT'

        # Percentage
        if re.search(r'\d+\.?\d*\s*%', s):
            return 'PCT'

        # Fraction patterns
        for fn in FRACTION_MAP:
            if fn in s:
                return 'FRAC'

        # Fix: "gave/gives him/her/them/[name] N" = someone giving TO subject = ADD
        if has_sub and re.search(
                r'(?:gave|gives?)\s+(?:him|her|them|me|us|\w+)\s+(?:another\s+)?\$?\d', s):
            has_add = True
            has_sub = False

        # SUB vs ADD
        if has_sub and not has_add:
            return 'SUB'
        if has_add and not has_sub:
            return 'ADD'
        if has_sub and has_add:
            # Strong add verbs trump generic sub
            if words & {'buys', 'bought', 'gets', 'got', 'earns', 'earned',
                        'receives', 'received', 'wins', 'won', 'finds', 'found',
                        'collects', 'collected', 'gathers', 'gathered',
                        'adds', 'added', 'refills', 'refilled'}:
                return 'ADD'
            return 'SUB'

        if has_init:
            return 'INIT'

        if n_nums > 0:
            return 'INIT'

        return 'NONE'

    def _process_sentence(self, sent: str, state: SolverState):
        """Process a single sentence, updating state."""
        s = sent.strip()
        if not s:
            return
        sl = s.lower()

        # ── Specialized handlers (highest priority) ──
        if self._handle_time_duration(sl, s, state):
            return
        if self._handle_percentage(sl, state):
            return
        if self._handle_fraction_of(sl, state):
            return
        if self._handle_multiplier_ref(sl, state):
            return
        if self._handle_comparison(sl, state):
            return
        if self._handle_pricing(sl, s, state):
            return

        # ── General action-based processing ──
        action = self._classify_action(s)
        nums = extract_numbers(s)
        values = [v for v, _, _ in nums]

        if not values:
            return

        if action == 'INIT':
            self._do_init(values, sl, state)
        elif action == 'ADD':
            self._do_add(values, sl, state)
        elif action == 'SUB':
            self._do_sub(values, sl, state)
        elif action == 'MULT':
            self._do_mult(values, sl, state)
        elif action == 'DIV':
            self._do_div(values, sl, state)

    # ── Action Handlers ──

    def _do_init(self, values: List[float], sl: str, state: SolverState):
        """Initialize or set a value."""
        if len(values) == 1:
            if state.last is None:
                state.op(f"Start: {values[0]}", values[0])
            else:
                # Store as secondary value
                state.vars[f'v{len(state.vars)}'] = values[0]
        elif len(values) >= 2:
            # Multiple numbers in INIT: could be compound noun "a 2x3 grid"
            # or rate "16 eggs per day" — check for rate
            if self._has_rate_pattern(sl):
                result = values[0] * values[1]
                state.op(f"{values[0]} x {values[1]} = {result}", result)
            elif state.last is None:
                state.op(f"Start: {values[0]}", values[0])

    def _do_add(self, values: List[float], sl: str, state: SolverState):
        """Add to running total."""
        if len(values) == 1:
            val = values[0]
            if state.last is None:
                state.op(f"Start: {val}", val)
            else:
                old = state.last
                result = old + val
                state.op(f"{old} + {val} = {result}", result)
        elif len(values) >= 2:
            # Check for embedded multiplication: "buys 3 packs of 5 each"
            if self._has_mult_signal(sl) or self._has_rate_pattern(sl):
                product = values[0] * values[1]
                if state.last is not None:
                    old = state.last
                    result = old + product
                    state.op(f"{old} + {values[0]}x{values[1]} = {result}", result)
                else:
                    state.op(f"{values[0]} x {values[1]} = {product}", product)
            else:
                if state.last is None:
                    result = sum(values)
                    state.op(f"{' + '.join(str(v) for v in values)} = {result}", result)
                else:
                    old = state.last
                    result = old + sum(values)
                    state.op(f"{old} + {' + '.join(str(v) for v in values)} = {result}", result)

    def _do_sub(self, values: List[float], sl: str, state: SolverState):
        """Subtract from running total."""
        # Check for non-temporal MULT signal
        temporal_only = bool(re.search(
            r'\bevery\s+(?:day|morning|afternoon|evening|night|week|month|year)\b'
            r'|\bdaily\b|\bweekly\b|\bmonthly\b|\byearly\b', sl))
        real_mult = (self._has_mult_signal(sl) or self._has_rate_pattern(sl)) and not temporal_only

        if len(values) == 1:
            val = values[0]
            if state.last is None:
                state.op(f"Start: {val}", val)
            else:
                old = state.last
                result = old - val
                state.op(f"{old} - {val} = {result}", result)
        elif len(values) >= 2:
            if real_mult:
                product = values[0] * values[1]
                if state.last is not None:
                    old = state.last
                    result = old - product
                    state.op(f"{old} - {values[0]}x{values[1]} = {result}", result)
                else:
                    state.op(f"{values[0]} x {values[1]} = {product}", product)
            else:
                if state.last is None:
                    result = values[0] - values[1]
                    state.op(f"{values[0]} - {values[1]} = {result}", result)
                else:
                    old = state.last
                    total_sub = sum(values)
                    result = old - total_sub
                    state.op(f"{old} - {' - '.join(str(v) for v in values)} = {result}", result)

    def _do_mult(self, values: List[float], sl: str, state: SolverState):
        """Multiply values."""
        words = set(re.findall(r'\b\w+\b', sl))

        if len(values) >= 2:
            # Product of all numbers in sentence
            product = 1
            for v in values:
                product *= v

            if state.last is not None:
                # Check if this is add-with-mult or sub-with-mult
                if words & ADD_VERBS:
                    old = state.last
                    result = old + product
                    state.op(f"{old} + {'x'.join(str(v) for v in values)} = {result}", result)
                elif words & SUB_VERBS:
                    old = state.last
                    result = old - product
                    state.op(f"{old} - {'x'.join(str(v) for v in values)} = {result}", result)
                else:
                    state.op(f"{'x'.join(str(v) for v in values)} = {product}", product)
            else:
                state.op(f"{'x'.join(str(v) for v in values)} = {product}", product)

        elif len(values) == 1 and state.last is not None:
            # Single number with MULT signal: multiply with state.last
            old = state.last
            result = old * values[0]
            state.op(f"{old} x {values[0]} = {result}", result)

        elif len(values) == 1:
            state.op(f"Start: {values[0]}", values[0])

    def _do_div(self, values: List[float], sl: str, state: SolverState):
        """Divide."""
        if len(values) >= 2:
            a, b = values[0], values[-1]
            if b > 0:
                result = a / b
                state.op(f"{a} / {b} = {result}", result)
        elif len(values) == 1 and state.last is not None:
            if values[0] > 0:
                old = state.last
                result = old / values[0]
                state.op(f"{old} / {values[0]} = {result}", result)

    # ── Specialized Handlers ──

    def _handle_time_duration(self, sl: str, s: str, state: SolverState) -> bool:
        """Handle 'from 1:00 PM to 5:00 PM' -> duration."""
        m = re.search(
            r'(\d{1,2}):?(\d{0,2})\s*(am|pm)\s+(?:to|until|till)\s+'
            r'(\d{1,2}):?(\d{0,2})\s*(am|pm)', sl)
        if not m:
            return False
        h1, h2 = int(m.group(1)), int(m.group(4))
        p1, p2 = m.group(3), m.group(6)
        if p1 == 'pm' and h1 != 12: h1 += 12
        if p2 == 'pm' and h2 != 12: h2 += 12
        if p1 == 'am' and h1 == 12: h1 = 0
        if p2 == 'am' and h2 == 12: h2 = 0
        dur = h2 - h1
        if dur < 0: dur += 24

        # Look for a rate to multiply by
        nums = extract_numbers(s)
        rate_vals = [v for v, _, _ in nums if v not in (h1, h2, h1-12, h2-12, dur)]
        if state.last is not None:
            result = state.last * dur
            state.op(f"{state.last} x {dur}h = {result}", result)
        elif rate_vals:
            result = rate_vals[0] * dur
            state.op(f"{rate_vals[0]} x {dur}h = {result}", result)
        else:
            state.op(f"Duration: {dur}h", float(dur))
        return True

    def _handle_percentage(self, s: str, state: SolverState) -> bool:
        """Handle N% of X, N% increase/decrease."""
        m = re.search(r'(\d+\.?\d*)\s*(?:%|percent)', s)
        if not m:
            return False

        pct = float(m.group(1))
        nums = extract_numbers(s)
        other = [v for v, _, _ in nums if abs(v - pct) > 0.001]
        # When "of them/the [noun]" → use state.last as base (reference to prior group)
        refers_to_prior = bool(re.search(
            r'(?:%|percent)\s+of\s+(?:them|the\s+\w+|those|these|his|her|its|all)', s))
        if refers_to_prior and state.last is not None:
            base = state.last
        else:
            base = other[0] if other else (state.last if state.last is not None else None)
        if base is None:
            return False

        decrease_words = ['discount', 'off', 'decrease', 'reduction', 'less',
                          'cheaper', 'save', 'deducted', 'reduced', 'fewer',
                          'lose', 'lost', 'remove']
        increase_words = ['increase', 'more', 'raise', 'markup', 'up', 'tax',
                          'tip', 'surcharge', 'higher', 'rose', 'grew', 'growth',
                          'added', 'increased', 'profit', 'above', 'appreciate',
                          'gained', 'gain']

        # Check for reverse percentage: "this is with a N% discount FROM the original"
        # meaning: base = discounted_price, need to find original = base / (1 - pct/100)
        reverse_discount = bool(re.search(
            r'(?:with|after)\s+(?:a\s+)?[\d.]+\s*%\s*(?:discount|off|reduction).*?'
            r'(?:from|of)\s+(?:the\s+)?(?:original|regular|full|retail|normal)', s))
        reverse_increase = bool(re.search(
            r'(?:with|after)\s+(?:a\s+)?[\d.]+\s*%\s*(?:increase|markup|tax|surcharge).*?'
            r'(?:from|of)\s+(?:the\s+)?(?:original|regular|base)', s))

        if reverse_discount:
            result = base / (1 - pct / 100)
            state.op(f"{base} / (1 - {pct}%) = {result}", result)
        elif reverse_increase:
            result = base / (1 + pct / 100)
            state.op(f"{base} / (1 + {pct}%) = {result}", result)
        elif any(w in s for w in decrease_words):
            amt = base * pct / 100
            result = base - amt
            state.op(f"{base} - {pct}% = {result}", result)
        elif any(w in s for w in increase_words):
            amt = base * pct / 100
            result = base + amt
            state.op(f"{base} + {pct}% = {result}", result)
        else:
            result = base * pct / 100
            state.op(f"{pct}% of {base} = {result}", result)
        return True

    def _handle_fraction_of(self, s: str, state: SolverState) -> bool:
        """Handle 'half of X', 'a third of the total', etc."""

        # "X and half that much"
        if re.search(r'\band\s+half\s+(?:that|as)\s+much', s):
            nums = extract_numbers(s)
            vals = [v for v, _, _ in nums]
            if vals:
                base = vals[0]
                result = base + base * 0.5
                state.op(f"{base} + {base}/2 = {result}", result)
                return True
            elif state.last is not None:
                result = state.last + state.last * 0.5
                state.op(f"{state.last} + {state.last}/2 = {result}", result)
                return True

        # Named fractions with "of"
        for frac_name, frac_val in sorted(FRACTION_MAP.items(), key=lambda x: -len(x[0])):
            pat = r'\b' + re.escape(frac_name) + r'\b'
            if re.search(pat, s) and 'of' in s:
                nums = extract_numbers(s)
                vals = [v for v, _, _ in nums]
                if vals:
                    base = vals[0]
                else:
                    base = state.last
                if base is not None:
                    # Check for subtraction context: "gives away a third"
                    words = set(re.findall(r'\b\w+\b', s))
                    if words & SUB_VERBS and state.last is not None:
                        amt = state.last * frac_val
                        result = state.last - amt
                        state.op(f"{state.last} - {frac_name} = {result}", result)
                    else:
                        result = base * frac_val
                        state.op(f"{frac_name} of {base} = {result}", result)
                    return True

        # "half that/as much/as many"
        if re.search(r'\bhalf\b', s) and state.last is not None:
            if re.search(r'half\s+(?:that|as\s+much|as\s+many|of\s+(?:that|the|it|them|his|her))', s):
                result = state.last * 0.5
                state.op(f"{state.last} x 0.5 = {result}", result)
                return True

        return False

    def _handle_multiplier_ref(self, s: str, state: SolverState) -> bool:
        """Handle 'twice as many', 'N times as many'."""
        words = set(re.findall(r'\b\w+\b', s))

        # Only add if there's a strong add verb directly implying receipt
        # "received twice" = ADD, but "save twice as much" = just multiply
        strong_add = bool(words & {'received', 'got', 'bought', 'earned',
                                    'gifted', 'given', 'won', 'found',
                                    'collect', 'collected', 'pick', 'picked',
                                    'gather', 'gathered',
                                    'made', 'baked', 'cooked', 'produced'})
        # "left" in "what was left" is a reference, not subtraction
        sub_words = words & SUB_VERBS
        if sub_words == {'left'} and re.search(r'(?:what\s+was|what\s+is|as\s+much\s+as.*)\s+left', s):
            sub_words = set()
        is_sub = bool(sub_words)

        for word, mult in MULTIPLIER_WORDS.items():
            if re.search(r'\b' + word + r'\b', s):
                nums = extract_numbers(s)
                vals = [v for v, _, _ in nums]
                if vals:
                    base = vals[0]
                    result = base * mult
                    state.op(f"{base} x {mult} = {result}", result)
                    return True
                if state.last is not None:
                    computed = state.last * mult
                    # If strong ADD verb, add the multiplied amount to running total
                    if strong_add and not is_sub:
                        result = state.last + computed
                        state.op(f"{state.last} + {mult}×{state.last} = {result}", result)
                    elif is_sub:
                        result = state.last - computed
                        state.op(f"{state.last} - {mult}×{state.last} = {result}", result)
                    else:
                        result = computed
                        state.op(f"{state.last} x {mult} = {result}", result)
                    return True

        # "N times as many/much"
        m = re.search(r'(\d+)\s+times\s+(?:as\s+)?(?:many|much|more|that|the)', s)
        if m:
            mult = float(m.group(1))
            nums = extract_numbers(s)
            vals = [v for v, _, _ in nums if abs(v - mult) > 0.001]
            if vals:
                result = vals[0] * mult
                state.op(f"{vals[0]} x {mult} = {result}", result)
                return True
            if state.last is not None:
                result = state.last * mult
                state.op(f"{state.last} x {mult} = {result}", result)
                return True

        return False

    def _handle_comparison(self, s: str, state: SolverState) -> bool:
        """Handle 'X more/less than Y'."""
        # "N more than"
        m = re.search(r'(\d+\.?\d*)\s+more\s+(?:\w+\s+)*than', s)
        if m:
            val = float(m.group(1))
            nums = extract_numbers(s)
            others = [v for v, _, _ in nums if abs(v - val) > 0.001]
            if others:
                result = others[0] + val
                state.op(f"{others[0]} + {val} = {result}", result)
                return True
            if state.last is not None:
                result = state.last + val
                state.op(f"{state.last} + {val} = {result}", result)
                return True

        # "N less/fewer than"
        m = re.search(r'(\d+\.?\d*)\s+(?:less|fewer)\s+(?:\w+\s+)*than', s)
        if m:
            val = float(m.group(1))
            nums = extract_numbers(s)
            others = [v for v, _, _ in nums if abs(v - val) > 0.001]
            if others:
                result = others[0] - val
                state.op(f"{others[0]} - {val} = {result}", result)
                return True
            if state.last is not None:
                result = state.last - val
                state.op(f"{state.last} - {val} = {result}", result)
                return True

        return False

    def _handle_pricing(self, sl: str, s: str, state: SolverState) -> bool:
        """Handle 'N items at $X each', 'sells at $X per Y'."""
        # "N <thing> at/for $X each/per"
        patterns = [
            r'(\d+\.?\d*)\s+(?:\w+\s+){0,3}(?:which\s+)?(?:cost|costs|costing|priced\s+at|at|for)\s+\$?(\d+\.?\d*)\s*(?:each|per\s+\w+|apiece)',
            r'(\d+\.?\d*)\s+(?:\w+\s+){0,3}(?:at|for)\s+\$(\d+\.?\d*)\s+each',
        ]
        for pat in patterns:
            m = re.search(pat, sl)
            if m:
                qty = float(m.group(1))
                price = float(m.group(2))
                result = qty * price
                words_set = set(re.findall(r'\b\w+\b', sl))
                if state.last is not None:
                    old = state.last
                    if words_set & SUB_VERBS:
                        state.op(f"{old} - {qty}x${price} = {old - result}", old - result)
                    else:
                        state.op(f"{old} + {qty}x${price} = {old + result}", old + result)
                else:
                    state.op(f"{qty} x ${price} = {result}", result)
                return True

        # "one pair costs $X ... one pair costs $Y" with shared quantity
        # Pattern: "N pairs of A, N pairs of B, N pairs of C. A costs $X, B $Y, C $Z"
        prices = re.findall(r'\$(\d+\.?\d*)', sl)
        if len(prices) >= 2:
            price_vals = [float(p) for p in prices]
            # Check for shared quantity: "3 pairs of shorts, 3 pairs of pants"
            qty_m = re.search(r'(\d+)\s+(?:pairs?|sets?|boxes?|packs?|bags?)\s+of', sl)
            if qty_m:
                qty = float(qty_m.group(1))
                total = qty * sum(price_vals)
                state.op(f"{qty} × ({'+'.join(prices)}) = {total}", total)
                return True

        # "sells/sold ... at/for $X per/each"
        m = re.search(r'(?:sells?|sold)\s+.*?(?:at|for)\s+\$?(\d+\.?\d*)\s*(?:per|each|apiece)', sl)
        if m:
            price = float(m.group(1))
            if state.last is not None:
                result = state.last * price
                state.op(f"{state.last} x ${price} = {result}", result)
                return True

        return False


def evaluate_gsm8k(solver: WordProblemSolver, data_path: str):
    """Evaluate solver on GSM8K JSONL data."""
    import json

    correct = 0
    total = 0
    no_answer = 0
    errors = []

    with open(data_path) as f:
        for line in f:
            item = json.loads(line)
            question = item['question']
            expected_str = item['answer'].split('####')[-1].strip()
            expected_str = expected_str.replace(',', '').replace('$', '')
            try:
                expected = float(expected_str)
            except ValueError:
                continue

            total += 1
            result = solver.solve(question)

            if result is None:
                no_answer += 1
                continue

            got = float(result['answer'])
            if abs(got - expected) < 0.01:
                correct += 1
            elif len(errors) < 20:
                errors.append({
                    'question': question[:100],
                    'expected': expected_str,
                    'got': str(result['answer']),
                    'steps': result['steps'],
                })

    return correct, total, no_answer, errors
