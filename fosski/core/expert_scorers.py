"""
Expert Scorers — Mixture of Experts for Multiple-Choice Benchmarks
===================================================================
15 specialized scoring functions, each a lightweight expert.
No neural networks — pure rules + KB + statistics.
Combined via additive fusion (same as Transformer residual stream).

Each scorer returns a score per choice. Higher = better.
Scorers are independent: can be mixed and matched per benchmark.
"""

import re
import math
from typing import List, Dict, Optional, Tuple
from collections import Counter


# ── Shared utilities ──

_STOP = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
    'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
    'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over',
    'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
    'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more',
    'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
    'same', 'so', 'than', 'too', 'very', 'and', 'but', 'or', 'if',
    'he', 'she', 'it', 'they', 'we', 'you', 'me', 'him', 'her', 'us',
    'them', 'his', 'its', 'my', 'your', 'our', 'their', 'this', 'that',
    'these', 'those', 'what', 'which', 'who', 'whom',
    'just', 'also', 'even', 'back', 'still', 'around', 'about',
}


def _content_words(text: str) -> set:
    return set(w for w in re.findall(r'\b[a-z]+\b', text.lower())
               if w not in _STOP and len(w) > 2)


def _bigrams(text: str) -> List[str]:
    words = [w for w in re.findall(r'\b[a-z]+\b', text.lower()) if len(w) > 1]
    return [words[i] + ' ' + words[i+1] for i in range(len(words) - 1)]


def _trigrams(text: str) -> List[str]:
    words = [w for w in re.findall(r'\b[a-z]+\b', text.lower()) if len(w) > 1]
    return [words[i] + ' ' + words[i+1] + ' ' + words[i+2]
            for i in range(len(words) - 2)]


# ═══════════════════════════════════════════════════════════════════
# Expert 1: Subject-Verb Agreement
# ═══════════════════════════════════════════════════════════════════

_SINGULAR_VERBS = re.compile(r'\b(he|she|it|this|that)\s+(is|was|has|does|goes|comes|makes|takes|runs|seems|appears|looks)\b', re.I)
_PLURAL_VERBS = re.compile(r'\b(they|we|these|those)\s+(are|were|have|do|go|come|make|take|run|seem|appear|look)\b', re.I)
_DISAGREEMENT = re.compile(r'\b(he|she|it)\s+(are|were|have|do)\b|\b(they|we)\s+(is|was|has|does)\b', re.I)


def score_subject_verb_agreement(choices: List[str]) -> List[float]:
    """Penalize choices with subject-verb disagreement."""
    scores = []
    for c in choices:
        s = 0.0
        # Reward agreement
        s += len(_SINGULAR_VERBS.findall(c)) * 0.5
        s += len(_PLURAL_VERBS.findall(c)) * 0.5
        # Penalize disagreement
        s -= len(_DISAGREEMENT.findall(c)) * 2.0
        scores.append(s)
    return scores


# ═══════════════════════════════════════════════════════════════════
# Expert 2: Temporal Scorer (Time sequence coherence)
# ═══════════════════════════════════════════════════════════════════

_TEMPORAL_ORDER = ['first', 'then', 'next', 'after', 'finally', 'last',
                   'before', 'during', 'while', 'until', 'later',
                   'previously', 'earlier', 'already', 'soon']

_TEMPORAL_MARKERS = re.compile(
    r'\b(first|then|next|after|finally|last|before|during|while|until|'
    r'later|previously|earlier|already|soon|yesterday|today|tomorrow|'
    r'morning|evening|night|day|week|month|year)\b', re.I)


def score_temporal(context: str, choices: List[str]) -> List[float]:
    """Score temporal coherence between context and choices."""
    ctx_markers = set(m.lower() for m in _TEMPORAL_MARKERS.findall(context))
    scores = []
    for c in choices:
        s = 0.0
        c_markers = set(m.lower() for m in _TEMPORAL_MARKERS.findall(c))
        # Continuity: if context has temporal markers, choices should too
        if ctx_markers:
            overlap = len(ctx_markers & c_markers)
            s += overlap * 0.5
            # Temporal progression: "first...then" is good
            if 'first' in ctx_markers and 'then' in c_markers:
                s += 1.0
            if 'before' in ctx_markers and 'after' in c_markers:
                s += 0.5
        scores.append(s)
    return scores


# ═══════════════════════════════════════════════════════════════════
# Expert 3: Causal Scorer (Cause-Effect relationships)
# ═══════════════════════════════════════════════════════════════════

_CAUSAL_MARKERS = re.compile(
    r'\b(because|therefore|thus|hence|consequently|so|since|'
    r'as a result|leads to|causes|due to|owing to|results in|'
    r'if.*then|when.*then)\b', re.I)

# Common cause-effect pairs (ConceptNet-derived)
CAUSAL_PAIRS = {
    ('heat', 'expand'): 1.0, ('heat', 'melt'): 1.0, ('heat', 'evaporate'): 1.0,
    ('cold', 'freeze'): 1.0, ('cold', 'contract'): 1.0, ('cold', 'ice'): 0.8,
    ('rain', 'wet'): 0.8, ('rain', 'flood'): 0.6, ('rain', 'grow'): 0.5,
    ('fire', 'burn'): 1.0, ('fire', 'smoke'): 0.8, ('fire', 'heat'): 0.8,
    ('sun', 'light'): 0.8, ('sun', 'heat'): 0.8, ('sun', 'grow'): 0.6,
    ('wind', 'blow'): 0.8, ('wind', 'dry'): 0.5, ('wind', 'cool'): 0.5,
    ('water', 'dissolve'): 0.8, ('water', 'rust'): 0.7, ('water', 'grow'): 0.6,
    ('gravity', 'fall'): 1.0, ('gravity', 'weight'): 0.8,
    ('friction', 'heat'): 0.8, ('friction', 'slow'): 0.7,
    ('pressure', 'compress'): 0.8, ('pressure', 'crush'): 0.7,
    ('electricity', 'shock'): 0.8, ('electricity', 'light'): 0.7,
}


def score_causal(context: str, choices: List[str]) -> List[float]:
    """Score causal coherence between context and choices."""
    ctx_words = _content_words(context)
    scores = []
    for c in choices:
        s = 0.0
        c_words = _content_words(c)
        # Check causal pairs
        for (cause, effect), weight in CAUSAL_PAIRS.items():
            if cause in ctx_words and effect in c_words:
                s += weight
            elif effect in ctx_words and cause in c_words:
                s += weight * 0.5  # Reverse direction weaker
        # Causal markers → reward coherent structure
        s += len(_CAUSAL_MARKERS.findall(c)) * 0.3
        scores.append(s)
    return scores


# ═══════════════════════════════════════════════════════════════════
# Expert 4: Negation Scorer
# ═══════════════════════════════════════════════════════════════════

_NEGATION = re.compile(
    r"\b(not|never|don'?t|doesn'?t|cannot|can'?t|won'?t|wouldn'?t|"
    r"shouldn'?t|isn'?t|aren'?t|wasn'?t|weren'?t|hardly|barely|"
    r"neither|nor|without)\b", re.IGNORECASE
)


def score_negation(context: str, choices: List[str]) -> List[float]:
    """Penalize choices with unexpected negations.

    In most QA benchmarks, the correct answer aligns with the
    positive framing of the question. Negations in one choice
    but not others signal implausibility.
    """
    ctx_neg_count = len(_NEGATION.findall(context))
    neg_counts = [len(_NEGATION.findall(c)) for c in choices]
    mean_neg = sum(neg_counts) / max(len(neg_counts), 1)

    scores = []
    for i, c in enumerate(choices):
        s = 0.0
        if neg_counts[i] > mean_neg + 0.5:
            # This choice has more negation than average → penalty
            s -= (neg_counts[i] - mean_neg) * 1.5
        # Double negation → even worse
        if neg_counts[i] >= 2 and ctx_neg_count == 0:
            s -= 1.0
        scores.append(s)
    return scores


# ═══════════════════════════════════════════════════════════════════
# Expert 5: Entity Continuity Scorer
# ═══════════════════════════════════════════════════════════════════


def _extract_entities(text: str) -> set:
    """Extract likely entities (capitalized words, nouns)."""
    # Named entities
    named = set(re.findall(r'\b[A-Z][a-z]+\b', text))
    # Common nouns that act as entities
    nouns = set(re.findall(r'\b(?:the|a|an)\s+(\w+)\b', text.lower()))
    return named | nouns


def score_entity_continuity(context: str, choices: List[str]) -> List[float]:
    """Reward choices that maintain entity continuity with context."""
    ctx_entities = _extract_entities(context)
    ctx_words = _content_words(context)
    scores = []
    for c in choices:
        s = 0.0
        c_entities = _extract_entities(c)
        c_words = _content_words(c)
        # Named entity continuity
        shared = ctx_entities & c_entities
        s += len(shared) * 2.0
        # Content word continuity
        s += len(ctx_words & c_words) * 0.3
        # New entity penalty (introducing unrelated entities)
        new_entities = c_entities - ctx_entities
        if len(new_entities) > 2:
            s -= (len(new_entities) - 2) * 0.5
        scores.append(s)
    return scores


# ═══════════════════════════════════════════════════════════════════
# Expert 6: Specificity Scorer (concrete > vague)
# ═══════════════════════════════════════════════════════════════════

_VAGUE_WORDS = {
    'something', 'things', 'stuff', 'whatever', 'somehow', 'somewhere',
    'anything', 'everything', 'nothing', 'anyone', 'someone', 'everyone',
    'nobody', 'various', 'many', 'several', 'certain', 'general',
    'particular', 'specific', 'different', 'other', 'another', 'etc',
}

_CONCRETE_MARKERS = re.compile(
    r'\b(\d+|red|blue|green|yellow|hot|cold|large|small|fast|slow|'
    r'north|south|east|west|iron|steel|wood|glass|plastic|metal|'
    r'water|air|rock|sand|soil|clay|copper|gold|silver|hydrogen|'
    r'oxygen|carbon|nitrogen|calcium)\b', re.I)


def score_specificity(choices: List[str]) -> List[float]:
    """Reward specific/concrete choices over vague ones."""
    scores = []
    for c in choices:
        s = 0.0
        c_lower = c.lower()
        words = set(c_lower.split())
        # Penalize vague words
        s -= len(words & _VAGUE_WORDS) * 0.5
        # Reward concrete markers
        s += len(_CONCRETE_MARKERS.findall(c)) * 0.3
        # Reward numbers (specific quantities)
        s += len(re.findall(r'\b\d+\b', c)) * 0.5
        scores.append(s)
    return scores


# ═══════════════════════════════════════════════════════════════════
# Expert 7: Lexical Overlap Scorer (IDF-weighted)
# ═══════════════════════════════════════════════════════════════════


def score_lexical_overlap(context: str, choices: List[str]) -> List[float]:
    """Score by IDF-weighted word overlap between context and choices."""
    ctx_words = _content_words(context)

    # Build IDF from choices
    all_choice_words = Counter()
    for c in choices:
        for w in _content_words(c):
            all_choice_words[w] += 1

    scores = []
    for c in choices:
        s = 0.0
        c_words = _content_words(c)
        for w in c_words & ctx_words:
            # IDF: rare words in choices matter more
            freq = all_choice_words.get(w, 1)
            s += 1.0 / freq
        scores.append(s)
    return scores


# ═══════════════════════════════════════════════════════════════════
# Expert 8: Bigram/Trigram Scorer
# ═══════════════════════════════════════════════════════════════════


def score_ngram_overlap(context: str, choices: List[str]) -> List[float]:
    """Score by n-gram overlap between context and choices."""
    ctx_bi = set(_bigrams(context))
    ctx_tri = set(_trigrams(context))
    scores = []
    for c in choices:
        s = 0.0
        c_bi = set(_bigrams(c))
        c_tri = set(_trigrams(c))
        s += len(ctx_bi & c_bi) * 1.0
        s += len(ctx_tri & c_tri) * 2.0
        scores.append(s)
    return scores


# ═══════════════════════════════════════════════════════════════════
# Expert 9: Sentiment Consistency Scorer
# ═══════════════════════════════════════════════════════════════════

_POSITIVE = {
    'good', 'great', 'better', 'best', 'excellent', 'wonderful',
    'beautiful', 'happy', 'glad', 'pleased', 'successful', 'helpful',
    'useful', 'effective', 'efficient', 'safe', 'healthy', 'clean',
    'easy', 'fast', 'strong', 'right', 'correct', 'proper', 'nice',
    'benefit', 'improve', 'increase', 'protect', 'prevent', 'cure',
}
_NEGATIVE = {
    'bad', 'worse', 'worst', 'terrible', 'horrible', 'ugly',
    'sad', 'angry', 'upset', 'failed', 'harmful', 'useless',
    'ineffective', 'dangerous', 'unhealthy', 'dirty', 'hard',
    'slow', 'weak', 'wrong', 'incorrect', 'improper', 'damage',
    'destroy', 'harm', 'hurt', 'kill', 'break', 'ruin', 'waste',
    'pollute', 'contaminate', 'infect', 'poison',
}


def score_sentiment_consistency(context: str, choices: List[str]) -> List[float]:
    """Reward choices whose sentiment aligns with context."""
    ctx_words = set(context.lower().split())
    ctx_pos = len(ctx_words & _POSITIVE)
    ctx_neg = len(ctx_words & _NEGATIVE)
    ctx_sentiment = ctx_pos - ctx_neg  # positive = good, negative = bad

    scores = []
    for c in choices:
        c_words = set(c.lower().split())
        c_pos = len(c_words & _POSITIVE)
        c_neg = len(c_words & _NEGATIVE)
        c_sentiment = c_pos - c_neg
        # Reward same direction
        if ctx_sentiment > 0 and c_sentiment > 0:
            scores.append(0.5)
        elif ctx_sentiment < 0 and c_sentiment < 0:
            scores.append(0.5)
        elif ctx_sentiment == 0:
            scores.append(0.0)
        else:
            scores.append(-0.3)
    return scores


# ═══════════════════════════════════════════════════════════════════
# Expert 10: Topic Coherence Scorer
# ═══════════════════════════════════════════════════════════════════

# Domain topic clusters — words that belong together
TOPIC_CLUSTERS = {
    'cooking': {'cook', 'bake', 'fry', 'boil', 'grill', 'roast', 'heat',
                'oven', 'stove', 'pan', 'pot', 'recipe', 'ingredient',
                'kitchen', 'food', 'meal', 'dish', 'serve', 'mix', 'stir'},
    'sports': {'game', 'play', 'team', 'score', 'win', 'lose', 'ball',
               'field', 'court', 'goal', 'match', 'competition', 'player',
               'coach', 'practice', 'train', 'race', 'run', 'swim'},
    'science': {'atom', 'molecule', 'cell', 'energy', 'force', 'mass',
                'light', 'heat', 'sound', 'wave', 'electron', 'proton',
                'nucleus', 'element', 'compound', 'reaction', 'gas',
                'liquid', 'solid', 'temperature', 'pressure', 'gravity'},
    'nature': {'plant', 'animal', 'tree', 'flower', 'leaf', 'root',
               'seed', 'soil', 'water', 'rain', 'sun', 'wind', 'forest',
               'ocean', 'river', 'mountain', 'desert', 'habitat'},
    'building': {'build', 'construct', 'wood', 'brick', 'concrete', 'steel',
                 'nail', 'screw', 'hammer', 'wall', 'roof', 'floor',
                 'foundation', 'structure', 'beam', 'pillar'},
    'cleaning': {'clean', 'wash', 'scrub', 'wipe', 'mop', 'sweep',
                 'rinse', 'soap', 'detergent', 'bleach', 'sponge',
                 'brush', 'vacuum', 'dust', 'polish', 'shine'},
    'health': {'doctor', 'medicine', 'hospital', 'disease', 'infection',
               'symptom', 'treatment', 'vaccine', 'virus', 'bacteria',
               'blood', 'heart', 'lung', 'brain', 'bone', 'muscle',
               'health', 'diet', 'exercise', 'immune'},
}


def score_topic_coherence(context: str, choices: List[str]) -> List[float]:
    """Reward choices that stay in the same topic domain as context."""
    ctx_words = _content_words(context)
    # Detect context topic(s)
    ctx_topics = {}
    for topic, cluster in TOPIC_CLUSTERS.items():
        overlap = len(ctx_words & cluster)
        if overlap > 0:
            ctx_topics[topic] = overlap

    if not ctx_topics:
        return [0.0] * len(choices)

    scores = []
    for c in choices:
        s = 0.0
        c_words = _content_words(c)
        for topic, ctx_strength in ctx_topics.items():
            cluster = TOPIC_CLUSTERS[topic]
            c_overlap = len(c_words & cluster)
            s += c_overlap * ctx_strength * 0.3
        scores.append(s)
    return scores


# ═══════════════════════════════════════════════════════════════════
# Expert 11: Anti-Adversarial Scorer
# ═══════════════════════════════════════════════════════════════════


def score_anti_adversarial(choices: List[str]) -> List[float]:
    """Detect and penalize adversarial/distractor patterns.

    Common adversarial tricks: excessive repetition, nonsense words,
    unnatural sentence structure, random topic shifts.
    """
    scores = []
    for c in choices:
        s = 0.0
        words = re.findall(r'\b[a-z]+\b', c.lower())
        if not words:
            scores.append(-1.0)
            continue
        # Repetition penalty
        unique_ratio = len(set(words)) / len(words) if words else 1.0
        if unique_ratio < 0.4:
            s -= 2.0  # Very repetitive
        elif unique_ratio < 0.6:
            s -= 0.5
        # Excessive sentence count
        periods = c.count('.')
        if periods > 5:
            s -= (periods - 5) * 0.2
        # Very short or very long relative to others
        scores.append(s)

    # Length outlier penalty
    if len(scores) >= 2:
        lengths = [len(c.split()) for c in choices]
        mean_len = sum(lengths) / len(lengths)
        for i, l in enumerate(lengths):
            if mean_len > 0:
                ratio = l / mean_len
                if ratio > 3.0 or ratio < 0.2:
                    scores[i] -= 1.0

    return scores


# ═══════════════════════════════════════════════════════════════════
# Expert 12: Cross-Choice Discrimination
# ═══════════════════════════════════════════════════════════════════


def score_cross_discrimination(context: str, choices: List[str]) -> List[float]:
    """Reward choices with unique context-matching words.

    If a word appears in only one choice AND matches the context,
    it's a strong discriminative signal.
    """
    ctx_words = _content_words(context)
    choice_words = [_content_words(c) for c in choices]
    n = len(choices)

    scores = []
    for i in range(n):
        s = 0.0
        for w in choice_words[i]:
            # Count how many OTHER choices contain this word
            in_others = sum(1 for j in range(n) if j != i and w in choice_words[j])
            if in_others == 0 and w in ctx_words:
                s += 3.0  # Unique to this choice AND in context
            elif in_others == 0:
                s += 0.5  # Unique but not in context
            elif w in ctx_words:
                s += 1.0 / (1 + in_others)  # Shared but matches context
        scores.append(s)
    return scores


# ═══════════════════════════════════════════════════════════════════
# Master Scorer: Combines all experts via additive fusion
# ═══════════════════════════════════════════════════════════════════

# Expert weights — tuned per benchmark type
EXPERT_WEIGHTS_DEFAULT = {
    'subject_verb': 0.5,
    'temporal': 0.3,
    'causal': 0.8,
    'negation': 1.0,
    'entity_continuity': 0.5,
    'specificity': 0.3,
    'lexical_overlap': 1.0,
    'ngram_overlap': 0.8,
    'sentiment': 0.3,
    'topic_coherence': 0.5,
    'anti_adversarial': 0.5,
    'cross_discrimination': 0.8,
}


def score_all_experts(context: str, choices: List[str],
                      weights: Optional[Dict[str, float]] = None) -> List[float]:
    """Run all experts and return combined scores per choice.

    Args:
        context: Question/context text
        choices: List of answer choices
        weights: Optional custom weights per expert

    Returns:
        List of scores, one per choice. Higher = more plausible.
    """
    w = weights or EXPERT_WEIGHTS_DEFAULT
    n = len(choices)
    combined = [0.0] * n

    # Run each expert and add weighted scores
    experts = [
        ('subject_verb', score_subject_verb_agreement(choices)),
        ('temporal', score_temporal(context, choices)),
        ('causal', score_causal(context, choices)),
        ('negation', score_negation(context, choices)),
        ('entity_continuity', score_entity_continuity(context, choices)),
        ('specificity', score_specificity(choices)),
        ('lexical_overlap', score_lexical_overlap(context, choices)),
        ('ngram_overlap', score_ngram_overlap(context, choices)),
        ('sentiment', score_sentiment_consistency(context, choices)),
        ('topic_coherence', score_topic_coherence(context, choices)),
        ('anti_adversarial', score_anti_adversarial(choices)),
        ('cross_discrimination', score_cross_discrimination(context, choices)),
    ]

    for name, scores in experts:
        weight = w.get(name, 0.5)
        for i in range(n):
            combined[i] += weight * scores[i]

    return combined
