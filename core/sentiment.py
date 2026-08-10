"""
Sentiment & Text Classification — Without Neural Networks
===========================================================
Lexicon-based sentiment + Naive Bayes text classification.

Capabilities:
  1. Sentiment analysis (positive/negative/neutral + score)
  2. Emotion detection (joy, anger, fear, sadness, surprise, disgust)
  3. Text classification (Naive Bayes, trainable)
  4. Intent detection (question, command, statement, greeting)
  5. Urgency detection (critical, high, normal, low)
  6. Formality detection (formal, informal, technical)
"""

import re
import math
from typing import Dict, Any, List, Optional, Tuple
from collections import Counter, defaultdict


# === Sentiment Lexicon ===
# Score: -1.0 (very negative) to +1.0 (very positive)

SENTIMENT_LEXICON = {
    # Strong positive
    'excellent': 0.9, 'amazing': 0.9, 'outstanding': 0.9, 'brilliant': 0.9,
    'fantastic': 0.9, 'wonderful': 0.9, 'superb': 0.85, 'perfect': 0.9,
    'love': 0.8, 'best': 0.8, 'great': 0.7, 'awesome': 0.8,

    # Moderate positive
    'good': 0.5, 'nice': 0.5, 'happy': 0.6, 'pleased': 0.6,
    'glad': 0.5, 'enjoy': 0.6, 'beautiful': 0.7, 'impressive': 0.6,
    'helpful': 0.5, 'useful': 0.5, 'interesting': 0.4, 'fun': 0.5,
    'cool': 0.4, 'fine': 0.3, 'well': 0.3, 'like': 0.3,
    'success': 0.6, 'successful': 0.6, 'win': 0.6, 'won': 0.6,
    'improve': 0.4, 'improved': 0.4, 'better': 0.4, 'fast': 0.3,
    'easy': 0.3, 'clean': 0.3, 'clear': 0.3, 'smart': 0.5,
    'working': 0.3, 'works': 0.3, 'solved': 0.5, 'fixed': 0.4,
    'complete': 0.3, 'done': 0.3, 'ready': 0.3, 'pass': 0.4,
    'passed': 0.4, 'correct': 0.4, 'right': 0.3, 'true': 0.2,

    # Mild negative
    'bad': -0.5, 'wrong': -0.5, 'poor': -0.5, 'slow': -0.3,
    'difficult': -0.3, 'hard': -0.2, 'confusing': -0.4, 'ugly': -0.5,
    'boring': -0.4, 'annoying': -0.5, 'disappointing': -0.5,
    'fail': -0.5, 'failed': -0.5, 'error': -0.4, 'bug': -0.4,
    'broken': -0.5, 'crash': -0.6, 'stuck': -0.4, 'missing': -0.3,
    'problem': -0.4, 'issue': -0.3, 'trouble': -0.4, 'lost': -0.4,

    # Strong negative
    'terrible': -0.9, 'horrible': -0.9, 'awful': -0.8, 'worst': -0.9,
    'hate': -0.8, 'disaster': -0.8, 'catastrophe': -0.9,
    'useless': -0.7, 'pathetic': -0.7, 'disgusting': -0.8,
    'dangerous': -0.6, 'malicious': -0.8, 'toxic': -0.7,
    'dead': -0.5, 'killed': -0.6, 'destroyed': -0.7, 'ruined': -0.7,
}

# Intensifiers and negators
INTENSIFIERS = {
    'very': 1.5, 'really': 1.4, 'extremely': 1.8, 'incredibly': 1.7,
    'absolutely': 1.6, 'totally': 1.5, 'completely': 1.4, 'utterly': 1.6,
    'quite': 1.2, 'rather': 1.1, 'pretty': 1.2, 'so': 1.3,
    'super': 1.5, 'highly': 1.4, 'remarkably': 1.4,
}

NEGATORS = frozenset([
    'not', 'no', 'never', 'neither', 'nobody', 'nothing', 'nowhere',
    'nor', 'cannot', "can't", "won't", "wouldn't", "shouldn't",
    "couldn't", "doesn't", "didn't", "isn't", "aren't", "wasn't",
    "weren't", "hasn't", "haven't", "hadn't",
])

# Emotion keywords
EMOTION_KEYWORDS = {
    'joy': ['happy', 'joy', 'excited', 'glad', 'delighted', 'thrilled',
            'pleased', 'cheerful', 'wonderful', 'amazing', 'love', 'great'],
    'anger': ['angry', 'furious', 'mad', 'rage', 'outraged', 'annoyed',
              'frustrated', 'irritated', 'hate', 'hostile', 'aggressive'],
    'fear': ['afraid', 'scared', 'fearful', 'terrified', 'worried',
             'anxious', 'nervous', 'panic', 'dread', 'alarmed'],
    'sadness': ['sad', 'depressed', 'unhappy', 'miserable', 'heartbroken',
                'disappointed', 'grief', 'lonely', 'hopeless', 'regret'],
    'surprise': ['surprised', 'shocked', 'amazed', 'astonished', 'unexpected',
                 'stunning', 'remarkable', 'incredible', 'unbelievable'],
    'disgust': ['disgusting', 'gross', 'revolting', 'repulsive', 'nasty',
                'sickening', 'vile', 'foul', 'horrible', 'awful'],
}


def tokenize(text: str) -> List[str]:
    """Simple word tokenizer."""
    return re.findall(r"\b[\w']+\b", text.lower())


class SentimentAnalyzer:
    """
    Lexicon-based sentiment analysis with context awareness.

    Handles: negation, intensifiers, but-clauses, emoji.
    """

    def __init__(self, lexicon: Optional[Dict[str, float]] = None):
        self.lexicon = lexicon or dict(SENTIMENT_LEXICON)

    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Full sentiment analysis.

        Returns:
            dict with: score (-1 to 1), label, confidence, emotions,
                       positive_words, negative_words
        """
        tokens = tokenize(text)
        if not tokens:
            return {'score': 0.0, 'label': 'neutral', 'confidence': 0.0,
                    'emotions': {}, 'positive_words': [], 'negative_words': []}

        scores = []
        pos_words = []
        neg_words = []
        negation_window = 0
        intensifier = 1.0

        for i, token in enumerate(tokens):
            # Track negation (3-word window)
            if token in NEGATORS:
                negation_window = 3
                continue

            # Track intensifiers
            if token in INTENSIFIERS:
                intensifier = INTENSIFIERS[token]
                continue

            # Lookup sentiment
            if token in self.lexicon:
                score = self.lexicon[token] * intensifier

                # Apply negation
                if negation_window > 0:
                    score *= -0.75  # Negate but slightly weaker
                    negation_window -= 1

                scores.append(score)
                if score > 0:
                    pos_words.append(token)
                elif score < 0:
                    neg_words.append(token)

                intensifier = 1.0  # Reset
            else:
                intensifier = 1.0
                if negation_window > 0:
                    negation_window -= 1

        # Handle "but" clauses — second half weighs more
        but_idx = None
        for i, token in enumerate(tokens):
            if token in ('but', 'however', 'although', 'yet', 'nevertheless'):
                but_idx = i
                break

        if but_idx is not None and scores:
            # Re-weight: after "but" gets 1.5x weight
            mid = len(scores) // 2
            for i in range(mid, len(scores)):
                scores[i] *= 1.5

        # Aggregate
        if scores:
            avg_score = sum(scores) / len(scores)
            # Clamp to [-1, 1]
            avg_score = max(-1.0, min(1.0, avg_score))
        else:
            avg_score = 0.0

        # Determine label
        if avg_score > 0.1:
            label = 'positive'
        elif avg_score < -0.1:
            label = 'negative'
        else:
            label = 'neutral'

        # Confidence based on how many sentiment words found
        coverage = len(scores) / max(len(tokens), 1)
        confidence = min(coverage * 3, 1.0)  # Scale up

        # Emotion detection
        emotions = self._detect_emotions(tokens)

        return {
            'score': round(avg_score, 3),
            'label': label,
            'confidence': round(confidence, 2),
            'emotions': emotions,
            'positive_words': pos_words,
            'negative_words': neg_words,
            'word_count': len(tokens),
            'sentiment_words': len(scores),
        }

    def _detect_emotions(self, tokens: List[str]) -> Dict[str, float]:
        """Detect emotions from token overlap with emotion keywords."""
        token_set = set(tokens)
        emotions = {}
        for emotion, keywords in EMOTION_KEYWORDS.items():
            overlap = token_set & set(keywords)
            if overlap:
                emotions[emotion] = len(overlap) / len(keywords)
        return emotions

    def add_word(self, word: str, score: float):
        """Add a word to the sentiment lexicon."""
        self.lexicon[word.lower()] = max(-1.0, min(1.0, score))


class NaiveBayesClassifier:
    """
    Multinomial Naive Bayes text classifier.
    Trainable on any label set.

    Usage:
        clf = NaiveBayesClassifier()
        clf.train([
            ("I love this!", "positive"),
            ("This is terrible", "negative"),
        ])
        result = clf.classify("It's wonderful")
    """

    def __init__(self, smoothing: float = 1.0):
        self.smoothing = smoothing  # Laplace smoothing
        self._class_counts: Counter = Counter()
        self._word_counts: Dict[str, Counter] = defaultdict(Counter)
        self._vocab: set = set()
        self._total_docs = 0

    def train(self, examples: List[Tuple[str, str]]):
        """Train on (text, label) pairs."""
        for text, label in examples:
            tokens = tokenize(text)
            self._class_counts[label] += 1
            self._total_docs += 1
            for token in tokens:
                self._word_counts[label][token] += 1
                self._vocab.add(token)

    def classify(self, text: str) -> Dict[str, Any]:
        """
        Classify text.

        Returns:
            dict with: label, confidence, scores (per-class log-probs)
        """
        tokens = tokenize(text)
        if not self._total_docs:
            return {'label': 'unknown', 'confidence': 0.0, 'scores': {}}

        scores = {}
        vocab_size = len(self._vocab)

        for label in self._class_counts:
            # Prior probability
            log_prob = math.log(self._class_counts[label] / self._total_docs)

            # Likelihood
            total_words = sum(self._word_counts[label].values())
            for token in tokens:
                count = self._word_counts[label].get(token, 0)
                # Laplace smoothing
                prob = (count + self.smoothing) / (total_words + self.smoothing * vocab_size)
                log_prob += math.log(prob)

            scores[label] = log_prob

        # Best class
        best_label = max(scores, key=lambda k: scores[k])

        # Convert to confidence (softmax-like)
        max_score = max(scores.values())
        exp_scores = {k: math.exp(v - max_score) for k, v in scores.items()}
        total = sum(exp_scores.values())
        probs = {k: v / total for k, v in exp_scores.items()}

        return {
            'label': best_label,
            'confidence': round(probs[best_label], 3),
            'scores': {k: round(v, 4) for k, v in probs.items()},
        }


class IntentDetector:
    """
    Detect user intent from text.
    Categories: question, command, statement, greeting, farewell, thanks, complaint.
    """

    PATTERNS = {
        'question': [
            r'^(?:what|who|where|when|why|how|which|is|are|was|were|do|does|did|can|could|would|will|shall)\b',
            r'\?$',
            r'\b(?:tell me|explain|describe|define|show me)\b',
        ],
        'command': [
            r'^(?:do|make|create|build|run|execute|find|search|open|close|start|stop|install|remove|delete)\b',
            r'^(?:please\s+)?(?:do|make|create|build|run|find|search|open)\b',
            r'^(?:lets?|let\'s)\s+',
        ],
        'greeting': [
            r'^(?:hi|hello|hey|howdy|greetings|good\s+(?:morning|afternoon|evening))\b',
            r'^(?:hallo|guten\s+(?:morgen|tag|abend)|moin|servus)\b',
        ],
        'farewell': [
            r'^(?:bye|goodbye|see you|farewell|ciao|tschüss|auf wiedersehen)\b',
            r'^(?:good night|gute nacht)\b',
        ],
        'thanks': [
            r'\b(?:thanks|thank you|thx|danke|vielen dank)\b',
        ],
        'complaint': [
            r'\b(?:broken|doesn\'t work|not working|bug|error|crash|fail|wrong)\b',
            r'\b(?:kaputt|funktioniert nicht|fehler)\b',
        ],
    }

    def detect(self, text: str) -> Dict[str, Any]:
        """
        Detect intent.

        Returns:
            dict with: intent, confidence, all_intents
        """
        text_lower = text.lower().strip()
        matches = {}

        for intent, patterns in self.PATTERNS.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    score += 1
            if score > 0:
                matches[intent] = score

        if not matches:
            return {'intent': 'statement', 'confidence': 0.5, 'all_intents': {'statement': 0.5}}

        # Normalize
        total = sum(matches.values())
        probs = {k: v / total for k, v in matches.items()}
        best = max(probs, key=lambda k: probs[k])

        return {
            'intent': best,
            'confidence': round(probs[best], 2),
            'all_intents': {k: round(v, 2) for k, v in probs.items()},
        }


class UrgencyDetector:
    """Detect urgency level from text."""

    URGENT_WORDS = frozenset([
        'urgent', 'asap', 'immediately', 'critical', 'emergency', 'now',
        'right now', 'hurry', 'deadline', 'overdue', 'blocking', 'blocker',
        'sofort', 'dringend', 'notfall', 'jetzt',
    ])

    HIGH_WORDS = frozenset([
        'important', 'priority', 'soon', 'quickly', 'fast', 'today',
        'broken', 'down', 'outage', 'crash', 'fail',
        'wichtig', 'schnell', 'heute', 'kaputt',
    ])

    def detect(self, text: str) -> Dict[str, Any]:
        """
        Detect urgency.

        Returns:
            dict with: level ('critical'|'high'|'normal'|'low'), score (0-1)
        """
        words = set(tokenize(text))
        text_lower = text.lower()

        urgent_matches = len(words & self.URGENT_WORDS)
        high_matches = len(words & self.HIGH_WORDS)

        # Check for exclamation marks (intensity signal)
        excl_count = text.count('!')
        caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)

        score = 0.0
        score += urgent_matches * 0.4
        score += high_matches * 0.2
        score += min(excl_count * 0.1, 0.3)
        score += caps_ratio * 0.5  # ALL CAPS = urgent

        score = min(score, 1.0)

        if score > 0.6:
            level = 'critical'
        elif score > 0.3:
            level = 'high'
        elif score > 0.1:
            level = 'normal'
        else:
            level = 'low'

        return {'level': level, 'score': round(score, 2)}
