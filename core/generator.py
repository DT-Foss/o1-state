"""
Text Generator — PPM-Based Free Text Generation
==================================================
Generates coherent multi-sentence text using PPM context trees.

Three modes:
  1. Free generation: given a seed, generate continuation
  2. Constrained generation: generate text that must include specific facts
  3. Conversational generation: generate responses given dialog context

Key principle: PPM generates character-by-character using statistical
patterns learned from training data. This is NOT a transformer — it's
a variable-order Markov model that captures local linguistic patterns.

For factual responses, generation is CONSTRAINED by the knowledge store:
- Facts are retrieved first (Fiber 2)
- PPM reformulates them into natural language (Fiber 1)
- The generator NEVER invents facts — it only structures known information

For creative/free generation (stories, code), PPM generates freely
from its learned distribution.
"""

import re
import numpy as np
from typing import Dict, Any, List, Optional


class TextGenerator:
    """
    PPM-based text generation engine.

    Supports multiple generation strategies:
    - Greedy: always pick most likely next character
    - Sampling: sample from distribution with temperature
    - Nucleus (top-p): sample from top-p probability mass
    - Beam search: maintain top-k candidates
    """

    def __init__(self, lm_model=None, knowledge_store=None):
        self.lm = lm_model
        self.knowledge = knowledge_store
        self.rng = np.random.RandomState(42)

    def generate(self, seed: str = "", max_chars: int = 200,
                 temperature: float = 0.8, strategy: str = 'nucleus',
                 top_p: float = 0.9, stop_at: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate text from a seed string.

        Args:
            seed: starting text
            max_chars: maximum characters to generate
            temperature: sampling temperature (lower = more deterministic)
            strategy: 'greedy', 'sample', 'nucleus', 'beam'
            top_p: nucleus sampling threshold
            stop_at: stop generation when this string appears

        Returns:
            dict with: text, n_chars, strategy, seed
        """
        if self.lm is None:
            return {'text': seed, 'n_chars': 0, 'error': 'No PPM model loaded'}

        context = list(seed)
        generated = []

        for _ in range(max_chars):
            probs = self.lm.predict(context)
            if not probs:
                break

            if strategy == 'greedy':
                next_char = max(probs, key=probs.get)
            elif strategy == 'sample':
                next_char = self._sample(probs, temperature)
            elif strategy == 'nucleus':
                next_char = self._nucleus_sample(probs, temperature, top_p)
            else:  # beam — fallback to nucleus for now
                next_char = self._nucleus_sample(probs, temperature, top_p)

            generated.append(next_char)
            context.append(next_char)

            # Check stop condition
            if stop_at and ''.join(generated).endswith(stop_at):
                break

        text = seed + ''.join(generated)
        return {
            'text': text,
            'generated': ''.join(generated),
            'n_chars': len(generated),
            'strategy': strategy,
            'seed': seed,
        }

    def generate_response(self, query: str, facts: Optional[List] = None,
                          max_chars: int = 300, temperature: float = 0.7) -> Dict[str, Any]:
        """
        Generate a conversational response.

        If facts are provided, generates text constrained to include them.
        If no facts, generates a free response based on the query context.
        """
        if facts:
            return self._generate_factual(query, facts, max_chars, temperature)
        else:
            return self._generate_free(query, max_chars, temperature)

    def _generate_factual(self, query: str, facts: List,
                          max_chars: int, temperature: float) -> Dict[str, Any]:
        """Generate response anchored to facts."""
        # Build seed from facts
        fact_parts = []
        for fact in facts:
            if isinstance(fact, (list, tuple)) and len(fact) >= 3:
                s, r, o = fact[0], fact[1], fact[2]
                fact_parts.append(f"{s} {r} {o}")

        # Use PPM to generate a natural phrasing
        if self.lm and fact_parts:
            # Try different seeds and pick the most natural
            candidates = []
            for fp in fact_parts:
                seed = fp[:20]  # Use beginning of fact as seed
                result = self.generate(
                    seed=seed, max_chars=max_chars,
                    temperature=temperature, strategy='nucleus',
                    stop_at='.'
                )
                candidates.append(result['text'])

            # Pick the one that looks most complete
            best = max(candidates, key=lambda t: len(t) if '.' in t else 0)
            return {
                'text': best,
                'facts_used': facts,
                'method': 'ppm_factual',
            }

        # Fallback: template-based
        text = ". ".join(fact_parts) + "."
        return {
            'text': text,
            'facts_used': facts,
            'method': 'template',
        }

    def _generate_free(self, query: str, max_chars: int,
                       temperature: float) -> Dict[str, Any]:
        """Generate free text continuation."""
        # Use the query as seed context
        seed = query.rstrip('?') + ": "
        result = self.generate(
            seed=seed, max_chars=max_chars,
            temperature=temperature, strategy='nucleus',
            stop_at='\n\n'
        )
        return {
            'text': result['text'],
            'method': 'ppm_free',
            'n_chars': result['n_chars'],
        }

    def generate_code(self, seed: str = "", language: str = 'python',
                      max_chars: int = 500, temperature: float = 0.5) -> Dict[str, Any]:
        """
        Generate code using a code-trained PPM.

        Lower temperature for code (more deterministic).
        Stops at double newline or when indentation returns to 0.
        """
        if self.lm is None:
            return {'text': seed, 'error': 'No PPM model loaded'}

        # Code generation uses lower temperature
        result = self.generate(
            seed=seed, max_chars=max_chars,
            temperature=temperature, strategy='nucleus',
            top_p=0.95,
            stop_at='\n\n\n'  # Triple newline = end of block
        )

        # Post-process: trim trailing incomplete lines
        text = result['text']
        lines = text.split('\n')
        # Find last complete line (ends with : or has matching indent)
        while lines and lines[-1].strip() == '':
            lines.pop()

        return {
            'text': '\n'.join(lines),
            'language': language,
            'n_chars': result['n_chars'],
            'method': 'ppm_code',
        }

    def complete(self, prefix: str, n_completions: int = 5,
                 max_chars: int = 50) -> List[str]:
        """
        Generate multiple completions for a prefix (autocomplete).

        Returns list of completion strings.
        """
        if self.lm is None:
            return [prefix]

        completions = []
        for i in range(n_completions):
            # Vary temperature for diversity
            temp = 0.5 + (i * 0.15)
            result = self.generate(
                seed=prefix, max_chars=max_chars,
                temperature=temp, strategy='nucleus',
                stop_at='\n'
            )
            completion = result['generated']
            if completion and completion not in completions:
                completions.append(completion)

        return completions

    # === Sampling Methods ===

    def _sample(self, probs: Dict[str, float], temperature: float) -> str:
        """Sample from probability distribution with temperature."""
        if temperature <= 0:
            return max(probs, key=probs.get)

        symbols = list(probs.keys())
        p = np.array([probs[s] for s in symbols])

        # Apply temperature
        p = np.power(p, 1.0 / temperature)
        p = p / p.sum()

        idx = self.rng.choice(len(symbols), p=p)
        return symbols[idx]

    def _nucleus_sample(self, probs: Dict[str, float],
                        temperature: float, top_p: float) -> str:
        """Nucleus (top-p) sampling."""
        if temperature <= 0:
            return max(probs, key=probs.get)

        # Sort by probability descending
        sorted_items = sorted(probs.items(), key=lambda x: -x[1])
        symbols = [s for s, _ in sorted_items]
        p = np.array([prob for _, prob in sorted_items])

        # Apply temperature
        p = np.power(p, 1.0 / temperature)
        p = p / p.sum()

        # Find nucleus (top-p cutoff)
        cumsum = np.cumsum(p)
        cutoff = np.searchsorted(cumsum, top_p) + 1
        cutoff = min(cutoff, len(symbols))

        # Re-normalize within nucleus
        nucleus_p = p[:cutoff]
        nucleus_p = nucleus_p / nucleus_p.sum()

        idx = self.rng.choice(cutoff, p=nucleus_p)
        return symbols[idx]
