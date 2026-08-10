#!/usr/bin/env python3
"""
Extract Residual Stream States from Qwen3-1.7B
================================================
Run forward pass on important words/phrases, capture hidden states
at layers 10 and 18 (middle and late). Save as pre-contextualized
embeddings for Hopfield lookup.

Layer 10: syntactic patterns, word relationships
Layer 18: semantic composition, fact recall
(These are the MLP knowledge layers per Mechanistic Interpretability)

One-time extraction — results are saved to disk.
"""

import os
import sys
import json
import numpy as np

# MLX imports
import mlx.core as mx
from mlx_lm import load

base = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base, 'data')

print("Loading Qwen3-1.7B...")
model, tokenizer = load('Qwen/Qwen3-1.7B')
print(f"  {len(model.layers)} layers, loaded")

# Load vocab for important words
vocab_path = os.path.join(data_dir, 'qwen3_1.7b_vocab.json')
with open(vocab_path) as f:
    token2id = json.load(f)
id2token = {v: k for k, v in token2id.items()}

# Important words/phrases to extract residuals for
# Mix of: common nouns, countries, capitals, concepts, relations
IMPORTANT_WORDS = [
    # Countries & capitals
    "France", "Paris", "Germany", "Berlin", "Japan", "Tokyo",
    "Spain", "Madrid", "Italy", "Rome", "China", "Beijing",
    "Russia", "Moscow", "Brazil", "Australia", "India", "Egypt",
    "Norway", "Oslo", "Sweden", "Stockholm", "Canada", "Ottawa",
    # Science
    "gravity", "planet", "star", "moon", "sun", "earth",
    "water", "oxygen", "hydrogen", "carbon", "energy", "light",
    "atom", "electron", "proton", "neutron", "molecule", "cell",
    "photosynthesis", "evolution", "DNA", "gene", "protein",
    # People
    "Shakespeare", "Einstein", "Newton", "Darwin", "Orwell",
    "Mozart", "Beethoven", "Tesla", "Edison", "Curie",
    # Animals
    "cat", "dog", "fish", "bird", "horse", "lion", "eagle",
    "whale", "spider", "snake", "elephant", "dolphin",
    # Common objects
    "book", "computer", "phone", "car", "house", "table",
    "chair", "door", "window", "key", "clock", "mirror",
    # Abstract concepts
    "time", "space", "love", "truth", "justice", "freedom",
    "knowledge", "power", "death", "life", "war", "peace",
    # Colors
    "red", "blue", "green", "yellow", "black", "white",
    # Food
    "bread", "rice", "milk", "apple", "coffee", "chocolate",
    # Body parts
    "heart", "brain", "blood", "bone", "eye", "hand",
    # Geography
    "mountain", "river", "ocean", "island", "desert", "forest",
    # Relations/questions
    "capital", "author", "language", "largest", "smallest",
    "color", "speed", "weight", "temperature", "distance",
]

# Also extract for common phrases (multi-word context)
IMPORTANT_PHRASES = [
    "capital of France",
    "who wrote",
    "largest planet",
    "speed of light",
    "boiling point of water",
    "color of sky",
    "the sun is",
    "water is made of",
    "gravity causes",
    "plants need",
]


def extract_hidden_states(text, layers_to_capture=[10, 18]):
    """Run forward pass and capture hidden states at specific layers.

    Returns: dict of layer_idx → numpy array (seq_len, hidden_dim)
    """
    tokens = tokenizer.encode(text)
    if not tokens:
        return {}

    x = mx.array([tokens])

    # Manual forward pass through embedding + layers
    h = model.model.embed_tokens(x)  # (1, seq_len, 2048)

    results = {}
    for i, layer in enumerate(model.layers):
        h = layer(h, mask=None, cache=None)
        if isinstance(h, tuple):
            h = h[0]
        if i in layers_to_capture:
            # Take last token's hidden state (most contextual)
            state = np.array(h[0, -1, :].tolist(), dtype=np.float32)
            results[i] = state

    return results


print(f"\nExtracting residuals for {len(IMPORTANT_WORDS)} words + {len(IMPORTANT_PHRASES)} phrases...")

# Storage: word → {layer_10: vec, layer_18: vec}
residuals = {}
n_done = 0

for word in IMPORTANT_WORDS:
    try:
        states = extract_hidden_states(word)
        if states:
            residuals[word.lower()] = {
                f'layer_{k}': v for k, v in states.items()
            }
            n_done += 1
    except Exception as e:
        print(f"  Skip '{word}': {e}")

    if n_done % 20 == 0 and n_done > 0:
        print(f"  {n_done}/{len(IMPORTANT_WORDS)} words done")

print(f"  Words: {n_done}/{len(IMPORTANT_WORDS)}")

n_phrases = 0
for phrase in IMPORTANT_PHRASES:
    try:
        states = extract_hidden_states(phrase)
        if states:
            residuals[phrase.lower()] = {
                f'layer_{k}': v for k, v in states.items()
            }
            n_phrases += 1
    except Exception as e:
        print(f"  Skip '{phrase}': {e}")

print(f"  Phrases: {n_phrases}/{len(IMPORTANT_PHRASES)}")

# Save as npz
# Flatten: key = "word__layer_10", value = 2048d vector
flat = {}
for key, layers in residuals.items():
    for layer_name, vec in layers.items():
        flat_key = f"{key}__{layer_name}"
        flat[flat_key] = vec

save_path = os.path.join(data_dir, 'qwen3_residuals.npz')
np.savez_compressed(save_path, **flat)
print(f"\nSaved {len(flat)} residual vectors to {save_path}")
print(f"  ({len(residuals)} entries × 2 layers = {len(flat)} vectors)")
print(f"  Each vector: 2048d float32")
