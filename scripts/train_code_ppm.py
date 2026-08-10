"""
Train PPM-5 on Python code corpus.
Output: data/ppm5_code.json
"""
import os
import sys
import time
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.language import PPMModel

CORPUS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      'data', 'code_corpus.txt')
OUTPUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      'data', 'ppm5_code.json')

print("Loading code corpus...")
with open(CORPUS, 'r', encoding='utf-8') as f:
    text = f.read()
print(f"Corpus: {len(text):,} chars")

# Train PPM-5
print("\nTraining PPM-5 on code corpus...")
t0 = time.time()
ppm = PPMModel(max_order=5)
ppm.train(list(text))
elapsed = time.time() - t0
print(f"Trained in {elapsed:.1f}s")
print(f"Vocab: {len(ppm.vocab)} symbols")
print(f"Total symbols: {ppm._total_symbols:,}")

# Save
print(f"\nSaving to {OUTPUT}...")
ppm.save(OUTPUT)
size = os.path.getsize(OUTPUT)
print(f"Saved: {size:,} bytes")

# Quick BPC test on a code snippet
test_code = """def fibonacci(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
"""
test_seq = list(test_code)
total_log = 0
for i in range(1, len(test_seq)):
    ctx = test_seq[:i]
    probs = ppm.predict(ctx)
    p = probs.get(test_seq[i], 1e-10)
    total_log += -math.log2(max(p, 1e-10))

bpc = total_log / len(test_seq)
print(f"\nBPC on fibonacci test: {bpc:.4f}")

# Generate a small code sample
print("\nGeneration test (seed='def '):")
seed = list("def ")
generated = ppm.generate(seed, n_tokens=100)
print("  " + ''.join(generated))
