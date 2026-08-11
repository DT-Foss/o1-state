#!/usr/bin/env python3
"""train_form_causal_v6.py — Runde 6: Ziel-Kette dominiert den Korpus.

v5-Befund: Das Modell generiert die Kausal-Form korrekt, aber die
6-Objekt-Kette bricht nach ~3 Gliedern in eine Schleife, weil world- und
chained-Ketten gemischt sind (nach "lung damage." konkurrieren
"And so lung damage causes breathlessness" und world-Fortsetzungen).

v6-Design:
  a) chained.causal (die Graphen, die die Engine prueft) REPEAT=16 —
     die Ziel-Ketten dominieren die Uebergangs-Statistik
  b) andere Graphen (world, code/*) REPEAT=4 — Vielfalt, aber nachrangig
  c) Faraday-Prosa auf 25% begrenzt (weniger Konkurrenz an Satzgrenzen)
  d) kanonische Formen (seed=0), \\n\\n-Blockgrenzen, fester Seed (v5-Erbe)
Checkpoint: data/hsslm_form_v6.pt, BPE: data/hsslm_bpe_v6.json.
"""

from __future__ import annotations

import glob
import json
import math
import random
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent  # FERTIG/ (scripts/v7)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from fertig.hsslm.bpe import BPETokenizer  # noqa: E402
from fertig.hsslm.model import HSSLMC  # noqa: E402
from fertig.pipeline import load_graph, verbalize  # noqa: E402

SEQ = 128
BATCH = 16
STEPS = 800
LR = 2.5e-4
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SAVE = str(ROOT / "data" / "hsslm_form_v7.pt")
BPE_SAVE = str(ROOT / "data" / "hsslm_bpe_v7.json")
CORPUS_SEED = 12345
REPEAT_ZIEL = 24     # chained.causal (Engine-Ziel-Graph)
REPEAT_ANDERE = 4    # world, code/*
FARADAY_RATIO = 0.25


def _canonical_forms(graph_path: str, per_entity: int = 40) -> list[str]:
    """Kanonische Ketten (seed=0, deterministische Opener), fester Seed."""
    try:
        vocab, stoi, adj, mech = load_graph(graph_path)
    except Exception:
        return []
    rng = random.Random(CORPUS_SEED)
    forms: set[str] = set()
    for entity in vocab:
        cur = stoi.get(entity)
        if cur is None:
            continue
        for _ in range(per_entity):
            if cur is None:
                break
            nbrs = adj.get(cur, {})
            if not nbrs:
                break
            nxt = rng.choice(list(nbrs))
            hops = [(cur, nxt)]
            cur2 = nxt
            seen = {cur}
            for _ in range(4):
                if cur2 in seen or cur2 is None:
                    break
                seen.add(cur2)
                nn = adj.get(cur2, {})
                if not nn:
                    break
                nxt2 = rng.choice(list(nn))
                hops.append((cur2, nxt2))
                cur2 = nxt2
            forms.add(verbalize(hops, vocab, mech, seed=0))
    return list(forms)


def build_corpus_v7(max_chars: int = 600_000) -> str:
    """Ziel-Graph (chained) x16, andere x4, Faraday 25%, \\n\\n-Getrennt."""
    parts: list[str] = []
    for path in sorted(glob.glob(str(ROOT / "data" / "*.causal"))):
        repeat = REPEAT_ZIEL if "chained" in Path(path).name else REPEAT_ANDERE
        for f in _canonical_forms(path):
            parts.extend([f] * repeat)
    for path in sorted(glob.glob(str(ROOT / "data" / "code" / "*.causal"))):
        for f in _canonical_forms(path):
            parts.extend([f] * REPEAT_ANDERE)
    causal_text = "\n\n".join(parts)
    faraday = ROOT / "data" / "faraday_candle.txt"
    if faraday.exists():
        prose = faraday.read_text(encoding="utf-8", errors="ignore")
        sents = re.split(r"(?<=[.!?])\s+", prose)
        paras, cur = [], []
        for s in sents:
            cur.append(s)
            if sum(len(x) for x in cur) > 200:
                paras.append(" ".join(cur))
                cur = []
        if cur:
            paras.append(" ".join(cur))
        cut = int(len(causal_text) * FARADAY_RATIO / (1 - FARADAY_RATIO))
        text = causal_text + "\n\n" + "\n\n".join(paras)[:cut]
    else:
        text = causal_text
    return text[:max_chars]


def make_batches(ids: torch.Tensor):
    n = ids.numel() - SEQ - 1
    while True:
        idx = torch.randint(0, n, (BATCH,))
        x = torch.stack([ids[i:i + SEQ] for i in idx])
        y = torch.stack([ids[i + 1:i + 1 + SEQ] for i in idx])
        yield x.to(DEVICE), y.to(DEVICE)


def lm_loss(model, x, y):
    was_training = model.training
    model.eval()
    out = model.forward(x)
    if was_training:
        model.train()
    logits = out["logits"]
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                           y.reshape(-1), ignore_index=0)


def main():
    print(f"device={DEVICE} seq={SEQ} batch={BATCH} steps={STEPS} "
          f"korpus=v7 (chained x{REPEAT_ZIEL}, code x{REPEAT_ANDERE}, "
          f"faraday {FARADAY_RATIO:.0%})")
    text = build_corpus_v7()
    bpe = BPETokenizer(vocab_size=400)
    bpe.fit(text)
    ids = torch.tensor(bpe.encode(text), dtype=torch.long)
    print(f"Korpus: {len(text):,} chars -> {ids.numel():,} BPE-Tokens "
          f"(vocab {bpe.VOCAB_SIZE}, merges {len(bpe.merges)})")
    Path(BPE_SAVE).write_text(json.dumps({
        "merges": bpe.merges, "vocab": bpe.vocab,
        "itos": {str(k): v for k, v in bpe.itos.items()}}))

    model = HSSLMC(config={
        "d_model": 256, "vocab_size": bpe.VOCAB_SIZE,
        "n_layers": 4, "hierarchical": False,
    }).to(DEVICE)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    batches = make_batches(ids)
    model.train()
    losses = []
    t0 = time.time()
    for step in range(1, STEPS + 1):
        x, y = next(batches)
        loss = lm_loss(model, x, y)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
        if step % 100 == 0 or step == 1:
            recent = sum(losses[-100:]) / len(losses[-100:])
            ppl = math.exp(min(recent, 20))
            print(f"  step {step:4d}  loss {loss.item():.3f}  "
                  f"avg100 {recent:.3f}  ppl {ppl:7.1f}  "
                  f"({(time.time()-t0):.0f}s)")
        if step % 200 == 0:
            torch.save(model.state_dict(), SAVE)
            print(f"  CHECKPOINT: {SAVE} (step {step})")

    model.eval()
    prompt = "Smoking causes"
    pids = torch.tensor([bpe.encode(prompt)], dtype=torch.long).to(DEVICE)
    with torch.no_grad():
        out = model.generate(pids, max_new_tokens=30, use_zeno=False,
                             use_foss_gate=False)
    print(f"\ngenerated: {bpe.decode(out[0].cpu())!r}")


if __name__ == "__main__":
    main()
