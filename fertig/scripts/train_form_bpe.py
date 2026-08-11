"""
train_form_bpe.py — HSSLM-C mit BPE auf dem 3.9M-Formen-Korpus.

BPE (Wortgrenzen erhalten) statt Char-Level: das Modell sieht Wort-
Struktur, lernt Satzformen statt Buchstaben-Suppe. Checkpointing alle
200 Steps — robust gegen Timeouts.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fertig.hsslm.model import HSSLMC  # noqa: E402
from fertig.hsslm.bpe import BPETokenizer  # noqa: E402

SEQ = 128
BATCH = 16
STEPS = 1600
LR = 2.5e-4
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SAVE = str(Path(__file__).resolve().parent.parent / "data" / "hsslm_form.pt")

EN_FACTUAL = Path("/Users/bhkmie/Documents/kimi/workspace/AI_Causal_Work/"
                  "docs/superpowers/sprache/corpora/en_factual")


def build_corpus(max_chars: int = 3_000_000) -> str:
    parts = []
    if EN_FACTUAL.exists():
        for f in sorted(EN_FACTUAL.glob("*.txt")):
            parts.append(f.read_text(encoding="utf-8", errors="ignore"))
    local = Path(__file__).resolve().parent.parent / "data"
    for f in ("faraday_candle.txt",):
        fp = local / f
        if fp.exists():
            parts.append(fp.read_text(encoding="utf-8", errors="ignore"))
    return " ".join(" ".join(parts).split())[:max_chars]


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
    print(f"device={DEVICE} seq={SEQ} batch={BATCH} steps={STEPS}")
    text = build_corpus()
    bpe = BPETokenizer(vocab_size=400)
    bpe.fit(text)
    ids = torch.tensor(bpe.encode(text), dtype=torch.long)
    print(f"Korpus: {len(text):,} chars -> {ids.numel():,} BPE-Tokens "
          f"(vocab {bpe.VOCAB_SIZE})")

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
        opt.zero_grad()
        loss.backward()
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
        out = model.generate(pids, max_new_tokens=40, use_zeno=False,
                             use_foss_gate=False)
    print(f"\ngenerated: {bpe.decode(out[0].cpu())!r}")


if __name__ == "__main__":
    main()
