"""
train_form_causal.py — HSSLM-C auf causal-zentriertem Formen-Korpus.

Erkenntnis aus Runde 1: der BPE-Lauf lernte nur lange Prosa (Darwin/
Faraday), die kurzen kausalen Formen waren <1% des Korpus. Hier wird
verbalize(seed) als AUGMENTIERUNG genutzt: jeder Kausalpfad wird mit
vielen Verknüpfer-Formen ausgesprochen ("Consequently...", "This...",
"As a result...") — genau die Formen, die die Form-Engine braucht.

Korpus: multi-seed verbalisierte Pfade aus allen .causal-Graphen
(chained, code) + Faraday-Prosa als Flüssigkeits-Basis (1:1).
"""

from __future__ import annotations

import glob
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fertig.hsslm.model import HSSLMC  # noqa: E402
from fertig.hsslm.bpe import BPETokenizer  # noqa: E402
from fertig.pipeline import load_graph, verbalize  # noqa: E402

SEQ = 128
BATCH = 16
STEPS = 1200
# bvn_rr: Random-Reshuffling (BvN-Pfad-Integral -> Permutations-Epochen).
# Renn-Sieger der Training-Dynamics (4.578 vs baseline 4.612, Codex).
LR = 2.5e-4
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
ROOT = Path(__file__).resolve().parent.parent
SAVE = str(ROOT / "data" / "hsslm_form.pt")
BPE_SAVE = str(ROOT / "data" / "hsslm_bpe.json")


def _causal_forms(graph_path: str, per_entity: int = 12,
                  seeds: int = 6) -> list[str]:
    """Alle Pfade eines Graphen in vielen Verknüpfer-Formen verbalisieren."""
    try:
        vocab, stoi, adj, mech = load_graph(graph_path)
    except Exception:
        return []
    rng = random.Random(hash(graph_path) % 2 ** 32)
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
            for s in range(seeds):
                forms.add(verbalize(hops, vocab, mech, seed=s))
    return list(forms)


def build_corpus(max_chars: int = 600_000) -> str:
    parts: list[str] = []
    # 1. Kausal-Formen: NUR saubere Graphen (chained). world.causal ist
    #    Web-Rauschen ("The since i was dubious...") — kontaminiert die
    #    Formen. Code-Graphen sind kein Standard-Format (0 Formen).
    for path in [ROOT / "data" / "chained.causal"]:
        parts.extend(_causal_forms(str(path), per_entity=50, seeds=12))
    text = " ".join(parts)
    # 2. Faraday-Prosa als Flüssigkeits-Basis (Proportion ~1:1)
    faraday = ROOT / "data" / "faraday_clean.txt"
    if faraday.exists():
        prose = faraday.read_text(encoding="utf-8", errors="ignore")
        # Mindestens 50% des Korpus als Prosa-Basis — causal-Formen
        # sind die Zielform, Prosa die Flüssigkeit.
        min_prose = max(len(text), max_chars // 2)
        cut = min(len(prose), min_prose)
        text += " " + prose[:cut]
    return " ".join(text.split())[:max_chars]


def make_batches(ids: torch.Tensor):
    """BvN Random-Reshuffling: Epochen = Permutationen der Startpositionen,
    Batch-Starts konsekutiv aus der Permutation (bessere endliche
    Konvergenz als iid; mathematische Rechtfertigung: BvN-Zerlegung der
    doppelt-stochastischen Batch-Mischmatrix M = Sum λₖ Pₖ)."""
    n = ids.numel() - SEQ - 1
    gen = torch.Generator().manual_seed(1001)
    perm = None
    ptr = 0
    while True:
        if perm is None or ptr + BATCH > perm.numel():
            perm = torch.randperm(n, generator=gen)
            ptr = 0
        idx = perm[ptr:ptr + BATCH]
        ptr += BATCH
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
    import json
    print(f"device={DEVICE} seq={SEQ} batch={BATCH} steps={STEPS}")
    text = build_corpus()
    bpe = BPETokenizer(vocab_size=400)
    bpe.fit(text)
    ids = torch.tensor(bpe.encode(text), dtype=torch.long)
    print(f"Korpus: {len(text):,} chars -> {ids.numel():,} BPE-Tokens "
          f"(vocab {bpe.VOCAB_SIZE}, merges {len(bpe.merges)})")
    # BPE deterministisch für die Engine persistieren
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
        out = model.generate(pids, max_new_tokens=30, use_zeno=False,
                             use_foss_gate=False)
    print(f"\ngenerated: {bpe.decode(out[0].cpu())!r}")


if __name__ == "__main__":
    main()
