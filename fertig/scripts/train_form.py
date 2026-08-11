"""
train_form.py — HSSLM-C auf dem FERTIG-Formen-Korpus trainieren.

Der Formen-Korpus = die deterministische Prosa aus dem .causal-Graphen
(verbalize) + Faraday — HSSLM lernt die FLÜSSIGE Form, FERTIG liefert
den Beleg. PURE LM-Loss (flat, keine Random-Aux — die Gradienten-
Vergiftung aus dem Base-Modell wird umgangen), char-level, MPS.

Meilenstein: loss fällt, Generierung driftet Richtung Korpus — dann ist
HSSLM bereit, Form-Varianten für das Utterance-IR zu erzeugen, die gegen
den Plan verifiziert werden.
"""

from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fertig.hsslm.model import HSSLMC  # noqa: E402
from fertig.hsslm.tokenizer import HierarchicalTokenizer  # noqa: E402
from fertig.pipeline import load_graph, verbalize, DEFAULT_GRAPH  # noqa: E402
from fertig import state_init  # noqa: E402

SEQ = 128
BATCH = 8
STEPS = 600
LR = 3e-4
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def build_form_corpus(graph_path: str, max_chars: int = 200_000) -> str:
    """Formen-Korpus: verbalisierte Kausal-Prosa (Beleg-Struktur!) +
    Faraday als Flüssigkeits-Quelle."""
    parts = []
    vocab, stoi, adj, mech = load_graph(graph_path)
    SM = state_init.initialize_symbol_state(len(vocab))
    for entity in vocab:
        hops = []
        cur = stoi.get(entity)
        seen = set()
        for _ in range(4):
            if cur is None or cur in seen:
                break
            seen.add(cur)
            nbrs = adj.get(cur, {})
            if not nbrs:
                break
            nxt = max(nbrs, key=nbrs.get)
            hops.append((cur, nxt))
            cur = nxt
        if hops:
            parts.append(verbalize(hops, vocab, mech, seed=0))
    text = " ".join(parts)
    faraday = Path(__file__).resolve().parent.parent / "data" / "faraday_candle.txt"
    if faraday.exists():
        text += " " + faraday.read_text(encoding="utf-8", errors="ignore")
    return " ".join(text.split())[:max_chars]


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


def train(graph_path: str = str(DEFAULT_GRAPH), steps: int = STEPS,
          save_path: str = "data/hsslm_form.pt") -> dict:
    print(f"device={DEVICE} seq={SEQ} batch={BATCH} steps={steps}")
    tk = HierarchicalTokenizer()
    text = build_form_corpus(graph_path)
    enc = tk.encode(text, add_bos=False, add_eos=False,
                    max_length=10_000_000)
    ids = enc["input_ids"].long()
    print(f"Formen-Korpus: {len(text)} chars -> {ids.numel()} tokens")

    model = HSSLMC(config={
        "d_model": 256, "vocab_size": tk.VOCAB_SIZE,
        "n_layers": 4, "hierarchical": False,
    }).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"HSSLM-C params: {n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    batches = make_batches(ids)
    model.train()
    losses = []
    t0 = time.time()
    for step in range(1, steps + 1):
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
            # Checkpoint: robust gegen Timeouts (keine halben Sachen)
            torch.save(model.state_dict(), save_path)
            print(f"  CHECKPOINT gespeichert: {save_path} (step {step})")

    fell = sum(losses[-20:]) / 20 < losses[0] - 0.3
    print(f"\nloss: {losses[0]:.3f} -> {sum(losses[-20:])/20:.3f} "
          f"({'FELL' if fell else 'FLAT'})")

    # Generierung: driftet sie Richtung Korpus?
    model.eval()
    prompt = "Smoking"
    pids = tk.encode(prompt, add_bos=True, add_eos=False)[
        "input_ids"].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model.generate(pids, max_new_tokens=80, use_zeno=True,
                             use_foss_gate=False)
    gen = tk.decode(out[0].cpu())
    print(f"\nprompt:    {prompt!r}")
    print(f"generated: {gen!r}")

    if fell:
        torch.save(model.state_dict(), save_path)
        print(f"Gewichte gespeichert: {save_path}")
    return {"loss_first": losses[0], "loss_last": sum(losses[-20:]) / 20,
            "fell": fell, "generated": gen}


if __name__ == "__main__":
    train()
