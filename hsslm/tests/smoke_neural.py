"""
Smoke test for HSSLM neural (S6 / Mamba-style) module.

Verifies:
1. Model builds, forward pass on random batch works.
2. Parameter count matches the claimed ~8.6M.
3. Loss decreases over a handful of steps on a mini-batch (flat LM mode,
   no hierarchical boundaries -- isolates the SSM core's learnability).
4. Detach-carry streaming test (F2-exactness pattern): forward on a full
   sequence in one shot vs. chunked with detached state carried forward.
   Reports max-abs-delta between the two logit streams.

Run:
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false python3 smoke_neural.py
"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
torch.set_num_threads(1)
torch.manual_seed(43)

from neural.model import HSSLM
from neural.config import HSSLMConfig


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def test_build_and_param_count():
    section("1. BUILD + PARAMETER COUNT")
    config = HSSLMConfig()
    model = HSSLM(config)
    model.print_parameter_summary()
    counts = model.get_parameter_count()
    total = counts["total"]
    claimed = 8_620_829
    delta = abs(total - claimed)
    pct = delta / claimed * 100
    print(f"\nClaimed: {claimed:,}  Measured: {total:,}  Delta: {delta:,} ({pct:.2f}%)")
    ok = pct < 1.0
    print(f"VERDICT: {'PASS' if ok else 'FAIL'} (within 1% of claimed 8.6M)")
    return model, ok


def test_forward_random_batch(model):
    section("2. FORWARD PASS ON RANDOM BATCH")
    B, L = 2, 32
    vocab = model.vocab_size
    input_ids = torch.randint(4, vocab, (B, L))  # avoid special tokens 0-3
    labels = torch.randint(4, vocab, (B, L))

    model.eval()
    with torch.no_grad():
        out = model(input_ids, labels=labels)  # flat mode, no boundaries
    logits = out["logits"]
    loss = out["loss"]
    print(f"input_ids: {tuple(input_ids.shape)}")
    print(f"logits:    {tuple(logits.shape)}  (expect ({B}, {L}, {vocab}))")
    print(f"loss:      {loss.item():.4f}")
    shape_ok = logits.shape == (B, L, vocab)
    finite_ok = torch.isfinite(logits).all().item() and torch.isfinite(loss).item()
    print(f"VERDICT: {'PASS' if (shape_ok and finite_ok) else 'FAIL'}")
    return shape_ok and finite_ok


def test_loss_decreases(model):
    section("3. LOSS DECREASE OVER MINI-BATCH STEPS (flat LM mode)")
    model.train()
    B, L = 4, 24
    vocab = model.vocab_size
    torch.manual_seed(1)
    input_ids = torch.randint(4, vocab, (B, L))
    labels = torch.randint(4, vocab, (B, L))

    opt = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95), weight_decay=0.1)
    losses = []
    n_steps = 30
    for step in range(n_steps):
        opt.zero_grad()
        out = model(input_ids, labels=labels)
        loss = out["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
        if step % 5 == 0 or step == n_steps - 1:
            print(f"  step {step:3d}: loss = {loss.item():.4f}")

    first_avg = sum(losses[:3]) / 3
    last_avg = sum(losses[-3:]) / 3
    print(f"\nFirst-3 avg: {first_avg:.4f}  Last-3 avg: {last_avg:.4f}")
    ok = last_avg < first_avg and all(torch.isfinite(torch.tensor(l)) for l in losses)
    print(f"VERDICT: {'PASS' if ok else 'FAIL'} (loss decreased, no NaN/Inf)")
    return ok


def test_detach_carry_streaming():
    """F2-exactness pattern: one-shot forward vs. chunked forward with
    detached state carried across chunk boundaries. Uses FLAT mode
    (no hierarchical composer) to isolate the SSM core's streaming property,
    since the hierarchical composers (word/phrase/sentence/discourse
    boundary pooling) are not designed to operate chunk-locally.
    """
    section("4. DETACH-CARRY STREAMING TEST (SSM core, flat mode)")
    torch.manual_seed(7)
    config = HSSLMConfig()
    config.hierarchical = False  # isolate the SSM core itself
    model = HSSLM(config)
    model.eval()

    B, L = 1, 24
    vocab = model.vocab_size
    input_ids = torch.randint(4, vocab, (B, L))

    with torch.no_grad():
        # One-shot: whole sequence at once, no external state
        out_full = model(input_ids, states=None)
        logits_full = out_full["logits"]  # (B, L, V)

        # Chunked: process in chunks of 6, carrying detached state forward
        chunk_size = 6
        states = None
        logits_chunks = []
        for start in range(0, L, chunk_size):
            end = min(start + chunk_size, L)
            chunk = input_ids[:, start:end]
            out_chunk = model(chunk, states=states)
            logits_chunks.append(out_chunk["logits"])
            # detach-carry: sever autograd graph, keep values
            states = [s.detach() if s is not None else None for s in out_chunk["states"]]

        logits_chunked = torch.cat(logits_chunks, dim=1)  # (B, L, V)

    assert logits_full.shape == logits_chunked.shape, (
        f"shape mismatch: {logits_full.shape} vs {logits_chunked.shape}"
    )
    delta = (logits_full - logits_chunked).abs()
    max_abs_delta = delta.max().item()
    mean_abs_delta = delta.mean().item()
    rel_delta = (delta / (logits_full.abs() + 1e-8)).max().item()

    print(f"logits_full:    {tuple(logits_full.shape)}")
    print(f"logits_chunked: {tuple(logits_chunked.shape)}")
    print(f"max |delta|:    {max_abs_delta:.3e}")
    print(f"mean |delta|:   {mean_abs_delta:.3e}")
    print(f"max relative:   {rel_delta:.3e}")

    exact = max_abs_delta < 1e-4
    print(f"\nVERDICT: {'EXACT (F2-pattern holds)' if exact else 'NOT EXACT -- reports the delta magnitude for the record'}")
    return exact, max_abs_delta


def test_detach_carry_hierarchical_note():
    section("4b. DETACH-CARRY WITH HIERARCHICAL COMPOSER (documents which composer breaks it)")
    torch.manual_seed(7)
    config = HSSLMConfig()
    config.hierarchical = True
    model = HSSLM(config)
    model.eval()

    # With hierarchical=True, forward requires 'boundaries' to activate the
    # composer path at all -- otherwise model.forward silently falls back to
    # flat mode (see model.py: `if self.hierarchical and boundaries is not None`).
    # This means the *streaming* interface (states=...) has no defined
    # interaction with the composer: boundaries are computed by the tokenizer
    # over word/sentence spans that assume the FULL sequence's absolute
    # token positions are known -- chunking breaks boundary indices unless
    # the caller re-derives per-chunk boundaries with correct offsets, and
    # DiscourseComposer keeps its own separate mutable `discourse_state`
    # instance attribute across calls (NOT threaded through the `states`
    # list at all), so chunk-to-chunk continuity is implicit/hidden rather
    # than an explicit, resettable, testable state per the states=[...] API.
    print("Finding (no numeric test needed -- structural):")
    print("  - HierarchicalComposer.forward requires `boundaries` computed on")
    print("    absolute token indices; chunking requires the caller to")
    print("    correctly re-offset word/sentence boundary tensors per chunk.")
    print("  - DiscourseComposer.discourse_state is an instance attribute,")
    print("    NOT part of the states=[...] list returned by model.forward().")
    print("    It persists across calls implicitly (module-level mutable")
    print("    state) rather than being explicit/functional. A caller doing")
    print("    detach-carry via states=[...] will NOT detach the discourse")
    print("    state -- it silently keeps accumulating gradients unless")
    print("    .reset() or manual .detach_() is called on it separately.")
    print("  - WordComposer/SentenceComposer boundary pooling loops over")
    print("    Python lists per-batch-item (no vectorized chunk-carry path).")
    return True


if __name__ == "__main__":
    results = {}

    model, results["param_count"] = test_build_and_param_count()
    results["forward"] = test_forward_random_batch(model)
    results["loss_decrease"] = test_loss_decreases(model)
    results["streaming_exact"], max_delta = test_detach_carry_streaming()
    test_detach_carry_hierarchical_note()

    section("SUMMARY")
    for k, v in results.items():
        print(f"  {k:20s}: {'PASS' if v else 'FAIL/NOTED'}")
    print(f"  streaming max|delta|: {max_delta:.3e}")
