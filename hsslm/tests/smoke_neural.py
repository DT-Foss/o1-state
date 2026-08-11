"""
Smoke test for HSSLM neural (S6 / Mamba-style) module -- post streaming fix.

Verifies:
1. Model builds, forward pass on random batch works.
2. Parameter count matches the claimed ~8.6M.
3. Loss decreases over a handful of steps on a mini-batch (flat LM mode).
4. Detach-carry streaming test (F2-exactness pattern): forward on a full
   sequence in one shot vs. chunked with detached state (SSM state + conv
   buffer + position offset) carried forward. Reports max-abs-delta.
5. Same test with the hierarchical composer active (discourse_state carried
   too), to measure whether/how much the composer breaks exactness.

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
from neural.core_engine import StateSpaceCore


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def detach_state_tree(states):
    """Detach an arbitrarily nested list/tuple-of-tensors state structure."""
    if states is None:
        return None
    if isinstance(states, torch.Tensor):
        return states.detach()
    if isinstance(states, (list, tuple)):
        cls = type(states)
        return cls(detach_state_tree(s) for s in states)
    return states


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
    input_ids = torch.randint(4, vocab, (B, L))
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


def test_detach_carry_streaming_core():
    """F2-exactness on the bare StateSpaceCore: one-shot vs chunked+detached,
    with the fixed conv-buffer carry (no position embedding involved here --
    isolates the SSM+conv streaming property from the embedding layer)."""
    section("4. DETACH-CARRY STREAMING TEST -- bare StateSpaceCore (post-fix)")
    torch.manual_seed(7)
    core = StateSpaceCore(n_layers=6, d_model=256, d_state=16, d_conv=4, expand=2, dt_rank=8, dropout=0.0)
    core.eval()

    B, L, D = 1, 24, 256
    x = torch.randn(B, L, D)

    with torch.no_grad():
        y_full, _ = core(x, states=None)

        chunk_size = 6
        states = None
        ys = []
        for start in range(0, L, chunk_size):
            end = min(start + chunk_size, L)
            xc = x[:, start:end, :]
            yc, states = core(xc, states)
            ys.append(yc)
            states = detach_state_tree(states)
        y_chunked = torch.cat(ys, dim=1)

    delta = (y_full - y_chunked).abs()
    max_abs_delta = delta.max().item()
    mean_abs_delta = delta.mean().item()
    print(f"max|delta|:  {max_abs_delta:.3e}")
    print(f"mean|delta|: {mean_abs_delta:.3e}")

    exact = max_abs_delta < 1e-4
    print(f"VERDICT: {'EXACT (F2-pattern holds)' if exact else 'NOT EXACT'}")
    return exact, max_abs_delta


def test_detach_carry_streaming_full_model():
    """F2-exactness on the full HSSLM in flat mode (embedding + SSM core +
    LM head), including the position_offset fix, one-shot vs chunked."""
    section("4b. DETACH-CARRY STREAMING TEST -- full HSSLM, flat mode")
    torch.manual_seed(7)
    config = HSSLMConfig()
    config.hierarchical = False
    model = HSSLM(config)
    model.eval()

    B, L = 1, 24
    vocab = model.vocab_size
    input_ids = torch.randint(4, vocab, (B, L))

    with torch.no_grad():
        out_full = model(input_ids, states=None, position_offset=0)
        logits_full = out_full["logits"]

        chunk_size = 6
        states = None
        logits_chunks = []
        for start in range(0, L, chunk_size):
            end = min(start + chunk_size, L)
            chunk = input_ids[:, start:end]
            out_chunk = model(chunk, states=states, position_offset=start)
            logits_chunks.append(out_chunk["logits"])
            states = detach_state_tree(out_chunk["states"])
        logits_chunked = torch.cat(logits_chunks, dim=1)

    delta = (logits_full - logits_chunked).abs()
    max_abs_delta = delta.max().item()
    mean_abs_delta = delta.mean().item()
    print(f"max|delta|:  {max_abs_delta:.3e}")
    print(f"mean|delta|: {mean_abs_delta:.3e}")

    exact = max_abs_delta < 1e-4
    print(f"VERDICT: {'EXACT (F2-pattern holds)' if exact else 'NOT EXACT'}")
    return exact, max_abs_delta


def test_detach_carry_streaming_hierarchical():
    """Same test with the hierarchical composer active: boundaries are
    re-offset per chunk, discourse_state is explicitly carried+detached.

    IMPORTANT structural finding: model.py computes `logits` from `hidden`
    (the raw SSM core output) in BOTH the hierarchical and flat branches --
    the composer's output only feeds `hierarchy`/`aux`, never `logits`. So
    comparing `logits` here would just re-measure the already-exact SSM core
    (test 4b) and vacuously "pass". To actually measure composer streaming
    behavior we compare the LAST discourse-level sentence representation
    between the one-shot and chunked runs instead.

    Also structural: WordComposer/SentenceComposer take boundaries local to
    whatever states they're given, and there is no canonical way to make a
    4-chunk boundary structure equal a 1-chunk boundary structure over the
    same tokens (different word/sentence counts by construction) -- so this
    number reflects the composer's genuine lack of a chunk-invariant design,
    not a numerical bug. Reported for the record per the lead's instruction:
    the flat core must be exact (it is, see 4b), the hierarchy MAY
    approximate as long as it's measured."""
    section("4c. DETACH-CARRY STREAMING TEST -- hierarchical composer (discourse repr.)")
    torch.manual_seed(7)
    config = HSSLMConfig()
    config.hierarchical = True
    model = HSSLM(config)
    model.eval()

    B, L = 1, 24
    vocab = model.vocab_size
    input_ids = torch.randint(4, vocab, (B, L))

    def make_boundaries(length, n_words):
        step = max(1, length // n_words)
        bounds = []
        pos = 0
        while pos < length:
            end = min(pos + step, length)
            bounds.append([pos, end])
            pos = end
        return torch.tensor(bounds, dtype=torch.long)

    full_word_b = [make_boundaries(L, n_words=4)]
    full_sent_b = [torch.tensor([[0, len(full_word_b[0])]], dtype=torch.long)]
    boundaries_full = {"word_boundaries": full_word_b, "sentence_boundaries": full_sent_b}

    with torch.no_grad():
        out_full = model(
            input_ids, boundaries=boundaries_full, states=None,
            position_offset=0, return_hierarchy=True,
        )
        disc_full = out_full["hierarchy"]["discourse"][:, -1, :]  # (B, D) last sentence repr

        chunk_size = 6
        states = None
        discourse_state = None
        last_disc_chunked = None
        for start in range(0, L, chunk_size):
            end = min(start + chunk_size, L)
            chunk = input_ids[:, start:end]
            chunk_len = end - start
            word_b = [make_boundaries(chunk_len, n_words=max(1, chunk_len // 3))]
            sent_b = [torch.tensor([[0, len(word_b[0])]], dtype=torch.long)]
            out_chunk = model(
                chunk,
                boundaries={"word_boundaries": word_b, "sentence_boundaries": sent_b},
                states=states,
                position_offset=start,
                discourse_state=discourse_state,
                return_hierarchy=True,
            )
            states = detach_state_tree(out_chunk["states"])
            ds = out_chunk.get("discourse_state")
            discourse_state = ds.detach() if ds is not None else None
            last_disc_chunked = out_chunk["hierarchy"]["discourse"][:, -1, :]

    delta = (disc_full - last_disc_chunked).abs()
    max_abs_delta = delta.max().item()
    mean_abs_delta = delta.mean().item()
    print(f"discourse repr max|delta|:  {max_abs_delta:.3e}")
    print(f"discourse repr mean|delta|: {mean_abs_delta:.3e}")
    print("NOTE: logits are unaffected by the composer (model.py computes")
    print("them from the raw SSM `hidden`, not from `hierarchy`) -- this is")
    print("the honest signal: the composer's own representation, not logits.")
    print("Delta reflects the composer's structural lack of chunk-invariant")
    print("boundary handling (different word/sentence counts by")
    print("construction across chunk vs one-shot), not a numerical bug in")
    print("the fixed SSM/conv/position streaming path (see 4b: exact).")
    return max_abs_delta


if __name__ == "__main__":
    results = {}

    model, results["param_count"] = test_build_and_param_count()
    results["forward"] = test_forward_random_batch(model)
    results["loss_decrease"] = test_loss_decreases(model)
    results["streaming_core_exact"], core_delta = test_detach_carry_streaming_core()
    results["streaming_full_exact"], full_delta = test_detach_carry_streaming_full_model()
    hier_delta = test_detach_carry_streaming_hierarchical()

    section("SUMMARY")
    for k, v in results.items():
        print(f"  {k:24s}: {'PASS' if v else 'FAIL'}")
    print(f"  streaming core max|delta|:  {core_delta:.3e}")
    print(f"  streaming full max|delta|:  {full_delta:.3e}")
    print(f"  streaming hier  max|delta|: {hier_delta:.3e}  (structural, see note above)")
