"""
Trainings-Harness fuer das Sprach-Leben (Baustein 2, Sprache-Haupt-Track).

These: HSSLM muss nichts WISSEN (Wissen liegt in der Datei/dem LiveStore),
es muss VERSPRACHLICHEN -- Struktur (trigger|mechanism|outcome, aus dem
Graphen) in fluessigen Satz uebersetzen. Die Trainingsdaten dafuer sind ein
Nebenprodukt des Builders (hsslm/data/graph_to_text.py).

Gemischtes Curriculum pro Chunk:
  - Rohes WT-103-LM-Streaming (HSSLMStreamer, gated -- surprise-gate wie
    gehabt, hsslm/neural/streaming.py) -- haelt die allgemeine Sprachfaehigkeit
    lebendig, unabhaengig vom Graphen.
  - Struktur->Satz-Paare aus graph_to_text_pairs.jsonl -- IMMER trainiert
    (kein Gate), das ist die Kernaufgabe: Verspracherlichung. Kein Gate
    hier, weil jedes Paar bereits durch den Builder + Foss-Gate kuratiert
    wurde (nicht rohes, ungefiltertes Streaming wie beim LM-Arm) -- ein
    zweites Surprise-Gate auf bereits-kuratierten Daten waere redundant.

Mischverhaeltnis: --mix (Default 0.5) = Anteil der Chunks, die ein
Struktur->Satz-Paar statt einen LM-Streaming-Chunk ziehen. Deterministisch
zyklisch (nicht zufaellig) fuer Reproduzierbarkeit: chunk c ist ein
Struktur-Chunk wenn (c * mix) % 1.0 < mix... praktischer: ein einfacher
Bresenham-artiger Zaehler, siehe mix_schedule().

Vokabular: Organism-5002 + <fact>/<say> = 5004 IDs (siehe
hsslm/data/graph_to_text.py's Format-Entscheidung-Docstring). LM-Head
bleibt weight-tied (HSSLMConfig.vocab_size=5004 durchgereicht, LMHead selbst
kennt nur vocab_size, keine Sonderbehandlung fuer die neuen IDs noetig).

Checkpointing: torch.save im Stil von portable_organism.save_snapshot (Feld
fuer Feld analog: model, opt, streamer_state, struct_epoch/struct_idx,
config, wall_s), atomar via tempfile+os.replace, alle --ckpt-every-chunks
Chunks. Resume via --resume <pfad>.

Status-JSON im pos_run-Stil (write_atomic, Felder wall_s/n_streamed/
tok_per_s/rss_gb/phase/arms-artige Unterstruktur), alle --status-every-s
Sekunden, damit ein externer Watcher (Wache) den Fortschritt lesen kann.

Usage (Smoke):
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \
    python3 speak_train.py --chunks 300 --mix 0.5 --sample-after-smoke
"""
import argparse
import json
import os
import shutil
import sys
import tempfile
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "hsslm"))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import torch
import torch.nn.functional as F

from neural.model import HSSLM
from neural.config import HSSLMConfig
from neural.streaming import HSSLMStreamer, detach_state_tree, IGNITION_CHUNKS, MIN_WINDOW

from length_extrap_v2 import build_vocab, tokenize

try:
    import psutil
    _PROC = psutil.Process()
    def _rss_gb(): return _PROC.memory_info().rss / 1e9
except ImportError:
    import resource
    def _rss_gb(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9

FACT_TOKEN = "<fact>"
SAY_TOKEN = "<say>"


# ===========================================================================
# Vocabulary: Organism 5000+2, extended with <fact>/<say> = 5004 total.
# ===========================================================================

def build_extended_vocab(text: str):
    """Organism vocabulary (length_extrap_v2.build_vocab) plus 2 structure
    marker IDs appended at the end. Returns (stoi, unk_id, mask_id,
    fact_id, say_id, total_ids)."""
    vocab, stoi, unk, mask = build_vocab(text)
    fact_id = mask + 1
    say_id = mask + 2
    stoi = dict(stoi)
    stoi[FACT_TOKEN] = fact_id
    stoi[SAY_TOKEN] = say_id
    total_ids = say_id + 1
    return stoi, unk, mask, fact_id, say_id, total_ids


def tokenize_structure(structure: str, stoi: dict, unk_id: int, fact_id: int, say_id: int):
    """Tokenize a '<fact> a | b | c <say>' string: the two markers get their
    dedicated IDs, everything else goes through the normal word tokenizer
    (length_extrap_v2's [a-zA-Z]+ regex via tokenize(), which silently drops
    non-letter chars including '|' -- fine, the marker IDs alone are enough
    structure signal, the pipe itself carries no separate meaning the model
    needs to reproduce)."""
    body = structure
    assert body.startswith(FACT_TOKEN) and body.endswith(SAY_TOKEN), \
        f"expected '{FACT_TOKEN} ... {SAY_TOKEN}', got: {structure!r}"
    inner = body[len(FACT_TOKEN):-len(SAY_TOKEN)].strip()
    inner_ids = tokenize(inner, stoi, unk_id)
    return [fact_id] + inner_ids + [say_id]


def tokenize_text(text: str, stoi: dict, unk_id: int):
    return tokenize(text, stoi, unk_id)


# ===========================================================================
# Struct->text pairs: load + deterministic mix schedule
# ===========================================================================

def load_pairs(path: str):
    pairs = []
    with open(path) as f:
        for line in f:
            pairs.append(json.loads(line))
    return pairs


def mix_schedule(n_chunks: int, mix: float):
    """Deterministic (not random) interleave: yields a bool per chunk index,
    True = struct chunk, False = LM chunk, with True-fraction == mix (up to
    rounding), spread evenly (Bresenham-style accumulator) rather than
    clumped, so a --mix 0.5 run alternates roughly 1:1 instead of front-
    loading all struct chunks first."""
    acc = 0.0
    out = []
    for _ in range(n_chunks):
        acc += mix
        if acc >= 1.0:
            acc -= 1.0
            out.append(True)
        else:
            out.append(False)
    return out


# ===========================================================================
# Checkpointing (portable_organism.save_snapshot style)
# ===========================================================================

def save_checkpoint(path, model, opt, streamer, struct_idx, wall_s, config, extra=None):
    ck = {
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "streamer_states": detach_state_tree(streamer.states),
        "streamer_discourse_state": streamer.discourse_state,
        "streamer_position": streamer.position,
        "streamer_window": list(streamer.window),
        "streamer_n_chunks": streamer.n_chunks,
        "streamer_n_bwd": streamer.n_bwd,
        "streamer_grad_tokens": streamer.grad_tokens,
        "struct_idx": struct_idx,
        "torch_rng": torch.get_rng_state(),
        "wall_s": wall_s,
        "config": config,
        "extra": extra or {},
    }
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}.pt"
    torch.save(ck, tmp)
    os.replace(tmp, path)


def load_checkpoint(path, model, opt, streamer):
    ck = torch.load(path, weights_only=False)
    model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["opt"])
    streamer.states = ck["streamer_states"]
    streamer.discourse_state = ck["streamer_discourse_state"]
    streamer.position = ck["streamer_position"]
    from collections import deque
    streamer.window = deque(ck["streamer_window"], maxlen=streamer.window.maxlen)
    streamer.n_chunks = ck["streamer_n_chunks"]
    streamer.n_bwd = ck["streamer_n_bwd"]
    streamer.grad_tokens = ck["streamer_grad_tokens"]
    torch.set_rng_state(ck["torch_rng"])
    return ck["struct_idx"], ck["wall_s"]


def write_atomic(path, obj):
    d = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp_")
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


# ===========================================================================
# Sampling (greedy decode, for the post-smoke qualitative check)
# ===========================================================================

@torch.no_grad()
def greedy_decode_structure(model, structure: str, stoi, unk_id, fact_id, say_id,
                             id_to_word, max_new_tokens=40):
    model.eval()
    input_ids_list = tokenize_structure(structure, stoi, unk_id, fact_id, say_id)
    input_ids = torch.tensor([input_ids_list], dtype=torch.long)

    states = None
    position = 0
    for t in range(input_ids.shape[1]):
        tok = input_ids[:, t:t + 1]
        out = model(tok, states=states, position_offset=position)
        states = out["states"]
        position += 1

    generated = []
    current = input_ids[:, -1:]
    for _ in range(max_new_tokens):
        out = model(current, states=states, position_offset=position)
        logits = out["logits"][:, -1, :]
        states = out["states"]
        position += 1
        next_id = int(torch.argmax(logits, dim=-1).item())
        generated.append(next_id)
        current = torch.tensor([[next_id]], dtype=torch.long)
        if next_id == fact_id:  # ran into another structure marker: stop
            break

    words = [id_to_word.get(i, f"<id{i}>") for i in generated]
    model.train()
    return " ".join(words)


# ===========================================================================
# Main training loop
# ===========================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=int, default=300, help="total training chunks (both arms combined)")
    ap.add_argument("--chunk-size", type=int, default=32, help="LM-arm chunk size in tokens")
    ap.add_argument("--mix", type=float, default=0.5, help="fraction of chunks that are struct->text pairs")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--d-model", type=int, default=256, help="HSSLM d_model (d256 per lead's spec)")
    ap.add_argument("--core", choices=["s6", "gssm"], default="s6",
                     help="HSSLM core recurrence: s6 (Mamba-style, default) or gssm (GSSM-SELECTIVE)")
    ap.add_argument("--pairs", default=os.path.join(REPO_ROOT, "hsslm", "data", "graph_to_text_pairs.jsonl"))
    ap.add_argument("--out-prefix", default=os.path.join(REPO_ROOT, "results", "hsslm_speak"))
    ap.add_argument("--resume", default=None)
    ap.add_argument("--ckpt-every-chunks", type=int, default=100)
    ap.add_argument("--status-every-s", type=float, default=10.0)
    ap.add_argument("--gate-q", type=float, default=0.75)
    ap.add_argument("--sample-after-smoke", action="store_true",
                     help="greedy-decode 3 structure inputs after training and print them")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(1)

    status_path = args.out_prefix + "_status.json"
    ckpt_path = args.out_prefix + "_ckpt.pt"
    metrics_path = args.out_prefix + "_metrics.jsonl"

    print("=" * 74)
    print("SPEAK-TRAIN: HSSLM d{} core={}  mix={}  chunks={}".format(
        args.d_model, args.core, args.mix, args.chunks))
    print("=" * 74)

    # --- Vocabulary + WT-103 text (for both arms) --------------------------
    print("\n[1] Loading WT-103 + building extended vocabulary (Organism 5000+2 + <fact>/<say>)...")
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
    text = "\n\n".join(ds["train"]["text"][:20000])[:20_000_000]
    stoi, unk_id, mask_id, fact_id, say_id, total_ids = build_extended_vocab(text)
    id_to_word = {v: k for k, v in stoi.items()}
    lm_ids = tokenize_text(text, stoi, unk_id)
    print(f"    vocab: {mask_id - 1} words + unk({unk_id}) + mask({mask_id}) + "
          f"fact({fact_id}) + say({say_id}) = {total_ids} total IDs")
    print(f"    LM stream tokens: {len(lm_ids):,}")

    # --- Struct->text pairs -------------------------------------------------
    print(f"\n[2] Loading struct->text pairs from {args.pairs} ...")
    pairs = load_pairs(args.pairs)
    print(f"    n pairs: {len(pairs)}")
    struct_examples = []
    for p in pairs:
        struct_ids = tokenize_structure(p["structure"], stoi, unk_id, fact_id, say_id)
        text_ids = tokenize_text(p["text"], stoi, unk_id)
        # training sequence: structure tokens followed by text tokens, next-
        # token objective over the WHOLE sequence (structure->text
        # continuation is exactly the behavior we want the model to learn --
        # predicting the next word of a sentence GIVEN the structure prefix).
        seq = struct_ids + text_ids
        if len(seq) < 2:
            continue
        struct_examples.append(seq)
    print(f"    n usable (>=2 tokens): {len(struct_examples)}")

    # --- Model + optimizer ---------------------------------------------------
    print(f"\n[3] Building HSSLM d_model={args.d_model} vocab_size={total_ids}...")
    config = HSSLMConfig()
    config.vocab_size = total_ids
    config.d_model = args.d_model
    config.core = args.core
    config.hierarchical = False
    config.dropout = 0.0
    config.pad_token_id = -1
    model = HSSLM(config)
    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"    parameters: {n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95), weight_decay=0.1)
    streamer = HSSLMStreamer(
        model, opt, vocab_size=total_ids, pad_token_id=-1,
        gate_q=args.gate_q, grad_clip=1.0,
    )

    struct_idx = 0
    wall_off = 0.0
    if args.resume and os.path.exists(args.resume):
        print(f"\n[resume] loading checkpoint {args.resume} ...")
        struct_idx, wall_off = load_checkpoint(args.resume, model, opt, streamer)
        print(f"    resumed at struct_idx={struct_idx}  wall_off={wall_off:.1f}s  "
              f"streamer.n_chunks={streamer.n_chunks}  streamer.position={streamer.position}")

    schedule = mix_schedule(args.chunks, args.mix)
    n_struct_chunks = sum(schedule)
    n_lm_chunks = len(schedule) - n_struct_chunks
    print(f"\n[4] Mix schedule: {n_struct_chunks} struct chunks, {n_lm_chunks} LM chunks "
          f"(mix={args.mix})")

    # --- Training loop --------------------------------------------------------
    print(f"\n[5] Training {args.chunks} chunks...")
    t0 = time.time()
    wall = lambda: time.time() - t0 + wall_off
    lm_pos = 0  # cursor into lm_ids
    lm_losses, struct_losses = [], []
    last_status = last_ckpt = time.time()
    status = {"pid": os.getpid(), "phase": "running",
              "config": vars(args), "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    def write_status():
        st = dict(status)
        st.update({
            "wall_s": round(wall(), 1),
            "n_chunks_done": chunk_idx + 1,
            "n_chunks_total": args.chunks,
            "rss_gb": round(_rss_gb(), 3),
            "disk_free_gb": round(shutil.disk_usage(".").free / 1e9, 1),
            "arms": {
                "lm": {"n_chunks": streamer.n_chunks, "n_bwd": streamer.n_bwd,
                       "grad_tokens": streamer.grad_tokens,
                       "gate_rate": streamer.n_bwd / max(1, streamer.n_chunks)},
                "struct": {"n_chunks": struct_idx, "n_pairs_total": len(struct_examples)},
            },
            "lm_loss_recent": lm_losses[-1] if lm_losses else None,
            "struct_loss_recent": struct_losses[-1] if struct_losses else None,
        })
        write_atomic(status_path, st)

    for chunk_idx in range(args.chunks):
        is_struct = schedule[chunk_idx]

        if is_struct and struct_examples:
            # --- Struct->text arm: always trained, no gate ------------------
            seq = struct_examples[struct_idx % len(struct_examples)]
            struct_idx += 1
            x = torch.tensor([seq[:-1]], dtype=torch.long)
            y = torch.tensor([seq[1:]], dtype=torch.long)
            out = model(x, labels=y, states=None, position_offset=0)
            loss = out["loss"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            struct_losses.append(float(loss.item()))
            rec = {"type": "struct", "chunk": chunk_idx, "loss": round(float(loss.item()), 4)}
        else:
            # --- LM streaming arm: gated (HSSLMStreamer) ---------------------
            if lm_pos + args.chunk_size + 1 > len(lm_ids):
                lm_pos = 0  # wrap
            chunk_ids = lm_ids[lm_pos:lm_pos + args.chunk_size + 1]
            lm_pos += args.chunk_size
            x = torch.tensor([chunk_ids[:-1]], dtype=torch.long)
            y = torch.tensor([chunk_ids[1:]], dtype=torch.long)
            surprise, gated, _ = streamer.step_gated(x, y)
            lm_losses.append(surprise)
            rec = {"type": "lm", "chunk": chunk_idx, "surprise": round(surprise, 4), "gated": gated}

        with open(metrics_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

        now = time.time()
        if now - last_status >= args.status_every_s:
            write_status()
            last_status = now
        if (chunk_idx + 1) % args.ckpt_every_chunks == 0:
            save_checkpoint(ckpt_path, model, opt, streamer, struct_idx, wall(), vars(args))
            last_ckpt = now

        if chunk_idx % 50 == 0 or chunk_idx == args.chunks - 1:
            lm_recent = sum(lm_losses[-10:]) / max(1, len(lm_losses[-10:])) if lm_losses else float("nan")
            st_recent = sum(struct_losses[-10:]) / max(1, len(struct_losses[-10:])) if struct_losses else float("nan")
            print(f"  chunk {chunk_idx:4d}  type={'struct' if is_struct else 'lm   '}  "
                  f"lm_loss(recent10)={lm_recent:.3f}  struct_loss(recent10)={st_recent:.3f}  "
                  f"gate_rate={streamer.n_bwd/max(1,streamer.n_chunks):.3f}")

    status["phase"] = "done"
    write_status()
    save_checkpoint(ckpt_path, model, opt, streamer, struct_idx, wall(), vars(args), extra={"final": True})

    print("\n" + "=" * 74)
    print("SUMMARY")
    print("=" * 74)
    print(f"n_params: {n_params:,}")
    print(f"LM arm:     n_chunks={streamer.n_chunks}  n_bwd={streamer.n_bwd}  "
          f"gate_rate={streamer.n_bwd/max(1,streamer.n_chunks):.3f}")
    if lm_losses:
        print(f"LM surprise: first10={sum(lm_losses[:10])/min(10,len(lm_losses)):.3f}  "
              f"last10={sum(lm_losses[-10:])/min(10,len(lm_losses[-10:])):.3f}")
    print(f"Struct arm: n_chunks={struct_idx}")
    if struct_losses:
        print(f"Struct loss: first10={sum(struct_losses[:10])/min(10,len(struct_losses)):.3f}  "
              f"last10={sum(struct_losses[-10:])/min(10,len(struct_losses[-10:])):.3f}")
    print(f"wall_s: {wall():.1f}")
    print(f"checkpoint: {ckpt_path}")
    print(f"status: {status_path}")
    print(f"metrics: {metrics_path}")

    if args.sample_after_smoke:
        print("\n" + "=" * 74)
        print("SAMPLES (greedy decode, 3 structure inputs)")
        print("=" * 74)
        sample_structs = [pairs[i]["structure"] for i in (0, len(pairs) // 2, len(pairs) - 1)]
        for s in sample_structs:
            decoded = greedy_decode_structure(model, s, stoi, unk_id, fact_id, say_id, id_to_word)
            print(f"\nSTRUCTURE: {s}")
            print(f"GENERATED: {decoded}")


if __name__ == "__main__":
    main()
