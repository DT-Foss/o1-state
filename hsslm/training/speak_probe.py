"""Qualitative decode probe for a running speak-train life.

Loads a COPY of the trainer's checkpoint (never the live file), rebuilds the
exact training vocabulary, and greedy-decodes (a) a few struct->text pairs
from the training file, (b) hand-written NOVEL structures the model has never
seen, (c) a free LM continuation from a plain-text prompt. Read-only with
respect to the life: no optimizer, no streamer state mutation on disk.

Usage (on the machine that hosts the life):
    python3 hsslm/training/speak_probe.py --ckpt results/probe_copies/speak_life_s6_ckpt.pt --core s6
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

import speak_train as st
from speak_train import HSSLM, HSSLMConfig

NOVEL_STRUCTURES = [
    "<fact> heavy rain | flooded | the river valley <say>",
    "<fact> the new engine | improved | fuel efficiency <say>",
    "<fact> the treaty | ended | the long war <say>",
]


@torch.no_grad()
def greedy_decode_banned(model, prompt_ids, id_to_word, banned_ids, stop_id=None,
                         max_new_tokens=40):
    """Greedy decode with a banned-ID set (UNK/mask): argmax over the REAL
    word distribution. Also reports the mean probability mass the model put
    on the banned IDs — the quantity the ban hides."""
    model.eval()
    input_ids = torch.tensor([prompt_ids], dtype=torch.long)
    states = None
    position = 0
    for t in range(input_ids.shape[1]):
        out = model(input_ids[:, t:t + 1], states=states, position_offset=position)
        states = out["states"]
        position += 1
    generated = []
    banned_mass = []
    current = input_ids[:, -1:]
    for _ in range(max_new_tokens):
        out = model(current, states=states, position_offset=position)
        logits = out["logits"][:, -1, :].clone()
        states = out["states"]
        position += 1
        probs = torch.softmax(logits, dim=-1)
        banned_mass.append(float(sum(probs[0, b] for b in banned_ids)))
        for b in banned_ids:
            logits[0, b] = float("-inf")
        next_id = int(torch.argmax(logits, dim=-1).item())
        generated.append(next_id)
        current = torch.tensor([[next_id]], dtype=torch.long)
        if stop_id is not None and next_id == stop_id:
            break
    text = " ".join(id_to_word.get(i, f"<id{i}>") for i in generated)
    return text, sum(banned_mass) / len(banned_mass)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to a COPY of the life's checkpoint")
    ap.add_argument("--core", choices=["s6", "gssm"], required=True)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--pairs", default=os.path.join(st.REPO_ROOT, "hsslm", "data", "graph_to_text_pairs.jsonl"))
    ap.add_argument("--pair-indices", default="0,500,1500")
    ap.add_argument("--lm-prompt", default="the game was released in")
    ap.add_argument("--max-new", type=int, default=40)
    ap.add_argument("--out", default=None, help="optional JSON output path")
    args = ap.parse_args()

    torch.set_num_threads(1)

    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
    text = "\n\n".join(ds["train"]["text"][:20000])[:20_000_000]
    stoi, unk_id, mask_id, fact_id, say_id, total_ids = st.build_extended_vocab(text)
    id_to_word = {v: k for k, v in stoi.items()}

    config = HSSLMConfig()
    config.vocab_size = total_ids
    config.d_model = args.d_model
    config.core = args.core
    config.hierarchical = False
    config.dropout = 0.0
    config.pad_token_id = -1
    model = HSSLM(config)

    ck = torch.load(args.ckpt, weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()
    meta = {
        "core": args.core,
        "struct_idx": ck.get("struct_idx"),
        "wall_s": round(ck.get("wall_s", 0.0), 1),
    }
    print(f"[probe] core={args.core} ckpt struct_idx={meta['struct_idx']} wall={meta['wall_s']}s")

    pairs = st.load_pairs(args.pairs)
    results = {"meta": meta, "seen_pairs": [], "novel": [], "lm": None}
    banned = [unk_id, mask_id]

    for idx in [int(i) for i in args.pair_indices.split(",")]:
        p = pairs[idx]
        prompt_ids = st.tokenize_structure(p["structure"], stoi, unk_id, fact_id, say_id)
        decoded, unk_mass = greedy_decode_banned(
            model, prompt_ids, id_to_word, banned, stop_id=fact_id,
            max_new_tokens=args.max_new)
        results["seen_pairs"].append({"idx": idx, "structure": p["structure"],
                                      "target_head": " ".join(p["text"].split()[:15]),
                                      "decoded": decoded, "unk_mass": round(unk_mass, 4)})
        print(f"\n[seen {idx}] {p['structure']}  (unk_mass={unk_mass:.3f})\n  target : {results['seen_pairs'][-1]['target_head']} ...\n  decoded: {decoded}")

    for s in NOVEL_STRUCTURES:
        prompt_ids = st.tokenize_structure(s, stoi, unk_id, fact_id, say_id)
        decoded, unk_mass = greedy_decode_banned(
            model, prompt_ids, id_to_word, banned, stop_id=fact_id,
            max_new_tokens=args.max_new)
        results["novel"].append({"structure": s, "decoded": decoded,
                                 "unk_mass": round(unk_mass, 4)})
        print(f"\n[novel] {s}  (unk_mass={unk_mass:.3f})\n  decoded: {decoded}")

    prompt_ids = st.tokenize_text(args.lm_prompt, stoi, unk_id)
    lm_out, unk_mass = greedy_decode_banned(
        model, prompt_ids, id_to_word, banned, max_new_tokens=args.max_new)
    results["lm"] = {"prompt": args.lm_prompt, "continuation": lm_out,
                     "unk_mass": round(unk_mass, 4)}
    print(f"\n[lm] '{args.lm_prompt}' (unk_mass={unk_mass:.3f}) -> {lm_out}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
