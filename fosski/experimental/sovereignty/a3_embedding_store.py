"""
A3EmbeddingStore -- Souveränitäts-Probe (revival-probe Task 19).

Serves core.reservoir_lm.EmbeddingStore's exact interface (encode, dim,
nearest, _load, emb property, _token2id) from the A3 organism's OWN
learned embed.weight (128d, 5002 rows: 5000-word frequency vocab + UNK +
MASK) instead of Qwen3-1.7B's frozen, pretrained 512d SVD-projected
embeddings. No Qwen data enters this path at any point -- that is the
entire point of the swap; see the module docstring's "SOVEREIGNTY
CONTRACT" section below for exactly what is and is not verified about
that claim.

================================================================
DIMENSION DESIGN -- the decision, and why (documented, not a placeholder)
================================================================
Qwen3-1.7B: 2048d raw -> SVD-projected to 512d (core/reservoir_lm.py's
EmbeddingStore, the ONLY path repl.py ever actually loads).
A3 organism (p78_reader_A3.pt): embed.weight is (5002, 128) -- 128d
native, no projection of any kind.

Three options were on the table (per the assignment):
  (a) Pad A3's 128d vectors to 512d with zeros.
  (b) A fixed-seed random projection A3's 128d -> 512d (Johnson-
      Lindenstrauss style, mirroring EmbeddingStore's OWN fallback
      projection path for when Qwen's SVD files are missing -- see
      reservoir_lm.py:59-64).
  (c) Make every consumption site dimension-agnostic: read dim from the
      embedding store itself rather than assuming 512.

CHOSEN: (c), because the kartierung (mapping) already found it is MOSTLY
free -- core/reservoir_lm.py's build_reservoir_lm() already derives
ReservoirLM's input_dim from store.dim, not a hardcoded 512
(`dim = store.dim; reservoir = ReservoirLM(input_dim=dim, ...)`,
reservoir_lm.py:409-410), and every consumption site the mapping found
in core/foss_pipeline.py and core/hopfield_bank.py reads
`self.emb_store.dim`, never a bare `512` constant -- WITH ONE EXCEPTION,
documented honestly below.

Options (a)/(b) were rejected because they would ADD information A3
never learned (512-64=448 zero-padded dimensions, or a random linear
mixing that Qwen's SVD-informed projection is NOT -- inventing a fake
"512d-shaped" vector for A3 would make the comparison dishonest: it
would test "does padding/projecting A3's real 128d signal into a bigger
shape help", not "does A3's own embedding replace Qwen's". Option (c)
is the one that tests what the assignment actually asks: can the
reservoir/attention/hopfield MACHINERY run on A3's own signal, at A3's
own native dimension, unmodified in its own logic.

ONE HONEST EXCEPTION FOUND DURING MAPPING, not resolved by (c):
core/foss_pipeline.py:_residual_hopfield_retrieve (line ~765-785) calls
`self.residual_hopfield.retrieve_from_512d(query_512, ...)` -- a method
NAME-hardcoded to 512d (core/residual_hopfield.py stores pre-extracted
Qwen transformer LAYER-18 RESIDUAL STATES, a completely different
Qwen-derived artifact from the embedding table itself, with no A3
equivalent to substitute -- A3 is a from-scratch organism with no
"layer 18 of a pretrained transformer" to have residuals of). This path
is DISABLED, not dimension-fixed, when running under A3 (see
build_reservoir_lm_a3's docstring below) -- fixing it would require
either a fake 512d vector (rejected per the reasoning above) or building
an A3-native residual-cache equivalent, which is out of scope for this
probe (no such artifact exists to build one from -- A3 has no "layer
18", it has 2 layers per its own config).

================================================================
SOVEREIGNTY CONTRACT -- what "no Qwen data" means and how it's checked
================================================================
This module never imports, opens, or references any qwen3_*.npy/.json
file. Its ONLY external input is p78_reader_A3.pt (the organism
checkpoint) and a vocabulary independently reconstructed via
length_extrap_v2.build_vocab() over the SAME WikiText-2 corpus
pos_run.py trained A3 against (see build_a3_vocab() below) -- the
vocabulary itself is not stored IN the checkpoint (checked directly:
ck['extra'] holds only {'source', 'arm', 'life'} provenance strings, no
vocab list), so it is reconstructed deterministically from the training
corpus + the exact same build_vocab() call pos_run.py itself made
(same VOCAB_MAX=5000, same frequency-sort, same regex tokenization) --
verified to produce exactly 5000 entries + UNK(5000) + MASK(5001),
matching embed.weight's (5002, 128) row count exactly.
"""
import json
import os

import numpy as np


class A3EmbeddingStore:
    """Same public interface as core.reservoir_lm.EmbeddingStore --
    encode(token), dim, nearest(vec, top_k), emb property, _load(),
    _token2id -- backed by A3's own learned embed.weight instead of
    Qwen3's pretrained, frozen embeddings."""

    def __init__(self, checkpoint_path, corpus_cache_dir=None):
        self._checkpoint_path = checkpoint_path
        self._corpus_cache_dir = corpus_cache_dir
        self._emb = None
        self._emb_normed = None
        self._token2id = None
        self._id2token = None
        self.dim = None
        self.unk_idx = None
        self.mask_idx = None

    def _load(self):
        if self._emb is not None:
            return
        import torch

        ck = torch.load(self._checkpoint_path, map_location="cpu", weights_only=False)
        weight = ck["organism"]["model"]["embed.weight"]
        self._emb = weight.detach().cpu().numpy().astype(np.float32)
        self.dim = self._emb.shape[1]

        norms = np.linalg.norm(self._emb, axis=1, keepdims=True)
        self._emb_normed = self._emb / np.clip(norms, 1e-8, None)

        vocab, stoi, unk_idx, mask_idx = build_a3_vocab()
        if self._emb.shape[0] != len(vocab) + 2:
            raise ValueError(
                "A3 embed.weight has {} rows but the reconstructed vocab "
                "has {} entries (+2 for UNK/MASK = {}) -- vocabulary "
                "reconstruction does not match this checkpoint; refusing "
                "to silently misalign token strings to the wrong "
                "embedding rows.".format(
                    self._emb.shape[0], len(vocab), len(vocab) + 2))
        self._token2id = stoi
        self._id2token = {v: k for k, v in stoi.items()}
        self.unk_idx = unk_idx
        self.mask_idx = mask_idx

    @property
    def emb(self):
        self._load()
        return self._emb

    @property
    def vocab_size(self):
        self._load()
        return self._emb.shape[0]

    def encode(self, token):
        """Token string -> embedding vector. A3's vocab is lowercase,
        alphabetic-only (length_extrap_v2.build_vocab's regex is
        `[a-zA-Z]+`, applied after .lower()) -- no BPE variants to try
        (unlike Qwen's subword tokenizer), just the one normalized form.
        Multi-word input is split and averaged, same contract as
        EmbeddingStore.encode()."""
        self._load()
        if ' ' in token or '_' in token:
            words = token.replace('_', ' ').split()
            vecs = []
            for w in words:
                v = self.encode(w)
                if v is not None:
                    vecs.append(v)
            if vecs:
                return np.mean(vecs, axis=0).astype(np.float32)
            return None
        tid = self._token2id.get(token.lower())
        if tid is not None:
            return self._emb[tid]
        return None

    def encode_text(self, text):
        self._load()
        words = text.lower().split()
        vecs = []
        for w in words:
            v = self.encode(w)
            if v is not None:
                vecs.append(v)
        if not vecs:
            return np.zeros(self.dim, dtype=np.float32)
        return np.mean(vecs, axis=0)

    def nearest(self, vec, top_k=10):
        self._load()
        vec_norm = vec / (np.linalg.norm(vec) + 1e-8)
        sims = self._emb_normed @ vec_norm
        top_idx = np.argpartition(sims, -top_k)[-top_k:]
        top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]
        return [(self._id2token.get(int(i), '?'), float(sims[i])) for i in top_idx]


def build_a3_vocab():
    """Deterministically reconstructs the exact vocabulary A3's training
    (pos_run.py) built via length_extrap_v2.build_vocab(load_wikitext2()[0])
    -- same corpus (Salesforce/wikitext, wikitext-2-raw-v1, offline HF
    cache), same VOCAB_MAX=5000, same frequency-sort (Python's sorted()
    is stable; dict insertion order is preserved since 3.7 -- this is
    deterministic given the same corpus, not a heuristic re-derivation).
    Returns (vocab_list, stoi_dict, unk_idx, mask_idx)."""
    import sys
    sys.path.insert(0, os.path.expanduser("~/livecausal_bridge"))
    from length_extrap_v2 import load_wikitext2, build_vocab  # noqa: E402

    train_text, _val_text = load_wikitext2()
    vocab, stoi, unk_idx, mask_idx = build_vocab(train_text)
    return vocab, stoi, unk_idx, mask_idx


def build_reservoir_lm_a3(checkpoint_path=None):
    """A3 counterpart to core.reservoir_lm.build_reservoir_lm() -- same
    return contract (ReservoirLM, EmbeddingStore-compatible object), same
    dimension-derivation logic (ReservoirLM's input_dim comes from
    store.dim, which is 128 here instead of 512 -- nothing in
    ReservoirLM's own construction assumes a fixed dimension, per the
    mapping in this module's docstring).

    NOTE on core/foss_pipeline.py's _residual_hopfield_retrieve: this
    function does NOT attempt to make that path work under A3 -- it has
    no A3 equivalent (see the module docstring's "ONE HONEST EXCEPTION"
    section). A caller wiring this into FossPipeline.configure() should
    pass residual_hopfield=None to disable that specific sub-path, which
    is what the repl.py swap flag built for this probe does.
    """
    if checkpoint_path is None:
        checkpoint_path = os.path.expanduser(
            "~/livecausal_bridge/p78_reader_A3.pt")
    if not os.path.exists(checkpoint_path):
        return None, None

    from core.reservoir_lm import ReservoirLM

    store = A3EmbeddingStore(checkpoint_path)
    store._load()
    dim = store.dim

    reservoir = ReservoirLM(input_dim=dim, reservoir_size=2048)
    return reservoir, store
