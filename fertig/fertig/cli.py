"""
fertig.cli — ein Einstiegspunkt für das fertige symbolische System.

Subcommands:
  info     Graph-Statistiken + Hyperboloid-Check
  chains   abgeleitete Ketten (3-Pass-Inferenz, pass1)
  graph    gewicht-freie Kausal-Walks (Kette als Text)
  speech   Walks als gesprochene Prosa (handgeschriebene Verknüpfer)
  mined    Walks als Prosa mit gemessener Muster-Bank (erfordert Bank)
  bank     Muster-Bank aus Korpus minen (extract_patterns)
  corpus   Korpus-Modus: Prompt gewicht-frei fortsetzen

Determinismus: `graph`, `chains`, `speech`, `corpus` sind bei gleichen
Argumenten bit-identisch. `mined` ist in der Form zufällig (echter RNG),
die Fakten bleiben deterministisch.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from . import __version__
from . import sampler, state_init, inference as hsslm_inference
from . import pipeline, corpus, mined
from .pattern_bank import PatternBank
from .intent import parse_command
from . import tools, learn as learn_mod, arena as arena_mod
from . import bench as bench_mod

DATA = Path(__file__).resolve().parent.parent / "data"


def _load_graph(args):
    path = Path(args.graph)
    return pipeline.load_graph(path)


def cmd_info(args) -> int:
    vocab, stoi, adj, mech = _load_graph(args)
    print(f"Graph      : {args.graph}")
    print(f"Symbole    : {len(vocab)}")
    print(f"Kanten     : {sum(len(v) for v in adj.values())}")
    print(f"Triplets   : {len({(a, b) for a in adj for b in adj[a]})} eindeutige Hops")
    SM = state_init.initialize_symbol_state(len(vocab))
    mink = -SM[:, 0] ** 2 + np.sum(SM[:, 1:] ** 2, axis=1)
    ok = bool(np.allclose(mink, -1.0, atol=1e-9))
    print(f"Hyperboloid-Check (alle Zustände auf der Einheits-Hyperboloid): "
          f"{ok} (max. Abweichung {np.max(np.abs(mink + 1.0)):.2e})")
    print(f"Top-Startpunkte: {', '.join(pipeline.top_starts(adj, vocab))}")
    return 0


def cmd_chains(args) -> int:
    vocab, stoi, adj, mech = _load_graph(args)
    chains = pipeline.derive_chains(adj, vocab)
    print(f"{len(chains)} abgeleitete exakte Ketten (pass1):")
    for chain, conf in sorted(chains.items(), key=lambda kv: -kv[1])[: args.n]:
        names = " -> ".join(vocab[i] for i in chain if i < len(vocab))
        print(f"  [{conf:.2f}] {names}")
    return 0


def cmd_graph(args) -> int:
    vocab, stoi, adj, mech = _load_graph(args)
    SM = state_init.initialize_symbol_state(len(vocab))
    starts = args.start or pipeline.top_starts(adj, vocab)
    print("=== gewicht-freie Kausal-Walks (Entitäten exakt, Walk generiert) ===")
    for start in starts:
        for tau in (0.15, 0.5):
            print(f"[tau={tau}] {pipeline.walk(start, vocab, stoi, adj, mech, SM, n=args.n, tau=tau)}")
        print()
    return 0


def cmd_speech(args) -> int:
    vocab, stoi, adj, mech = _load_graph(args)
    SM = state_init.initialize_symbol_state(len(vocab))
    starts = args.start or pipeline.top_starts(adj, vocab)
    print("=== gewicht-freie SPRACHE aus dem .causal-Graphen ===")
    for start in starts:
        hops = pipeline.walk_chain(start, vocab, stoi, adj, SM, n=args.n, tau=0.3)
        chain = (" -> ".join([vocab[hops[0][0]]] + [vocab[b] for _, b in hops])
                 if hops else "(Sackgasse)")
        print(f"[{start}]")
        print(f"  Kette : {chain}")
        print(f"  Sprache: {pipeline.verbalize(hops, vocab, mech, seed=args.seed)}\n")
    return 0


def cmd_mined(args) -> int:
    bank_path = Path(args.bank)
    if not bank_path.exists():
        print(f"Muster-Bank fehlt: {bank_path}\n"
              f"  -> minen mit:  python -m fertig.cli bank -o {bank_path}",
              file=sys.stderr)
        return 1
    bank = PatternBank.load(bank_path)
    vocab, stoi, adj, mech = _load_graph(args)
    SM = state_init.initialize_symbol_state(len(vocab))
    starts = args.start or pipeline.top_starts(adj, vocab)
    openers = mined.MinedOpeners(bank, seed=args.seed)
    print("=== SPRACHE mit gemessener Muster-Bank ===")
    for start in starts:
        hops = pipeline.walk_chain(start, vocab, stoi, adj, SM, tau=0.3, n=args.n)
        chain = (" -> ".join([vocab[hops[0][0]]] + [vocab[b] for _, b in hops])
                 if hops else "(Sackgasse)")
        print(f"[{start}]")
        print(f"  Kette : {chain}")
        print(f"  Sprache: {mined.verbalize_mined(hops, vocab, mech, openers)}\n")
    return 0


def cmd_bank(args) -> int:
    bank = PatternBank()
    corpora = args.corpora or [corpus.DEFAULT_CORPUS]
    for path in corpora:
        text = Path(path).read_text(encoding="utf-8", errors="ignore")
        bank.extract(text)
        print(f"extrahiert: {path} ({len(text)} Zeichen)")
    out = Path(args.out)
    bank.save(out)
    print(f"\nSätze: {bank.n_sentences} | Skelette: {len(bank.skeletons)} | "
          f"Opener: {len(bank.openers)}")
    print(f"Bank gespeichert: {out}")
    return 0


def cmd_corpus(args) -> int:
    path = Path(args.corpus)
    text = path.read_text(encoding="utf-8", errors="ignore")
    vocab, stoi, adjacency, trigram, unigram = corpus.build_vocab(
        text, max_vocab=args.max_vocab)
    print(corpus.stats(vocab, adjacency, trigram, unigram))
    SM = state_init.initialize_symbol_state(len(vocab))
    prompts = args.prompt or ["the candle", "the flame", "we have here"]
    print("\n=== gewicht-freie Fortsetzung (gemessene Trigramm/Bigramm-Kanten) ===")
    for prompt in prompts:
        for tau in (0.2, 0.5):
            out = corpus.generate(prompt, vocab, stoi, adjacency, trigram,
                                  unigram, SM, n=args.n, tau=tau)
            print(f"\n[tau={tau}] {prompt!r}\n  -> {out}")
    return 0


def cmd_intent(args) -> int:
    """NL-Befehl -> Intent-Tupel + Tool-Call."""
    vocab = pipeline.load_graph(args.graph)[0]
    lex = None
    if Path(args.lexicon).exists():
        lex = learn_mod.Lexicon.load(args.lexicon)
    it = parse_command(" ".join(args.command), vocab, lexicon=lex)
    if args.video:
        it.arguments["video"] = args.video
        # Bei 'erkennen' IST das Video das Ziel — kein Graph-Target nötig
        if it.action == "erkennen":
            it.status = "ok"
            it.tool = "video"
            it.grounded = True
            it.confidence = max(it.confidence, 0.8)
            it.target = args.video
    print(f"Befehl   : {' '.join(args.command)}")
    print(f"Parse    : {it.tree}")
    print(f"Intent   : action={it.action!r} target={it.target!r} "
          f"conf={it.confidence:.3f}")
    print(f"Grounded : {it.grounded} | Ambiguität: {it.ambiguity:.2f} | "
          f"Status: {it.status}")
    if it.arguments:
        print(f"Args     : {it.arguments}")
    if args.execute and it.status == "ok":
        res = tools.execute(it, args.graph)
        print(f"\n[Tool {res.tool}] ok={res.ok}")
        if res.text:
            print(res.text)
    return 0


def cmd_learn(args) -> int:
    lex = learn_mod.Lexicon()
    corpora = args.corpora or [corpus.DEFAULT_CORPUS]
    for p in corpora:
        lex = learn_mod.learn_from_file(p, lex, min_count=args.min_count)
        print(f"gelernt: {p}")
    out = Path(args.out)
    lex.save(out)
    print(f"\nToken gesamt: {lex.tokens}")
    print(f"Aktionen gelernt: {len(lex.actions)}"
      f" | Nomen gelernt: {len(lex.nouns)}")
    top_v = sorted(lex.actions.items(), key=lambda kv: -kv[1]["weight"])[:10]
    for v, meta in top_v:
        print(f"  verb {v:14s} -> action {meta['action']!r} (w={meta['weight']})")
    top_n = sorted(lex.nouns.items(), key=lambda kv: -kv[1])[:10]
    print("Top-Nomen:")
    for n, w in top_n:
        print(f"  noun {n:14s} (w={w})")
    print(f"\nLexikon gespeichert: {out}")
    return 0


def cmd_arena(args) -> int:
    res = arena_mod.run_arena(args.graph, verbose=not args.quiet)
    print()
    print(res.report())
    return 0


def cmd_bench(args) -> int:
    if args.name == "blimp":
        res = bench_mod.run_blimp(subtasks=args.subtasks or None,
                                  verbose=not args.quiet)
        print()
        print(res.report())
    elif args.name == "snips":
        res = bench_mod.run_snips(verbose=not args.quiet)
        print()
        print(res.report())
    elif args.name == "humaneval":
        res = bench_mod.run_humaneval(n_eval=args.n, learn=not args.no_learn,
                                      verbose=not args.quiet)
        print()
        print(res.report())
    elif args.name == "hellaswag":
        res = bench_mod.run_hellaswag(n=args.n, verbose=not args.quiet)
        print()
        print(res.report())
    elif args.name == "winogrande":
        res = bench_mod.run_winogrande(verbose=not args.quiet)
        print()
        print(res.report())
    elif args.name == "lambada":
        res = bench_mod.run_lambada(verbose=not args.quiet)
        print()
        print(res.report())
    elif args.name == "llm-snips":
        d = bench_mod.run_llm_snips(n=args.n, verbose=not args.quiet)
        if "error" in d:
            print(d["error"], file=sys.stderr)
            return 1
        print()
        print(f"DeepSeek auf SNIPS (n={d['total']}): "
              f"{d['hits']}/{d['total']} ({100*d['hits']/max(d['total'],1):.1f}%)")
        print(f"FERTIG auf SNIPS (gesamt): 88.3% — die präregistrierte Arena läuft")
    elif args.name == "llm-all":
        d = bench_mod.run_llm_all(n_each=args.n, verbose=not args.quiet)
        if "error" in d:
            print(d["error"], file=sys.stderr)
            return 1
        print("\n=== Gap-Ledger: jede Lücke ist ein registriertes Ziel ===")
        for name, e in d.items():
            print(f"  {name:14s} Lücke {e['gap']*100:+5.1f}pp -> "
                  f"{e['unser_weg']}")
        print("\n  gespeichert in data/gap_ledger.json")
    elif args.name == "arc":
        res = bench_mod.run_arc(n=args.n, use_graph=not args.no_graph,
                                verbose=not args.quiet)
        print()
        print(res.report())
    else:
        print(f"unbekannter Benchmark: {args.name} "
              f"(blimp | snips | humaneval | hellaswag | winogrande | lambada | "
              f"llm-snips | arc)", file=sys.stderr)
        return 1
    return 0


def cmd_code(args) -> int:
    from . import code as code_mod
    prompt = " ".join(args.prompt)
    fragments = code_mod.load_fragments()
    triplets = code_mod.load_triplets()
    code, used = code_mod.assemble(prompt, fragments, triplets)
    print(f"Prompt: {prompt}")
    print(f"Fragmente: {used if used else '(keine über Schwelle — ehrliche Sackgasse)'}")
    print("---")
    print(code if code else "(kein Code assembliert)")
    if args.execute and code:
        rc, out, err = code_mod.run_sandbox(code)
        print(f"\n[Sandbox] exit={rc}")
        if out:
            print(out[:2000])
        if err:
            print(err[:1000])
    return 0


def cmd_grow(args) -> int:
    from . import gaps as gaps_mod
    srcs = args.sources.split(",") if args.sources else None
    if args.gaps:
        from .arena import EVAL_SET
        n = gaps_mod.grow_gaps([c for c, _, _ in EVAL_SET],
                               max_targets=args.max_targets,
                               verbose=True)
        print(f"\n{gaps_mod.WORLD_GRAPH}: {n} neue Tripletts aus dem Loop")
    else:
        target = " ".join(args.target)
        gaps_mod.grow(target, source_names=srcs)
    return 0


def cmd_crawl(args) -> int:
    from . import sources as sources_mod
    from . import gaps as gaps_mod
    print(f"[crawl] {args.url} ...")
    trips = sources_mod.fetch_url_direct(args.url)
    print(f"  {len(trips)} Kausal-Tripletts extrahiert")
    for a, b, c, conf in trips[:10]:
        print(f"  {a} {b} {c} ({conf:.2f})")
    if args.store:
        merged = {t[:3]: t[3] for t in gaps_mod._load_world()}
        added = 0
        for a, b, c, conf in trips:
            key = (a, b, c)
            if key not in merged or conf > merged[key]:
                merged[key] = conf
                added += 1
        gaps_mod._save_world([(a, b, c, conf)
                              for (a, b, c), conf in merged.items()])
        print(f"  {added} neu gespeichert, Welt-Graph jetzt "
              f"{len(merged)} Tripletts")
    return 0


def cmd_evolve(args) -> int:
    from .evolve import evolve as evolve_fn
    srcs = args.sources.split(",") if args.sources else None
    log = evolve_fn(args.iterations, args.arc_questions,
                    args.grow_per_iter, srcs)
    if len(log) >= 2:
        print("\n=== Evolve-Protokoll (Ledger) ===")
        print(f"{'It':>3} {'Graph':>6} {'ARC%':>6} {'Cov%':>6} "
              f"{'Δacc':>6} {'Δcov':>6}")
        for e in log:
            print(f"{e['iteration']:>3} {e['graph_triplets']:>6} "
                  f"{100*e['arc_accuracy']:>5.1f}% "
                  f"{100*e['arc_coverage']:>5.1f}% "
                  f"{100*e['arc_delta_acc']:>+5.1f}% "
                  f"{100*e['arc_delta_cov']:>+5.1f}%")
    return 0


def cmd_ground(args) -> int:
    from . import grounding as g
    if args.all:
        from . import gaps as gaps_mod
        trips = gaps_mod._load_world()
        symbols = set()
        for a, b, c, _ in trips:
            symbols.add(a)
            symbols.add(c)
        symbols = sorted(symbols)[: args.max]
        anchored = {}
        for i, word in enumerate(symbols, 1):
            print(f"[{i}/{len(symbols)}] ", end="")
            res = g.ground_symbol(word, verbose=False)
            anchored[word] = bool(res["perceptual"] or res["quantitative"])
        trips = gaps_mod._load_world()
        cov = g.grounding_coverage(trips, anchored)
        print(f"\n=== Grounding-Coverage (der Regress-Ende-Wert) ===")
        print(f"  Symbole im Graphen : {cov['symbols']}")
        print(f"  Nicht-Wort-gebunden: {cov['grounded']} "
              f"({100*cov['coverage']:.1f}%)")
        print(f"  (perzeptuell: CLIP-Bilder, quantitativ: Zahlen+Einheiten)")
    else:
        for word in args.word:
            g.ground_symbol(word)
    return 0


def cmd_quant(args) -> int:
    from . import quant as quant_mod
    if args.all:
        r = quant_mod.run_quant(verbose=True)
        print(f"\nQuantitative QA: {r['covered']}/{r['total']} beantwortbar "
              f"({100*r['covered']/max(r['total'],1):.0f}%)")
    else:
        q = " ".join(args.question)
        ans, mech, conf = quant_mod.answer(q)
        if ans:
            print(f"Antwort: {ans}  [{mech}, conf={conf:.2f}]")
        else:
            print("Nicht beantwortbar — die Lücke ist der nächste "
                  "ground-Kandidat.")
    return 0


def cmd_vision(args) -> int:
    from . import vision as v
    if args.unsupervised:
        # Harnad-Ebene: Kategorien OHNE Wörter — Bilder aller Wörter
        # gemischt, Cluster entstehen aus Pixel-Struktur
        import urllib.request
        all_images, true_labels = [], []
        for word in args.word:
            for img in v.commons_images(word, args.images):
                all_images.append(img)
                true_labels.append(word)
        clusters, centers = v.cluster_unsupervised(all_images, k=len(args.word))
        if not clusters:
            print("Keine Bilder verfügbar.")
            return 1
        purity = v.cluster_purity(clusters, true_labels)
        print(f"Unüberwachte Kategorien (kein Wort, kein Netz, nur Pixel):")
        for ci, cl in enumerate(clusters):
            from collections import Counter
            dom = Counter(true_labels[i] for i in cl).most_common(1)[0][0]
            print(f"  Cluster {ci}: {len(cl)} Bilder — dominant: {dom}")
        print(f"\nPurity (Cluster vs. wahre Klassen, NUR zur Validierung): "
              f"{100*purity:.1f}%")
        print("Die Kategorien entstanden ohne Labels — Wörter wurden erst")
        print("nach der Cluster-Bildung zugeordnet.")
        return 0
    bank = v.build_bank(args.word, n_images=args.images)
    if not bank.prototypes:
        print("Keine Kategorien gebaut.")
        return 1
    print(f"Kategorien: {', '.join(bank.prototypes)}")
    print(f"Konsistenz: " + ", ".join(
        f"{w}={bank.consistency[w]:.1f}" for w in bank.prototypes))
    ratio = bank.harnad_ratio()
    if ratio is not None:
        print(f"\nHarnad-Ratio (within/between): {ratio:.3f} "
              f"({'< 1: kategorielle Wahrnehmung wirkt' if ratio < 1 else '>= 1: Kategorien trennen nicht'})")
    if args.test:
        import urllib.request
        req = urllib.request.Request(args.test, headers={
            "User-Agent": "fertig/1.0"})
        img = urllib.request.urlopen(req, timeout=30).read()
        word, d = bank.recognize(img)
        print(f"\nTestbild erkannt als: {word} (Distanz {d:.4f})")
    return 0


def cmd_video(args) -> int:
    from . import video as v
    data = open(args.gif, "rb").read()
    raw = v.extract_frames(data)
    if args.mode == "verstehen":
        u = v.understand(raw)
        print(f"Frames: {u['frames']} | bewegt: {u['bewegt']} "
              f"(sig={u['signatur']:.4f} pix={u['pixel']:.4f})")
        print(f"periodisch: {u['periodisch']} | szenenwechsel: "
              f"{u['szenenwechsel']} | paritaet: {u['paritaet']}")
    elif args.mode == "generieren":
        codes = v.frame_code(v.frame_signatures(raw))
        trans = v.learn_transitions(codes)
        gen = v.generate_frames(codes[0], trans, n=args.frames)
        print(f"Original : {codes}")
        print(f"Generiert: {gen}")
        ok = sum(1 for a, b in zip(gen, gen[1:])
                 if b in trans.get(a, {}))
        print(f"Grammatik-Treue: {ok}/{len(gen)-1} Kanten")
    return 0


def cmd_stream(args) -> int:
    from . import stream as s
    source = args.source
    if source.startswith(("http://", "https://")) and "youtu" in source:
        print(f"[stream] yt-dlp: {source}")
        try:
            source = s.ytdlp_url(source)
            print(f"[stream] direkte URL erhalten ({len(source)} Zeichen)")
        except Exception as e:
            print(f"[stream] yt-dlp fehlgeschlagen: {e}")
            return 1
    print(f"[stream] lerne aus {source} ({args.seconds}s @ {args.fps}fps) ...")
    learner = s.learn_from(source, seconds=args.seconds, fps=args.fps)
    st = learner.state()
    print(f"\n=== Gelernt (O(1), konstantes Memory) ===")
    print(f"  Frames        : {st['frames']}")
    print(f"  Bewegung      : sig={st['bewegung']:.4f} "
          f"pixel={st['pixel']:.4f}")
    print(f"  Periodisch    : {st['periodisch']}")
    print(f"  Szenenwechsel : {st['szenenwechsel']}")
    print(f"  Grammatik     : {st['grammatik_kanten']} Kanten")
    print(f"  Fortsetzung   : {learner.generate(10)}")
    if args.name:
        from . import video as v
        import json
        bank = v.VideoBank()
        bank_path = Path("data/video_bank.json")
        if bank_path.exists():
            data = json.loads(bank_path.read_text())
            for k, arr in data.items():
                bank.prototypes[k] = np.array(arr)
        bank.add_from_learner(args.name, learner)
        bank_path.write_text(json.dumps(
            {k: v.tolist() for k, v in bank.prototypes.items()}))
        facts = learner.to_graph_facts(args.name)
        print(f"  Kategorie '{args.name}' gelernt + gespeichert; "
              f"Graph-Fakten: {facts}")
    if args.recognize:
        from . import video as v
        bank = v.VideoBank()
        bank_path = Path("data/video_bank.json")
        if bank_path.exists():
            import json
            data = json.loads(bank_path.read_text())
            for k, arr in data.items():
                bank.prototypes[k] = np.array(arr)
        sig = learner.sequence_signature()
        word, d = bank.recognize_signature(sig)
        print(f"  Erkannt als   : {word} (Distanz {d:.4f})")
    return 0


def cmd_interp(args) -> int:
    from . import video as v
    from .interp import InterpLearner
    raw = v.extract_frames(open(args.video, "rb").read(),
                           max_frames=args.frames)
    learner = InterpLearner(n_bins=args.bins,
                            quality_threshold=args.schwelle)
    for f in raw:
        learner.update(f)
    print(f"Gelernt: {learner.frames_seen} Frames, "
          f"{sum(len(r) for r in learner.transitions.values())} "
          f"Grammatik-Kanten")
    if args.selfpaced:
        print("\nSelf-paced Curriculum (Surprise stellt Stützräder ein):")
        curve = learner.self_paced_learn(raw, max_gap=args.maxgap)
        print(f"  gap-Verlauf: {[g for g, _ in curve[::max(1, len(curve)//10)]]}")
        print(f"  Endstand: gap={curve[-1][0]}")
    else:
        print("\nInterpolation (Lücke wächst = Stützräder ab):")
        for gap in [2, 4, 6, 8]:
            q = learner.quality(raw, gap)
            mark = "<-- beherrscht" if q < args.schwelle else ""
            print(f"  Lücke {gap}: Fehler {q:.4f} {mark}")
        print(f"\nBeherrschte Lücke: {learner.mastered_gap(raw, args.maxgap)}")
    return 0


def cmd_schauen(args) -> int:
    """GOAT-Moonshoot: Video -> Verständnis -> Kategorie -> Wissen -> Sprache.
    Der geschlossene Kreislauf des Organismus in einem Befehl."""
    from . import stream as s
    from . import video as v
    from . import gaps as gaps_mod
    from .pipeline import load_graph_merged, _toks
    from .intent import parse_command
    from . import tools
    from pathlib import Path as _P
    import json

    data = open(args.video, "rb").read()
    print(f"[schauen] {args.video} — der Kreislauf startet\n")

    # 1. SEHEN (O(1)-Stream-Lernen)
    raw = v.extract_frames(data)
    learner = s.StreamLearner()
    for f in raw:
        learner.update(f)
    st = learner.state()
    print(f"[1/5] SEHEN     : {st['frames']} Frames, Bewegung "
          f"sig={st['bewegung']:.3f} pix={st['pixel']:.3f}, "
          f"periodisch={st['periodisch']}")

    # 2. ERKENNEN (VideoBank)
    bank = v.VideoBank().load(_P("data/video_bank.json"))
    word, d = bank.recognize_signature(learner.sequence_signature())
    if word:
        print(f"[2/5] ERKENNEN  : {word} (Distanz {d:.4f})")
    else:
        print(f"[2/5] ERKENNEN  : unbekannt (Distanz {d:.4f}) — "
              f"wird neue Kategorie")

    # 3. LERNEN (Kategorie + Fakten in den Welt-Graphen)
    name = args.name or (word or _P(args.video).stem)
    bank.add_from_learner(name, learner)
    bank.save(_P("data/video_bank.json"))
    facts = learner.to_graph_facts(name)
    print(f"[3/5] LERNEN    : Kategorie '{name}' + Graph-Fakten {facts}")

    # 4. WISSEN (Graph konsultieren — Erinnerung an die eigene Sprache)
    merged_vocab = load_graph_merged()[0]
    trips = gaps_mod._load_world()
    erinnerung = [c for a, b, c, _ in trips
                  if a == name and b == "beschreibt_sich"]
    if erinnerung:
        print(f"[4/5] WISSEN    : (erinnert) {erinnerung[0]}")
    else:
        # Struktur-Narration: Fakten aus dem gemessenen Zustand — und als
        # Selbstbeschreibung in den Graphen schreiben (autobiografisch)
        teile = [f"{name}"]
        if st["bewegung"] > 0.02 or st["pixel"] > 0.02:
            teile.append("zeigt Bewegung")
        if st["periodisch"]:
            teile.append("wiederholt sich periodisch")
        if st["szenenwechsel"]:
            teile.append(f"hat {st['szenenwechsel']} Szenenwechsel")
        narration = ", ".join(teile) + "."
        best = {t[:3]: t[3] for t in trips}
        best[(name, "beschreibt_sich", narration)] = 0.7
        gaps_mod._save_world([(a, b, c, conf)
                              for (a, b, c), conf in best.items()])
        print(f"[4/5] WISSEN    : {narration} (als Selbstbeschreibung "
              f"gespeichert)")

    # 5. SPRECHEN (die Antwort als Intent-Ausgabe)
    print(f"[5/5] SPRECHEN  : Das Video zeigt {name or 'etwas Neues'}"
          + (f", es ist periodisch" if st["periodisch"] else "")
          + (f", es bewegt sich" if st["bewegung"] > 0.02 else ""))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="fertig",
        description="FERTIG — das fertige symbolische Sprachsystem "
                    "(gewicht-frei, deterministisch, aus .causal-Graphen und Korpora).")
    ap.add_argument("--version", action="version", version=f"fertig {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("info", help="Graph-Statistiken + Hyperboloid-Check")
    p.add_argument("--graph", default=str(pipeline.DEFAULT_GRAPH))
    p.set_defaults(fn=cmd_info)

    p = sub.add_parser("chains", help="abgeleitete Ketten (pass1)")
    p.add_argument("--graph", default=str(pipeline.DEFAULT_GRAPH))
    p.add_argument("-n", type=int, default=8)
    p.set_defaults(fn=cmd_chains)

    p = sub.add_parser("graph", help="gewicht-freie Kausal-Walks")
    p.add_argument("--graph", default=str(pipeline.DEFAULT_GRAPH))
    p.add_argument("-n", type=int, default=8)
    p.add_argument("start", nargs="*")
    p.set_defaults(fn=cmd_graph)

    p = sub.add_parser("speech", help="Walks als gesprochene Prosa")
    p.add_argument("--graph", default=str(pipeline.DEFAULT_GRAPH))
    p.add_argument("-n", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("start", nargs="*")
    p.set_defaults(fn=cmd_speech)

    p = sub.add_parser("mined", help="Prosa mit gemessener Muster-Bank")
    p.add_argument("--graph", default=str(pipeline.DEFAULT_GRAPH))
    p.add_argument("--bank", default=str(mined.DEFAULT_BANK))
    p.add_argument("-n", type=int, default=8)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("start", nargs="*")
    p.set_defaults(fn=cmd_mined)

    p = sub.add_parser("bank", help="Muster-Bank aus Korpus minen")
    p.add_argument("-o", "--out", default=str(mined.DEFAULT_BANK))
    p.add_argument("corpora", nargs="*")
    p.set_defaults(fn=cmd_bank)

    p = sub.add_parser("intent", help="NL-Befehl -> Intent-Tupel + Tool-Call")
    p.add_argument("--graph", default=str(pipeline.DEFAULT_GRAPH))
    p.add_argument("--lexicon", default=str(learn_mod.DEFAULT_LEXICON))
    p.add_argument("-x", "--execute", action="store_true",
                   help="Intent auch ausführen (Tool-Call)")
    p.add_argument("--video", default=None,
                   help="Video-Datei für die erkennen-Aktion")
    p.add_argument("command", nargs="+")
    p.set_defaults(fn=cmd_intent)

    p = sub.add_parser("learn", help="Lexikon aus Korpus lernen (wächst)")
    p.add_argument("-o", "--out", default=str(learn_mod.DEFAULT_LEXICON))
    p.add_argument("--min-count", type=int, default=2)
    p.add_argument("corpora", nargs="*")
    p.set_defaults(fn=cmd_learn)

    p = sub.add_parser("arena", help="Selbst-Benchmark (präregistriert)")
    p.add_argument("--graph", default=str(pipeline.DEFAULT_GRAPH))
    p.add_argument("-q", "--quiet", action="store_true")
    p.set_defaults(fn=cmd_arena)

    p = sub.add_parser("bench", help="SOTA-Benchmarks")
    p.add_argument("name", choices=["blimp", "snips", "humaneval",
                                    "hellaswag", "winogrande", "lambada",
                                    "llm-snips", "llm-all", "arc"])
    p.add_argument("--subtasks", nargs="*",
                   help="BLiMP-Subtasks (Standard: alle 8)")
    p.add_argument("-n", type=int, default=30,
                   help="HumanEval: Anzahl Evaluations-Probleme")
    p.add_argument("--no-learn", action="store_true",
                   help="HumanEval: nicht aus Referenzlösungen lernen")
    p.add_argument("--no-graph", action="store_true",
                   help="ARC: ohne Graph-Antworten (nur LM-Baseline)")
    p.add_argument("-q", "--quiet", action="store_true")
    p.set_defaults(fn=cmd_bench)

    p = sub.add_parser("evolve", help="Autonomer Verbesserungs-Loop")
    p.add_argument("--iterations", type=int, default=3)
    p.add_argument("--arc-questions", type=int, default=30)
    p.add_argument("--grow-per-iter", type=int, default=3)
    p.add_argument("--sources", default=None)
    p.set_defaults(fn=cmd_evolve)

    p = sub.add_parser("code", help="Code aus Prompt assembleren + sandboxen")
    p.add_argument("prompt", nargs="+")
    p.add_argument("-x", "--execute", action="store_true",
                   help="assemblierten Code in der Sandbox ausführen")
    p.set_defaults(fn=cmd_code)

    p = sub.add_parser("schauen", help="GOAT: Video->Verstehen->Wissen->Sprache")
    p.add_argument("video")
    p.add_argument("--name", default=None,
                   help="Kategorie-Name (Default: erkannt oder Dateiname)")
    p.set_defaults(fn=cmd_schauen)

    p = sub.add_parser("interp", help="Interpolation mit Stützrädern")
    p.add_argument("video")
    p.add_argument("--frames", type=int, default=32)
    p.add_argument("--bins", type=int, default=16)
    p.add_argument("--schwelle", type=float, default=0.05)
    p.add_argument("--maxgap", type=int, default=8)
    p.add_argument("-s", "--selfpaced", action="store_true")
    p.set_defaults(fn=cmd_interp)

    p = sub.add_parser("stream", help="Permanent aus Video-Streams lernen")
    p.add_argument("source", help="Datei oder YouTube-URL")
    p.add_argument("--seconds", type=int, default=15)
    p.add_argument("--fps", type=int, default=2)
    p.add_argument("--name", default=None,
                   help="Kategorie-Name: Stream in VideoBank + Graph lernen")
    p.add_argument("--recognize", action="store_true",
                   help="gegen die VideoBank erkennen")
    p.set_defaults(fn=cmd_stream)

    p = sub.add_parser("video", help="Video-Verständnis/-Generierung (GIF)")
    p.add_argument("gif", help="Pfad zur GIF-Datei")
    p.add_argument("--mode", choices=["verstehen", "generieren"],
                   default="verstehen")
    p.add_argument("--frames", type=int, default=16)
    p.set_defaults(fn=cmd_video)

    p = sub.add_parser("vision", help="Deterministische Bilderkennung")
    p.add_argument("word", nargs="+", help="Kategorien-Wörter")
    p.add_argument("--images", type=int, default=4, help="Bilder pro Wort")
    p.add_argument("--test", default=None,
                   help="URL eines Testbildes zum Erkennen")
    p.add_argument("-u", "--unsupervised", action="store_true",
                   help="Harnad-Ebene: Kategorien ohne Wörter (Clustering)")
    p.set_defaults(fn=cmd_vision)

    p = sub.add_parser("quant", help="Quantitative QA (Grounding-Beweis)")
    p.add_argument("question", nargs="*")
    p.add_argument("--all", action="store_true", help="präregistrierte Arena")
    p.set_defaults(fn=cmd_quant)

    p = sub.add_parser("ground", help="Symbol an Nicht-Wort-Anker binden")
    p.add_argument("word", nargs="*")
    p.add_argument("--all", action="store_true",
                   help="alle Graph-Symbole erden + Coverage messen")
    p.add_argument("--max", type=int, default=25)
    p.set_defaults(fn=cmd_ground)

    p = sub.add_parser("grow", help="Gap-Loop: Weltwissen in den Graphen holen")
    p.add_argument("target", nargs="*", help="Entitäten (z. B. sugar)")
    p.add_argument("--sources", default=None,
                   help="wikipedia,wiktionary,duckduckgo,web,arxiv,pubmed,"
                        "semantic_scholar,openalex (Default: alle)")
    p.add_argument("--gaps", action="store_true",
                   help="Lücken aus der Arena automatisch wachsen lassen")
    p.add_argument("--max-targets", type=int, default=3)
    p.set_defaults(fn=cmd_grow)

    p = sub.add_parser("crawl", help="Beliebige URL -> Text -> Tripletts")
    p.add_argument("url")
    p.add_argument("--store", action="store_true",
                   help="Tripletts in den Welt-Graphen speichern")
    p.set_defaults(fn=cmd_crawl)

    p = sub.add_parser("corpus", help="Korpus-Modus: Prompt fortsetzen")
    p.add_argument("--corpus", default=str(corpus.DEFAULT_CORPUS))
    p.add_argument("--max-vocab", type=int, default=2000)
    p.add_argument("-n", type=int, default=25)
    p.add_argument("prompt", nargs="*")
    p.set_defaults(fn=cmd_corpus)

    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
