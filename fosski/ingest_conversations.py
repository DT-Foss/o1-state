#!/usr/bin/env python3
"""
FOSS-KI Conversation Ingestor — Full 10-Layer Passive Distillation
===================================================================
Watches Claude Code JSONL files. Every message goes through ALL extractors
in a single pass. No LLM needed — the text IS from an LLM.

Layers:
  0. Topic Map         — domain frequency from user questions
  1. Facts             — triplets from "X is a Y", "X uses Y", etc.
  2. FLM Corpus        — raw assistant text → language model training
  3. Formulation       — explanation templates, connectors, transitions
  4. Reasoning Chains  — step-by-step structure extraction
  5. Common Sense      — "X can/cannot Y" rules
  6. Word Relations    — similar_to, contrast, synonym, antonym
  7. Code Fragments    — code blocks with language + intent
  8. User Patterns     — how David communicates (vocab, abbreviations)
  9. Corrections       — "nein", "falsch", "das stimmt nicht" → anti-hallucination

Usage:
    python3 ingest_conversations.py                     # watch mode
    python3 ingest_conversations.py --backfill          # process existing
    python3 ingest_conversations.py --backfill --watch   # both
"""

import sys
import os
import json
import time
import re
import logging
import argparse
import signal
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
CLAUDE_PROJECTS_DIR = os.path.expanduser('~/.claude/projects')
SEEN_FILE = os.path.join(DATA_DIR, 'ingest_seen.json')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(DATA_DIR, 'ingest.log')),
    ]
)
log = logging.getLogger('ingest')

_shutdown = False

def _signal_handler(sig, frame):
    global _shutdown
    log.info(f"Shutdown ({sig}). Finishing current batch...")
    _shutdown = True

signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ============================================================================
# JSONL PARSER — extract user+assistant message PAIRS
# ============================================================================

def parse_conversation_pairs(path, seen_uuids):
    """
    Parse JSONL and yield (uuid, role, text, prev_user_text) tuples.
    Pairs each assistant message with the preceding user message for context.
    """
    results = []
    last_user_text = ''
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                msg_type = obj.get('type', '')
                uuid = obj.get('uuid', '')

                if msg_type == 'user':
                    content = obj.get('message', {}).get('content', '')
                    if isinstance(content, str):
                        # Skip system/command messages
                        if content.startswith(('<local-command', '<command-',
                                               '<system-reminder')):
                            continue
                        # Strip XML tags from user content
                        clean = re.sub(r'<[^>]+>', '', content).strip()
                        if clean and len(clean) > 5:
                            last_user_text = clean
                            if uuid not in seen_uuids:
                                results.append((uuid, 'user', clean, ''))

                elif msg_type == 'assistant':
                    if uuid in seen_uuids:
                        continue
                    content = obj.get('message', {}).get('content', '')
                    texts = []
                    if isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict) and block.get('type') == 'text':
                                t = block.get('text', '').strip()
                                if t and len(t) > 15:
                                    texts.append(t)
                    elif isinstance(content, str) and len(content) > 15:
                        texts.append(content.strip())

                    if texts:
                        full = '\n\n'.join(texts)
                        results.append((uuid, 'assistant', full, last_user_text))

    except Exception as e:
        log.warning(f"Error reading {os.path.basename(path)}: {e}")

    return results


# ============================================================================
# LAYER 0 — TOPIC MAP (domain frequency from user questions)
# ============================================================================

_DOMAIN_KEYWORDS = {
    'programming': ['python', 'javascript', 'rust', 'code', 'function', 'class',
                     'import', 'def ', 'bug', 'error', 'compile', 'debug', 'git',
                     'api', 'docker', 'test', 'deploy', 'database', 'sql', 'regex'],
    'math': ['integral', 'derivative', 'matrix', 'vector', 'eigenvalue', 'proof',
             'theorem', 'equation', 'formula', 'probability', 'statistics'],
    'ai_ml': ['neural', 'model', 'training', 'inference', 'llm', 'transformer',
              'embedding', 'tokenizer', 'gradient', 'loss', 'epoch', 'batch',
              'attention', 'diffusion', 'finetune', 'rlhf', 'markov'],
    'crypto': ['cipher', 'encrypt', 'decrypt', 'hash', 'key', 'aes', 'rsa',
               'nist', 'pqc', 'lattice', 'casi', 'distinguisher'],
    'security': ['exploit', 'vulnerability', 'cve', 'payload', 'shellcode',
                 'evasion', 'pentest', 'opsec', 'forensic', 'malware'],
    'networking': ['tcp', 'udp', 'http', 'dns', 'port', 'socket', 'proxy',
                   'firewall', 'packet', 'ais', 'nmea', 'maritime'],
    'language': ['grammar', 'noun', 'verb', 'sentence', 'word', 'syntax',
                 'semantics', 'translation', 'formulation', 'text'],
    'science': ['physics', 'chemistry', 'biology', 'quantum', 'relativity',
                'entropy', 'thermodynamic', 'gravitational'],
}

def extract_topics(text):
    """Classify text into domains. Returns list of matched domains."""
    text_lower = text.lower()
    matched = []
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in text_lower)
        if hits >= 2:
            matched.append((domain, hits))
    matched.sort(key=lambda x: -x[1])
    return [d for d, _ in matched[:3]]


# ============================================================================
# LAYER 1 — FACT EXTRACTION (triplets)
# ============================================================================

_FACT_PATTERNS = [
    # "X is a/an/the Y"
    (r'(?:^|\. )([A-Z][a-zA-Z0-9_ -]{1,40})\s+is\s+(?:a|an|the)\s+'
     r'([a-zA-Z0-9_ ,()-]{2,80}?)(?:\.|,|\n|$)', 'type'),
    # "X uses/supports/provides Y"
    (r'(?:^|\. )([A-Z][a-zA-Z0-9_ -]{1,40})\s+'
     r'(?:uses|supports|provides|implements|requires)\s+'
     r'([a-zA-Z0-9_ -]{2,60})', 'uses'),
    # "X was created/developed by Y"
    (r'([A-Z][a-zA-Z0-9_ -]{1,40})\s+was\s+'
     r'(?:created|developed|designed|invented|founded|introduced)\s+by\s+'
     r'([A-Z][a-zA-Z0-9_ .-]{2,40})', 'creator'),
    # "X consists of Y"
    (r'([A-Z][a-zA-Z0-9_ -]{1,40})\s+'
     r'(?:consists of|contains|includes|comprises)\s+'
     r'([a-zA-Z0-9_ ,()-]{2,80})', 'contains'),
    # "X is used for Y"
    (r'([A-Z][a-zA-Z0-9_ -]{1,40})\s+'
     r'(?:is used for|is designed for|is meant for|enables)\s+'
     r'([a-zA-Z0-9_ ,()-]{2,80})', 'used_for'),
    # "X returns Y"
    (r'([A-Za-z_]\w+)\s*\(\)\s+returns?\s+([a-zA-Z0-9_ -]{2,40})', 'returns'),
]

def extract_facts(text, kb):
    """Layer 1: Extract factual triplets from text."""
    stored = 0
    for pattern, relation in _FACT_PATTERNS:
        for m in re.finditer(pattern, text):
            subj = m.group(1).strip().rstrip('.')
            obj = m.group(2).strip().rstrip('.')
            if 2 < len(subj) <= 50 and 2 < len(obj) <= 80:
                kb.store_fact(subj.lower(), relation, obj.lower())
                stored += 1

    # Code: def function_name(params)
    for m in re.finditer(r'def\s+(\w{2,30})\s*\(([^)]{0,100})\)', text):
        func, params = m.group(1), m.group(2).strip()
        if not func.startswith('_'):
            kb.store_fact(func, 'type', 'function')
            if params:
                kb.store_fact(func, 'parameters', params[:80])
            stored += 1

    # Code: class ClassName(Parent)
    for m in re.finditer(r'class\s+([A-Z]\w{1,30})\s*(?:\(([^)]*)\))?:', text):
        cls, parent = m.group(1), m.group(2)
        kb.store_fact(cls.lower(), 'type', 'class')
        if parent and parent.strip():
            kb.store_fact(cls.lower(), 'inherits', parent.strip().lower())
        stored += 1

    return stored


# ============================================================================
# LAYER 3 — FORMULATION PATTERNS (explanation templates)
# ============================================================================

_FORMULATION_STARTERS = [
    'the reason', 'this means', 'in other words', 'specifically',
    'the key insight', 'the main difference', 'essentially',
    'in practice', 'the advantage', 'the problem is', 'note that',
    'this is because', 'the idea is', 'more precisely',
    'for example', 'to put it simply', 'the tradeoff',
    'what this means is', 'the important thing',
    'this works because', 'the way this works',
]

def extract_formulations(text, kb):
    """Layer 3: Extract explanation/formulation patterns."""
    stored = 0
    sentences = re.split(r'(?<=[.!?])\s+', text)
    for sent in sentences:
        sent_lower = sent.lower().strip()
        for starter in _FORMULATION_STARTERS:
            if sent_lower.startswith(starter) and len(sent) > 30:
                # Store as formulation template
                # Replace specific nouns with {X} to make it a template
                template = sent[:200]
                kb.store_fact(starter, 'formulation_example', template)
                stored += 1
                break
    return stored


# ============================================================================
# LAYER 4 — REASONING CHAINS (step-by-step structure)
# ============================================================================

_STEP_PATTERNS = [
    r'(?:first|step 1)[,:]?\s+(.+?)(?:\.|\n)',
    r'(?:second|step 2|then|next)[,:]?\s+(.+?)(?:\.|\n)',
    r'(?:third|step 3|after that)[,:]?\s+(.+?)(?:\.|\n)',
    r'(?:finally|lastly|step \d+)[,:]?\s+(.+?)(?:\.|\n)',
]

_REASONING_INDICATORS = re.compile(
    r'\b(?:first.*(?:then|next|second)|step \d.*step \d|'
    r'because.*therefore|if.*then.*otherwise)\b', re.DOTALL | re.IGNORECASE
)

def extract_reasoning(text, kb):
    """Layer 4: Extract reasoning chain structures."""
    stored = 0
    if not _REASONING_INDICATORS.search(text):
        return 0

    # Extract numbered steps
    steps = []
    for m in re.finditer(r'(?:^|\n)\s*(?:\d+[\.)]\s*|[-*]\s+)(.{10,200})', text):
        step = m.group(1).strip().rstrip('.')
        if step:
            steps.append(step)

    if len(steps) >= 2:
        # Store the reasoning chain structure
        chain = ' → '.join(steps[:6])
        kb.store_fact('reasoning_chain', 'pattern', chain[:300])
        stored += 1

        # Classify reasoning type
        text_lower = text.lower()
        if any(w in text_lower for w in ['debug', 'error', 'fix', 'bug']):
            kb.store_fact('debugging', 'reasoning_example', chain[:200])
        elif any(w in text_lower for w in ['design', 'architect', 'pattern']):
            kb.store_fact('design', 'reasoning_example', chain[:200])
        elif any(w in text_lower for w in ['optimize', 'performance', 'speed']):
            kb.store_fact('optimization', 'reasoning_example', chain[:200])
        stored += 1

    return stored


# ============================================================================
# LAYER 5 — COMMON SENSE (can/cannot rules)
# ============================================================================

def extract_common_sense(text, kb):
    """Layer 5: Extract common sense rules from text."""
    stored = 0

    # "X can/cannot Y"
    for m in re.finditer(
            r'([A-Za-z][a-zA-Z_ -]{1,30})\s+(can(?:not|\'t)?)\s+'
            r'([a-zA-Z_ -]{2,60})', text):
        subj = m.group(1).strip().lower()
        ability = 'can' if 'not' not in m.group(2) and "n't" not in m.group(2) else 'cannot'
        obj = m.group(3).strip().rstrip('.').lower()
        if len(subj) > 2 and len(obj) > 2:
            kb.store_fact(subj, ability, obj)
            stored += 1

    # "X should/must Y"
    for m in re.finditer(
            r'(?:you |we )?(?:should|must|need to)\s+'
            r'([a-zA-Z_ -]{2,80})', text):
        rule = m.group(1).strip().rstrip('.').lower()
        if len(rule) > 5:
            kb.store_fact('best_practice', 'rule', rule[:100])
            stored += 1

    return stored


# ============================================================================
# LAYER 6 — WORD RELATIONS (similar, contrast, synonym)
# ============================================================================

_RELATION_PATTERNS = [
    (r'([A-Za-z][a-zA-Z_ -]{1,30})\s+is\s+similar\s+to\s+'
     r'([A-Za-z][a-zA-Z_ -]{1,30})', 'similar_to'),
    (r'unlike\s+([A-Za-z][a-zA-Z_ -]{1,30}),\s*'
     r'([A-Za-z][a-zA-Z_ -]{1,30})', 'contrast'),
    (r'([A-Za-z][a-zA-Z_ -]{1,30})\s+(?:is the opposite of|differs from)\s+'
     r'([A-Za-z][a-zA-Z_ -]{1,30})', 'antonym'),
    (r'([A-Za-z][a-zA-Z_ -]{1,30})\s*(?:\(also (?:called|known as)\s+'
     r'([A-Za-z][a-zA-Z_ -]{1,30})\))', 'synonym'),
    (r'([A-Za-z][a-zA-Z_ -]{1,30}),?\s+also\s+known\s+as\s+'
     r'([A-Za-z][a-zA-Z_ -]{1,30})', 'synonym'),
    (r'([A-Za-z][a-zA-Z_ -]{1,30})\s+is\s+(?:a type|a kind|a form)\s+of\s+'
     r'([A-Za-z][a-zA-Z_ -]{1,30})', 'hypernym'),
    (r'([A-Za-z][a-zA-Z_ -]{1,30})\s+(?:vs\.?|versus)\s+'
     r'([A-Za-z][a-zA-Z_ -]{1,30})', 'comparison'),
]

def extract_word_relations(text, kb):
    """Layer 6: Extract word-level semantic relations."""
    stored = 0
    for pattern, relation in _RELATION_PATTERNS:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            a = m.group(1).strip().lower()
            b = m.group(2).strip().rstrip('.').lower()
            if 2 < len(a) <= 40 and 2 < len(b) <= 40:
                kb.store_fact(a, relation, b)
                stored += 1
    return stored


# ============================================================================
# LAYER 7 — CODE FRAGMENTS (with language + intent)
# ============================================================================

def extract_code_fragments(text, user_context, kb):
    """Layer 7: Extract code blocks with language and intent from context."""
    stored = 0
    for m in re.finditer(r'```(\w*)\n(.*?)```', text, re.DOTALL):
        lang = m.group(1).lower() or 'unknown'
        code = m.group(2).strip()
        if not code or len(code) < 20:
            continue

        # Detect language from content if not specified
        if lang == 'unknown':
            if 'def ' in code or 'import ' in code:
                lang = 'python'
            elif 'function ' in code or 'const ' in code or '=>' in code:
                lang = 'javascript'
            elif '#!/bin/' in code:
                lang = 'bash'
            elif '#include' in code:
                lang = 'c'
            elif 'fn ' in code and '-> ' in code:
                lang = 'rust'

        # Extract function/class names from code
        for fm in re.finditer(r'def\s+(\w{2,30})', code):
            func = fm.group(1)
            if not func.startswith('_'):
                kb.store_fact(func, 'type', 'code_template')
                kb.store_fact(func, 'language', lang)
                stored += 1

        for cm in re.finditer(r'class\s+([A-Z]\w{1,30})', code):
            cls = cm.group(1)
            kb.store_fact(cls.lower(), 'type', 'code_template')
            kb.store_fact(cls.lower(), 'language', lang)
            stored += 1

        # Store code snippet hash → intent mapping (from user question)
        if user_context and len(user_context) > 10:
            # Extract short intent from user question
            intent = user_context[:120].replace('\n', ' ').strip()
            code_id = f"code_{hash(code[:200]) & 0xFFFFFFFF:08x}"
            kb.store_fact(code_id, 'intent', intent)
            kb.store_fact(code_id, 'language', lang)
            stored += 1

    return stored


# ============================================================================
# LAYER 8 — USER COMMUNICATION PATTERNS
# ============================================================================

def extract_user_patterns(text, kb, user_vocab):
    """Layer 8: Analyze David's communication style."""
    # Count word frequencies
    words = re.findall(r'\b\w{3,}\b', text.lower())
    for w in words:
        user_vocab[w] += 1

    stored = 0

    # Detect abbreviations (ALL CAPS, 3-5 chars, must look like real acronyms)
    _ABBR_STOPWORDS = {
        # English
        'THE', 'AND', 'FOR', 'NOT', 'BUT', 'ALL', 'ARE', 'WAS', 'HAS', 'CAN',
        'DID', 'GET', 'SET', 'PUT', 'USE', 'NEW', 'OLD', 'BIG', 'END', 'RUN',
        'FIX', 'YOU', 'HIS', 'HER', 'ITS', 'OUR', 'WHO', 'HOW', 'WHY', 'YES',
        'GOT', 'HAD', 'LET', 'SAY', 'SEE', 'TRY', 'WAY', 'MAY', 'DAY', 'ANY',
        'NOW', 'OWN', 'TOO', 'ADD', 'AGO', 'BAD', 'FAR', 'FEW', 'RAW', 'LOW',
        # German (David writes in German)
        'DAS', 'DIE', 'DER', 'UND', 'ICH', 'IST', 'DEN', 'VON', 'MIT', 'AUF',
        'DEM', 'HAT', 'EIN', 'NUR', 'WIR', 'WAS', 'MAN', 'ODE', 'BIN', 'MIR',
        'DIR', 'IHN', 'IHR', 'WIE', 'NUN', 'MAL', 'GUT', 'NEU', 'NIE', 'TAG',
        'DICH', 'ALLE', 'AUCH', 'NOCH', 'DANN', 'KEIN', 'JEDE', 'HIER', 'DOCH',
        'ABER', 'MACH', 'HALT', 'LASS', 'GUCK', 'WIRD', 'NACH', 'ODER', 'WENN',
        'SICH', 'SIND', 'EINE', 'DASS', 'HAST', 'MUSS', 'WEIL', 'SEHR',
    }
    for m in re.finditer(r'\b([A-Z]{3,5})\b', text):
        abbr = m.group(1)
        if abbr not in _ABBR_STOPWORDS:
            kb.store_fact(abbr, 'type', 'abbreviation_used_by_user')
            stored += 1

    # Detect German (David writes in German sometimes)
    german_words = {'und', 'oder', 'ist', 'nicht', 'das', 'ein', 'eine', 'wie',
                    'was', 'warum', 'aber', 'auch', 'noch', 'schon', 'dann',
                    'halt', 'nein', 'ja', 'doch', 'lass', 'mach', 'guck'}
    n_german = sum(1 for w in words if w in german_words)
    if n_german >= 3:
        kb.store_fact('user_language', 'includes', 'german')
        stored += 1

    return stored


# ============================================================================
# LAYER 9 — CORRECTIONS (anti-hallucination gold)
# ============================================================================

_CORRECTION_PATTERNS = [
    r'(?:nein|no|falsch|wrong|incorrect|that\'s not right|das stimmt nicht|'
    r'quatsch|unsinn|bullshit|that\'s wrong|not quite)',
]

def extract_corrections(user_text, prev_assistant_text, kb):
    """
    Layer 9: Detect when user corrects the assistant.
    Stores (wrong_claim, corrected_to, right_answer) triplets.
    """
    stored = 0
    if not user_text or not prev_assistant_text:
        return 0

    user_lower = user_text.lower()

    # Check if user is correcting
    is_correction = any(re.search(p, user_lower) for p in _CORRECTION_PATTERNS)
    if not is_correction:
        return 0

    # The correction is valuable — store it
    # Extract the corrected information from the user's message
    # (everything after the correction marker)
    correction_text = user_text
    for marker in ['nein', 'no,', 'falsch', 'wrong,', 'not quite,',
                    'das stimmt nicht', 'actually']:
        idx = user_lower.find(marker)
        if idx >= 0:
            after = user_text[idx + len(marker):].strip().lstrip(',').strip()
            if after and len(after) > 5:
                correction_text = after
                break

    # Store the correction
    wrong_summary = prev_assistant_text[:150].replace('\n', ' ')
    kb.store_fact('correction', 'wrong_claim', wrong_summary)
    kb.store_fact('correction', 'right_answer', correction_text[:200])
    stored += 2

    return stored


# ============================================================================
# MASTER INGEST — all layers in one pass
# ============================================================================

def ingest_message(uuid, role, text, user_context, kb, flm,
                   stats, topic_counter, user_vocab,
                   prev_assistant_text='', live_train=False):
    """Run ALL extraction layers on a single message.
    live_train=True: train FLM immediately (watch mode).
    live_train=False: only persist corpus to file (backfill, train after).
    """

    if role == 'user':
        # === USER MESSAGE ===
        # Layer 0: Topic map
        topics = extract_topics(text)
        for t in topics:
            topic_counter[t] += 1
            stats['topics'] += 1

        # Layer 8: User communication patterns
        stats['user_patterns'] += extract_user_patterns(text, kb, user_vocab)

        # Layer 9: Corrections (needs previous assistant text)
        stats['corrections'] += extract_corrections(
            text, prev_assistant_text, kb)

    elif role == 'assistant':
        # === ASSISTANT MESSAGE ===
        # Layer 1: Fact extraction
        stats['facts'] += extract_facts(text, kb)

        # Layer 2: FLM corpus — persist to file + optionally train live
        if len(text) > 50:
            clean = re.sub(r'```\w*\n.*?```', '', text, flags=re.DOTALL)
            clean = re.sub(r'`[^`]+`', '', clean)
            clean = clean.strip()
            if 50 < len(clean) < 10000:
                # Always persist to corpus file
                corpus_path = os.path.join(DATA_DIR, 'flm_corpus.txt')
                with open(corpus_path, 'a', encoding='utf-8') as f:
                    f.write(clean + '\n\n')
                stats['flm_chars'] += len(clean)
                # In watch mode: train FLM immediately on new text
                if live_train and flm:
                    flm.train(clean[:5000])

        # Layer 3: Formulation patterns
        stats['formulations'] += extract_formulations(text, kb)

        # Layer 4: Reasoning chains
        stats['reasoning'] += extract_reasoning(text, kb)

        # Layer 5: Common sense rules
        stats['common_sense'] += extract_common_sense(text, kb)

        # Layer 6: Word relations
        stats['word_relations'] += extract_word_relations(text, kb)

        # Layer 7: Code fragments
        stats['code'] += extract_code_fragments(text, user_context, kb)

    stats['messages'] += 1


def make_stats():
    """Create fresh stats dict."""
    return {
        'messages': 0, 'facts': 0, 'flm_chars': 0, 'formulations': 0,
        'reasoning': 0, 'common_sense': 0, 'word_relations': 0,
        'code': 0, 'user_patterns': 0, 'corrections': 0, 'topics': 0,
    }


def stats_summary(stats):
    """One-line stats summary (only non-zero)."""
    parts = []
    for k, v in stats.items():
        if v > 0 and k != 'messages':
            if k == 'flm_chars':
                parts.append(f"FLM={v//1024}KB")
            else:
                parts.append(f"{k}={v}")
    return ' | '.join(parts)


# ============================================================================
# FILE PROCESSING
# ============================================================================

def ingest_file(path, kb, flm, seen, stats, topic_counter, user_vocab,
                live_train=False):
    """Process one JSONL file — all messages through all layers."""
    pairs = parse_conversation_pairs(path, seen)
    if not pairs:
        return

    prev_assistant = ''
    for uuid, role, text, user_ctx in pairs:
        if _shutdown:
            break

        ingest_message(uuid, role, text, user_ctx, kb, flm,
                       stats, topic_counter, user_vocab,
                       prev_assistant_text=prev_assistant,
                       live_train=live_train)

        if role == 'assistant':
            prev_assistant = text
        elif role == 'user':
            prev_assistant = ''  # reset after user speaks again

        seen.add(uuid)


def load_seen():
    if os.path.exists(SEEN_FILE):
        try:
            with open(SEEN_FILE, 'r') as f:
                return set(json.load(f))
        except Exception:
            pass
    return set()


def save_seen(seen):
    with open(SEEN_FILE, 'w') as f:
        json.dump(list(seen), f)


def find_jsonl_files():
    files = []
    if not os.path.exists(CLAUDE_PROJECTS_DIR):
        return files
    for root, _, fnames in os.walk(CLAUDE_PROJECTS_DIR):
        for fn in fnames:
            if fn.endswith('.jsonl'):
                files.append(os.path.join(root, fn))
    return sorted(files, key=lambda p: os.path.getmtime(p), reverse=True)


def _save_brain(kb, flm, brain_path):
    from core.brain import BrainSnapshot
    snap = BrainSnapshot.capture_from_parts(knowledge_store=kb, lm_model=flm)
    snap.save(brain_path, compressed=True)
    size_kb = os.path.getsize(brain_path) / 1024
    log.info(f"  Brain saved: {kb.n_facts} facts, {size_kb:.0f}KB")


def _save_topic_map(topic_counter):
    """Save domain frequency map."""
    path = os.path.join(DATA_DIR, 'topic_map.json')
    with open(path, 'w') as f:
        json.dump(dict(topic_counter.most_common(50)), f, indent=2)


def _save_user_vocab(user_vocab, top_n=500):
    """Save David's most-used words."""
    path = os.path.join(DATA_DIR, 'user_vocab.json')
    top = dict(user_vocab.most_common(top_n))
    with open(path, 'w') as f:
        json.dump(top, f, indent=2)


# ============================================================================
# BACKFILL + WATCH
# ============================================================================

def run_backfill(kb, flm, seen, max_files=0):
    files = find_jsonl_files()
    n_files = max_files if max_files > 0 else len(files)
    log.info(f"Backfill: {len(files)} JSONL files, processing {n_files}")

    stats = make_stats()
    topic_counter = Counter()
    user_vocab = Counter()

    for path in files[:n_files]:
        if _shutdown:
            break
        size_mb = os.path.getsize(path) / (1024 * 1024)
        if size_mb > 200:  # Only skip truly massive files (>200MB)
            log.info(f"  Skip {os.path.basename(path)} ({size_mb:.0f}MB)")
            continue
        ingest_file(path, kb, flm, seen, stats, topic_counter, user_vocab)
        if stats['messages'] > 0 and stats['messages'] % 500 == 0:
            log.info(f"  Progress: {stats['messages']} msgs | {stats_summary(stats)}")

    log.info(f"Backfill done: {stats['messages']} msgs | {stats_summary(stats)}")

    # === SAVE BRAIN IMMEDIATELY — before FLM training which can take hours ===
    brain_path = os.path.join(DATA_DIR, 'foss-ki.brain')
    _save_brain(kb, None, brain_path)  # save facts now, FLM separately

    # === TRAIN FLM on accumulated corpus (chunked to manage RAM) ===
    if flm:
        corpus_path = os.path.join(DATA_DIR, 'flm_corpus.txt')
        if os.path.exists(corpus_path):
            corpus_size = os.path.getsize(corpus_path)
            # PPM-Trie grows O(n * max_order) nodes. At order 12 and 15MB
            # that's 180M node insertions. Train in 10KB chunks with
            # paragraph boundaries so context doesn't span chunks.
            chunk_size = 10_000  # 10KB chunks (safe for RAM)
            log.info(f"  Training FLM on corpus ({corpus_size // 1024}KB) "
                     f"in {chunk_size // 1024}KB chunks...")
            trained_chars = 0
            n_chunks = 0
            with open(corpus_path, 'r', encoding='utf-8') as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    flm.train(chunk)
                    trained_chars += len(chunk)
                    n_chunks += 1
                    if n_chunks % 100 == 0:
                        log.info(f"    FLM chunk {n_chunks}: "
                                 f"{trained_chars // 1024}KB trained")
                    if _shutdown:
                        break
            log.info(f"  FLM trained: {trained_chars // 1024}KB "
                     f"in {n_chunks} chunks, vocab={len(flm.vocab)}")

    # Save topic map and vocab
    _save_topic_map(topic_counter)
    _save_user_vocab(user_vocab)

    # Log top topics
    if topic_counter:
        top3 = topic_counter.most_common(5)
        log.info(f"  Top domains: {', '.join(f'{d}={n}' for d, n in top3)}")

    return stats


def run_watch(kb, flm, seen, brain_path, interval=30):
    log.info(f"Watch mode (every {interval}s)...")
    file_sizes = {}
    cycle = 0
    topic_counter = Counter()
    user_vocab = Counter()

    # Load existing topic map
    topic_path = os.path.join(DATA_DIR, 'topic_map.json')
    if os.path.exists(topic_path):
        try:
            with open(topic_path, 'r') as f:
                topic_counter.update(json.load(f))
        except Exception:
            pass

    while not _shutdown:
        cycle += 1
        files = find_jsonl_files()
        stats = make_stats()

        for path in files[:20]:
            if _shutdown:
                break
            try:
                size = os.path.getsize(path)
            except OSError:
                continue

            if size != file_sizes.get(path, 0):
                ingest_file(path, kb, flm, seen, stats,
                            topic_counter, user_vocab,
                            live_train=True)
                file_sizes[path] = size

        if stats['messages'] > 0:
            log.info(f"Watch {cycle}: +{stats['messages']} msgs | "
                     f"{stats_summary(stats)}")
            save_seen(seen)
            if stats['messages'] >= 5:
                _save_brain(kb, flm, brain_path)
                _save_topic_map(topic_counter)

        if cycle % 20 == 0:
            save_seen(seen)

        for _ in range(interval):
            if _shutdown:
                break
            time.sleep(1)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='FOSS-KI Full 10-Layer Conversation Ingestor')
    parser.add_argument('--backfill', action='store_true',
                        help='Process existing conversation files')
    parser.add_argument('--watch', action='store_true',
                        help='Watch for new conversations')
    parser.add_argument('--max-files', type=int, default=0,
                        help='Max files for backfill (0=all)')
    parser.add_argument('--interval', type=int, default=30,
                        help='Watch interval in seconds')
    args = parser.parse_args()

    if not args.backfill and not args.watch:
        args.watch = True

    brain_path = os.path.join(DATA_DIR, 'foss-ki.brain')

    # Bootstrap
    from core.agent import Agent
    agent = Agent(workspace_dir=os.path.dirname(os.path.abspath(__file__)))
    n = agent.bootstrap(use_brain=True)
    kb = agent.dialog.knowledge

    # Harvest stdlib (once)
    from core.harvester import KnowledgeHarvester
    h = KnowledgeHarvester(kb)
    h.harvest_all(verbose=False)

    # Inference
    from core.inference import InferenceEngine
    inf = InferenceEngine(kb)
    inf.run(store_inferred=True)
    log.info(f"Bootstrapped: {kb.n_facts} facts")

    # Create FLM (empty — will be trained after backfill or from brain)
    try:
        from core.foss_lm import FossLanguageModel
        flm = FossLanguageModel()
        log.info("FLM created (empty, will train after backfill)")
    except Exception as e:
        log.warning(f"FLM failed: {e}")
        flm = None

    seen = load_seen()
    log.info(f"Previously seen: {len(seen)} messages")

    files = find_jsonl_files()
    log.info(f"Found {len(files)} conversation files")

    if args.backfill:
        run_backfill(kb, flm, seen, max_files=args.max_files)
        save_seen(seen)
        _save_brain(kb, flm, brain_path)

    if args.watch:
        try:
            run_watch(kb, flm, seen, brain_path, interval=args.interval)
        finally:
            save_seen(seen)
            _save_brain(kb, flm, brain_path)
            log.info(f"Shutdown. KB: {kb.n_facts} facts")


if __name__ == '__main__':
    main()
