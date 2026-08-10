#!/usr/bin/env python3
"""
bAbI Real Benchmark — Facebook AI Research 20 Tasks
=====================================================
Uses the REAL bAbI dataset (tasks_1-20_v1-2).
Tests tasks 1-5 (basic reasoning) against our knowledge store.

Format per line:
  ID TEXT
  ID QUESTION\tANSWER\tSUPPORTING_FACT_IDS
Story resets when ID goes back to 1.
"""

import sys
import os
import re
import json
import time
import copy
from datetime import datetime
from typing import List, Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BABI_DIR = os.path.join(BASE_DIR, 'data', 'external_benchmarks', 'tasks_1-20_v1-2', 'en')


# ─── Parser ───────────────────────────────────────────────────────────

# Movement verbs → location extraction
_MOVE_PATTERNS = [
    re.compile(r'(\w+)\s+(?:travelled|journeyed|went|moved|went\s+back)\s+to\s+the\s+(\w+)'),
    re.compile(r'(\w+)\s+(?:travelled|journeyed|went|moved|went\s+back)\s+to\s+(\w+)'),
]

_PICKUP_PATTERN = re.compile(r'(\w+)\s+(?:picked up|got|grabbed|took)\s+the\s+(\w+)')
_DROP_PATTERN = re.compile(r'(\w+)\s+(?:dropped|left|put down|discarded)\s+the\s+(\w+)')
_GIVE_PATTERN = re.compile(r'(\w+)\s+(?:gave|handed|passed)\s+the\s+(\w+)\s+to\s+(\w+)')


def parse_babi_file(filepath: str) -> List[Dict]:
    """Parse bAbI file into list of stories, each with facts and questions."""
    stories = []
    current_facts = {}  # line_id → text
    current_state = {}  # entity → {location: str, carrying: set}
    current_questions = []

    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Split ID from rest
            parts = line.split(' ', 1)
            line_id = int(parts[0])
            text = parts[1]

            # Story reset
            if line_id == 1 and current_questions:
                stories.append({
                    'questions': current_questions,
                })
                current_facts = {}
                current_state = {}
                current_questions = []

            if line_id == 1 and not current_questions:
                current_facts = {}
                current_state = {}
                current_questions = []

            # Is this a question or a fact?
            if '\t' in text:
                # Question: QUESTION\tANSWER\tSUPPORTING_IDS
                q_parts = text.split('\t')
                question = q_parts[0].strip()
                answer = q_parts[1].strip() if len(q_parts) > 1 else ''
                current_questions.append({
                    'question': question,
                    'answer': answer,
                    'state_snapshot': copy.deepcopy(current_state),
                })
            else:
                # Fact sentence — update world state
                current_facts[line_id] = text
                _update_state(current_state, text)

    # Last story
    if current_questions:
        stories.append({'questions': current_questions})

    return stories


_SPATIAL_PATTERN = re.compile(
    r'The\s+(\w+)\s+is\s+(?:to\s+the\s+)?(north|south|east|west)\s+of\s+the\s+(\w+)')

_FEAR_PATTERN = re.compile(r'(\w+)\s+(?:is|are)\s+afraid\s+of\s+(\w+)')
_ISA_PATTERN = re.compile(r'(\w+)\s+is\s+a\s+(\w+)')

# Number words for Task 7
_NUM_WORDS = {0: 'none', 1: 'one', 2: 'two', 3: 'three', 4: 'four',
              5: 'five', 6: 'six', 7: 'seven', 8: 'eight', 9: 'nine', 10: 'ten'}


def _update_state(state: dict, text: str):
    """Update world state from a bAbI fact sentence."""
    # Movement
    for pat in _MOVE_PATTERNS:
        m = pat.search(text)
        if m:
            entity = m.group(1)
            location = m.group(2)
            if entity not in state:
                state[entity] = {'location': None, 'carrying': set(),
                                 'location_history': [], 'gave': [], 'received': []}
            # Track location history for "before" questions (Task 3/14)
            old_loc = state[entity].get('location')
            if old_loc is not None:
                state[entity].setdefault('location_history', []).append(old_loc)
            state[entity]['location'] = location
            # Move carried objects too
            for obj_name in list(state[entity].get('carrying', set())):
                if obj_name in state:
                    old_obj_loc = state[obj_name].get('location')
                    if old_obj_loc is not None:
                        state[obj_name].setdefault('location_history', []).append(old_obj_loc)
                    state[obj_name]['location'] = location
            return

    # Pick up
    m = _PICKUP_PATTERN.search(text)
    if m:
        entity, obj = m.group(1), m.group(2)
        if entity not in state:
            state[entity] = {'location': None, 'carrying': set(),
                             'location_history': [], 'gave': [], 'received': []}
        state[entity]['carrying'].add(obj)
        if obj not in state:
            state[obj] = {'location': state[entity].get('location'), 'carrying': set(),
                          'location_history': [], 'gave': [], 'received': []}
        state[obj]['location'] = state[entity]['location']
        return

    # Drop
    m = _DROP_PATTERN.search(text)
    if m:
        entity, obj = m.group(1), m.group(2)
        if entity in state:
            state[entity].get('carrying', set()).discard(obj)
        return

    # Give (Task 5)
    m = _GIVE_PATTERN.search(text)
    if m:
        giver, obj, receiver = m.group(1), m.group(2), m.group(3)
        if giver in state:
            state[giver].get('carrying', set()).discard(obj)
        else:
            state[giver] = {'location': None, 'carrying': set(),
                            'location_history': [], 'gave': [], 'received': []}
        if receiver not in state:
            state[receiver] = {'location': None, 'carrying': set(),
                               'location_history': [], 'gave': [], 'received': []}
        state[receiver]['carrying'].add(obj)
        # Track give actions for "who gave what to whom"
        state[giver].setdefault('gave', []).append((obj, receiver))
        state[receiver].setdefault('received', []).append((obj, giver))
        if obj in state:
            state[obj]['location'] = state.get(receiver, {}).get('location')
        return

    # Spatial relations (Task 4): "The bedroom is north of the bathroom"
    m = _SPATIAL_PATTERN.search(text)
    if m:
        room1, direction, room2 = m.group(1), m.group(2), m.group(3)
        key = '__spatial__'
        if key not in state:
            state[key] = {}
        state[key][(room1.lower(), direction)] = room2.lower()
        # Store inverse
        opposites = {'north': 'south', 'south': 'north', 'east': 'west', 'west': 'east'}
        state[key][(room2.lower(), opposites[direction])] = room1.lower()
        return

    # Is-A facts (Task 15): "Emily is a cat"
    m = _ISA_PATTERN.search(text)
    if m:
        entity, category = m.group(1), m.group(2)
        key = '__isa__'
        if key not in state:
            state[key] = {}
        state[key][entity.lower()] = category.lower()
        return

    # Fear relations (Task 15): "Cats are afraid of wolves"
    m = _FEAR_PATTERN.search(text)
    if m:
        fearer, feared = m.group(1), m.group(2)
        key = '__fears__'
        if key not in state:
            state[key] = {}
        state[key][fearer.lower()] = feared.lower()
        return


def answer_from_state(state: dict, question: str) -> Optional[str]:
    """Answer a bAbI question from world state."""
    q = question.lower().strip().rstrip('?').strip()

    # "Where is X?" → state[X].location
    m = re.match(r'where\s+is\s+(?:the\s+)?(\w+)', q)
    if m:
        entity = m.group(1).title()
        for name, info in state.items():
            if name.startswith('__'):
                continue
            if name.lower() == entity.lower():
                return info.get('location')
            if entity.lower() in {c.lower() for c in info.get('carrying', set())}:
                return info.get('location')
        return None

    # "Where was X before the Y?" → location_history (Task 3/14)
    m = re.match(r'where\s+was\s+(?:the\s+)?(\w+)\s+before\s+the\s+(\w+)', q)
    if m:
        entity = m.group(1).title()
        before_loc = m.group(2).lower()
        for name, info in state.items():
            if name.startswith('__'):
                continue
            if name.lower() == entity.lower():
                history = info.get('location_history', [])
                current = info.get('location', '')
                full_path = history + ([current] if current else [])
                # Find the location just before before_loc
                for i, loc in enumerate(full_path):
                    if loc and loc.lower() == before_loc and i > 0:
                        return full_path[i - 1]
                return None
        return None

    # "What is X carrying?"
    m = re.match(r'what\s+is\s+(\w+)\s+carrying', q)
    if m:
        entity = m.group(1).title()
        for name, info in state.items():
            if name.startswith('__'):
                continue
            if name.lower() == entity.lower():
                carrying = info.get('carrying', set())
                if carrying:
                    return ','.join(sorted(carrying))
                return 'nothing'
        return 'nothing'

    # "How many objects is X carrying?" (Task 7)
    m = re.match(r'how\s+many\s+objects?\s+is\s+(\w+)\s+carrying', q)
    if m:
        entity = m.group(1).title()
        for name, info in state.items():
            if name.startswith('__'):
                continue
            if name.lower() == entity.lower():
                count = len(info.get('carrying', set()))
                return _NUM_WORDS.get(count, str(count))
        return 'none'

    # "What did X give to Y?" (Task 5)
    m = re.match(r'what\s+did\s+(\w+)\s+give\s+to\s+(\w+)', q)
    if m:
        giver = m.group(1).title()
        receiver = m.group(2).title()
        for name, info in state.items():
            if name.startswith('__'):
                continue
            if name.lower() == giver.lower():
                for obj, recv in info.get('gave', []):
                    if recv.lower() == receiver.lower():
                        return obj.lower()
        return None

    # "Who gave the X to Y?" (Task 5)
    m = re.match(r'who\s+gave\s+the\s+(\w+)\s+to\s+(\w+)', q)
    if m:
        obj = m.group(1).title()
        receiver = m.group(2).title()
        for name, info in state.items():
            if name.startswith('__'):
                continue
            if name.lower() == receiver.lower():
                for recv_obj, from_who in info.get('received', []):
                    if recv_obj.lower() == obj.lower():
                        return from_who.lower()
        return None

    # "Who did X give the Y to?" (Task 5 variant)
    m = re.match(r'who\s+did\s+(\w+)\s+give\s+the\s+(\w+)\s+to', q)
    if m:
        giver = m.group(1).title()
        obj = m.group(2).title()
        for name, info in state.items():
            if name.startswith('__'):
                continue
            if name.lower() == giver.lower():
                for gave_obj, recv in info.get('gave', []):
                    if gave_obj.lower() == obj.lower():
                        return recv.lower()
        return None

    # "Is X in the Y?" → yes/no
    m = re.match(r'is\s+(\w+)\s+in\s+the\s+(\w+)', q)
    if m:
        entity, location = m.group(1).title(), m.group(2)
        for name, info in state.items():
            if name.startswith('__'):
                continue
            if name.lower() == entity.lower():
                if info.get('location', '').lower() == location.lower():
                    return 'yes'
                return 'no'
        return 'no'

    # "What is the Y [direction] of?" (Task 4 spatial)
    m = re.match(r'what\s+is\s+(?:the\s+)?(\w+)\s+(north|south|east|west)\s+of', q)
    if m:
        room = m.group(1).lower()
        direction = m.group(2).lower()
        spatial = state.get('__spatial__', {})
        result = spatial.get((room, direction))
        return result

    # "What is [direction] of the Y?" (Task 4 variant)
    m = re.match(r'what\s+is\s+(north|south|east|west)\s+of\s+the\s+(\w+)', q)
    if m:
        direction = m.group(1).lower()
        room = m.group(2).lower()
        spatial = state.get('__spatial__', {})
        result = spatial.get((room, direction))
        return result

    # "What is X afraid of?" (Task 15 deduction)
    m = re.match(r'what\s+is\s+(\w+)\s+afraid\s+of', q)
    if m:
        entity = m.group(1).lower()
        fears = state.get('__fears__', {})
        isa = state.get('__isa__', {})
        # Direct fear
        if entity in fears:
            return fears[entity]
        # Deduction: entity is-a category, category fears X
        category = isa.get(entity)
        if category:
            # Try plural forms
            for cat_form in [category, category + 's', category.rstrip('s')]:
                if cat_form in fears:
                    return fears[cat_form]
        return None

    return None


# ─── Task Runners ─────────────────────────────────────────────────────

TASK_FILES = {
    1: 'qa1_single-supporting-fact_test.txt',
    2: 'qa2_two-supporting-facts_test.txt',
    3: 'qa3_three-supporting-facts_test.txt',
    4: 'qa4_two-arg-relations_test.txt',
    5: 'qa5_three-arg-relations_test.txt',
    6: 'qa6_yes-no-questions_test.txt',
    7: 'qa7_counting_test.txt',
    8: 'qa8_lists-sets_test.txt',
    14: 'qa14_time-reasoning_test.txt',
    15: 'qa15_basic-deduction_test.txt',
}


def run_task(task_id: int, limit: Optional[int] = None) -> Dict:
    """Run a single bAbI task."""
    filename = TASK_FILES.get(task_id)
    if not filename:
        return {'task': task_id, 'skipped': True, 'reason': 'Not implemented'}

    filepath = os.path.join(BABI_DIR, filename)
    if not os.path.exists(filepath):
        return {'task': task_id, 'skipped': True, 'reason': 'File not found'}

    stories = parse_babi_file(filepath)

    results = []
    for story in stories:
        for qa in story['questions']:
            if limit and len(results) >= limit:
                break

            t0 = time.time()
            predicted = answer_from_state(qa['state_snapshot'], qa['question'])
            elapsed = time.time() - t0

            expected = qa['answer'].lower()
            got = (predicted or '').lower()
            # Normalize comma-separated lists (order-independent)
            if ',' in expected or ',' in got:
                exp_set = set(s.strip() for s in expected.split(','))
                got_set = set(s.strip() for s in got.split(','))
                correct = exp_set == got_set
            else:
                correct = got == expected

            results.append({
                'question': qa['question'],
                'expected': expected,
                'got': got,
                'correct': correct,
                'time_ms': round(elapsed * 1000, 2),
            })

        if limit and len(results) >= limit:
            break

    total = len(results)
    correct_count = sum(1 for r in results if r['correct'])
    accuracy = correct_count / total if total > 0 else 0.0

    return {
        'task': task_id,
        'total': total,
        'correct': correct_count,
        'accuracy': round(accuracy, 4),
        'results': results,
    }


def main():
    print("=" * 70)
    print("  bAbI REAL BENCHMARK — Facebook AI Research 20 Tasks")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    if not os.path.exists(BABI_DIR):
        print(f"\n  ERROR: bAbI data not found at {BABI_DIR}")
        return

    limit = 200  # 200 questions per task

    all_results = {}
    total_correct = 0
    total_questions = 0

    for task_id in sorted(TASK_FILES.keys()):
        result = run_task(task_id, limit=limit)
        all_results[f'task_{task_id}'] = result

        if result.get('skipped'):
            print(f"\n  Task {task_id:>2}: SKIPPED ({result['reason']})")
        else:
            pct = result['accuracy']
            total_correct += result['correct']
            total_questions += result['total']
            bar = '#' * int(pct * 20) + '.' * (20 - int(pct * 20))
            print(f"  Task {task_id:>2}: {result['correct']:>4}/{result['total']:<4} [{bar}] {pct:.1%}")

            # Show first 3 errors
            errors = [r for r in result['results'] if not r['correct']]
            for e in errors[:3]:
                print(f"           Q: {e['question']}")
                print(f"           Expected: {e['expected']}, Got: {e['got']}")

    overall = total_correct / total_questions if total_questions > 0 else 0.0
    print(f"\n{'=' * 70}")
    print(f"  OVERALL: {total_correct}/{total_questions} ({overall:.1%})")
    print(f"  Tasks tested: {len([r for r in all_results.values() if not r.get('skipped')])}")
    print(f"{'=' * 70}")

    # Save
    results_dir = os.path.join(BASE_DIR, 'benchmarks', 'results')
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f'babi_real_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    with open(path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"  Saved: {path}")


if __name__ == '__main__':
    main()
