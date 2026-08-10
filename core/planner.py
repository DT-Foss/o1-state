"""
Goal Planner — Decompose Goals Into Executable Steps
======================================================
Claude's 4 + unsere 5 (Epistemic) + diese 6: GOAL DECOMPOSITION.

"Build a web scraper" is not ONE task. It's:
  1. Import libraries (requests, bs4)
  2. Fetch URL
  3. Parse HTML
  4. Extract data
  5. Output result

Without decomposition, you try to do everything at once and fail.
WITH decomposition, each step is testable independently.

This is what turns "code generation" into "problem solving."

Plan types:
  - SEQUENTIAL: A then B then C
  - CONDITIONAL: if X then A else B
  - ITERATIVE: repeat A until condition
  - PARALLEL: A and B independently, then join

No external dependencies.
"""

import re
import time
from typing import Dict, Any, List, Optional
from enum import Enum


class StepStatus(Enum):
    PENDING = 'pending'
    RUNNING = 'running'
    DONE = 'done'
    FAILED = 'failed'
    SKIPPED = 'skipped'


class PlanStep:
    """A single step in a plan."""

    def __init__(self, description: str, step_type: str = 'action',
                 depends_on: List[int] = None):
        self.description = description
        self.step_type = step_type  # 'action', 'check', 'decision', 'loop'
        self.depends_on = depends_on or []
        self.status = StepStatus.PENDING
        self.result: Any = None
        self.error: str = ''
        self.start_time: float = 0
        self.end_time: float = 0

    @property
    def elapsed(self) -> float:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0

    def __repr__(self):
        return f'Step({self.description!r}, status={self.status.value})'


class Plan:
    """An execution plan with steps and dependencies."""

    def __init__(self, goal: str, steps: List[PlanStep] = None):
        self.goal = goal
        self.steps = steps or []
        self.created_at = time.time()
        self.status = 'created'

    def add_step(self, description: str, step_type: str = 'action',
                 depends_on: List[int] = None) -> int:
        """Add a step and return its index."""
        step = PlanStep(description, step_type, depends_on)
        self.steps.append(step)
        return len(self.steps) - 1

    def can_execute(self, step_idx: int) -> bool:
        """Can this step be executed (all dependencies met)?"""
        step = self.steps[step_idx]
        if step.status != StepStatus.PENDING:
            return False
        for dep_idx in step.depends_on:
            if dep_idx >= len(self.steps):
                return False
            if self.steps[dep_idx].status != StepStatus.DONE:
                return False
        return True

    def next_steps(self) -> List[int]:
        """Return indices of steps ready to execute."""
        return [i for i in range(len(self.steps)) if self.can_execute(i)]

    def mark_done(self, step_idx: int, result: Any = None):
        self.steps[step_idx].status = StepStatus.DONE
        self.steps[step_idx].result = result
        self.steps[step_idx].end_time = time.time()

    def mark_failed(self, step_idx: int, error: str = ''):
        self.steps[step_idx].status = StepStatus.FAILED
        self.steps[step_idx].error = error
        self.steps[step_idx].end_time = time.time()

    def mark_running(self, step_idx: int):
        self.steps[step_idx].status = StepStatus.RUNNING
        self.steps[step_idx].start_time = time.time()

    @property
    def is_complete(self) -> bool:
        return all(s.status in (StepStatus.DONE, StepStatus.SKIPPED)
                   for s in self.steps)

    @property
    def is_failed(self) -> bool:
        return any(s.status == StepStatus.FAILED for s in self.steps)

    @property
    def progress(self) -> float:
        if not self.steps:
            return 0.0
        done = sum(1 for s in self.steps
                   if s.status in (StepStatus.DONE, StepStatus.SKIPPED))
        return done / len(self.steps)

    def summary(self) -> str:
        """Human-readable plan summary."""
        lines = [f'Plan: {self.goal}',
                 f'Progress: {self.progress:.0%} ({len(self.steps)} steps)',
                 '']
        for i, step in enumerate(self.steps):
            icon = {
                StepStatus.PENDING: '○',
                StepStatus.RUNNING: '►',
                StepStatus.DONE: '✓',
                StepStatus.FAILED: '✗',
                StepStatus.SKIPPED: '–',
            }[step.status]
            deps = f' (after {step.depends_on})' if step.depends_on else ''
            lines.append(f'  {icon} {i}: {step.description}{deps}')
            if step.error:
                lines.append(f'       ERROR: {step.error}')
        return '\n'.join(lines)


class GoalPlanner:
    """
    Decompose a high-level goal into an executable plan.

    Uses pattern matching + domain knowledge to break down goals.
    """

    # Known decomposition patterns
    PATTERNS = {
        # Code generation patterns
        r'(?:build|create|write|make)\s+(?:a\s+)?(\w+\s+)?(?:script|program|tool|function)': [
            ('Parse requirements from goal', 'action'),
            ('Identify required imports/libraries', 'action'),
            ('Design function signatures', 'action'),
            ('Implement core logic', 'action'),
            ('Add error handling', 'action'),
            ('Write test cases', 'check'),
            ('Validate output', 'check'),
        ],

        # Data processing patterns
        r'(?:process|analyze|parse|extract)\s+(?:data|file|text|csv|json)': [
            ('Read input data', 'action'),
            ('Validate input format', 'check'),
            ('Transform/parse data', 'action'),
            ('Apply analysis/extraction', 'action'),
            ('Format output', 'action'),
            ('Validate results', 'check'),
        ],

        # Debugging patterns
        r'(?:fix|debug|repair|solve)\s+(?:the\s+)?(?:bug|error|issue|problem)': [
            ('Reproduce the error', 'action'),
            ('Read error message/traceback', 'action'),
            ('Identify the root cause line', 'action'),
            ('Predict what the fix should be', 'action'),
            ('Apply the fix', 'action'),
            ('Run tests to verify', 'check'),
            ('Check for regressions', 'check'),
        ],

        # Learning/research patterns
        r'(?:learn|understand|research|study)\s+(?:about\s+)?(.+)': [
            ('Define what we need to know', 'action'),
            ('Search existing knowledge', 'action'),
            ('Identify gaps', 'check'),
            ('Acquire missing information', 'action'),
            ('Verify new information', 'check'),
            ('Integrate into knowledge base', 'action'),
        ],

        # Improvement patterns
        r'(?:improve|optimize|refactor|enhance)\s+(.+)': [
            ('Measure current performance/quality', 'action'),
            ('Identify bottlenecks/issues', 'action'),
            ('Design improvements', 'action'),
            ('Implement changes', 'action'),
            ('Measure new performance', 'check'),
            ('Compare before vs after', 'check'),
            ('Decide: keep or revert', 'decision'),
        ],

        # Testing patterns
        r'(?:test|verify|validate|check)\s+(.+)': [
            ('Identify what to test', 'action'),
            ('Write test cases', 'action'),
            ('Run tests', 'action'),
            ('Analyze results', 'check'),
            ('Fix failures if any', 'action'),
            ('Re-run to confirm', 'check'),
        ],
    }

    # Task decomposition for common atomic operations
    ATOMIC_TASKS = {
        'read_file': ['Open file', 'Read contents', 'Close file', 'Return contents'],
        'write_file': ['Validate path', 'Open file', 'Write contents', 'Close file', 'Verify written'],
        'http_request': ['Construct URL', 'Set headers', 'Send request', 'Check status', 'Parse response'],
        'sort_data': ['Validate input', 'Choose algorithm', 'Execute sort', 'Verify sorted'],
        'search': ['Define search terms', 'Execute search', 'Rank results', 'Return top matches'],
    }

    def plan(self, goal: str) -> Plan:
        """
        Create an execution plan for a goal.

        Returns:
            Plan with ordered, dependency-linked steps.
        """
        goal_lower = goal.lower().strip()

        # Try pattern matching first
        for pattern, steps_template in self.PATTERNS.items():
            if re.search(pattern, goal_lower, re.I):
                return self._plan_from_template(goal, steps_template)

        # Fallback: decompose based on goal structure
        return self._plan_generic(goal)

    def plan_code_task(self, task: str, context: Dict[str, Any] = None) -> Plan:
        """
        Specialized planner for code generation/modification tasks.

        Args:
            task: what to do
            context: available info (files, functions, errors, etc.)
        """
        context = context or {}
        plan = Plan(goal=task)

        # Always start with understanding
        plan.add_step('Understand the requirement', 'action')

        # If there's existing code, read it first
        if context.get('file'):
            plan.add_step(f'Read {context["file"]}', 'action', depends_on=[0])

        # If there's an error, analyze it
        if context.get('error'):
            plan.add_step('Analyze error message', 'action', depends_on=[0])
            plan.add_step('Predict root cause', 'action',
                         depends_on=[len(plan.steps) - 1])

        # Core work
        design_idx = plan.add_step('Design solution', 'action',
                                   depends_on=[len(plan.steps) - 1])
        predict_idx = plan.add_step('Predict expected output', 'action',
                                    depends_on=[design_idx])
        impl_idx = plan.add_step('Implement solution', 'action',
                                 depends_on=[design_idx])

        # Verify
        exec_idx = plan.add_step('Execute and capture output', 'action',
                                 depends_on=[impl_idx])
        plan.add_step('Compare output with prediction', 'check',
                     depends_on=[predict_idx, exec_idx])

        # Fix loop
        plan.add_step('If mismatch: trace cause and fix', 'action',
                     depends_on=[len(plan.steps) - 1])

        return plan

    def plan_self_improve(self, target: str) -> Plan:
        """
        Plan a self-improvement cycle.
        This is THE plan that ties all 6 capabilities together.
        """
        plan = Plan(goal=f'Self-improve: {target}')

        # Fähigkeit 5: Epistemic — what do we know/not know?
        ep_idx = plan.add_step('Check epistemic state: what do we know about this?', 'action')

        # Fähigkeit 6: Decompose — break it down
        dec_idx = plan.add_step('Decompose goal into subtasks', 'action',
                                depends_on=[ep_idx])

        # For each subtask: the 4-capability loop
        pred_idx = plan.add_step('PREDICT: what should the output be?', 'action',
                                 depends_on=[dec_idx])
        exec_idx = plan.add_step('EXECUTE: generate and run code', 'action',
                                 depends_on=[pred_idx])
        comp_idx = plan.add_step('COMPARE: does output match prediction?', 'check',
                                 depends_on=[pred_idx, exec_idx])
        attr_idx = plan.add_step('ATTRIBUTE: if wrong, find root cause', 'action',
                                 depends_on=[comp_idx])
        plan.add_step('FIX: apply correction and re-execute', 'action',
                     depends_on=[attr_idx])
        plan.add_step('UPDATE: record what we learned (epistemic)', 'action',
                     depends_on=[len(plan.steps) - 1])

        return plan

    def _plan_from_template(self, goal: str, steps_template: List[tuple]) -> Plan:
        """Create plan from a matched template."""
        plan = Plan(goal=goal)
        prev_idx = -1
        for desc, step_type in steps_template:
            depends = [prev_idx] if prev_idx >= 0 else []
            prev_idx = plan.add_step(desc, step_type, depends)
        return plan

    def _plan_generic(self, goal: str) -> Plan:
        """Generic decomposition for unrecognized goals."""
        plan = Plan(goal=goal)

        # Universal 4-step plan
        understand = plan.add_step('Understand the goal', 'action')
        design = plan.add_step('Design an approach', 'action', depends_on=[understand])
        execute = plan.add_step('Execute the approach', 'action', depends_on=[design])
        plan.add_step('Verify the result', 'check', depends_on=[execute])

        return plan

    def decompose_atomic(self, task_type: str) -> List[str]:
        """Get atomic decomposition for a known task type."""
        return self.ATOMIC_TASKS.get(task_type, [task_type])
