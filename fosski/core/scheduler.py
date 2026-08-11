"""
Scheduler — Task Planning & Automation Engine
===============================================
An AI that replaces ChatGPT must be able to PLAN and EXECUTE
multi-step workflows autonomously.

Capabilities:
  1. Task queue with priorities and dependencies
  2. DAG-based execution ordering (topological sort)
  3. Cron-like recurring tasks
  4. Retry with backoff on failure
  5. Event triggers (file change, time, condition)
  6. Pipeline composition (chain tasks)

This is what makes FOSS-KI proactive, not just reactive.
"""

import time
import threading
import heapq
from enum import Enum
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field


class TaskStatus(Enum):
    PENDING = 'pending'
    RUNNING = 'running'
    COMPLETED = 'completed'
    FAILED = 'failed'
    BLOCKED = 'blocked'      # Waiting on dependency
    CANCELLED = 'cancelled'
    RETRYING = 'retrying'


class Priority(Enum):
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4


@dataclass(order=True)
class Task:
    """A schedulable task with metadata."""
    priority: int
    created_at: float = field(compare=True)
    task_id: str = field(compare=False)
    name: str = field(compare=False)
    fn: Optional[Callable] = field(compare=False, default=None)
    args: Dict = field(compare=False, default_factory=dict)
    status: TaskStatus = field(compare=False, default=TaskStatus.PENDING)
    depends_on: List[str] = field(compare=False, default_factory=list)
    max_retries: int = field(compare=False, default=0)
    retry_count: int = field(compare=False, default=0)
    retry_delay: float = field(compare=False, default=1.0)
    result: Any = field(compare=False, default=None)
    error: Optional[str] = field(compare=False, default=None)
    started_at: Optional[float] = field(compare=False, default=None)
    completed_at: Optional[float] = field(compare=False, default=None)
    tags: List[str] = field(compare=False, default_factory=list)


class TaskScheduler:
    """
    Priority queue scheduler with dependency resolution.

    Usage:
        sched = TaskScheduler()
        sched.add('fetch', fetch_data, priority=Priority.HIGH)
        sched.add('process', process_data, depends_on=['fetch'])
        sched.add('report', generate_report, depends_on=['process'])
        results = sched.run_all()
    """

    def __init__(self):
        self._tasks: Dict[str, Task] = {}
        self._queue: List[Task] = []
        self._counter = 0
        self._results: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def add(self, name: str, fn: Callable = None,
            args: Optional[Dict] = None,
            priority: Priority = Priority.NORMAL,
            depends_on: Optional[List[str]] = None,
            max_retries: int = 0,
            retry_delay: float = 1.0,
            tags: Optional[List[str]] = None) -> str:
        """
        Add a task to the scheduler.

        Returns task_id.
        """
        with self._lock:
            self._counter += 1
            task_id = f"task_{self._counter:04d}"

            task = Task(
                priority=priority.value,
                created_at=time.time(),
                task_id=task_id,
                name=name,
                fn=fn,
                args=args or {},
                depends_on=depends_on or [],
                max_retries=max_retries,
                retry_delay=retry_delay,
                tags=tags or [],
            )
            self._tasks[task_id] = task
            heapq.heappush(self._queue, task)

        return task_id

    def get_task(self, task_id: str) -> Optional[Task]:
        return self._tasks.get(task_id)

    def cancel(self, task_id: str) -> bool:
        """Cancel a pending task."""
        task = self._tasks.get(task_id)
        if task and task.status == TaskStatus.PENDING:
            task.status = TaskStatus.CANCELLED
            return True
        return False

    def get_execution_order(self) -> List[str]:
        """
        Topological sort of tasks respecting dependencies.
        Returns list of task_ids in execution order.
        """
        # Build adjacency list
        graph: Dict[str, List[str]] = {tid: [] for tid in self._tasks}
        in_degree: Dict[str, int] = {tid: 0 for tid in self._tasks}

        for tid, task in self._tasks.items():
            for dep in task.depends_on:
                if dep in graph:
                    graph[dep].append(tid)
                    in_degree[tid] += 1

        # Kahn's algorithm with priority
        ready = []
        for tid in self._tasks:
            if in_degree[tid] == 0:
                task = self._tasks[tid]
                heapq.heappush(ready, (task.priority, task.created_at, tid))

        order = []
        while ready:
            _, _, tid = heapq.heappop(ready)
            order.append(tid)

            for next_tid in graph.get(tid, []):
                in_degree[next_tid] -= 1
                if in_degree[next_tid] == 0:
                    next_task = self._tasks[next_tid]
                    heapq.heappush(ready, (next_task.priority, next_task.created_at, next_tid))

        # Check for cycles
        if len(order) != len(self._tasks):
            remaining = set(self._tasks.keys()) - set(order)
            raise ValueError(f"Dependency cycle detected involving: {remaining}")

        return order

    def run_all(self) -> Dict[str, Any]:
        """
        Execute all tasks in dependency order.

        Returns dict of task_id → result.
        """
        try:
            order = self.get_execution_order()
        except ValueError as e:
            return {'error': str(e)}

        results = {}
        for task_id in order:
            task = self._tasks[task_id]

            if task.status == TaskStatus.CANCELLED:
                continue

            # Check dependencies
            deps_ok = all(
                self._tasks.get(dep, Task(0, 0, '', '')).status == TaskStatus.COMPLETED
                for dep in task.depends_on
            )
            if not deps_ok:
                task.status = TaskStatus.BLOCKED
                task.error = "Dependency failed"
                continue

            # Inject dependency results into args
            for dep_id in task.depends_on:
                if dep_id in results:
                    task.args[f'_dep_{dep_id}'] = results[dep_id]

            # Execute with retries
            result = self._execute_task(task)
            results[task_id] = result

        self._results = results
        return results

    def _execute_task(self, task: Task) -> Any:
        """Execute a single task with retry logic."""
        if task.fn is None:
            task.status = TaskStatus.COMPLETED
            task.result = None
            return None

        for attempt in range(task.max_retries + 1):
            task.status = TaskStatus.RUNNING
            task.started_at = time.time()

            try:
                result = task.fn(**task.args)
                task.status = TaskStatus.COMPLETED
                task.result = result
                task.completed_at = time.time()
                return result
            except Exception as e:
                task.error = str(e)
                task.retry_count = attempt + 1

                if attempt < task.max_retries:
                    task.status = TaskStatus.RETRYING
                    time.sleep(task.retry_delay * (2 ** attempt))
                else:
                    task.status = TaskStatus.FAILED
                    return None

        return None

    def get_summary(self) -> Dict[str, Any]:
        """Get execution summary."""
        statuses = {}
        for task in self._tasks.values():
            s = task.status.value
            statuses[s] = statuses.get(s, 0) + 1

        total_time = 0
        for task in self._tasks.values():
            if task.started_at and task.completed_at:
                total_time += task.completed_at - task.started_at

        return {
            'total_tasks': len(self._tasks),
            'statuses': statuses,
            'total_time': round(total_time, 3),
            'tasks': [
                {
                    'id': t.task_id,
                    'name': t.name,
                    'status': t.status.value,
                    'priority': t.priority,
                    'retries': t.retry_count,
                    'error': t.error,
                }
                for t in self._tasks.values()
            ],
        }

    def clear(self):
        """Clear all tasks."""
        with self._lock:
            self._tasks.clear()
            self._queue.clear()
            self._results.clear()
            self._counter = 0


class Pipeline:
    """
    Linear pipeline: chain functions where output of one feeds into the next.

    Usage:
        pipe = Pipeline("data_pipeline")
        pipe.add_step("fetch", fetch_data)
        pipe.add_step("clean", clean_data)
        pipe.add_step("analyze", analyze)
        result = pipe.run(initial_input="raw_data.csv")
    """

    def __init__(self, name: str = "pipeline"):
        self.name = name
        self.steps: List[Dict] = []

    def add_step(self, name: str, fn: Callable,
                 args: Optional[Dict] = None) -> 'Pipeline':
        """Add a step. Returns self for chaining."""
        self.steps.append({
            'name': name,
            'fn': fn,
            'args': args or {},
        })
        return self

    def run(self, initial_input: Any = None) -> Dict:
        """
        Run the pipeline. Each step receives the previous step's output.

        Returns:
            dict with: result, steps, elapsed, errors
        """
        current = initial_input
        step_results = []
        errors = []
        start = time.time()

        for i, step in enumerate(self.steps):
            step_start = time.time()
            try:
                args = dict(step['args'])
                args['_input'] = current
                current = step['fn'](**args)
                step_results.append({
                    'name': step['name'],
                    'status': 'completed',
                    'elapsed': round(time.time() - step_start, 3),
                })
            except Exception as e:
                errors.append({
                    'step': step['name'],
                    'error': str(e),
                })
                step_results.append({
                    'name': step['name'],
                    'status': 'failed',
                    'error': str(e),
                    'elapsed': round(time.time() - step_start, 3),
                })
                break  # Pipeline stops on error

        return {
            'result': current,
            'steps': step_results,
            'elapsed': round(time.time() - start, 3),
            'errors': errors,
            'completed': len(errors) == 0,
        }


class EventTrigger:
    """
    Simple event-condition-action trigger.

    Usage:
        trigger = EventTrigger("high_cpu")
        trigger.condition = lambda ctx: ctx.get('cpu') > 90
        trigger.action = lambda ctx: alert("CPU high!")
        trigger.check({'cpu': 95})  # → fires
    """

    def __init__(self, name: str, condition: Optional[Callable] = None,
                 action: Optional[Callable] = None,
                 cooldown: float = 60.0):
        self.name = name
        self.condition = condition
        self.action = action
        self.cooldown = cooldown
        self._last_fired: float = 0
        self.fire_count: int = 0

    def check(self, context: Dict) -> bool:
        """Check condition and fire action if met."""
        if self.condition is None or self.action is None:
            return False

        now = time.time()
        if now - self._last_fired < self.cooldown:
            return False

        if self.condition(context):
            self.action(context)
            self._last_fired = now
            self.fire_count += 1
            return True

        return False
