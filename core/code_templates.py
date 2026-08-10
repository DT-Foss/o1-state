"""
Code Template Library — Building Blocks for Constraint-Based Code Generation
==============================================================================
Instead of generating code character-by-character with PPM (which can't plan
over 50+ lines), we use composable templates + constraint matching.

Strategy: User intent → constraint extraction → template selection → composition → AST validation.

Each template is a function that generates valid Python code given parameters.
Templates are composable: a "REST API" template uses "function", "try_except", etc.
"""


# ── Data Structure Templates ──

def template_list_operations(name='process_list', operation='filter'):
    """Generate list processing functions."""
    ops = {
        'filter': f'''def {name}(items, condition):
    """Filter items by condition."""
    return [item for item in items if condition(item)]
''',
        'map': f'''def {name}(items, transform):
    """Apply transform to each item."""
    return [transform(item) for item in items]
''',
        'reduce': f'''def {name}(items, combine, initial=None):
    """Reduce items to single value."""
    result = initial if initial is not None else items[0]
    start = 0 if initial is not None else 1
    for i in range(start, len(items)):
        result = combine(result, items[i])
    return result
''',
        'flatten': f'''def {name}(nested):
    """Flatten a nested list."""
    result = []
    for item in nested:
        if isinstance(item, list):
            result.extend({name}(item))
        else:
            result.append(item)
    return result
''',
        'unique': f'''def {name}(items):
    """Remove duplicates while preserving order."""
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
''',
        'chunk': f'''def {name}(items, size):
    """Split list into chunks of given size."""
    return [items[i:i + size] for i in range(0, len(items), size)]
''',
        'zip_longest': f'''def {name}(*iterables, fillvalue=None):
    """Zip iterables, filling shorter ones with fillvalue."""
    max_len = max(len(it) for it in iterables)
    result = []
    for i in range(max_len):
        row = []
        for it in iterables:
            row.append(it[i] if i < len(it) else fillvalue)
        result.append(tuple(row))
    return result
''',
    }
    return ops.get(operation, ops['filter'])


def template_dict_operations(name='process_dict', operation='merge'):
    """Generate dictionary processing functions."""
    ops = {
        'merge': f'''def {name}(*dicts):
    """Merge multiple dictionaries."""
    result = {{}}
    for d in dicts:
        result.update(d)
    return result
''',
        'invert': f'''def {name}(d):
    """Swap keys and values."""
    return {{v: k for k, v in d.items()}}
''',
        'filter_keys': f'''def {name}(d, keys):
    """Keep only specified keys."""
    return {{k: v for k, v in d.items() if k in keys}}
''',
        'group_by': f'''def {name}(items, key_func):
    """Group items by key function."""
    groups = {{}}
    for item in items:
        key = key_func(item)
        if key not in groups:
            groups[key] = []
        groups[key].append(item)
    return groups
''',
    }
    return ops.get(operation, ops['merge'])


# ── String Templates ──

def template_string_operations(name='process_string', operation='reverse'):
    """Generate string processing functions."""
    ops = {
        'reverse': f'''def {name}(s):
    """Reverse a string."""
    return s[::-1]
''',
        'capitalize_words': f'''def {name}(s):
    """Capitalize first letter of each word."""
    return ' '.join(word.capitalize() for word in s.split())
''',
        'remove_duplicates': f'''def {name}(s):
    """Remove duplicate characters while preserving order."""
    seen = set()
    result = []
    for c in s:
        if c not in seen:
            seen.add(c)
            result.append(c)
    return ''.join(result)
''',
        'count_words': f'''def {name}(s):
    """Count word frequencies."""
    words = s.lower().split()
    freq = {{}}
    for word in words:
        freq[word] = freq.get(word, 0) + 1
    return freq
''',
        'is_anagram': f'''def {name}(s1, s2):
    """Check if two strings are anagrams."""
    return sorted(s1.lower().replace(' ', '')) == sorted(s2.lower().replace(' ', ''))
''',
        'caesar_cipher': f'''def {name}(text, shift):
    """Apply Caesar cipher encryption."""
    result = []
    for c in text:
        if c.isalpha():
            base = ord('A') if c.isupper() else ord('a')
            result.append(chr((ord(c) - base + shift) % 26 + base))
        else:
            result.append(c)
    return ''.join(result)
''',
    }
    return ops.get(operation, ops['reverse'])


# ── Algorithm Templates ──

def template_sorting(name='sort', algorithm='quicksort'):
    """Generate sorting algorithm implementations."""
    algos = {
        'quicksort': f'''def {name}(arr):
    """Quicksort — O(n log n) average, O(n^2) worst."""
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return {name}(left) + middle + {name}(right)
''',
        'mergesort': f'''def {name}(arr):
    """Merge sort — O(n log n) guaranteed."""
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = {name}(arr[:mid])
    right = {name}(arr[mid:])
    return _merge(left, right)

def _merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result
''',
        'bubblesort': f'''def {name}(arr):
    """Bubble sort — O(n^2), simple."""
    arr = arr.copy()
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr
''',
        'insertion_sort': f'''def {name}(arr):
    """Insertion sort — O(n^2), good for small/nearly sorted arrays."""
    arr = arr.copy()
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr
''',
    }
    return algos.get(algorithm, algos['quicksort'])


def template_search(name='search', algorithm='binary'):
    """Generate search algorithm implementations."""
    algos = {
        'binary': f'''def {name}(arr, target):
    """Binary search — O(log n), requires sorted array."""
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
''',
        'linear': f'''def {name}(arr, target):
    """Linear search — O(n)."""
    for i, item in enumerate(arr):
        if item == target:
            return i
    return -1
''',
        'dfs': f'''def {name}(graph, start, target):
    """Depth-first search on a graph (adjacency dict)."""
    visited = set()
    stack = [start]
    while stack:
        node = stack.pop()
        if node == target:
            return True
        if node not in visited:
            visited.add(node)
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    stack.append(neighbor)
    return False
''',
        'bfs': f'''def {name}(graph, start, target):
    """Breadth-first search on a graph (adjacency dict)."""
    from collections import deque
    visited = set([start])
    queue = deque([start])
    while queue:
        node = queue.popleft()
        if node == target:
            return True
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return False
''',
    }
    return algos.get(algorithm, algos['binary'])


# ── Data Processing Templates ──

def template_csv_operations(name='process_csv', operation='read'):
    """Generate CSV handling code."""
    ops = {
        'read': f'''import csv

def {name}(filepath):
    """Read CSV file and return list of dicts."""
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)
''',
        'write': f'''import csv

def {name}(filepath, data, fieldnames=None):
    """Write list of dicts to CSV file."""
    if not data:
        return
    if fieldnames is None:
        fieldnames = list(data[0].keys())
    with open(filepath, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
''',
        'filter': f'''import csv

def {name}(filepath, column, value):
    """Read CSV and filter rows where column equals value."""
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return [row for row in reader if row.get(column) == value]
''',
    }
    return ops.get(operation, ops['read'])


def template_json_operations(name='process_json', operation='read'):
    """Generate JSON handling code."""
    ops = {
        'read': f'''import json

def {name}(filepath):
    """Read JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)
''',
        'write': f'''import json

def {name}(filepath, data, indent=2):
    """Write data to JSON file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
''',
        'flatten': f'''def {name}(nested, prefix=''):
    """Flatten nested JSON to dot-notation keys."""
    result = {{}}
    for key, value in nested.items():
        full_key = f"{{prefix}}.{{key}}" if prefix else key
        if isinstance(value, dict):
            result.update({name}(value, full_key))
        else:
            result[full_key] = value
    return result
''',
    }
    return ops.get(operation, ops['read'])


# ── Network Templates ──

def template_http(name='http_request', operation='get'):
    """Generate HTTP request code."""
    ops = {
        'get': f'''import urllib.request
import json

def {name}(url, headers=None):
    """Make HTTP GET request."""
    req = urllib.request.Request(url)
    if headers:
        for key, value in headers.items():
            req.add_header(key, value)
    with urllib.request.urlopen(req) as response:
        data = response.read().decode('utf-8')
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            return data
''',
        'post': f'''import urllib.request
import json

def {name}(url, data, headers=None):
    """Make HTTP POST request."""
    body = json.dumps(data).encode('utf-8')
    req = urllib.request.Request(url, data=body, method='POST')
    req.add_header('Content-Type', 'application/json')
    if headers:
        for key, value in headers.items():
            req.add_header(key, value)
    with urllib.request.urlopen(req) as response:
        return json.loads(response.read().decode('utf-8'))
''',
        'server': f'''from http.server import HTTPServer, BaseHTTPRequestHandler
import json

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        response = {{"status": "ok", "path": self.path}}
        self.wfile.write(json.dumps(response).encode())

def {name}(port=8080):
    """Start a simple HTTP server."""
    server = HTTPServer(('', port), Handler)
    print(f"Server running on port {{port}}")
    server.serve_forever()
''',
    }
    return ops.get(operation, ops['get'])


# ── Class Templates ──

def template_class(name='MyClass', features=None):
    """Generate a class with specified features."""
    if features is None:
        features = ['init', 'repr']

    parts = [f'class {name}:']
    parts.append(f'    """A {name} class."""')
    parts.append('')

    if 'init' in features:
        parts.append(f'    def __init__(self, **kwargs):')
        parts.append(f'        for key, value in kwargs.items():')
        parts.append(f'            setattr(self, key, value)')
        parts.append('')

    if 'repr' in features:
        parts.append(f'    def __repr__(self):')
        parts.append(f'        attrs = ", ".join(f"{{k}}={{v!r}}" for k, v in self.__dict__.items())')
        parts.append(f'        return f"{name}({{attrs}})"')
        parts.append('')

    if 'eq' in features:
        parts.append(f'    def __eq__(self, other):')
        parts.append(f'        if not isinstance(other, {name}):')
        parts.append(f'            return False')
        parts.append(f'        return self.__dict__ == other.__dict__')
        parts.append('')

    if 'to_dict' in features:
        parts.append(f'    def to_dict(self):')
        parts.append(f'        """Convert to dictionary."""')
        parts.append(f'        return dict(self.__dict__)')
        parts.append('')

    if 'from_dict' in features:
        parts.append(f'    @classmethod')
        parts.append(f'    def from_dict(cls, d):')
        parts.append(f'        """Create from dictionary."""')
        parts.append(f'        return cls(**d)')
        parts.append('')

    return '\n'.join(parts) + '\n'


# ── Utility Templates ──

def template_timer():
    """Generate a timing decorator."""
    return '''import time
import functools

def timer(func):
    """Decorator that measures execution time."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"{func.__name__} took {elapsed:.4f}s")
        return result
    return wrapper
'''


def template_retry():
    """Generate a retry decorator."""
    return '''import time
import functools

def retry(max_attempts=3, delay=1.0, backoff=2.0):
    """Decorator that retries on exception."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            current_delay = delay
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    if attempts == max_attempts:
                        raise
                    print(f"Retry {attempts}/{max_attempts} after {current_delay}s: {e}")
                    time.sleep(current_delay)
                    current_delay *= backoff
        return wrapper
    return decorator
'''


def template_cache():
    """Generate a simple cache decorator."""
    return '''import functools

def cache(func):
    """Simple memoization cache decorator."""
    memo = {}
    @functools.wraps(func)
    def wrapper(*args):
        if args not in memo:
            memo[args] = func(*args)
        return memo[args]
    return wrapper
'''


def template_logger():
    """Generate a logging setup."""
    return '''import logging

def setup_logger(name, level=logging.INFO, logfile=None):
    """Setup a logger with console and optional file output."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler (optional)
    if logfile:
        fh = logging.FileHandler(logfile)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
'''


# ── Template Registry ──

TEMPLATE_REGISTRY = {
    # Algorithms
    'sort a list': lambda: template_sorting(),
    'sort list': lambda: template_sorting(),
    'sort array': lambda: template_sorting(),
    'sort': lambda: template_sorting(),
    'quicksort': lambda: template_sorting(algorithm='quicksort'),
    'mergesort': lambda: template_sorting(algorithm='mergesort'),
    'bubblesort': lambda: template_sorting(algorithm='bubblesort'),
    'insertion sort': lambda: template_sorting(algorithm='insertion_sort'),
    'binary search': lambda: template_search(algorithm='binary'),
    'linear search': lambda: template_search(algorithm='linear'),
    'dfs': lambda: template_search(algorithm='dfs'),
    'bfs': lambda: template_search(algorithm='bfs'),
    'depth first': lambda: template_search(algorithm='dfs'),
    'breadth first': lambda: template_search(algorithm='bfs'),

    'sort by': lambda: """def sort_by_key(items, key_index=1):
    \"\"\"Sort a list of tuples/lists by a specific element.\"\"\"
    return sorted(items, key=lambda x: x[key_index])

# Example:
# data = [('Alice', 25), ('Bob', 20), ('Charlie', 30)]
# sort_by_key(data, 1)  # Sort by age → [('Bob', 20), ('Alice', 25), ('Charlie', 30)]
""",
    'sort tuple': lambda: """def sort_tuples(tuples, key_index=1):
    \"\"\"Sort tuples by the specified element (default: second element).\"\"\"
    return sorted(tuples, key=lambda x: x[key_index])

# Example:
# data = [('Alice', 25), ('Bob', 20), ('Charlie', 30)]
# sort_tuples(data)  # → [('Bob', 20), ('Alice', 25), ('Charlie', 30)]
""",
    'second element': lambda: """def sort_by_second(items):
    \"\"\"Sort a list of tuples by the second element.\"\"\"
    return sorted(items, key=lambda x: x[1])

# Example:
# data = [('Alice', 25), ('Bob', 20), ('Charlie', 30)]
# sort_by_second(data)  # → [('Bob', 20), ('Alice', 25), ('Charlie', 30)]
""",

    # Data structures
    'filter': lambda: template_list_operations(operation='filter'),
    'map': lambda: template_list_operations(operation='map'),
    'reduce': lambda: template_list_operations(operation='reduce'),
    'flatten': lambda: template_list_operations(operation='flatten'),
    'unique': lambda: template_list_operations(operation='unique'),
    'deduplicate': lambda: template_list_operations(operation='unique'),
    'chunk': lambda: template_list_operations(operation='chunk'),
    'merge dict': lambda: template_dict_operations(operation='merge'),
    'invert dict': lambda: template_dict_operations(operation='invert'),
    'group by': lambda: template_dict_operations(operation='group_by'),

    # Data structure implementations
    'stack': lambda: """class Stack:
    \"\"\"Stack data structure (LIFO).\"\"\"
    def __init__(self):
        self._items = []

    def push(self, item):
        self._items.append(item)

    def pop(self):
        if self.is_empty():
            raise IndexError('pop from empty stack')
        return self._items.pop()

    def peek(self):
        if self.is_empty():
            raise IndexError('peek at empty stack')
        return self._items[-1]

    def is_empty(self):
        return len(self._items) == 0

    def size(self):
        return len(self._items)

    def __repr__(self):
        return f'Stack({self._items})'
""",
    'queue': lambda: """from collections import deque

class Queue:
    \"\"\"Queue data structure (FIFO).\"\"\"
    def __init__(self):
        self._items = deque()

    def enqueue(self, item):
        self._items.append(item)

    def dequeue(self):
        if self.is_empty():
            raise IndexError('dequeue from empty queue')
        return self._items.popleft()

    def peek(self):
        if self.is_empty():
            raise IndexError('peek at empty queue')
        return self._items[0]

    def is_empty(self):
        return len(self._items) == 0

    def size(self):
        return len(self._items)
""",
    'linked list': lambda: """class Node:
    def __init__(self, data, next_node=None):
        self.data = data
        self.next = next_node

class LinkedList:
    def __init__(self):
        self.head = None

    def append(self, data):
        if not self.head:
            self.head = Node(data)
            return
        current = self.head
        while current.next:
            current = current.next
        current.next = Node(data)

    def prepend(self, data):
        self.head = Node(data, self.head)

    def delete(self, data):
        if not self.head:
            return
        if self.head.data == data:
            self.head = self.head.next
            return
        current = self.head
        while current.next:
            if current.next.data == data:
                current.next = current.next.next
                return
            current = current.next

    def __iter__(self):
        current = self.head
        while current:
            yield current.data
            current = current.next

    def __repr__(self):
        return ' -> '.join(str(x) for x in self)
""",

    # String
    'reverse string': lambda: template_string_operations(operation='reverse'),
    'reverse a string': lambda: template_string_operations(operation='reverse'),
    'reverses a string': lambda: template_string_operations(operation='reverse'),
    'string reversal': lambda: template_string_operations(operation='reverse'),
    'capitalize': lambda: template_string_operations(operation='capitalize_words'),
    'word count': lambda: template_string_operations(operation='count_words'),
    'count words': lambda: template_string_operations(operation='count_words'),
    'anagram': lambda: template_string_operations(operation='is_anagram'),
    'caesar': lambda: template_string_operations(operation='caesar_cipher'),
    'cipher': lambda: template_string_operations(operation='caesar_cipher'),

    # File/Data
    'read csv': lambda: template_csv_operations(operation='read'),
    'read a csv': lambda: template_csv_operations(operation='read'),
    'parse csv': lambda: template_csv_operations(operation='read'),
    'write csv': lambda: template_csv_operations(operation='write'),
    'filter csv': lambda: template_csv_operations(operation='filter'),
    'read json': lambda: template_json_operations(operation='read'),
    'read a json': lambda: template_json_operations(operation='read'),
    'parse json': lambda: template_json_operations(operation='read'),
    'write json': lambda: template_json_operations(operation='write'),
    'flatten json': lambda: template_json_operations(operation='flatten'),

    # Network
    'http get': lambda: template_http(operation='get'),
    'http post': lambda: template_http(operation='post'),
    'http server': lambda: template_http(operation='server'),
    'web server': lambda: template_http(operation='server'),
    'api': lambda: template_http(operation='get'),

    # Utilities
    'timer': lambda: template_timer(),
    'retry': lambda: template_retry(),
    'cache': lambda: template_cache(),
    'memoize': lambda: template_cache(),
    'logger': lambda: template_logger(),
    'logging': lambda: template_logger(),

    # Data structures
    'binary tree': lambda: """class TreeNode:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class BinaryTree:
    def __init__(self):
        self.root = None

    def insert(self, val):
        if not self.root:
            self.root = TreeNode(val)
            return
        queue = [self.root]
        while queue:
            node = queue.pop(0)
            if not node.left:
                node.left = TreeNode(val)
                return
            if not node.right:
                node.right = TreeNode(val)
                return
            queue.extend([node.left, node.right])

    def inorder(self, node=None, first=True):
        if first:
            node = self.root
        if not node:
            return []
        return self.inorder(node.left, False) + [node.val] + self.inorder(node.right, False)
""",
    'min heap': lambda: """class MinHeap:
    def __init__(self):
        self.heap = []

    def push(self, val):
        self.heap.append(val)
        i = len(self.heap) - 1
        while i > 0:
            parent = (i - 1) // 2
            if self.heap[i] < self.heap[parent]:
                self.heap[i], self.heap[parent] = self.heap[parent], self.heap[i]
                i = parent
            else:
                break

    def pop(self):
        if not self.heap:
            raise IndexError("pop from empty heap")
        if len(self.heap) == 1:
            return self.heap.pop()
        root = self.heap[0]
        self.heap[0] = self.heap.pop()
        self._sift_down(0)
        return root

    def _sift_down(self, i):
        n = len(self.heap)
        while True:
            smallest = i
            left, right = 2 * i + 1, 2 * i + 2
            if left < n and self.heap[left] < self.heap[smallest]:
                smallest = left
            if right < n and self.heap[right] < self.heap[smallest]:
                smallest = right
            if smallest == i:
                break
            self.heap[i], self.heap[smallest] = self.heap[smallest], self.heap[i]
            i = smallest

    def peek(self):
        return self.heap[0] if self.heap else None
""",
    'balanced parentheses': lambda: """def is_balanced(s):
    \"\"\"Check if parentheses/brackets are balanced.\"\"\"
    stack = []
    pairs = {')': '(', ']': '[', '}': '{'}
    for c in s:
        if c in '([{':
            stack.append(c)
        elif c in ')]}':
            if not stack or stack[-1] != pairs[c]:
                return False
            stack.pop()
    return len(stack) == 0


# Examples
print(is_balanced("({[]})"))   # True
print(is_balanced("([)]"))     # False
print(is_balanced("((()))"))   # True
""",
    'merge sorted': lambda: """def merge_sorted(a, b):
    \"\"\"Merge two sorted lists into one sorted list.\"\"\"
    result = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            result.append(a[i])
            i += 1
        else:
            result.append(b[j])
            j += 1
    result.extend(a[i:])
    result.extend(b[j:])
    return result


# Example
print(merge_sorted([1, 3, 5], [2, 4, 6]))  # [1, 2, 3, 4, 5, 6]
""",
    'merge two sorted': lambda: """def merge_sorted(a, b):
    \"\"\"Merge two sorted lists into one sorted list.\"\"\"
    result = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            result.append(a[i])
            i += 1
        else:
            result.append(b[j])
            j += 1
    result.extend(a[i:])
    result.extend(b[j:])
    return result


# Example
print(merge_sorted([1, 3, 5], [2, 4, 6]))  # [1, 2, 3, 4, 5, 6]
""",
    'sorted lists': lambda: """def merge_sorted(a, b):
    \"\"\"Merge two sorted lists into one sorted list.\"\"\"
    result = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            result.append(a[i])
            i += 1
        else:
            result.append(b[j])
            j += 1
    result.extend(a[i:])
    result.extend(b[j:])
    return result


# Example
print(merge_sorted([1, 3, 5], [2, 4, 6]))  # [1, 2, 3, 4, 5, 6]
""",
    'find duplicates': lambda: """def find_duplicates(lst):
    \"\"\"Find all duplicate elements in a list.\"\"\"
    seen = set()
    duplicates = set()
    for item in lst:
        if item in seen:
            duplicates.add(item)
        seen.add(item)
    return list(duplicates)


# Example
print(find_duplicates([1, 2, 3, 2, 4, 5, 3]))  # [2, 3]
""",
    'duplicates': lambda: """def find_duplicates(lst):
    \"\"\"Find all duplicate elements in a list.\"\"\"
    seen = set()
    duplicates = set()
    for item in lst:
        if item in seen:
            duplicates.add(item)
        seen.add(item)
    return list(duplicates)


# Example
print(find_duplicates([1, 2, 3, 2, 4, 5, 3]))  # [2, 3]
""",
    'two stacks': lambda: """class QueueFromStacks:
    \"\"\"Queue implemented using two stacks.\"\"\"
    def __init__(self):
        self.in_stack = []
        self.out_stack = []

    def enqueue(self, val):
        self.in_stack.append(val)

    def dequeue(self):
        if not self.out_stack:
            while self.in_stack:
                self.out_stack.append(self.in_stack.pop())
        if not self.out_stack:
            raise IndexError("dequeue from empty queue")
        return self.out_stack.pop()

    def peek(self):
        if not self.out_stack:
            while self.in_stack:
                self.out_stack.append(self.in_stack.pop())
        return self.out_stack[-1] if self.out_stack else None

    def is_empty(self):
        return not self.in_stack and not self.out_stack
""",
    'longest common substring': lambda: """def longest_common_substring(s1, s2):
    \"\"\"Find the longest common substring between two strings.\"\"\"
    m, n = len(s1), len(s2)
    longest = 0
    end_idx = 0
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                curr[j] = prev[j-1] + 1
                if curr[j] > longest:
                    longest = curr[j]
                    end_idx = i
        prev = curr
    return s1[end_idx - longest:end_idx]


# Example
print(longest_common_substring("abcdef", "zbcdf"))  # "bcd"
""",

    # Classes
    'class': lambda: template_class(),
    'dataclass': lambda: template_class(features=['init', 'repr', 'eq', 'to_dict', 'from_dict']),

    # Sorting variants (with spaces)
    'bubble sort': lambda: template_sorting(algorithm='bubblesort'),
    'merge sort': lambda: template_sorting(algorithm='mergesort'),
    'quick sort': lambda: template_sorting(algorithm='quicksort'),

    # Math / numeric
    'fibonacci': lambda: """def fibonacci(n):
    \"\"\"Return the nth Fibonacci number.\"\"\"
    if n <= 0:
        return 0
    if n == 1:
        return 1
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b


# Examples
for i in range(10):
    print(f"F({i}) = {fibonacci(i)}")
""",
    'prime': lambda: """def is_prime(n):
    \"\"\"Check if n is a prime number.\"\"\"
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


# Examples
primes = [n for n in range(100) if is_prime(n)]
print(f"Primes under 100: {primes}")
""",
    'factorial': lambda: """def factorial(n):
    \"\"\"Calculate n! (factorial of n).\"\"\"
    if n < 0:
        raise ValueError("Factorial not defined for negative numbers")
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


# Examples
for i in range(10):
    print(f"{i}! = {factorial(i)}")
""",
    'read a file': lambda: """def read_file(path):
    \"\"\"Read and return file contents.\"\"\"
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def read_lines(path):
    \"\"\"Read file and return list of lines.\"\"\"
    with open(path, 'r', encoding='utf-8') as f:
        return [line.rstrip('\\n') for line in f]


def read_file_safe(path):
    \"\"\"Read file with error handling.\"\"\"
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"File not found: {path}")
        return None
    except PermissionError:
        print(f"Permission denied: {path}")
        return None
""",
    'read file': lambda: """def read_file(path):
    \"\"\"Read and return file contents.\"\"\"
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def read_lines(path):
    \"\"\"Read file and return list of lines.\"\"\"
    with open(path, 'r', encoding='utf-8') as f:
        return [line.rstrip('\\n') for line in f]


def read_file_safe(path):
    \"\"\"Read file with error handling.\"\"\"
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"File not found: {path}")
        return None
    except PermissionError:
        print(f"Permission denied: {path}")
        return None
""",
    'write to a file': lambda: """def write_file(path, content):
    \"\"\"Write content to file.\"\"\"
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)


def append_file(path, content):
    \"\"\"Append content to file.\"\"\"
    with open(path, 'a', encoding='utf-8') as f:
        f.write(content)
""",
    'write a file': lambda: """def write_file(path, content):
    \"\"\"Write content to file.\"\"\"
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)


def append_file(path, content):
    \"\"\"Append content to file.\"\"\"
    with open(path, 'a', encoding='utf-8') as f:
        f.write(content)
""",
    'write file': lambda: """def write_file(path, content):
    \"\"\"Write content to file.\"\"\"
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)


def append_file(path, content):
    \"\"\"Append content to file.\"\"\"
    with open(path, 'a', encoding='utf-8') as f:
        f.write(content)
""",
    'web scraper': lambda: """import urllib.request
from html.parser import HTMLParser


class LinkExtractor(HTMLParser):
    \"\"\"Extract all links from HTML.\"\"\"
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            for name, value in attrs:
                if name == 'href':
                    self.links.append(value)


def scrape_links(url):
    \"\"\"Fetch URL and extract all links.\"\"\"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as resp:
        html = resp.read().decode('utf-8')
    parser = LinkExtractor()
    parser.feed(html)
    return parser.links


# Example
# links = scrape_links('https://example.com')
# for link in links:
#     print(link)
""",
    'calculator': lambda: """def calculator(expression):
    \"\"\"Evaluate a simple math expression safely.\"\"\"
    allowed = set('0123456789+-*/().% ')
    if not all(c in allowed for c in expression):
        raise ValueError("Invalid characters in expression")
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return result
    except (SyntaxError, ZeroDivisionError) as e:
        raise ValueError(f"Invalid expression: {e}")


# Examples
print(calculator("2 + 3 * 4"))   # 14
print(calculator("(10 + 5) / 3")) # 5.0
print(calculator("2 ** 10"))      # 1024
""",
    'todo list': lambda: """class TodoList:
    \"\"\"Simple TODO list manager.\"\"\"
    def __init__(self):
        self.tasks = []

    def add(self, task, priority=3):
        self.tasks.append({'task': task, 'priority': priority, 'done': False})

    def complete(self, index):
        if 0 <= index < len(self.tasks):
            self.tasks[index]['done'] = True

    def remove(self, index):
        if 0 <= index < len(self.tasks):
            self.tasks.pop(index)

    def show(self, show_done=False):
        for i, t in enumerate(self.tasks):
            if not show_done and t['done']:
                continue
            status = '[x]' if t['done'] else '[ ]'
            print(f"{i}: {status} (P{t['priority']}) {t['task']}")

    def pending(self):
        return [t for t in self.tasks if not t['done']]


# Example
todo = TodoList()
todo.add("Buy groceries", priority=2)
todo.add("Write report", priority=1)
todo.add("Clean desk", priority=3)
todo.show()
""",
    'todo': lambda: """class TodoList:
    \"\"\"Simple TODO list manager.\"\"\"
    def __init__(self):
        self.tasks = []

    def add(self, task, priority=3):
        self.tasks.append({'task': task, 'priority': priority, 'done': False})

    def complete(self, index):
        if 0 <= index < len(self.tasks):
            self.tasks[index]['done'] = True

    def remove(self, index):
        if 0 <= index < len(self.tasks):
            self.tasks.pop(index)

    def show(self, show_done=False):
        for i, t in enumerate(self.tasks):
            if not show_done and t['done']:
                continue
            status = '[x]' if t['done'] else '[ ]'
            print(f"{i}: {status} (P{t['priority']}) {t['task']}")

    def pending(self):
        return [t for t in self.tasks if not t['done']]


# Example
todo = TodoList()
todo.add("Buy groceries", priority=2)
todo.add("Write report", priority=1)
todo.add("Clean desk", priority=3)
todo.show()
""",
    'password': lambda: """import secrets
import string


def generate_password(length=16, uppercase=True, digits=True, special=True):
    \"\"\"Generate a secure random password.\"\"\"
    chars = string.ascii_lowercase
    if uppercase:
        chars += string.ascii_uppercase
    if digits:
        chars += string.digits
    if special:
        chars += string.punctuation
    return ''.join(secrets.choice(chars) for _ in range(length))


# Examples
print(generate_password())          # Full strength
print(generate_password(12, special=False))  # No special chars
""",
    'matrix': lambda: """def matrix_multiply(a, b):
    \"\"\"Multiply two matrices (lists of lists).\"\"\"
    rows_a, cols_a = len(a), len(a[0])
    rows_b, cols_b = len(b), len(b[0])
    if cols_a != rows_b:
        raise ValueError(f"Cannot multiply {rows_a}x{cols_a} by {rows_b}x{cols_b}")
    result = [[0] * cols_b for _ in range(rows_a)]
    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(cols_a):
                result[i][j] += a[i][k] * b[k][j]
    return result


def matrix_transpose(m):
    \"\"\"Transpose a matrix.\"\"\"
    return [list(row) for row in zip(*m)]


# Example
a = [[1, 2], [3, 4]]
b = [[5, 6], [7, 8]]
print(matrix_multiply(a, b))  # [[19, 22], [43, 50]]
print(matrix_transpose(a))    # [[1, 3], [2, 4]]
""",
}


def find_template(request: str) -> str:
    """Find the best matching template for a natural language request."""
    request_lower = request.lower()

    # Direct keyword match
    best_match = None
    best_score = 0

    for keyword, gen_func in TEMPLATE_REGISTRY.items():
        if keyword in request_lower:
            score = len(keyword)  # Longer match = more specific
            if score > best_score:
                best_score = score
                best_match = gen_func

    if best_match:
        return best_match()

    return None
