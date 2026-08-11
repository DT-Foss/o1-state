"""
Data Processing — CSV, JSON, Text Analysis Without Pandas
===========================================================
Pure Python data processing for FOSS-KI.

ChatGPT/Claude can analyze data. So must we.

Capabilities:
  1. CSV reading/writing with type inference
  2. JSON flattening and querying
  3. Basic statistics (mean, median, std, percentiles)
  4. Data filtering and grouping
  5. Frequency analysis
  6. Correlation (Pearson)
  7. Data transformation (sort, deduplicate, sample)
"""

import csv
import json
import math
import io
import re
from typing import List, Dict, Any, Optional, Tuple, Callable
from collections import Counter, defaultdict


# === Statistics ===

def mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def median(values: List[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    if n % 2 == 0:
        return (s[n//2 - 1] + s[n//2]) / 2
    return s[n//2]


def variance(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return sum((x - m) ** 2 for x in values) / (len(values) - 1)


def std(values: List[float]) -> float:
    return math.sqrt(variance(values))


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (p / 100)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] * (c - k) + s[c] * (k - f)


def mode(values: List) -> Any:
    if not values:
        return None
    counts = Counter(values)
    return counts.most_common(1)[0][0]


def pearson_correlation(x: List[float], y: List[float]) -> float:
    """Pearson correlation coefficient."""
    n = min(len(x), len(y))
    if n < 2:
        return 0.0
    mx, my = mean(x[:n]), mean(y[:n])
    sx, sy = std(x[:n]), std(y[:n])
    if sx == 0 or sy == 0:
        return 0.0
    cov = sum((x[i] - mx) * (y[i] - my) for i in range(n)) / (n - 1)
    return cov / (sx * sy)


def describe(values: List[float]) -> Dict[str, float]:
    """Descriptive statistics summary."""
    if not values:
        return {}
    return {
        'count': len(values),
        'mean': round(mean(values), 4),
        'std': round(std(values), 4),
        'min': min(values),
        'p25': round(percentile(values, 25), 4),
        'median': round(median(values), 4),
        'p75': round(percentile(values, 75), 4),
        'max': max(values),
        'sum': sum(values),
    }


# === CSV Processing ===

class DataTable:
    """
    In-memory table with column operations.
    Think of it as a minimal DataFrame without pandas.
    """

    def __init__(self, rows: Optional[List[Dict]] = None,
                 columns: Optional[List[str]] = None):
        self.rows: List[Dict] = rows or []
        self._columns = columns

    @property
    def columns(self) -> List[str]:
        if self._columns:
            return self._columns
        if self.rows:
            return list(self.rows[0].keys())
        return []

    def __len__(self):
        return len(self.rows)

    def __repr__(self):
        return f"DataTable({len(self.rows)} rows, {len(self.columns)} cols)"

    @classmethod
    def from_csv(cls, data: str, delimiter: str = ',',
                 has_header: bool = True) -> 'DataTable':
        """Parse CSV string into DataTable."""
        reader = csv.reader(io.StringIO(data), delimiter=delimiter)
        rows_list = list(reader)
        if not rows_list:
            return cls()

        if has_header:
            headers = rows_list[0]
            rows = []
            for row in rows_list[1:]:
                d = {}
                for i, h in enumerate(headers):
                    d[h.strip()] = _infer_type(row[i].strip()) if i < len(row) else None
                rows.append(d)
            return cls(rows, headers)
        else:
            headers = [f"col_{i}" for i in range(len(rows_list[0]))]
            rows = []
            for row in rows_list:
                d = {headers[i]: _infer_type(row[i].strip()) if i < len(row) else None
                     for i in range(len(headers))}
                rows.append(d)
            return cls(rows, headers)

    @classmethod
    def from_json(cls, data: str) -> 'DataTable':
        """Parse JSON array of objects into DataTable."""
        parsed = json.loads(data)
        if isinstance(parsed, list):
            return cls(parsed)
        elif isinstance(parsed, dict):
            return cls([parsed])
        return cls()

    def to_csv(self, delimiter: str = ',') -> str:
        """Export to CSV string."""
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=self.columns, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(self.rows)
        return output.getvalue()

    def to_json(self, indent: int = 2) -> str:
        """Export to JSON string."""
        return json.dumps(self.rows, indent=indent, default=str)

    def head(self, n: int = 5) -> 'DataTable':
        """First n rows."""
        return DataTable(self.rows[:n], self._columns)

    def tail(self, n: int = 5) -> 'DataTable':
        """Last n rows."""
        return DataTable(self.rows[-n:], self._columns)

    def column(self, name: str) -> List:
        """Get all values from a column."""
        return [row.get(name) for row in self.rows]

    def column_numeric(self, name: str) -> List[float]:
        """Get numeric values from a column (skip non-numeric)."""
        values = []
        for row in self.rows:
            v = row.get(name)
            if isinstance(v, (int, float)):
                values.append(float(v))
        return values

    def select(self, *columns: str) -> 'DataTable':
        """Select specific columns."""
        new_rows = [{c: row.get(c) for c in columns} for row in self.rows]
        return DataTable(new_rows, list(columns))

    def filter(self, fn: Callable) -> 'DataTable':
        """Filter rows by predicate function."""
        return DataTable([r for r in self.rows if fn(r)], self._columns)

    def sort_by(self, column: str, reverse: bool = False) -> 'DataTable':
        """Sort by column."""
        sorted_rows = sorted(
            self.rows,
            key=lambda r: (r.get(column) is None, r.get(column)),
            reverse=reverse
        )
        return DataTable(sorted_rows, self._columns)

    def group_by(self, column: str) -> Dict[Any, 'DataTable']:
        """Group rows by column value."""
        groups = defaultdict(list)
        for row in self.rows:
            key = row.get(column)
            groups[key].append(row)
        return {k: DataTable(v, self._columns) for k, v in groups.items()}

    def aggregate(self, column: str, fn: str = 'mean') -> float:
        """Aggregate a numeric column (mean, sum, min, max, count, std, median)."""
        values = self.column_numeric(column)
        if not values:
            return 0.0
        ops = {
            'mean': lambda v: mean(v),
            'sum': lambda v: sum(v),
            'min': lambda v: min(v),
            'max': lambda v: max(v),
            'count': lambda v: len(v),
            'std': lambda v: std(v),
            'median': lambda v: median(v),
        }
        return ops.get(fn, ops['mean'])(values)

    def add_column(self, name: str, fn: Callable) -> 'DataTable':
        """Add a computed column."""
        new_rows = []
        for row in self.rows:
            r = dict(row)
            r[name] = fn(row)
            new_rows.append(r)
        cols = list(self.columns) + [name] if name not in self.columns else self.columns
        return DataTable(new_rows, cols)

    def unique(self, column: str) -> List:
        """Unique values in a column."""
        seen = set()
        result = []
        for row in self.rows:
            v = row.get(column)
            if v not in seen:
                seen.add(v)
                result.append(v)
        return result

    def value_counts(self, column: str) -> List[Tuple[Any, int]]:
        """Frequency count for a column."""
        return Counter(row.get(column) for row in self.rows).most_common()

    def deduplicate(self, columns: Optional[List[str]] = None) -> 'DataTable':
        """Remove duplicate rows (by specified columns or all)."""
        seen = set()
        result = []
        for row in self.rows:
            if columns:
                key = tuple(row.get(c) for c in columns)
            else:
                key = tuple(sorted(row.items()))
            if key not in seen:
                seen.add(key)
                result.append(row)
        return DataTable(result, self._columns)

    def sample(self, n: int = 5) -> 'DataTable':
        """Random sample of n rows."""
        import random
        sampled = random.sample(self.rows, min(n, len(self.rows)))
        return DataTable(sampled, self._columns)

    def describe_column(self, column: str) -> Dict:
        """Full statistics for a column."""
        values = self.column_numeric(column)
        if values:
            return describe(values)
        # Categorical
        all_values = self.column(column)
        return {
            'count': len(all_values),
            'unique': len(set(all_values)),
            'top': mode(all_values),
            'freq': Counter(all_values).most_common(1)[0][1] if all_values else 0,
        }

    def correlation(self, col1: str, col2: str) -> float:
        """Pearson correlation between two numeric columns."""
        return pearson_correlation(
            self.column_numeric(col1),
            self.column_numeric(col2)
        )

    def summary(self) -> str:
        """Human-readable table summary."""
        lines = [f"DataTable: {len(self.rows)} rows × {len(self.columns)} columns"]
        lines.append(f"Columns: {', '.join(self.columns)}")
        for col in self.columns:
            values = self.column_numeric(col)
            if values:
                lines.append(f"  {col}: mean={mean(values):.2f}, std={std(values):.2f}, "
                           f"min={min(values)}, max={max(values)}")
            else:
                uniq = len(set(self.column(col)))
                lines.append(f"  {col}: {uniq} unique values")
        return '\n'.join(lines)

    def print_table(self, max_rows: int = 10) -> str:
        """Format as ASCII table."""
        if not self.rows:
            return "(empty table)"

        cols = self.columns
        # Calculate column widths
        widths = {c: len(c) for c in cols}
        for row in self.rows[:max_rows]:
            for c in cols:
                v = str(row.get(c, ''))
                widths[c] = max(widths[c], min(len(v), 30))

        # Header
        header = ' | '.join(c.ljust(widths[c]) for c in cols)
        sep = '-+-'.join('-' * widths[c] for c in cols)
        lines = [header, sep]

        # Rows
        for row in self.rows[:max_rows]:
            line = ' | '.join(str(row.get(c, '')).ljust(widths[c])[:widths[c]] for c in cols)
            lines.append(line)

        if len(self.rows) > max_rows:
            lines.append(f"... ({len(self.rows) - max_rows} more rows)")

        return '\n'.join(lines)


# === Helpers ===

def _infer_type(value: str) -> Any:
    """Infer Python type from string value."""
    if not value:
        return None
    # Try int
    try:
        return int(value)
    except ValueError:
        pass
    # Try float
    try:
        return float(value)
    except ValueError:
        pass
    # Boolean
    if value.lower() in ('true', 'yes'):
        return True
    if value.lower() in ('false', 'no'):
        return False
    # None
    if value.lower() in ('null', 'none', 'na', 'n/a', ''):
        return None
    return value


def flatten_json(data: Any, prefix: str = '') -> Dict[str, Any]:
    """Flatten nested JSON into dot-notation keys."""
    result = {}
    if isinstance(data, dict):
        for key, value in data.items():
            new_key = f"{prefix}.{key}" if prefix else key
            result.update(flatten_json(value, new_key))
    elif isinstance(data, list):
        for i, item in enumerate(data):
            new_key = f"{prefix}[{i}]"
            result.update(flatten_json(item, new_key))
    else:
        result[prefix] = data
    return result


def json_query(data: Any, path: str) -> Any:
    """Query JSON data with dot-notation path. Supports [n] for arrays."""
    parts = re.findall(r'(\w+)|\[(\d+)\]', path)
    current = data
    for key, idx in parts:
        if key:
            if isinstance(current, dict):
                current = current.get(key)
            else:
                return None
        elif idx:
            if isinstance(current, list) and int(idx) < len(current):
                current = current[int(idx)]
            else:
                return None
        if current is None:
            return None
    return current
