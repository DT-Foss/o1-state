"""
Knowledge Harvester — Extract facts from Python stdlib and installed packages
==============================================================================
Ported from FORGE's knowledge_harvester.py. Crawls Python modules via
inspect and extracts (Subject, Relation, Object) triplets about functions,
classes, methods, and their semantic properties.

This is the fastest path to 10K+ facts without any LLM calls.
Pure introspection, deterministic, millisecond extraction.
"""

import inspect
import importlib
import logging

logger = logging.getLogger(__name__)

# Modules to harvest (stdlib + common packages if installed)
STDLIB_MODULES = [
    'os', 'sys', 'json', 'math', 'datetime', 're', 'shutil',
    'itertools', 'collections', 'pathlib', 'csv', 'sqlite3',
    'hashlib', 'base64', 'zlib', 'argparse', 'logging',
    'socket', 'http', 'urllib', 'email', 'html', 'xml',
    'subprocess', 'threading', 'multiprocessing', 'asyncio',
    'struct', 'array', 'queue', 'heapq', 'bisect',
    'functools', 'operator', 'string', 'textwrap',
    'io', 'os.path', 'tempfile', 'glob', 'fnmatch',
    'pickle', 'shelve', 'copy', 'pprint',
    'unittest', 'doctest', 'typing', 'abc', 'enum',
    'decimal', 'fractions', 'random', 'statistics',
    'time', 'calendar', 'locale',
]

OPTIONAL_MODULES = [
    'numpy', 'pandas', 'requests', 'flask', 'pytest',
    'matplotlib', 'sqlalchemy', 'cryptography',
]

# Semantic detection patterns (from FORGE's _extract_semantics)
SEMANTIC_PATTERNS = [
    (['serialize', 'json', 'dump', 'encode', 'marshal'], 'performs', 'serialization'),
    (['deserialize', 'load', 'decode', 'parse', 'unmarshal'], 'performs', 'deserialization'),
    (['database', 'sqlite', 'query', 'sql', 'execute'], 'interacts_with', 'database'),
    (['http', 'url', 'request', 'api', 'route', 'endpoint'], 'provides', 'api_endpoint'),
    (['file', 'read', 'write', 'open', 'close', 'stream'], 'performs', 'file_io'),
    (['hash', 'digest', 'checksum', 'md5', 'sha'], 'performs', 'hashing'),
    (['encrypt', 'decrypt', 'cipher', 'aes', 'rsa'], 'performs', 'cryptography'),
    (['compress', 'decompress', 'zip', 'gzip', 'zlib'], 'performs', 'compression'),
    (['sort', 'sorted', 'order', 'rank', 'heap'], 'performs', 'sorting'),
    (['search', 'find', 'match', 'regex', 'pattern'], 'performs', 'searching'),
    (['thread', 'lock', 'mutex', 'semaphore', 'concurrent'], 'uses', 'concurrency'),
    (['socket', 'tcp', 'udp', 'connect', 'bind', 'listen'], 'uses', 'networking'),
    (['test', 'assert', 'mock', 'fixture'], 'performs', 'testing'),
    (['log', 'logger', 'handler', 'formatter'], 'performs', 'logging'),
    (['path', 'directory', 'folder', 'walk', 'glob'], 'operates_on', 'filesystem'),
    (['date', 'time', 'timestamp', 'datetime', 'timezone'], 'handles', 'temporal_data'),
    (['random', 'seed', 'choice', 'shuffle', 'sample'], 'generates', 'random_data'),
    (['math', 'sin', 'cos', 'sqrt', 'pi', 'factorial'], 'computes', 'mathematics'),
]


class KnowledgeHarvester:
    """
    Extract facts from Python modules via introspection.
    No LLM, no API, pure inspect. Deterministic output.
    """

    def __init__(self, knowledge_store=None):
        self.knowledge = knowledge_store
        self._seen = set()

    def harvest_module(self, module_name):
        """Harvest facts from a single module. Returns list of (S, R, O)."""
        facts = []
        try:
            module = importlib.import_module(module_name)
        except (ImportError, ModuleNotFoundError):
            return []

        for name, obj in inspect.getmembers(module):
            if name.startswith('_'):
                continue

            if inspect.isfunction(obj):
                facts.append((module_name, 'provides_function', name))
                facts.extend(self._extract_semantics(name, obj))

            elif inspect.isclass(obj):
                facts.append((module_name, 'defines_class', name))
                # Harvest class methods
                for m_name, m_obj in inspect.getmembers(obj):
                    if m_name.startswith('_'):
                        continue
                    if inspect.isfunction(m_obj) or inspect.ismethod(m_obj):
                        facts.append((name, 'has_method', m_name))
                        facts.extend(self._extract_semantics(m_name, m_obj))

            elif inspect.isbuiltin(obj):
                facts.append((module_name, 'provides_builtin', name))

        return facts

    def _extract_semantics(self, name, obj):
        """Extract semantic facts from name and docstring."""
        facts = []
        doc = (inspect.getdoc(obj) or '').lower()
        name_lower = name.lower()

        for keywords, relation, target in SEMANTIC_PATTERNS:
            if any(kw in doc or kw in name_lower for kw in keywords):
                facts.append((name, relation, target))

        return facts

    def harvest_stdlib(self, verbose=False):
        """Harvest all stdlib modules. Returns total new facts stored."""
        return self._harvest_list(STDLIB_MODULES, verbose=verbose)

    def harvest_optional(self, verbose=False):
        """Harvest optional/3rd-party modules. Returns total new facts stored."""
        return self._harvest_list(OPTIONAL_MODULES, verbose=verbose)

    def harvest_all(self, verbose=True):
        """Harvest everything available."""
        n1 = self.harvest_stdlib(verbose=verbose)
        n2 = self.harvest_optional(verbose=verbose)
        return n1 + n2

    def _harvest_list(self, modules, verbose=False):
        """Harvest a list of modules, store into knowledge store."""
        total = 0
        for mod_name in modules:
            facts = self.harvest_module(mod_name)
            batch = []
            for s, r, o in facts:
                key = (s.lower(), r.lower(), o.lower())
                if key in self._seen:
                    continue
                self._seen.add(key)
                batch.append((s, r, o))
            if batch and self.knowledge:
                self.knowledge.store_facts_fast(batch)
            total += len(batch)
            if verbose and len(batch) > 0:
                print(f"  {mod_name}: {len(batch)} facts")

        return total
