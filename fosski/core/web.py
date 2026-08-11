"""
Web & API Access — Real-World Data Pipeline
=============================================
FOSS-KI needs access to the outside world.
Without this, it's a brain in a jar.

Capabilities:
  1. HTTP client: GET/POST with timeout, retries, headers
  2. API registry: known public APIs with endpoints + parsers
  3. Web search: DuckDuckGo HTML scraping (no API key needed)
  4. Content extraction: HTML → text, JSON → structured data
  5. URL fetch: raw content retrieval with size limits
  6. RSS/Atom feed parsing: news and updates

Design:
  - Zero API keys required for core functionality
  - Public APIs (github.com/public-apis/public-apis) as data sources
  - Rate limiting built in (don't hammer endpoints)
  - Content caching to reduce repeat fetches
"""

import json
import re
import time
import urllib.request
import urllib.parse
import urllib.error
from html.parser import HTMLParser
from typing import Dict, Any, List, Optional
from collections import OrderedDict


class TextExtractor(HTMLParser):
    """Extract readable text from HTML, stripping tags and scripts."""

    def __init__(self):
        super().__init__()
        self._text = []
        self._skip = False
        self._skip_tags = {'script', 'style', 'noscript', 'svg', 'head'}

    def handle_starttag(self, tag, attrs):  # noqa: ARG002
        if tag in self._skip_tags:
            self._skip = True
        if tag in ('br', 'p', 'div', 'li', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'tr'):
            self._text.append('\n')

    def handle_endtag(self, tag):
        if tag in self._skip_tags:
            self._skip = False

    def handle_data(self, data):
        if not self._skip:
            text = data.strip()
            if text:
                self._text.append(text)

    def get_text(self) -> str:
        raw = ' '.join(self._text)
        # Collapse whitespace
        raw = re.sub(r'\n\s*\n', '\n\n', raw)
        raw = re.sub(r' {2,}', ' ', raw)
        return raw.strip()


class LRUCache:
    """Simple LRU cache for web responses."""

    def __init__(self, maxsize: int = 100, ttl: int = 300):
        self._cache: OrderedDict = OrderedDict()
        self._maxsize = maxsize
        self._ttl = ttl

    def get(self, key: str) -> Optional[Any]:
        if key in self._cache:
            value, ts = self._cache[key]
            if time.time() - ts < self._ttl:
                self._cache.move_to_end(key)
                return value
            else:
                del self._cache[key]
        return None

    def put(self, key: str, value: Any):
        if key in self._cache:
            del self._cache[key]
        self._cache[key] = (value, time.time())
        if len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)


class WebClient:
    """
    HTTP client with caching, rate limiting, retries.
    Uses only stdlib (urllib) — no requests dependency.
    """

    def __init__(self, timeout: int = 15, max_retries: int = 2,
                 rate_limit: float = 1.0, cache_ttl: int = 300,
                 max_response_size: int = 1_000_000):
        """
        Args:
            timeout: request timeout in seconds
            max_retries: retry count on failure
            rate_limit: minimum seconds between requests to same host
            cache_ttl: cache entry lifetime in seconds
            max_response_size: max bytes to read (1MB default)
        """
        self.timeout = timeout
        self.max_retries = max_retries
        self.rate_limit = rate_limit
        self.max_response_size = max_response_size
        self.cache = LRUCache(maxsize=200, ttl=cache_ttl)
        self._last_request: Dict[str, float] = {}  # host → timestamp
        self.user_agent = 'FOSS-KI/1.0 (Research AI; +https://github.com/foss-ki)'

    def get(self, url: str, headers: Optional[Dict] = None,
            use_cache: bool = True) -> Dict[str, Any]:
        """
        HTTP GET with caching and rate limiting.

        Returns:
            dict with: success, status, body, content_type, url, cached, elapsed
        """
        # Check cache
        if use_cache:
            cached = self.cache.get(url)
            if cached is not None:
                return {**cached, 'cached': True}

        # Rate limit
        self._rate_limit_wait(url)

        # Build request
        req_headers = {'User-Agent': self.user_agent}
        if headers:
            req_headers.update(headers)

        req = urllib.request.Request(url, headers=req_headers)

        # Execute with retries
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                start = time.time()
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    body = resp.read(self.max_response_size)
                    content_type = resp.headers.get('Content-Type', '')
                    charset = 'utf-8'
                    if 'charset=' in content_type:
                        charset = content_type.split('charset=')[-1].split(';')[0].strip()

                    try:
                        text = body.decode(charset, errors='replace')
                    except (LookupError, UnicodeDecodeError):
                        text = body.decode('utf-8', errors='replace')

                    result = {
                        'success': True,
                        'status': resp.status,
                        'body': text,
                        'content_type': content_type,
                        'url': url,
                        'cached': False,
                        'elapsed': round(time.time() - start, 3),
                        'size': len(body),
                    }

                    if use_cache:
                        self.cache.put(url, result)
                    return result

            except urllib.error.HTTPError as e:
                last_error = f"HTTP {e.code}: {e.reason}"
                if e.code in (429, 503):
                    time.sleep(2 ** attempt)
                    continue
                break
            except urllib.error.URLError as e:
                last_error = f"URL error: {e.reason}"
                break
            except TimeoutError:
                last_error = f"Timeout after {self.timeout}s"
                if attempt < self.max_retries:
                    continue
                break
            except Exception as e:
                last_error = str(e)
                break

        return {
            'success': False,
            'error': last_error,
            'url': url,
            'cached': False,
        }

    def post(self, url: str, data: Optional[Dict] = None,
             json_body: Optional[Dict] = None,
             headers: Optional[Dict] = None) -> Dict[str, Any]:
        """HTTP POST with JSON or form data."""
        self._rate_limit_wait(url)

        req_headers = {'User-Agent': self.user_agent}
        if headers:
            req_headers.update(headers)

        if json_body is not None:
            body = json.dumps(json_body).encode('utf-8')
            req_headers['Content-Type'] = 'application/json'
        elif data is not None:
            body = urllib.parse.urlencode(data).encode('utf-8')
            req_headers['Content-Type'] = 'application/x-www-form-urlencoded'
        else:
            body = b''

        req = urllib.request.Request(url, data=body, headers=req_headers, method='POST')

        try:
            start = time.time()
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                resp_body = resp.read(self.max_response_size)
                text = resp_body.decode('utf-8', errors='replace')
                return {
                    'success': True,
                    'status': resp.status,
                    'body': text,
                    'content_type': resp.headers.get('Content-Type', ''),
                    'url': url,
                    'elapsed': round(time.time() - start, 3),
                }
        except Exception as e:
            return {'success': False, 'error': str(e), 'url': url}

    def _rate_limit_wait(self, url: str):
        """Enforce per-host rate limiting."""
        try:
            host = urllib.parse.urlparse(url).netloc
        except Exception:
            return
        now = time.time()
        last = self._last_request.get(host, 0)
        wait = self.rate_limit - (now - last)
        if wait > 0:
            time.sleep(wait)
        self._last_request[host] = time.time()


class ContentExtractor:
    """Extract structured content from raw web responses."""

    @staticmethod
    def html_to_text(html: str, max_length: int = 10000) -> str:
        """Convert HTML to readable text."""
        parser = TextExtractor()
        try:
            parser.feed(html)
        except Exception:
            # Fallback: crude tag stripping
            text = re.sub(r'<[^>]+>', ' ', html)
            text = re.sub(r'\s+', ' ', text)
            return text[:max_length].strip()
        return parser.get_text()[:max_length]

    @staticmethod
    def extract_links(html: str, base_url: str = '') -> List[Dict[str, str]]:
        """Extract links from HTML."""
        links = []
        for m in re.finditer(r'<a\s[^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', html, re.I | re.S):
            href = m.group(1)
            text = re.sub(r'<[^>]+>', '', m.group(2)).strip()
            if href.startswith('/') and base_url:
                href = base_url.rstrip('/') + href
            if href.startswith('http'):
                links.append({'url': href, 'text': text})
        return links[:100]

    @staticmethod
    def extract_json(text: str) -> Optional[Any]:
        """Try to parse response as JSON."""
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return None

    @staticmethod
    def extract_rss(xml: str) -> List[Dict[str, str]]:
        """Basic RSS/Atom feed parser — no lxml needed."""
        items = []

        # RSS items
        for item_match in re.finditer(r'<item>(.*?)</item>', xml, re.S):
            item_xml = item_match.group(1)
            title = _xml_tag(item_xml, 'title')
            link = _xml_tag(item_xml, 'link')
            desc = _xml_tag(item_xml, 'description')
            pub_date = _xml_tag(item_xml, 'pubDate')
            items.append({
                'title': title or '',
                'link': link or '',
                'description': _strip_html(desc) if desc else '',
                'date': pub_date or '',
            })

        # Atom entries
        if not items:
            for entry_match in re.finditer(r'<entry>(.*?)</entry>', xml, re.S):
                entry_xml = entry_match.group(1)
                title = _xml_tag(entry_xml, 'title')
                link_m = re.search(r'<link[^>]*href=["\']([^"\']+)["\']', entry_xml)
                link = link_m.group(1) if link_m else ''
                summary = _xml_tag(entry_xml, 'summary') or _xml_tag(entry_xml, 'content')
                updated = _xml_tag(entry_xml, 'updated')
                items.append({
                    'title': title or '',
                    'link': link,
                    'description': _strip_html(summary) if summary else '',
                    'date': updated or '',
                })

        return items[:50]


# === Public API Registry ===
# Curated from github.com/public-apis/public-apis
# Only FREE, NO-AUTH-REQUIRED APIs

API_REGISTRY = {
    # Knowledge / Facts
    'wikipedia': {
        'name': 'Wikipedia',
        'base': 'https://en.wikipedia.org/api/rest_v1',
        'endpoints': {
            'summary': '/page/summary/{title}',
            'search': '/page/search/{query}',
        },
        'category': 'knowledge',
    },
    'wikidata': {
        'name': 'Wikidata',
        'base': 'https://www.wikidata.org/w/api.php',
        'endpoints': {
            'search': '?action=wbsearchentities&search={query}&language=en&format=json',
            'entity': '?action=wbgetentities&ids={id}&format=json&props=labels|descriptions|claims',
        },
        'category': 'knowledge',
    },
    'duckduckgo': {
        'name': 'DuckDuckGo Instant Answers',
        'base': 'https://api.duckduckgo.com',
        'endpoints': {
            'search': '/?q={query}&format=json&no_html=1&skip_disambig=1',
        },
        'category': 'search',
    },

    # Weather
    'open_meteo': {
        'name': 'Open-Meteo',
        'base': 'https://api.open-meteo.com/v1',
        'endpoints': {
            'forecast': '/forecast?latitude={lat}&longitude={lon}&current_weather=true',
        },
        'category': 'weather',
    },

    # Country / Geo
    'restcountries': {
        'name': 'REST Countries',
        'base': 'https://restcountries.com/v3.1',
        'endpoints': {
            'name': '/name/{query}',
            'code': '/alpha/{code}',
            'all': '/all?fields=name,capital,population,region',
        },
        'category': 'geo',
    },
    'ip_api': {
        'name': 'IP Geolocation',
        'base': 'http://ip-api.com',
        'endpoints': {
            'lookup': '/json/{ip}',
            'self': '/json/',
        },
        'category': 'geo',
    },

    # Science / Math
    'numbers_api': {
        'name': 'Numbers API',
        'base': 'http://numbersapi.com',
        'endpoints': {
            'trivia': '/{number}',
            'math': '/{number}/math',
            'date': '/{month}/{day}/date',
        },
        'category': 'science',
    },
    'arxiv': {
        'name': 'arXiv',
        'base': 'http://export.arxiv.org/api',
        'endpoints': {
            'search': '/query?search_query=all:{query}&max_results=5',
        },
        'category': 'science',
    },

    # Dev / Code
    'github_api': {
        'name': 'GitHub',
        'base': 'https://api.github.com',
        'endpoints': {
            'user': '/users/{user}',
            'repo': '/repos/{owner}/{repo}',
            'search_repos': '/search/repositories?q={query}&sort=stars&per_page=5',
            'search_code': '/search/code?q={query}&per_page=5',
        },
        'category': 'dev',
    },

    # News
    'hacker_news': {
        'name': 'Hacker News',
        'base': 'https://hacker-news.firebaseio.com/v0',
        'endpoints': {
            'top': '/topstories.json',
            'item': '/item/{id}.json',
        },
        'category': 'news',
    },

    # Exchange rates
    'exchangerate': {
        'name': 'Exchange Rate',
        'base': 'https://open.er-api.com/v6',
        'endpoints': {
            'latest': '/latest/{currency}',
        },
        'category': 'finance',
    },

    # DNS / Network
    'dns': {
        'name': 'DNS over HTTPS (Google)',
        'base': 'https://dns.google',
        'endpoints': {
            'resolve': '/resolve?name={domain}&type={type}',
        },
        'category': 'network',
    },
}


class APIClient:
    """
    Query registered public APIs by intent.

    Examples:
        client.query('wikipedia', 'summary', title='Python_(programming_language)')
        client.query('open_meteo', 'forecast', lat=59.91, lon=10.75)
        client.search('what is the population of France')
    """

    def __init__(self, web: Optional[WebClient] = None):
        self.web = web or WebClient()
        self.extract = ContentExtractor()

    def query(self, api_name: str, endpoint: str,
              **kwargs) -> Dict[str, Any]:
        """
        Query a registered API.

        Args:
            api_name: key from API_REGISTRY
            endpoint: endpoint name
            **kwargs: template variables

        Returns:
            dict with: success, data, raw, api, endpoint
        """
        api = API_REGISTRY.get(api_name)
        if not api:
            return {'success': False, 'error': f'Unknown API: {api_name}'}

        ep_template = api['endpoints'].get(endpoint)
        if not ep_template:
            return {'success': False, 'error': f'Unknown endpoint: {endpoint}'}

        # Build URL
        url = api['base'] + ep_template
        for key, value in kwargs.items():
            url = url.replace('{' + key + '}', urllib.parse.quote(str(value)))

        # Fetch
        result = self.web.get(url)
        if not result['success']:
            return result

        # Parse response
        data = self.extract.extract_json(result['body'])
        if data is None:
            data = result['body']

        return {
            'success': True,
            'data': data,
            'raw': result['body'][:2000],
            'api': api_name,
            'endpoint': endpoint,
            'url': url,
            'elapsed': result.get('elapsed', 0),
            'cached': result.get('cached', False),
        }

    def search(self, query: str) -> Dict[str, Any]:
        """
        Smart search across multiple APIs based on query intent.

        Returns best result from:
        1. DuckDuckGo Instant Answers (broad knowledge)
        2. Wikipedia (encyclopedic)
        3. Domain-specific APIs (weather, geo, etc.)
        """
        q = query.lower().strip()
        results = []

        # Weather queries
        if any(w in q for w in ('weather', 'temperature', 'forecast', 'rain', 'wind')):
            geo = self._extract_location_coords(q)
            if geo:
                r = self.query('open_meteo', 'forecast', lat=geo[0], lon=geo[1])
                if r['success']:
                    return self._format_weather(r['data'], q)

        # Country queries
        if any(w in q for w in ('country', 'capital', 'population', 'currency')):
            country_name = self._extract_entity(q,
                ['country', 'capital', 'population', 'currency', 'of', 'the',
                 'what', 'is', 'are', 'how', 'many', 'people', 'in'])
            if country_name:
                r = self.query('restcountries', 'name', query=country_name)
                if r['success'] and isinstance(r['data'], list) and r['data']:
                    return self._format_country(r['data'][0], q)

        # Exchange rate
        if any(w in q for w in ('exchange', 'rate', 'convert', 'currency', 'usd', 'eur', 'gbp')):
            currency = 'USD'
            for c in ('USD', 'EUR', 'GBP', 'JPY', 'NOK', 'SEK', 'CHF'):
                if c.lower() in q:
                    currency = c
                    break
            r = self.query('exchangerate', 'latest', currency=currency)
            if r['success']:
                return self._format_exchange(r['data'], q)

        # GitHub queries
        if any(w in q for w in ('github', 'repository', 'repo', 'open source', 'library', 'package')):
            search_q = self._extract_entity(q,
                ['github', 'repository', 'repo', 'find', 'search', 'for',
                 'open', 'source', 'library', 'package', 'the', 'best'])
            if search_q:
                r = self.query('github_api', 'search_repos', query=search_q)
                if r['success']:
                    return self._format_github(r['data'], q)

        # arXiv queries
        if any(w in q for w in ('paper', 'arxiv', 'research', 'publication', 'scientific')):
            search_q = self._extract_entity(q,
                ['paper', 'arxiv', 'research', 'publication', 'scientific',
                 'find', 'search', 'for', 'about', 'on', 'recent', 'latest'])
            if search_q:
                r = self.query('arxiv', 'search', query=search_q)
                if r['success']:
                    return self._format_arxiv(r['raw'], q)

        # IP lookup
        if re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', q):
            ip_m = re.search(r'(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})', q)
            ip = ip_m.group(1) if ip_m else ''
            r = self.query('ip_api', 'lookup', ip=ip)
            if r['success']:
                return r

        # DuckDuckGo fallback (broad search)
        r = self.query('duckduckgo', 'search', query=query)
        if r['success'] and isinstance(r['data'], dict):
            abstract = r['data'].get('Abstract') or r['data'].get('AbstractText')
            if abstract:
                results.append({
                    'source': 'DuckDuckGo',
                    'answer': abstract,
                    'url': r['data'].get('AbstractURL', ''),
                })

        # Wikipedia fallback
        wiki_q = self._extract_entity(q,
            ['what', 'who', 'where', 'when', 'which', 'is', 'are', 'was',
             'were', 'the', 'a', 'an', 'tell', 'me', 'about', 'describe'])
        if wiki_q:
            title = wiki_q.replace(' ', '_')
            r = self.query('wikipedia', 'summary', title=title)
            if r['success'] and isinstance(r['data'], dict):
                extract = r['data'].get('extract')
                if extract:
                    results.append({
                        'source': 'Wikipedia',
                        'answer': extract,
                        'url': r['data'].get('content_urls', {}).get('desktop', {}).get('page', ''),
                    })

        if results:
            best = max(results, key=lambda r: len(r.get('answer', '')))
            return {
                'success': True,
                'answer': best['answer'],
                'source': best['source'],
                'url': best.get('url', ''),
                'all_results': results,
            }

        return {
            'success': False,
            'error': 'No results found',
            'query': query,
        }

    def fetch_url(self, url: str) -> Dict[str, Any]:
        """Fetch a URL and extract readable content."""
        result = self.web.get(url)
        if not result['success']:
            return result

        content_type = result.get('content_type', '')

        if 'json' in content_type:
            data = self.extract.extract_json(result['body'])
            return {
                'success': True,
                'type': 'json',
                'data': data,
                'url': url,
            }

        if 'xml' in content_type or 'rss' in content_type or 'atom' in content_type:
            items = self.extract.extract_rss(result['body'])
            return {
                'success': True,
                'type': 'feed',
                'items': items,
                'url': url,
            }

        # HTML or text
        text = self.extract.html_to_text(result['body'])
        links = self.extract.extract_links(result['body'], url)

        return {
            'success': True,
            'type': 'html',
            'text': text,
            'links': links[:20],
            'url': url,
        }

    def hacker_news_top(self, count: int = 5) -> List[Dict]:
        """Fetch top Hacker News stories."""
        r = self.query('hacker_news', 'top')
        if not r['success']:
            return []

        ids = r['data'][:count] if isinstance(r['data'], list) else []
        stories = []
        for story_id in ids:
            sr = self.query('hacker_news', 'item', id=story_id)
            if sr['success'] and isinstance(sr['data'], dict):
                stories.append({
                    'title': sr['data'].get('title', ''),
                    'url': sr['data'].get('url', ''),
                    'score': sr['data'].get('score', 0),
                    'by': sr['data'].get('by', ''),
                })
        return stories

    # === Formatters ===

    def _format_weather(self, data: Any, query: str) -> Dict[str, Any]:
        if not isinstance(data, dict):
            return {'success': False, 'error': 'Bad weather data'}
        cw = data.get('current_weather', {})
        return {
            'success': True,
            'answer': (f"Temperature: {cw.get('temperature', '?')}°C, "
                      f"Wind: {cw.get('windspeed', '?')} km/h, "
                      f"Weather code: {cw.get('weathercode', '?')}"),
            'source': 'Open-Meteo',
            'data': cw,
        }

    def _format_country(self, data: Any, query: str) -> Dict[str, Any]:
        if not isinstance(data, dict):
            return {'success': False, 'error': 'Bad country data'}
        name = data.get('name', {})
        common_name = name.get('common', '?') if isinstance(name, dict) else str(name)
        capital = data.get('capital', ['?'])
        capital_str = capital[0] if isinstance(capital, list) and capital else str(capital)
        return {
            'success': True,
            'answer': (f"{common_name}: "
                      f"Capital: {capital_str}, "
                      f"Population: {data.get('population', '?'):,}, "
                      f"Region: {data.get('region', '?')}"),
            'source': 'REST Countries',
            'data': data,
        }

    def _format_exchange(self, data: Any, query: str) -> Dict[str, Any]:
        if not isinstance(data, dict):
            return {'success': False, 'error': 'Bad exchange data'}
        base = data.get('base_code', '?')
        rates = data.get('rates', {})
        # Show top currencies
        show = ['EUR', 'USD', 'GBP', 'JPY', 'NOK', 'SEK', 'CHF']
        rate_str = ', '.join(f"{c}: {rates.get(c, '?')}" for c in show if c != base)
        return {
            'success': True,
            'answer': f"1 {base} = {rate_str}",
            'source': 'Exchange Rate API',
            'data': {c: rates.get(c) for c in show if c in rates},
        }

    def _format_github(self, data: Any, query: str) -> Dict[str, Any]:
        if not isinstance(data, dict):
            return {'success': False, 'error': 'Bad GitHub data'}
        items = data.get('items', [])[:5]
        results = []
        for repo in items:
            results.append(
                f"- {repo.get('full_name', '?')} "
                f"({repo.get('stargazers_count', 0)} stars): "
                f"{repo.get('description', '')[:100]}"
            )
        return {
            'success': True,
            'answer': '\n'.join(results) if results else 'No repos found',
            'source': 'GitHub',
            'data': items,
        }

    def _format_arxiv(self, xml: str, query: str) -> Dict[str, Any]:
        entries = []
        for entry_m in re.finditer(r'<entry>(.*?)</entry>', xml, re.S):
            e = entry_m.group(1)
            title = _xml_tag(e, 'title')
            summary = _xml_tag(e, 'summary')
            link_m = re.search(r'<id>(.*?)</id>', e)
            link = link_m.group(1) if link_m else ''
            entries.append({
                'title': (title or '').strip().replace('\n', ' '),
                'summary': (summary or '')[:200].strip().replace('\n', ' '),
                'url': link,
            })

        if entries:
            lines = []
            for e in entries[:5]:
                lines.append(f"- {e['title']}")
                if e['summary']:
                    lines.append(f"  {e['summary']}...")
            return {
                'success': True,
                'answer': '\n'.join(lines),
                'source': 'arXiv',
                'data': entries,
            }
        return {'success': False, 'error': 'No arXiv results'}

    # === Helpers ===

    @staticmethod
    def _extract_entity(query: str, stop_words: List[str]) -> str:
        """Extract the main entity from a query by removing stop words."""
        words = query.lower().split()
        entity_words = [w for w in words if w not in stop_words and len(w) > 1]
        return ' '.join(entity_words).strip()

    @staticmethod
    def _extract_location_coords(query: str) -> Optional[tuple]:
        """Extract coordinates from known city names (basic lookup)."""
        known = {
            'oslo': (59.91, 10.75),
            'london': (51.51, -0.13),
            'berlin': (52.52, 13.40),
            'paris': (48.86, 2.35),
            'new york': (40.71, -74.01),
            'tokyo': (35.68, 139.69),
            'rome': (41.90, 12.50),
            'moscow': (55.76, 37.62),
            'sydney': (-33.87, 151.21),
            'tromsø': (69.65, 18.96),
            'bergen': (60.39, 5.32),
            'stavanger': (58.97, 5.73),
            'trondheim': (63.43, 10.40),
            'hamburg': (53.55, 9.99),
            'munich': (48.14, 11.58),
            'stockholm': (59.33, 18.07),
            'helsinki': (60.17, 24.94),
            'washington': (38.91, -77.04),
            'san francisco': (37.77, -122.42),
        }
        q = query.lower()
        for city, coords in known.items():
            if city in q:
                return coords

        # Try extracting lat/lon from query
        m = re.search(r'(\-?\d+\.?\d*)\s*[,/]\s*(\-?\d+\.?\d*)', query)
        if m:
            return (float(m.group(1)), float(m.group(2)))

        return None


# === XML Helpers ===

def _xml_tag(xml: str, tag: str) -> Optional[str]:
    """Extract content of an XML tag (simple, no nesting)."""
    m = re.search(rf'<{tag}[^>]*>(.*?)</{tag}>', xml, re.S)
    if m:
        return m.group(1).strip()
    return None


def _strip_html(text: str) -> str:
    """Strip HTML tags from text."""
    return re.sub(r'<[^>]+>', '', text).strip()


# === Tool Integration ===
# Functions to register as tools in the ToolRegistry

def _tool_web_search(query: str) -> str:
    """Search the web for information."""
    client = APIClient()
    result = client.search(query)
    if result.get('success') and result.get('answer'):
        source = result.get('source', 'Web')
        return f"[{source}] {result['answer']}"
    return f"No results for: {query}"


def _tool_fetch_url(url: str) -> str:
    """Fetch and extract content from a URL."""
    client = APIClient()
    result = client.fetch_url(url)
    if result.get('success'):
        if result.get('type') == 'json':
            return json.dumps(result['data'], indent=2)[:5000]
        elif result.get('type') == 'feed':
            lines = [f"- {item['title']}: {item['link']}" for item in result.get('items', [])[:10]]
            return '\n'.join(lines)
        else:
            return result.get('text', '')[:5000]
    return f"Failed to fetch: {result.get('error', 'unknown error')}"


def _tool_weather(location: str) -> str:
    """Get current weather for a location."""
    client = APIClient()
    result = client.search(f'weather in {location}')
    if result.get('success') and result.get('answer'):
        return result['answer']
    return f"Could not get weather for: {location}"


def _tool_news(count: str = '5') -> str:
    """Get top Hacker News stories."""
    client = APIClient()
    stories = client.hacker_news_top(int(count))
    if stories:
        lines = [f"- [{s['score']}] {s['title']}" for s in stories]
        return '\n'.join(lines)
    return "Could not fetch news"


def register_web_tools(registry):
    """Register web tools into an existing ToolRegistry."""
    from .tools import Tool

    registry.register(Tool(
        name='web_search',
        description='Search the web for current information, facts, definitions',
        fn=_tool_web_search,
        args=[{'name': 'query', 'type': 'str', 'description': 'search query', 'required': True}],
        category='web',
    ))

    registry.register(Tool(
        name='fetch_url',
        description='Fetch and extract content from a URL or website',
        fn=_tool_fetch_url,
        args=[{'name': 'url', 'type': 'str', 'description': 'URL to fetch', 'required': True}],
        category='web',
    ))

    registry.register(Tool(
        name='weather',
        description='Get current weather for a city or location',
        fn=_tool_weather,
        args=[{'name': 'location', 'type': 'str', 'description': 'city name', 'required': True}],
        category='web',
    ))

    registry.register(Tool(
        name='news',
        description='Get top tech news from Hacker News',
        fn=_tool_news,
        args=[{'name': 'count', 'type': 'str', 'description': 'number of stories', 'required': False}],
        category='web',
    ))
