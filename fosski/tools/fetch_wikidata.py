#!/usr/bin/env python3
"""Fetch structured facts from Wikidata SPARQL endpoint.

Generates structured text that the TripletExtractor can easily parse.
This is the "structured source" complement to Wikipedia text extraction.
"""

import urllib.request
import urllib.parse
import json
import os
import time

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

# Wikidata SPARQL queries for different fact types
QUERIES = {
    'capitals': '''
        SELECT ?countryLabel ?capitalLabel WHERE {
          ?country wdt:P31 wd:Q6256;
                   wdt:P36 ?capital.
          SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
        }
        LIMIT 250
    ''',
    'birthplaces': '''
        SELECT ?personLabel ?placeLabel WHERE {
          VALUES ?person {
            wd:Q937 wd:Q935 wd:Q7186 wd:Q1035 wd:Q7311
            wd:Q307 wd:Q7060 wd:Q7259 wd:Q34970 wd:Q3454165
            wd:Q9021 wd:Q46884 wd:Q44942 wd:Q17714 wd:Q7604
          }
          ?person wdt:P19 ?place.
          SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
        }
    ''',
    'inventors': '''
        SELECT ?inventionLabel ?inventorLabel WHERE {
          VALUES ?invention {
            wd:Q11035 wd:Q75 wd:Q5290 wd:Q11016 wd:Q11469
            wd:Q15078788 wd:Q11012 wd:Q11032 wd:Q5375 wd:Q11660
          }
          ?invention wdt:P61 ?inventor.
          SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
        }
    ''',
    'languages': '''
        SELECT ?countryLabel ?languageLabel WHERE {
          ?country wdt:P31 wd:Q6256;
                   wdt:P37 ?language.
          SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
        }
        LIMIT 300
    ''',
    'landmarks': '''
        SELECT ?landmarkLabel ?locationLabel WHERE {
          VALUES ?landmark {
            wd:Q243 wd:Q12501 wd:Q10285 wd:Q9141 wd:Q9202
            wd:Q37200 wd:Q18712428 wd:Q46239
          }
          ?landmark wdt:P131 ?location.
          SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
        }
    ''',
}


def run_sparql(query):
    """Run SPARQL query against Wikidata."""
    url = 'https://query.wikidata.org/sparql'
    data = urllib.parse.urlencode({'query': query, 'format': 'json'}).encode()
    req = urllib.request.Request(url, data, headers={
        'User-Agent': 'FOSS-KI/1.0 (research; david@foss.com.de)',
        'Accept': 'application/sparql-results+json',
    })
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read().decode('utf-8'))
        return result['results']['bindings']


def bindings_to_sentences(bindings, template, label_keys):
    """Convert SPARQL bindings to natural language sentences."""
    sentences = []
    seen = set()
    for b in bindings:
        values = []
        for key in label_keys:
            val = b.get(key, {}).get('value', '')
            values.append(val)
        if all(values):
            key = tuple(v.lower() for v in values)
            if key not in seen:
                seen.add(key)
                sentences.append(template.format(*values))
    return sentences


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    out_path = os.path.join(DATA_DIR, 'wikidata_facts.txt')

    all_sentences = []

    # Capitals
    print("Fetching capitals...")
    bindings = run_sparql(QUERIES['capitals'])
    sents = bindings_to_sentences(
        bindings,
        "{1} is the capital of {0}.",
        ['countryLabel', 'capitalLabel']
    )
    all_sentences.extend(sents)
    print(f"  {len(sents)} capital facts")
    time.sleep(1)

    # Birthplaces
    print("Fetching birthplaces...")
    bindings = run_sparql(QUERIES['birthplaces'])
    sents = bindings_to_sentences(
        bindings,
        "{0} was born in {1}.",
        ['personLabel', 'placeLabel']
    )
    all_sentences.extend(sents)
    print(f"  {len(sents)} birthplace facts")
    time.sleep(1)

    # Inventors
    print("Fetching inventors...")
    bindings = run_sparql(QUERIES['inventors'])
    sents = bindings_to_sentences(
        bindings,
        "{1} invented {0}.",
        ['inventionLabel', 'inventorLabel']
    )
    all_sentences.extend(sents)
    print(f"  {len(sents)} invention facts")
    time.sleep(1)

    # Languages
    print("Fetching official languages...")
    bindings = run_sparql(QUERIES['languages'])
    sents = bindings_to_sentences(
        bindings,
        "The official language of {0} is {1}.",
        ['countryLabel', 'languageLabel']
    )
    all_sentences.extend(sents)
    print(f"  {len(sents)} language facts")
    time.sleep(1)

    # Write
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(all_sentences))

    print(f"\nTotal: {len(all_sentences)} structured sentences")
    print(f"Saved to: {out_path}")


if __name__ == '__main__':
    main()
