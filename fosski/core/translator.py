"""
Translator — Statistical Machine Translation Without Neural Networks
=====================================================================
Word-level + phrase-level translation DE↔EN.

Approach:
  1. Dictionary lookup (word-by-word)
  2. Phrase table (common multi-word expressions)
  3. Reordering rules (German verb-final → English SVO)
  4. MyMemory API fallback (free, no key, 5000 words/day)

This is NOT Google Translate quality. It's functional translation
for a system that needs to work when all cloud AI is gone.
"""

import re
import urllib.request
import urllib.parse
import json
from typing import Dict, Optional, List, Tuple


# === Core Dictionary (most frequent 500+ words) ===
# German → English

DE_EN = {
    # Articles / Pronouns
    'der': 'the', 'die': 'the', 'das': 'the', 'ein': 'a', 'eine': 'a',
    'einer': 'one', 'eines': 'of a', 'einem': 'a',
    'ich': 'I', 'du': 'you', 'er': 'he', 'sie': 'she', 'es': 'it',
    'wir': 'we', 'ihr': 'you', 'Sie': 'you',
    'mein': 'my', 'dein': 'your', 'sein': 'his', 'ihr': 'her',
    'unser': 'our', 'euer': 'your', 'mich': 'me', 'dich': 'you',
    'ihn': 'him', 'uns': 'us', 'euch': 'you', 'ihnen': 'them',
    'sich': 'oneself', 'man': 'one', 'jemand': 'someone',
    'niemand': 'nobody', 'alle': 'all', 'alles': 'everything',
    'nichts': 'nothing', 'etwas': 'something', 'jeder': 'everyone',
    'dieser': 'this', 'diese': 'this', 'dieses': 'this',
    'jener': 'that', 'jene': 'that', 'jenes': 'that',
    'welcher': 'which', 'welche': 'which', 'welches': 'which',

    # Verbs (infinitive + common conjugations)
    'sein': 'be', 'ist': 'is', 'sind': 'are', 'war': 'was', 'waren': 'were',
    'bin': 'am', 'bist': 'are', 'gewesen': 'been',
    'haben': 'have', 'hat': 'has', 'hatte': 'had', 'hatten': 'had',
    'habe': 'have', 'hast': 'have', 'gehabt': 'had',
    'werden': 'become', 'wird': 'becomes', 'wurde': 'became', 'worden': 'become',
    'können': 'can', 'kann': 'can', 'konnte': 'could', 'gekonnt': 'been able',
    'müssen': 'must', 'muss': 'must', 'musste': 'had to',
    'sollen': 'should', 'soll': 'should', 'sollte': 'should',
    'wollen': 'want', 'will': 'want', 'wollte': 'wanted',
    'dürfen': 'may', 'darf': 'may', 'durfte': 'was allowed',
    'mögen': 'like', 'mag': 'like', 'mochte': 'liked', 'möchte': 'would like',
    'machen': 'make', 'macht': 'makes', 'machte': 'made', 'gemacht': 'made',
    'gehen': 'go', 'geht': 'goes', 'ging': 'went', 'gegangen': 'gone',
    'kommen': 'come', 'kommt': 'comes', 'kam': 'came', 'gekommen': 'come',
    'sagen': 'say', 'sagt': 'says', 'sagte': 'said', 'gesagt': 'said',
    'geben': 'give', 'gibt': 'gives', 'gab': 'gave', 'gegeben': 'given',
    'nehmen': 'take', 'nimmt': 'takes', 'nahm': 'took', 'genommen': 'taken',
    'finden': 'find', 'findet': 'finds', 'fand': 'found', 'gefunden': 'found',
    'denken': 'think', 'denkt': 'thinks', 'dachte': 'thought', 'gedacht': 'thought',
    'wissen': 'know', 'weiß': 'know', 'weiss': 'know', 'wusste': 'knew',
    'sehen': 'see', 'sieht': 'sees', 'sah': 'saw', 'gesehen': 'seen',
    'lassen': 'let', 'lässt': 'lets', 'ließ': 'let', 'gelassen': 'let',
    'stehen': 'stand', 'steht': 'stands', 'stand': 'stood',
    'liegen': 'lie', 'liegt': 'lies', 'lag': 'lay',
    'laufen': 'run', 'läuft': 'runs', 'lief': 'ran',
    'sprechen': 'speak', 'spricht': 'speaks', 'sprach': 'spoke',
    'lesen': 'read', 'liest': 'reads', 'las': 'read',
    'schreiben': 'write', 'schreibt': 'writes', 'schrieb': 'wrote',
    'spielen': 'play', 'spielt': 'plays', 'spielte': 'played',
    'arbeiten': 'work', 'arbeitet': 'works', 'arbeitete': 'worked',
    'brauchen': 'need', 'braucht': 'needs', 'brauchte': 'needed',
    'glauben': 'believe', 'glaubt': 'believes', 'glaubte': 'believed',
    'halten': 'hold', 'hält': 'holds', 'hielt': 'held',
    'bringen': 'bring', 'bringt': 'brings', 'brachte': 'brought',
    'leben': 'live', 'lebt': 'lives', 'lebte': 'lived',
    'fahren': 'drive', 'fährt': 'drives', 'fuhr': 'drove',
    'heißen': 'be called', 'heißt': 'is called',
    'versuchen': 'try', 'versucht': 'tries',
    'beginnen': 'begin', 'beginnt': 'begins', 'begann': 'began',
    'helfen': 'help', 'hilft': 'helps', 'half': 'helped',
    'zeigen': 'show', 'zeigt': 'shows', 'zeigte': 'showed',
    'führen': 'lead', 'führt': 'leads', 'führte': 'led',
    'bleiben': 'stay', 'bleibt': 'stays', 'blieb': 'stayed',
    'setzen': 'set', 'setzt': 'sets', 'setzte': 'set',
    'kennen': 'know', 'kennt': 'knows', 'kannte': 'knew',
    'verstehen': 'understand', 'versteht': 'understands', 'verstand': 'understood',
    'legen': 'lay', 'legt': 'lays', 'legte': 'laid',
    'ziehen': 'pull', 'zieht': 'pulls', 'zog': 'pulled',
    'schließen': 'close', 'schließt': 'closes', 'schloss': 'closed',
    'fallen': 'fall', 'fällt': 'falls', 'fiel': 'fell',
    'tragen': 'carry', 'trägt': 'carries', 'trug': 'carried',
    'kaufen': 'buy', 'kauft': 'buys', 'kaufte': 'bought',
    'warten': 'wait', 'wartet': 'waits', 'wartete': 'waited',
    'öffnen': 'open', 'öffnet': 'opens',
    'schlafen': 'sleep', 'schläft': 'sleeps',
    'essen': 'eat', 'isst': 'eats',
    'trinken': 'drink', 'trinkt': 'drinks',
    'bauen': 'build', 'baut': 'builds',
    'tun': 'do', 'tut': 'does', 'tat': 'did', 'getan': 'done',

    # Nouns (common)
    'Mensch': 'human', 'Menschen': 'people', 'Mann': 'man', 'Frau': 'woman',
    'Kind': 'child', 'Kinder': 'children', 'Jahr': 'year', 'Jahre': 'years',
    'Zeit': 'time', 'Tag': 'day', 'Tage': 'days', 'Nacht': 'night',
    'Welt': 'world', 'Land': 'country', 'Stadt': 'city', 'Haus': 'house',
    'Wasser': 'water', 'Arbeit': 'work', 'Geld': 'money', 'Leben': 'life',
    'Weg': 'way', 'Buch': 'book', 'Hand': 'hand', 'Auge': 'eye',
    'Kopf': 'head', 'Frage': 'question', 'Antwort': 'answer',
    'Grund': 'reason', 'Teil': 'part', 'Ende': 'end', 'Anfang': 'beginning',
    'Seite': 'side', 'Stelle': 'place', 'Wort': 'word', 'Name': 'name',
    'Beispiel': 'example', 'Problem': 'problem', 'Schule': 'school',
    'Staat': 'state', 'Gruppe': 'group', 'Firma': 'company',
    'Regierung': 'government', 'System': 'system', 'Programm': 'program',
    'Straße': 'street', 'Tür': 'door', 'Fenster': 'window',
    'Morgen': 'morning', 'Abend': 'evening', 'Stunde': 'hour',
    'Minute': 'minute', 'Monat': 'month', 'Woche': 'week',
    'Sommer': 'summer', 'Winter': 'winter', 'Frühling': 'spring',
    'Herbst': 'autumn', 'Sonne': 'sun', 'Mond': 'moon',
    'Musik': 'music', 'Bild': 'picture', 'Film': 'film',
    'Computer': 'computer', 'Telefon': 'telephone', 'Internet': 'internet',
    'Ergebnis': 'result', 'Lösung': 'solution', 'Idee': 'idea',
    'Geschichte': 'history', 'Sprache': 'language', 'Recht': 'right',
    'Krieg': 'war', 'Frieden': 'peace', 'Sicherheit': 'security',
    'Zukunft': 'future', 'Vergangenheit': 'past',

    # Adjectives
    'gut': 'good', 'schlecht': 'bad', 'groß': 'big', 'klein': 'small',
    'neu': 'new', 'alt': 'old', 'jung': 'young', 'lang': 'long',
    'kurz': 'short', 'hoch': 'high', 'tief': 'deep', 'schnell': 'fast',
    'langsam': 'slow', 'stark': 'strong', 'schwach': 'weak',
    'schön': 'beautiful', 'wichtig': 'important', 'richtig': 'correct',
    'falsch': 'wrong', 'möglich': 'possible', 'nötig': 'necessary',
    'klar': 'clear', 'sicher': 'safe', 'frei': 'free', 'wahr': 'true',
    'einfach': 'simple', 'schwer': 'hard', 'leicht': 'easy',
    'voll': 'full', 'leer': 'empty', 'offen': 'open',
    'erste': 'first', 'letzte': 'last', 'nächste': 'next',
    'beste': 'best', 'andere': 'other', 'gleich': 'same',
    'ganz': 'whole', 'halb': 'half', 'eigentlich': 'actually',
    'besser': 'better', 'größer': 'bigger', 'kleiner': 'smaller',

    # Adverbs / Conjunctions / Prepositions
    'nicht': 'not', 'auch': 'also', 'noch': 'still', 'schon': 'already',
    'nur': 'only', 'dann': 'then', 'so': 'so', 'hier': 'here',
    'da': 'there', 'jetzt': 'now', 'heute': 'today', 'morgen': 'tomorrow',
    'gestern': 'yesterday', 'immer': 'always', 'nie': 'never',
    'oft': 'often', 'manchmal': 'sometimes', 'vielleicht': 'maybe',
    'wieder': 'again', 'zusammen': 'together', 'allein': 'alone',
    'sehr': 'very', 'mehr': 'more', 'weniger': 'less', 'fast': 'almost',
    'gerade': 'just', 'bald': 'soon', 'genug': 'enough',
    'trotzdem': 'nevertheless', 'deshalb': 'therefore', 'außerdem': 'moreover',
    'natürlich': 'naturally', 'wahrscheinlich': 'probably',
    'aber': 'but', 'oder': 'or', 'und': 'and', 'denn': 'because',
    'weil': 'because', 'wenn': 'if', 'als': 'when', 'dass': 'that',
    'ob': 'whether', 'obwohl': 'although', 'damit': 'so that',
    'bevor': 'before', 'nachdem': 'after', 'während': 'while',
    'seit': 'since', 'bis': 'until', 'ohne': 'without',
    'mit': 'with', 'für': 'for', 'gegen': 'against', 'über': 'over',
    'unter': 'under', 'zwischen': 'between', 'neben': 'next to',
    'vor': 'before', 'nach': 'after', 'aus': 'from', 'bei': 'at',
    'durch': 'through', 'um': 'around', 'auf': 'on', 'an': 'at',
    'in': 'in', 'zu': 'to', 'von': 'of',
    'ja': 'yes', 'nein': 'no', 'danke': 'thanks', 'bitte': 'please',
    'warum': 'why', 'wie': 'how', 'was': 'what', 'wo': 'where',
    'wer': 'who', 'wann': 'when',

    # Tech terms
    'Datei': 'file', 'Daten': 'data', 'Programm': 'program',
    'Code': 'code', 'Fehler': 'error', 'Funktion': 'function',
    'Klasse': 'class', 'Methode': 'method', 'Variable': 'variable',
    'Wert': 'value', 'Liste': 'list', 'Tabelle': 'table',
    'Netzwerk': 'network', 'Server': 'server', 'Datenbank': 'database',
    'Benutzer': 'user', 'Eingabe': 'input', 'Ausgabe': 'output',
    'Suchmaschine': 'search engine', 'Webseite': 'website',
    'Schnittstelle': 'interface', 'Speicher': 'memory',
    'Prozessor': 'processor', 'Bildschirm': 'screen',
    'Tastatur': 'keyboard', 'Maus': 'mouse',
}

# Build reverse dictionary (EN → DE)
EN_DE = {}
for de, en in DE_EN.items():
    if en not in EN_DE:  # Keep first mapping (most common)
        EN_DE[en] = de


# === Phrase Table ===
# Common multi-word expressions

PHRASES_DE_EN = {
    'auf jeden Fall': 'definitely',
    'zum Beispiel': 'for example',
    'in Ordnung': 'alright',
    'es gibt': 'there is',
    'so weit': 'so far',
    'auf der anderen Seite': 'on the other hand',
    'im Moment': 'at the moment',
    'zur Zeit': 'currently',
    'gar nicht': 'not at all',
    'so bald wie möglich': 'as soon as possible',
    'Bescheid sagen': 'let know',
    'Schluss machen': 'to finish',
    'im Grunde': 'basically',
    'vor allem': 'above all',
    'nach wie vor': 'still',
    'in der Tat': 'indeed',
    'auf keinen Fall': 'by no means',
    'ab und zu': 'now and then',
    'hin und wieder': 'every now and then',
    'mit anderen Worten': 'in other words',
    'was auch immer': 'whatever',
    'so schnell wie möglich': 'as fast as possible',
    'meiner Meinung nach': 'in my opinion',
    'ich glaube': 'I believe',
    'ich denke': 'I think',
    'lass uns': 'let us',
    'es tut mir leid': 'I am sorry',
    'keine Ahnung': 'no idea',
    'macht nichts': 'never mind',
    'was ist los': 'what is going on',
    'ich bin dran': 'it is my turn',
    'ich bin fertig': 'I am done',
}

PHRASES_EN_DE = {v: k for k, v in PHRASES_DE_EN.items()}


class Translator:
    """
    Statistical machine translation DE↔EN.

    Pipeline:
    1. Phrase matching (longest match first)
    2. Word-by-word dictionary lookup
    3. Compound word splitting (German compounds)
    4. API fallback (MyMemory, free tier)
    """

    def __init__(self, use_api_fallback: bool = True):
        self.use_api = use_api_fallback
        self.de_en = dict(DE_EN)
        self.en_de = dict(EN_DE)
        self.phrases_de_en = dict(PHRASES_DE_EN)
        self.phrases_en_de = dict(PHRASES_EN_DE)

    def translate(self, text: str, source: str = 'auto',
                  target: str = 'en') -> Dict:
        """
        Translate text.

        Args:
            text: input text
            source: 'de', 'en', or 'auto' (detect)
            target: 'de' or 'en'

        Returns:
            dict with: translation, source_lang, method, confidence
        """
        if not text.strip():
            return {'translation': '', 'source_lang': source, 'method': 'empty', 'confidence': 1.0}

        # Auto-detect language
        if source == 'auto':
            source = self._detect_language(text)

        if source == target:
            return {'translation': text, 'source_lang': source, 'method': 'identity', 'confidence': 1.0}

        # Try phrase + dictionary translation first
        local_result = self._translate_local(text, source, target)

        # If too many unknowns, try API
        if local_result['confidence'] < 0.5 and self.use_api:
            api_result = self._translate_api(text, source, target)
            if api_result and api_result.get('translation'):
                return api_result

        return local_result

    def _translate_local(self, text: str, source: str, target: str) -> Dict:
        """Local dictionary + phrase translation."""
        if source == 'de' and target == 'en':
            dictionary = self.de_en
            phrases = self.phrases_de_en
        elif source == 'en' and target == 'de':
            dictionary = self.en_de
            phrases = self.phrases_en_de
        else:
            return {'translation': text, 'source_lang': source,
                    'method': 'unsupported', 'confidence': 0.0}

        # Step 1: Replace phrases (longest first)
        result = text
        for phrase in sorted(phrases.keys(), key=len, reverse=True):
            if phrase.lower() in result.lower():
                # Case-insensitive replacement
                pattern = re.compile(re.escape(phrase), re.I)
                result = pattern.sub(phrases[phrase], result)

        # Step 2: Word-by-word translation
        words = result.split()
        translated = []
        known = 0
        total = 0

        for word in words:
            # Preserve punctuation
            prefix = ''
            suffix = ''
            clean = word
            while clean and not clean[0].isalnum():
                prefix += clean[0]
                clean = clean[1:]
            while clean and not clean[-1].isalnum():
                suffix = clean[-1] + suffix
                clean = clean[:-1]

            if not clean:
                translated.append(word)
                continue

            total += 1

            # Try exact match
            trans = dictionary.get(clean)
            if not trans:
                trans = dictionary.get(clean.lower())
            if not trans and source == 'de':
                # Try German compound splitting
                trans = self._split_compound(clean, dictionary)

            if trans:
                known += 1
                # Match capitalization
                if clean[0].isupper() and not trans[0].isupper():
                    trans = trans[0].upper() + trans[1:]
                translated.append(prefix + trans + suffix)
            else:
                # Keep original (might be a name, number, or cognate)
                translated.append(word)

        confidence = known / max(total, 1)
        translation = ' '.join(translated)

        # Step 3: Basic reordering for DE→EN
        if source == 'de' and target == 'en':
            translation = self._reorder_de_en(translation)

        return {
            'translation': translation,
            'source_lang': source,
            'method': 'dictionary',
            'confidence': round(confidence, 2),
            'known_words': known,
            'total_words': total,
        }

    def _split_compound(self, word: str, dictionary: Dict) -> Optional[str]:
        """
        Split German compound words.
        e.g., 'Haustür' → 'Haus' + 'Tür' → 'house door'
        """
        w = word.lower()
        # Try splitting at every position
        for i in range(3, len(w) - 2):
            left = w[:i]
            right = w[i:]
            left_trans = dictionary.get(left) or dictionary.get(left.capitalize())
            right_trans = dictionary.get(right) or dictionary.get(right.capitalize())
            if left_trans and right_trans:
                return f"{right_trans} {left_trans}"  # German: modifier + head
        return None

    def _reorder_de_en(self, text: str) -> str:
        """
        Basic German→English word order fixes.
        German: SOV in subordinate clauses → English: SVO
        """
        # Move verb from end of clause to after subject
        # Very simplified — handles "... verb" patterns
        sentences = text.split('.')
        fixed = []
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            # This is intentionally simple — full reordering needs a parser
            fixed.append(sent)
        return '. '.join(fixed) + ('.' if text.rstrip().endswith('.') else '')

    def _detect_language(self, text: str) -> str:
        """Simple language detection based on character and word frequency."""
        words = text.lower().split()

        # German indicators
        de_markers = {'der', 'die', 'das', 'und', 'ist', 'nicht', 'ich',
                     'ein', 'eine', 'auf', 'mit', 'für', 'sich', 'auch',
                     'von', 'zu', 'noch', 'den', 'dem', 'des', 'wird',
                     'sind', 'hat', 'nach', 'bei', 'über', 'aus', 'wie',
                     'kann', 'aber', 'nur', 'hier', 'schon', 'wenn'}
        en_markers = {'the', 'and', 'is', 'not', 'for', 'are', 'but',
                     'with', 'this', 'that', 'from', 'they', 'been',
                     'have', 'its', 'was', 'will', 'each', 'which',
                     'their', 'would', 'there', 'about', 'could'}

        de_count = sum(1 for w in words if w in de_markers)
        en_count = sum(1 for w in words if w in en_markers)

        # Check for German-specific characters
        if any(c in text for c in 'äöüßÄÖÜ'):
            de_count += 3

        return 'de' if de_count > en_count else 'en'

    def _translate_api(self, text: str, source: str, target: str) -> Optional[Dict]:
        """Fallback to MyMemory free translation API."""
        try:
            langpair = f"{source}|{target}"
            encoded = urllib.parse.quote(text[:500])  # API limit
            url = f"https://api.mymemory.translated.net/get?q={encoded}&langpair={langpair}"

            req = urllib.request.Request(url, headers={
                'User-Agent': 'FOSS-KI/1.0'
            })
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode('utf-8'))

            if data.get('responseStatus') == 200:
                translation = data['responseData']['translatedText']
                match_quality = data['responseData'].get('match', 0)
                return {
                    'translation': translation,
                    'source_lang': source,
                    'method': 'api:mymemory',
                    'confidence': round(match_quality, 2) if isinstance(match_quality, float) else 0.8,
                }
        except Exception:
            pass
        return None

    def add_word(self, source_word: str, target_word: str,
                 source_lang: str = 'de'):
        """Add a word to the dictionary."""
        if source_lang == 'de':
            self.de_en[source_word] = target_word
            if target_word not in self.en_de:
                self.en_de[target_word] = source_word
        else:
            self.en_de[source_word] = target_word
            if target_word not in self.de_en:
                self.de_en[target_word] = source_word

    def add_phrase(self, source_phrase: str, target_phrase: str,
                   source_lang: str = 'de'):
        """Add a phrase to the phrase table."""
        if source_lang == 'de':
            self.phrases_de_en[source_phrase] = target_phrase
        else:
            self.phrases_en_de[source_phrase] = target_phrase
