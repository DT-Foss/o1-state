"""
Natural Language Generation — Sentence Planning + Surface Realization
======================================================================
Pre-Transformer NLG pipeline (Reiter & Dale, 2000):

1. Content Determination: WHAT to say (from facts/frames)
2. Discourse Planning: HOW to organize (RST relations)
3. Sentence Planning: HOW to express each unit
4. Surface Realization: Generate grammatical sentences

This replaces the simple template-based Composer with a proper
NLG pipeline that produces COHERENT multi-sentence text.

RST Relations (Mann & Thompson, 1988):
- Elaboration: adds detail to nucleus
- Contrast: opposing information
- Cause: reason for nucleus
- Sequence: temporal/logical ordering
- Summary: condensed version
- Background: context for understanding

Combined with Case-Based Reasoning (CBR):
- For open questions, find similar stored responses and adapt them
"""

import re
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict


# ============================================================
# RST Discourse Relations
# ============================================================

class RSTRelation:
    """A rhetorical relation between text spans."""
    ELABORATION = "elaboration"
    CONTRAST = "contrast"
    CAUSE = "cause"
    RESULT = "result"
    SEQUENCE = "sequence"
    BACKGROUND = "background"
    SUMMARY = "summary"
    CONDITION = "condition"
    PURPOSE = "purpose"
    CONCESSION = "concession"


# Connectives for each relation
CONNECTIVES = {
    RSTRelation.ELABORATION: [
        "Specifically, ", "In particular, ", "More precisely, ",
        "That is, ", "For example, ", "",  # empty = no connective needed
    ],
    RSTRelation.CONTRAST: [
        "However, ", "On the other hand, ", "In contrast, ",
        "Unlike this, ", "Nevertheless, ", "But ",
    ],
    RSTRelation.CAUSE: [
        "This is because ", "The reason is that ",
        "Due to this, ", "As a result of ",
    ],
    RSTRelation.RESULT: [
        "As a result, ", "Therefore, ", "Consequently, ",
        "This means that ", "Thus, ",
    ],
    RSTRelation.SEQUENCE: [
        "Then, ", "Next, ", "After that, ", "Subsequently, ",
        "Following this, ", "Finally, ",
    ],
    RSTRelation.BACKGROUND: [
        "To understand this, ", "For context, ",
        "It is important to note that ", "",
    ],
    RSTRelation.SUMMARY: [
        "In summary, ", "Overall, ", "To sum up, ",
        "In conclusion, ",
    ],
    RSTRelation.CONDITION: [
        "If this is the case, ", "Under these conditions, ",
        "Provided that ", "When ",
    ],
    RSTRelation.PURPOSE: [
        "In order to ", "So that ", "This is done to ",
        "The purpose is to ",
    ],
    RSTRelation.CONCESSION: [
        "Although ", "Despite this, ", "Even though ",
        "While this is true, ",
    ],
}


# ============================================================
# Discourse Planner
# ============================================================

class DiscoursePlan:
    """A plan for organizing content into coherent text."""

    def __init__(self):
        self.units: List[Dict[str, Any]] = []  # content units
        self.relations: List[Tuple[int, int, str]] = []  # (from_idx, to_idx, relation)

    def add_unit(self, content: str, role: str = "nucleus",
                 importance: float = 1.0) -> int:
        """Add a content unit. Returns its index."""
        idx = len(self.units)
        self.units.append({
            'content': content,
            'role': role,  # 'nucleus' or 'satellite'
            'importance': importance,
            'index': idx,
        })
        return idx

    def add_relation(self, from_idx: int, to_idx: int, relation: str):
        """Add a rhetorical relation between units."""
        self.relations.append((from_idx, to_idx, relation))

    def get_ordered_units(self) -> List[Dict[str, Any]]:
        """Get units in presentation order."""
        if not self.units:
            return []

        # Build dependency graph
        order: List[int] = []
        used = set()

        # Start with highest-importance nucleus
        nuclei = [(u['importance'], u['index']) for u in self.units if u['role'] == 'nucleus']
        nuclei.sort(reverse=True)

        if nuclei:
            start = nuclei[0][1]
        else:
            start = 0

        # Separate summary units (they go last)
        summary_indices = set()
        for _, to_idx, rel in self.relations:
            if rel == RSTRelation.SUMMARY:
                summary_indices.add(to_idx)

        # BFS from start, following relations (skip summaries)
        queue = [start]
        while queue:
            idx = queue.pop(0)
            if idx in used or idx in summary_indices:
                continue
            used.add(idx)
            order.append(idx)

            # Find related units
            for f_idx, to_idx, _ in self.relations:
                if f_idx == idx and to_idx not in used:
                    queue.append(to_idx)
                elif to_idx == idx and f_idx not in used:
                    queue.append(f_idx)

        # Add any remaining non-summary units
        for u in self.units:
            if u['index'] not in used and u['index'] not in summary_indices:
                order.append(u['index'])

        # Summaries go last
        for idx in summary_indices:
            order.append(idx)

        return [self.units[i] for i in order]

    def get_relation_for(self, unit_idx: int) -> Optional[str]:
        """Get the incoming relation for a unit."""
        for from_idx, to_idx, rel in self.relations:
            if to_idx == unit_idx:
                return rel
        return None


# ============================================================
# Sentence Planner
# ============================================================

class SentencePlanner:
    """
    Plans how to express content as sentences.

    Takes facts/content and creates sentence specifications,
    handling:
    - Aggregation (combining related facts into one sentence)
    - Referring expressions (pronoun/name selection)
    - Sentence variety (varying structure)
    """

    def __init__(self):
        self._last_subject = None
        self._sentence_count = 0

    def plan_from_facts(self, facts: List[Tuple[str, str, str]],
                        query: str = "") -> DiscoursePlan:
        """
        Create a discourse plan from a list of (S, R, O) facts.

        Groups related facts, determines relations, orders for coherence.
        """
        plan = DiscoursePlan()

        if not facts:
            return plan

        # Group facts by subject
        by_subject: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        for s, r, o in facts:
            by_subject[s].append((r, o))

        # Determine primary subject (most facts, or query subject)
        primary = max(by_subject, key=lambda s: len(by_subject[s]))

        # Build plan: identity/type first, then details
        priority_rels = ['type', 'identity', 'is_a', 'description',
                        'occupation', 'capital', 'population', 'location',
                        'birthplace', 'born', 'nationality']

        # Nucleus: primary subject introduction
        primary_facts = by_subject[primary]
        intro_facts = []
        detail_facts = []

        for r, o in primary_facts:
            if r in priority_rels:
                intro_facts.append((r, o))
            else:
                detail_facts.append((r, o))

        # Create introduction sentence(s)
        if intro_facts:
            intro_text = self._aggregate_facts(primary, intro_facts)
            idx_intro = plan.add_unit(intro_text, "nucleus", importance=1.0)
        else:
            idx_intro = plan.add_unit(
                self._single_fact(primary, detail_facts[0][0], detail_facts[0][1]),
                "nucleus", importance=1.0
            )
            detail_facts = detail_facts[1:]

        # Elaboration units for detail facts
        prev_idx = idx_intro
        for i, (r, o) in enumerate(detail_facts):
            text = self._single_fact(primary, r, o, use_pronoun=(i > 0))
            idx = plan.add_unit(text, "satellite", importance=0.5)
            plan.add_relation(prev_idx, idx, RSTRelation.ELABORATION)
            prev_idx = idx

        # Add facts from other subjects
        for subject, subject_facts in by_subject.items():
            if subject == primary:
                continue
            text = self._aggregate_facts(subject, subject_facts)
            idx = plan.add_unit(text, "satellite", importance=0.3)
            plan.add_relation(idx_intro, idx, RSTRelation.ELABORATION)

        return plan

    def plan_explanation(self, topic: str, points: List[str]) -> DiscoursePlan:
        """
        Create a plan for explaining a topic.

        Uses Background → Elaboration → Summary structure.
        """
        plan = DiscoursePlan()

        if not points:
            return plan

        # Background/introduction
        idx_bg = plan.add_unit(
            f"{topic} involves several important aspects.",
            "nucleus", importance=1.0
        )

        # Points as elaborations or sequence
        prev_idx = idx_bg
        for i, point in enumerate(points):
            rel = RSTRelation.SEQUENCE if i > 0 else RSTRelation.ELABORATION
            idx = plan.add_unit(point, "satellite", importance=0.6)
            plan.add_relation(prev_idx, idx, rel)
            prev_idx = idx

        # Summary if enough points
        if len(points) >= 3:
            idx_sum = plan.add_unit(
                f"These aspects together define {topic}.",
                "satellite", importance=0.4
            )
            plan.add_relation(idx_bg, idx_sum, RSTRelation.SUMMARY)

        return plan

    def _aggregate_facts(self, subject: str, facts: List[Tuple[str, str]]) -> str:
        """Combine related facts into a single sentence."""
        if not facts:
            return ""

        if len(facts) == 1:
            return self._single_fact(subject, facts[0][0], facts[0][1])

        # Group aggregatable relations
        sentences = []

        # Identity/type facts aggregate well
        type_facts = [(r, o) for r, o in facts if r in ('type', 'identity', 'is_a', 'occupation')]
        other_facts = [(r, o) for r, o in facts if r not in ('type', 'identity', 'is_a', 'occupation')]

        if type_facts:
            types = [o for _, o in type_facts]
            if len(types) == 1:
                sentences.append(f"{subject} is a {types[0]}")
            elif len(types) == 2:
                sentences.append(f"{subject} is a {types[0]} and {types[1]}")
            else:
                sentences.append(f"{subject} is a {', '.join(types[:-1])}, and {types[-1]}")

        # Border/list facts aggregate
        list_facts = [(r, o) for r, o in other_facts if r in ('borders', 'part_of', 'found_in')]
        remaining = [(r, o) for r, o in other_facts if r not in ('borders', 'part_of', 'found_in')]

        if list_facts:
            grouped = defaultdict(list)
            for r, o in list_facts:
                grouped[r].append(o)
            for r, objects in grouped.items():
                verb = self._relation_verb(r)
                if len(objects) == 1:
                    sentences.append(f"{subject} {verb} {objects[0]}")
                elif len(objects) == 2:
                    sentences.append(f"{subject} {verb} {objects[0]} and {objects[1]}")
                else:
                    sentences.append(f"{subject} {verb} {', '.join(objects[:-1])}, and {objects[-1]}")

        # Remaining facts as individual sentences
        for r, o in remaining[:3]:  # Cap at 3 to avoid run-on
            sentences.append(self._single_fact(subject, r, o, no_period=True))

        return '. '.join(sentences) + '.' if sentences else ""

    def _single_fact(self, subject: str, relation: str, obj: str,
                     use_pronoun: bool = False, no_period: bool = False) -> str:
        """Generate a single sentence from a fact."""
        s = "It" if use_pronoun else subject
        verb = self._relation_verb(relation)
        sentence = f"{s} {verb} {obj}"
        return sentence if no_period else sentence + "."

    @staticmethod
    def _relation_verb(relation: str) -> str:
        """Convert a relation name to a verb phrase."""
        verbs = {
            'capital': "has its capital in",
            'population': "has a population of",
            'language': "speaks",
            'borders': "borders",
            'location': "is located in",
            'birthplace': "was born in",
            'born': "was born in",
            'founder': "was founded by",
            'creator': "was created by",
            'discoverer': "was discovered by",
            'inventor': "was invented by",
            'author': "was written by",
            'type': "is a",
            'identity': "is a",
            'is_a': "is a",
            'description': "is",
            'known_as': "is also known as",
            'founded': "was founded in",
            'formula': "has the formula",
            'symbol': "has the symbol",
            'part_of': "is part of",
            'currency': "uses the currency",
            'died': "died in",
            'nationality': "is from",
            'occupation': "is a",
            'country': "is in",
            'has_property': "is",
            'can': "can",
            'needs': "needs",
            'used_for': "is used for",
            'causes': "causes",
            'found_in': "is found in",
            'opposite_of': "is the opposite of",
        }
        return verbs.get(relation, f"has {relation}")


# ============================================================
# Surface Realizer
# ============================================================

class SurfaceRealizer:
    """
    Converts a discourse plan into fluent text.

    Adds connectives based on RST relations,
    handles sentence variety, and ensures grammatical output.
    """

    def __init__(self, lm_model=None):
        """
        Args:
            lm_model: optional PPM for scoring alternative phrasings
        """
        self.lm = lm_model
        self._connective_idx: Dict[str, int] = defaultdict(int)

    def realize(self, plan: DiscoursePlan) -> str:
        """Convert a discourse plan to text."""
        if not plan.units:
            return ""

        ordered = plan.get_ordered_units()
        paragraphs: List[str] = []
        current_paragraph: List[str] = []

        for i, unit in enumerate(ordered):
            content = unit['content'].strip()
            if not content:
                continue

            # Get relation to determine connective
            relation = plan.get_relation_for(unit['index'])

            if i == 0:
                # First unit: no connective
                current_paragraph.append(content)
            elif relation:
                connective = self._pick_connective(relation)
                # Start new paragraph on major relation changes
                if relation in (RSTRelation.CONTRAST, RSTRelation.BACKGROUND, RSTRelation.SUMMARY):
                    if current_paragraph:
                        paragraphs.append(' '.join(current_paragraph))
                        current_paragraph = []

                if connective:
                    # Ensure sentence doesn't start with lowercase after connective
                    if content[0].islower():
                        content = content[0].upper() + content[1:]
                    sentence = connective + content
                else:
                    sentence = content

                current_paragraph.append(sentence)
            else:
                current_paragraph.append(content)

        if current_paragraph:
            paragraphs.append(' '.join(current_paragraph))

        return '\n\n'.join(paragraphs)

    def _pick_connective(self, relation: str) -> str:
        """Pick a connective, cycling through options for variety."""
        options = CONNECTIVES.get(relation, [""])
        idx = self._connective_idx[relation] % len(options)
        self._connective_idx[relation] += 1
        return options[idx]


# ============================================================
# Case-Based Reasoning (CBR)
# ============================================================

class CaseBase:
    """
    Case-Based Reasoning for open-ended questions.

    Stores question-answer pairs as cases. When a new question
    arrives, finds the most similar stored case and adapts
    its answer.

    This handles questions like "Why is democracy important?"
    without needing free text generation — by finding and
    adapting similar stored explanations.

    Usage:
        cb = CaseBase()
        cb.add_case("Why is education important?",
                     "Education is important because it develops critical thinking, ...")
        answer = cb.retrieve_and_adapt("Why is learning important?")
    """

    def __init__(self):
        self.cases: List[Dict[str, Any]] = []
        self._word_idf: Dict[str, float] = {}

    def add_case(self, question: str, answer: str,
                 category: str = "", tags: Optional[List[str]] = None):
        """Store a case (question-answer pair)."""
        self.cases.append({
            'question': question,
            'answer': answer,
            'category': category,
            'tags': tags or [],
            'q_words': set(self._normalize(question).split()),
        })
        self._update_idf()

    def add_cases_bulk(self, cases: List[Tuple[str, str]]):
        """Add multiple (question, answer) pairs."""
        for q, a in cases:
            self.cases.append({
                'question': q,
                'answer': a,
                'category': '',
                'tags': [],
                'q_words': set(self._normalize(q).split()),
            })
        self._update_idf()

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[Dict, float]]:
        """
        Find the most similar cases to a query.

        Returns list of (case, similarity_score) pairs.
        """
        if not self.cases:
            return []

        query_words = set(self._normalize(query).split())
        scored = []

        for case in self.cases:
            sim = self._similarity(query_words, case['q_words'])
            if sim > 0:
                scored.append((case, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def retrieve_and_adapt(self, query: str) -> Optional[str]:
        """
        Find the best matching case and adapt its answer.

        Adaptation: substitutes the old subject with the new one.
        """
        results = self.retrieve(query, top_k=1)
        if not results:
            return None

        best_case, similarity = results[0]
        if similarity < 0.2:
            return None

        # Extract subjects from both questions
        old_subject = self._extract_subject(best_case['question'])
        new_subject = self._extract_subject(query)

        answer = best_case['answer']

        # Adapt: replace old subject with new subject
        if old_subject and new_subject and old_subject != new_subject:
            answer = answer.replace(old_subject, new_subject)
            answer = answer.replace(old_subject.lower(), new_subject.lower())
            answer = answer.replace(old_subject.upper(), new_subject.upper())

        return answer

    def _similarity(self, words_a: set, words_b: set) -> float:
        """TF-IDF weighted Jaccard similarity."""
        if not words_a or not words_b:
            return 0.0

        # Weighted intersection
        common = words_a & words_b
        if not common:
            return 0.0

        # IDF weighting
        common_weight = sum(self._word_idf.get(w, 1.0) for w in common)
        all_weight = sum(self._word_idf.get(w, 1.0) for w in words_a | words_b)

        return common_weight / max(all_weight, 1e-10)

    def _update_idf(self):
        """Update IDF weights from stored cases."""
        doc_freq: Dict[str, int] = defaultdict(int)
        n = len(self.cases)
        for case in self.cases:
            for word in case['q_words']:
                doc_freq[word] += 1

        import math
        for word, df in doc_freq.items():
            self._word_idf[word] = math.log((n + 1) / (df + 1)) + 1

    @staticmethod
    def _normalize(text: str) -> str:
        """Normalize text for comparison."""
        text = text.lower().strip()
        text = re.sub(r'[^\w\s]', '', text)
        # Remove stop words
        stops = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be',
                'to', 'of', 'in', 'for', 'on', 'with', 'it', 'do', 'does'}
        words = [w for w in text.split() if w not in stops and len(w) > 1]
        return ' '.join(words)

    @staticmethod
    def _extract_subject(question: str) -> Optional[str]:
        """Extract the main subject from a question."""
        q = question.strip().rstrip('?').strip()

        # "Why is X important?"
        m = re.search(r'(?:why|how)\s+is\s+(.+?)\s+(?:important|good|bad|useful|necessary)', q, re.I)
        if m:
            return m.group(1).strip()

        # "What is X?"
        m = re.search(r'what\s+is\s+(.+?)(?:\s+(?:like|about))?$', q, re.I)
        if m:
            return m.group(1).strip()

        # "Tell me about X"
        m = re.search(r'(?:tell|explain|describe)\s+(?:me\s+)?(?:about\s+)?(.+)', q, re.I)
        if m:
            return m.group(1).strip()

        # Last resort: last noun phrase
        words = q.split()
        if words:
            return words[-1]
        return None


# ============================================================
# NLG Pipeline (combines everything)
# ============================================================

class NLGPipeline:
    """
    Complete Natural Language Generation pipeline.

    Combines:
    - SentencePlanner (content → discourse plan)
    - SurfaceRealizer (plan → text)
    - CaseBase (open-ended question handling)

    Usage:
        nlg = NLGPipeline()

        # From facts
        text = nlg.generate_from_facts([
            ("Germany", "capital", "Berlin"),
            ("Germany", "population", "83 million"),
            ("Germany", "borders", "France"),
        ])

        # From case base
        nlg.case_base.add_case("Why is education important?",
            "Education develops critical thinking and ...")
        text = nlg.answer_open_question("Why is learning important?")
    """

    def __init__(self, lm_model=None):
        self.planner = SentencePlanner()
        self.realizer = SurfaceRealizer(lm_model)
        self.case_base = CaseBase()

        # Pre-load some general cases
        self._load_default_cases()

    def generate_from_facts(self, facts: List[Tuple[str, str, str]],
                           query: str = "") -> str:
        """Generate coherent text from fact triplets."""
        plan = self.planner.plan_from_facts(facts, query)
        return self.realizer.realize(plan)

    def generate_explanation(self, topic: str, points: List[str]) -> str:
        """Generate an explanation with proper discourse structure."""
        plan = self.planner.plan_explanation(topic, points)
        return self.realizer.realize(plan)

    def answer_open_question(self, question: str) -> Optional[str]:
        """Try to answer an open-ended question via CBR."""
        return self.case_base.retrieve_and_adapt(question)

    def _load_default_cases(self):
        """Load default cases for common open-ended questions."""
        defaults = [
            ("Why is democracy important?",
             "Democracy is important because it gives citizens the power to choose their leaders and influence policy. It protects individual rights and freedoms through constitutional safeguards. Democratic systems provide accountability through regular elections and transparency. They also allow peaceful transitions of power and encourage civic participation."),

            ("Why is education important?",
             "Education is important because it develops critical thinking and problem-solving skills. It provides knowledge and skills needed for career success. Education promotes social mobility and reduces inequality. It also builds informed citizens who can participate effectively in democratic society."),

            ("Why is science important?",
             "Science is important because it helps us understand how the natural world works through systematic observation and experimentation. It drives technological innovation and improves quality of life. Science provides evidence-based solutions to global challenges like disease, climate change, and food security."),

            ("How does the internet work?",
             "The internet works by connecting millions of computer networks worldwide using standardized protocols like TCP/IP. Data is broken into packets, routed through multiple network nodes, and reassembled at the destination. DNS servers translate human-readable domain names into IP addresses. The infrastructure includes undersea cables, data centers, and local networks that together form a global communication system."),

            ("What is artificial intelligence?",
             "Artificial intelligence is the field of computer science focused on creating systems that can perform tasks normally requiring human intelligence. This includes learning from data, recognizing patterns, making decisions, and understanding language. AI ranges from narrow systems designed for specific tasks to the goal of general intelligence that can handle any cognitive task."),

            ("Why is climate change a problem?",
             "Climate change is a problem because rising global temperatures cause more extreme weather events, sea level rise, and ecosystem disruption. It threatens food security through changing rainfall patterns and crop failures. It disproportionately affects vulnerable populations and developing nations. The economic costs of inaction far exceed the costs of mitigation and adaptation."),

            ("What makes a good leader?",
             "A good leader demonstrates integrity, clear communication, and the ability to inspire others toward shared goals. They make informed decisions while remaining open to feedback and new information. Good leaders take responsibility for outcomes and empower their team members to grow. They balance confidence with humility and short-term needs with long-term vision."),

            ("Why is exercise important?",
             "Exercise is important because it strengthens the cardiovascular system and reduces the risk of chronic diseases. It improves mental health by reducing stress, anxiety, and depression through endorphin release. Regular physical activity enhances sleep quality, cognitive function, and energy levels. It also helps maintain healthy weight and bone density."),

            ("Why is privacy important?",
             "Privacy is important because it protects individual autonomy and the freedom to think and act without surveillance. It prevents abuse of power by limiting what governments and corporations can know about citizens. Privacy enables free speech by allowing people to express unpopular opinions without fear. It is a fundamental human right recognized by the Universal Declaration of Human Rights."),

            ("Why is diversity important?",
             "Diversity is important because it brings together different perspectives, experiences, and ideas that lead to better problem-solving. Diverse teams consistently outperform homogeneous ones in creativity and innovation. It promotes fairness and equal opportunity in society. Exposure to diverse viewpoints also reduces prejudice and builds empathy."),

            ("How does gravity work?",
             "Gravity is a fundamental force of nature that attracts objects with mass toward each other. According to Newton, the gravitational force between two objects is proportional to their masses and inversely proportional to the square of the distance between them. Einstein's general relativity describes gravity as the curvature of spacetime caused by mass and energy. On Earth, gravity accelerates objects at approximately 9.8 meters per second squared."),

            ("How does evolution work?",
             "Evolution works through natural selection, where organisms with traits better suited to their environment are more likely to survive and reproduce. Random genetic mutations introduce variation in a population. Over many generations, beneficial traits become more common while harmful ones decrease. This process has produced the enormous diversity of life on Earth over billions of years."),

            ("Why is sleep important?",
             "Sleep is important because it allows the brain to consolidate memories, process emotions, and clear metabolic waste. During sleep, the body repairs tissues, strengthens the immune system, and regulates hormones. Chronic sleep deprivation increases the risk of obesity, diabetes, cardiovascular disease, and mental health disorders. Adults typically need 7-9 hours of quality sleep per night."),

            ("Why is water important?",
             "Water is important because it makes up about 60 percent of the human body and is essential for nearly every biological function. It regulates body temperature, transports nutrients, lubricates joints, and removes waste. Water is also crucial for agriculture, industry, and energy production. Access to clean water is a fundamental human need and remains a global challenge."),

            ("How does photosynthesis work?",
             "Photosynthesis is the process by which plants convert sunlight, water, and carbon dioxide into glucose and oxygen. Light energy is captured by chlorophyll in the chloroplasts of plant cells. This energy drives a series of chemical reactions that split water molecules and fix carbon dioxide into organic sugars. The oxygen released as a byproduct is essential for most life on Earth."),

            ("Why is reading important?",
             "Reading is important because it develops vocabulary, improves comprehension, and strengthens analytical thinking. It exposes readers to new ideas, cultures, and perspectives they might never encounter otherwise. Regular reading has been shown to reduce stress, improve empathy, and slow cognitive decline with age. It also builds knowledge that compounds over time, making subsequent learning easier."),

            ("Why is mathematics important?",
             "Mathematics is important because it provides the universal language for describing patterns, structures, and relationships in the natural world. It underpins all of science, engineering, economics, and technology. Mathematical thinking develops logical reasoning and problem-solving skills applicable to any domain. From cryptography to medicine to space exploration, mathematics is the foundation of modern civilization."),
        ]
        self.case_base.add_cases_bulk(defaults)
