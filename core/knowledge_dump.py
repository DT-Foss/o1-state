"""
Knowledge Dump — Fill FOSS-KI with everything we know
=======================================================
Massive structured knowledge injection. Every fact is a (S, R, O) triplet.
This goes way beyond countries — science, tech, history, CS, math, people.

Call inject_all(knowledge_store) to load everything.
"""


def inject_all(ks) -> int:
    """Inject all knowledge domains. Returns total facts added."""
    n = 0
    n += _inject_cs(ks)
    n += _inject_science(ks)
    n += _inject_math(ks)
    n += _inject_history(ks)
    n += _inject_tech(ks)
    n += _inject_people(ks)
    n += _inject_philosophy(ks)
    n += _inject_music(ks)
    n += _inject_literature(ks)
    n += _inject_geography_extra(ks)
    n += _inject_biology(ks)
    n += _inject_physics(ks)
    n += _inject_chemistry(ks)
    n += _inject_economics(ks)
    n += _inject_space(ks)
    n += _inject_animals(ks)
    n += _inject_sports(ks)
    n += _inject_food(ks)
    n += _inject_art(ks)
    n += _inject_religion(ks)
    n += _inject_earth_science(ks)
    n += _inject_medicine(ks)
    n += _inject_law(ks)
    n += _inject_languages(ks)
    n += _inject_tech_companies(ks)
    n += _inject_historical_events(ks)
    return n


def _store(ks, facts):
    """Store facts, return count."""
    ks.store_facts(facts)
    return len(facts)


def _inject_cs(ks) -> int:
    """Computer Science fundamentals."""
    return _store(ks, [
        # Programming languages
        ("Python", "creator", "Guido van Rossum"),
        ("Python", "first_released", "1991"),
        ("Python", "paradigm", "multi-paradigm"),
        ("Python", "typing", "dynamically typed"),
        ("Python", "known_for", "readability and simplicity"),
        ("JavaScript", "creator", "Brendan Eich"),
        ("JavaScript", "first_released", "1995"),
        ("JavaScript", "paradigm", "multi-paradigm"),
        ("JavaScript", "known_for", "web browser scripting"),
        ("JavaScript", "standardized_as", "ECMAScript"),
        ("C", "creator", "Dennis Ritchie"),
        ("C", "first_released", "1972"),
        ("C", "known_for", "systems programming and operating systems"),
        ("C++", "creator", "Bjarne Stroustrup"),
        ("C++", "first_released", "1985"),
        ("C++", "based_on", "C"),
        ("Java", "creator", "James Gosling"),
        ("Java", "first_released", "1995"),
        ("Java", "known_for", "write once run anywhere"),
        ("Java", "paradigm", "object-oriented"),
        ("Rust", "creator", "Graydon Hoare"),
        ("Rust", "first_released", "2010"),
        ("Rust", "known_for", "memory safety without garbage collection"),
        ("Go", "creator", "Rob Pike, Ken Thompson, Robert Griesemer"),
        ("Go", "first_released", "2009"),
        ("Go", "organization", "Google"),
        ("Haskell", "first_released", "1990"),
        ("Haskell", "paradigm", "purely functional"),
        ("Ruby", "creator", "Yukihiro Matsumoto"),
        ("Ruby", "first_released", "1995"),
        ("Lisp", "creator", "John McCarthy"),
        ("Lisp", "first_released", "1958"),
        ("Lisp", "known_for", "artificial intelligence research"),
        ("SQL", "first_released", "1974"),
        ("SQL", "known_for", "relational database queries"),
        ("TypeScript", "creator", "Anders Hejlsberg"),
        ("TypeScript", "organization", "Microsoft"),
        ("TypeScript", "based_on", "JavaScript"),
        ("Swift", "creator", "Chris Lattner"),
        ("Swift", "organization", "Apple"),
        ("Kotlin", "organization", "JetBrains"),

        # Data structures
        ("array", "type", "data structure"),
        ("array", "access_time", "O(1)"),
        ("array", "insertion_time", "O(n)"),
        ("linked list", "type", "data structure"),
        ("linked list", "access_time", "O(n)"),
        ("linked list", "insertion_time", "O(1)"),
        ("hash table", "type", "data structure"),
        ("hash table", "average_lookup", "O(1)"),
        ("hash table", "worst_case_lookup", "O(n)"),
        ("hash table", "known_for", "key-value storage"),
        ("binary tree", "type", "data structure"),
        ("binary tree", "search_time", "O(log n)"),
        ("binary search tree", "type", "data structure"),
        ("binary search tree", "property", "left < root < right"),
        ("red-black tree", "type", "self-balancing binary search tree"),
        ("B-tree", "type", "data structure"),
        ("B-tree", "used_in", "databases and file systems"),
        ("heap", "type", "data structure"),
        ("heap", "property", "parent >= children (max-heap)"),
        ("graph", "type", "data structure"),
        ("graph", "components", "vertices and edges"),
        ("trie", "type", "data structure"),
        ("trie", "known_for", "prefix-based string search"),
        ("stack", "type", "data structure"),
        ("stack", "property", "last in first out (LIFO)"),
        ("queue", "type", "data structure"),
        ("queue", "property", "first in first out (FIFO)"),

        # Algorithms
        ("quicksort", "type", "sorting algorithm"),
        ("quicksort", "time_complexity", "O(n log n) average"),
        ("quicksort", "worst_case", "O(n^2)"),
        ("quicksort", "method", "divide and conquer with pivot"),
        ("merge sort", "type", "sorting algorithm"),
        ("merge sort", "time_complexity", "O(n log n)"),
        ("merge sort", "property", "stable sort"),
        ("bubble sort", "type", "sorting algorithm"),
        ("bubble sort", "time_complexity", "O(n^2)"),
        ("bubble sort", "known_for", "simplest sorting algorithm"),
        ("binary search", "type", "search algorithm"),
        ("binary search", "time_complexity", "O(log n)"),
        ("binary search", "requirement", "sorted input"),
        ("dijkstra", "type", "shortest path algorithm"),
        ("dijkstra", "creator", "Edsger Dijkstra"),
        ("dijkstra", "time_complexity", "O(V^2) or O(E log V) with heap"),
        ("A* search", "type", "pathfinding algorithm"),
        ("A* search", "based_on", "dijkstra with heuristic"),
        ("depth-first search", "type", "graph traversal"),
        ("depth-first search", "uses", "stack"),
        ("breadth-first search", "type", "graph traversal"),
        ("breadth-first search", "uses", "queue"),
        ("dynamic programming", "type", "algorithm design technique"),
        ("dynamic programming", "principle", "optimal substructure + overlapping subproblems"),

        # OS and systems
        ("Linux", "creator", "Linus Torvalds"),
        ("Linux", "first_released", "1991"),
        ("Linux", "type", "operating system kernel"),
        ("Unix", "creators", "Ken Thompson and Dennis Ritchie"),
        ("Unix", "first_released", "1969"),
        ("Unix", "organization", "Bell Labs"),
        ("Windows", "organization", "Microsoft"),
        ("macOS", "organization", "Apple"),
        ("TCP/IP", "type", "network protocol suite"),
        ("HTTP", "type", "application layer protocol"),
        ("HTTP", "port", "80"),
        ("HTTPS", "port", "443"),
        ("DNS", "type", "domain name resolution"),
        ("DNS", "port", "53"),
        ("SSH", "type", "secure remote access protocol"),
        ("SSH", "port", "22"),

        # Concepts
        ("recursion", "type", "programming concept"),
        ("recursion", "definition", "a function that calls itself"),
        ("recursion", "requires", "base case and recursive case"),
        ("object-oriented programming", "type", "programming paradigm"),
        ("object-oriented programming", "principles", "encapsulation, inheritance, polymorphism, abstraction"),
        ("functional programming", "type", "programming paradigm"),
        ("functional programming", "principles", "immutability, pure functions, higher-order functions"),
        ("machine learning", "type", "field of computer science"),
        ("machine learning", "definition", "algorithms that learn from data"),
        ("deep learning", "type", "subset of machine learning"),
        ("deep learning", "based_on", "artificial neural networks"),
        ("neural network", "type", "computational model"),
        ("neural network", "inspired_by", "biological neurons"),
        ("transformer", "type", "neural network architecture"),
        ("transformer", "introduced_by", "Vaswani et al. 2017"),
        ("transformer", "known_for", "self-attention mechanism"),
        ("gradient descent", "type", "optimization algorithm"),
        ("gradient descent", "used_in", "training neural networks"),
        ("backpropagation", "type", "algorithm"),
        ("backpropagation", "used_for", "computing gradients in neural networks"),
        ("Big O notation", "type", "asymptotic analysis"),
        ("Big O notation", "definition", "upper bound on algorithm growth rate"),
        ("NP-completeness", "type", "complexity class"),
        ("NP-completeness", "definition", "problems verifiable in polynomial time but not known to be solvable in polynomial time"),
        ("Turing machine", "creator", "Alan Turing"),
        ("Turing machine", "type", "theoretical computational model"),
        ("halting problem", "type", "undecidable problem"),
        ("halting problem", "proved_by", "Alan Turing"),
    ])


def _inject_science(ks) -> int:
    """General science facts."""
    return _store(ks, [
        ("speed of light", "value", "299,792,458 m/s"),
        ("speed of light", "symbol", "c"),
        ("speed of sound", "value", "343 m/s in air at 20°C"),
        ("gravity", "acceleration", "9.81 m/s² on Earth"),
        ("gravity", "discovered_by", "Isaac Newton"),
        ("Earth", "age", "4.54 billion years"),
        ("Earth", "diameter", "12,742 km"),
        ("Earth", "distance_from_sun", "149.6 million km"),
        ("Earth", "mass", "5.972 × 10^24 kg"),
        ("Sun", "type", "G-type main-sequence star"),
        ("Sun", "age", "4.6 billion years"),
        ("Sun", "distance_from_earth", "149.6 million km"),
        ("Sun", "surface_temperature", "5,778 K"),
        ("Moon", "distance_from_earth", "384,400 km"),
        ("Moon", "orbital_period", "27.3 days"),
        ("Mars", "type", "planet"),
        ("Mars", "known_for", "the Red Planet"),
        ("Mars", "distance_from_sun", "227.9 million km"),
        ("Jupiter", "type", "planet"),
        ("Jupiter", "known_for", "largest planet in solar system"),
        ("Saturn", "type", "planet"),
        ("Saturn", "known_for", "ring system"),
        ("Milky Way", "type", "barred spiral galaxy"),
        ("Milky Way", "diameter", "100,000 light-years"),
        ("Milky Way", "stars", "100-400 billion"),
        ("Big Bang", "type", "cosmological theory"),
        ("Big Bang", "age", "13.8 billion years ago"),
        ("black hole", "type", "astronomical object"),
        ("black hole", "property", "gravity so strong light cannot escape"),
        ("DNA", "structure", "double helix"),
        ("DNA", "function", "stores genetic information"),
        ("evolution", "proposed_by", "Charles Darwin"),
        ("evolution", "publication", "On the Origin of Species (1859)"),
        ("evolution", "mechanism", "natural selection"),
        ("photosynthesis", "type", "biological process"),
        ("photosynthesis", "equation", "6CO2 + 6H2O → C6H12O6 + 6O2"),
        ("photosynthesis", "performed_by", "plants, algae, cyanobacteria"),
        ("cell", "type", "basic unit of life"),
        ("cell", "discovered_by", "Robert Hooke (1665)"),
        ("mitochondria", "function", "powerhouse of the cell, produces ATP"),
        ("atom", "components", "protons, neutrons, electrons"),
        ("atom", "discovered_by", "concept from Democritus, modern model by Rutherford/Bohr"),
        ("periodic table", "creator", "Dmitri Mendeleev"),
        ("periodic table", "created", "1869"),
        ("periodic table", "elements", "118 known elements"),
        ("water", "formula", "H2O"),
        ("water", "boiling_point", "100°C at standard pressure"),
        ("water", "freezing_point", "0°C at standard pressure"),
    ])


def _inject_math(ks) -> int:
    """Mathematics."""
    return _store(ks, [
        ("pi", "value", "3.14159265358979..."),
        ("pi", "definition", "ratio of circumference to diameter of a circle"),
        ("pi", "type", "irrational number"),
        ("e", "value", "2.71828182845904..."),
        ("e", "definition", "base of natural logarithm"),
        ("e", "known_for", "compound interest and exponential growth"),
        ("golden ratio", "value", "1.61803398875..."),
        ("golden ratio", "symbol", "φ (phi)"),
        ("Pythagorean theorem", "formula", "a² + b² = c²"),
        ("Pythagorean theorem", "author", "Pythagoras"),
        ("Euler's identity", "formula", "e^(iπ) + 1 = 0"),
        ("Euler's identity", "known_for", "most beautiful equation in mathematics"),
        ("prime number", "definition", "natural number greater than 1 with no positive divisors other than 1 and itself"),
        ("Fibonacci sequence", "definition", "each number is the sum of the two preceding ones"),
        ("Fibonacci sequence", "starts_with", "0, 1, 1, 2, 3, 5, 8, 13, 21, 34"),
        ("calculus", "inventors", "Isaac Newton and Gottfried Leibniz"),
        ("calculus", "branches", "differential and integral"),
        ("derivative", "definition", "rate of change of a function"),
        ("integral", "definition", "area under a curve"),
        ("set theory", "founder", "Georg Cantor"),
        ("infinity", "types", "countable and uncountable"),
        ("Gödel's incompleteness theorems", "author", "Kurt Gödel"),
        ("Gödel's incompleteness theorems", "year", "1931"),
        ("Gödel's incompleteness theorems", "states", "any consistent formal system cannot prove all true statements about natural numbers"),
        ("Fermat's Last Theorem", "author", "Pierre de Fermat"),
        ("Fermat's Last Theorem", "proved_by", "Andrew Wiles (1995)"),
        ("Fermat's Last Theorem", "states", "no three positive integers satisfy a^n + b^n = c^n for n > 2"),
        ("Riemann hypothesis", "type", "unsolved problem"),
        ("Riemann hypothesis", "about", "distribution of prime numbers"),
        ("linear algebra", "studies", "vectors, matrices, linear transformations"),
        ("matrix multiplication", "time_complexity", "O(n^3) naive, O(n^2.37) Strassen-like"),
        ("eigenvalue", "definition", "scalar λ where Av = λv"),
        ("probability", "range", "0 to 1"),
        ("Bayes' theorem", "formula", "P(A|B) = P(B|A)P(A) / P(B)"),
        ("normal distribution", "known_for", "bell curve"),
        ("normal distribution", "parameters", "mean (μ) and standard deviation (σ)"),
    ])


def _inject_history(ks) -> int:
    """World history."""
    return _store(ks, [
        ("World War I", "started", "1914"),
        ("World War I", "ended", "1918"),
        ("World War I", "trigger", "assassination of Archduke Franz Ferdinand"),
        ("World War II", "started", "1939"),
        ("World War II", "ended", "1945"),
        ("World War II", "allied_powers", "USA, UK, USSR, France, China"),
        ("Cold War", "period", "1947-1991"),
        ("Cold War", "between", "USA and Soviet Union"),
        ("Moon landing", "date", "July 20, 1969"),
        ("Moon landing", "mission", "Apollo 11"),
        ("Moon landing", "first_person", "Neil Armstrong"),
        ("French Revolution", "started", "1789"),
        ("French Revolution", "ended", "1799"),
        ("French Revolution", "caused", "end of absolute monarchy in France"),
        ("Industrial Revolution", "period", "1760-1840"),
        ("Industrial Revolution", "started_in", "Great Britain"),
        ("Renaissance", "period", "14th to 17th century"),
        ("Renaissance", "started_in", "Italy"),
        ("Renaissance", "known_for", "revival of art, science, and classical learning"),
        ("Roman Empire", "founded", "27 BC"),
        ("Roman Empire", "fell", "476 AD (Western)"),
        ("Roman Empire", "capital", "Rome"),
        ("Ancient Greece", "known_for", "democracy, philosophy, Olympics"),
        ("Ancient Egypt", "known_for", "pyramids, pharaohs, hieroglyphics"),
        ("Silk Road", "type", "ancient trade route"),
        ("Silk Road", "connected", "China to the Mediterranean"),
        ("printing press", "inventor", "Johannes Gutenberg"),
        ("printing press", "invented", "around 1440"),
        ("printing press", "impact", "mass production of books, spread of knowledge"),
        ("Magna Carta", "signed", "1215"),
        ("Magna Carta", "significance", "limited the power of the English monarch"),
        ("American Revolution", "period", "1775-1783"),
        ("American Revolution", "result", "independence of United States from Britain"),
        ("Berlin Wall", "built", "1961"),
        ("Berlin Wall", "fell", "November 9, 1989"),
        ("Berlin Wall", "divided", "East and West Berlin"),
    ])


def _inject_tech(ks) -> int:
    """Technology and companies."""
    return _store(ks, [
        ("internet", "predecessor", "ARPANET (1969)"),
        ("internet", "protocol", "TCP/IP"),
        ("World Wide Web", "creator", "Tim Berners-Lee"),
        ("World Wide Web", "created", "1989"),
        ("World Wide Web", "organization", "CERN"),
        ("Google", "founders", "Larry Page and Sergey Brin"),
        ("Google", "founded", "1998"),
        ("Google", "known_for", "search engine"),
        ("Apple", "founders", "Steve Jobs, Steve Wozniak, Ronald Wayne"),
        ("Apple", "founded", "1976"),
        ("Apple", "known_for", "iPhone, Mac, iPad"),
        ("Microsoft", "founders", "Bill Gates and Paul Allen"),
        ("Microsoft", "founded", "1975"),
        ("Microsoft", "known_for", "Windows, Office, Azure"),
        ("Amazon", "founder", "Jeff Bezos"),
        ("Amazon", "founded", "1994"),
        ("Amazon", "known_for", "e-commerce and cloud computing (AWS)"),
        ("Tesla", "CEO", "Elon Musk"),
        ("Tesla", "founded", "2003"),
        ("Tesla", "known_for", "electric vehicles"),
        ("Bitcoin", "creator", "Satoshi Nakamoto"),
        ("Bitcoin", "created", "2009"),
        ("Bitcoin", "type", "cryptocurrency"),
        ("blockchain", "type", "distributed ledger technology"),
        ("blockchain", "property", "immutable, decentralized"),
        ("artificial intelligence", "coined_by", "John McCarthy (1956)"),
        ("artificial intelligence", "type", "field of computer science"),
        ("GPT", "creator", "OpenAI"),
        ("GPT", "type", "large language model"),
        ("transistor", "invented", "1947"),
        ("transistor", "inventors", "Bardeen, Brattain, Shockley"),
        ("transistor", "organization", "Bell Labs"),
        ("Moore's law", "states", "transistor count doubles every two years"),
        ("Moore's law", "author", "Gordon Moore (1965)"),
        ("CRISPR", "type", "gene editing technology"),
        ("CRISPR", "discovered_by", "Jennifer Doudna and Emmanuelle Charpentier"),
        ("quantum computing", "type", "computing paradigm"),
        ("quantum computing", "uses", "qubits instead of classical bits"),
        ("quantum computing", "property", "superposition and entanglement"),
        ("5G", "type", "mobile network technology"),
        ("5G", "speed", "up to 10 Gbps theoretical"),
        ("Git", "creator", "Linus Torvalds"),
        ("Git", "created", "2005"),
        ("Git", "type", "distributed version control system"),
        ("Docker", "first_released", "2013"),
        ("Docker", "type", "containerization platform"),
        ("Kubernetes", "creator", "Google"),
        ("Kubernetes", "type", "container orchestration"),
    ])


def _inject_people(ks) -> int:
    """Famous people."""
    return _store(ks, [
        ("Albert Einstein", "born", "1879"),
        ("Albert Einstein", "died", "1955"),
        ("Albert Einstein", "nationality", "German"),
        ("Albert Einstein", "known_for", "theory of relativity, E=mc²"),
        ("Albert Einstein", "Nobel Prize", "1921 (photoelectric effect)"),
        ("Isaac Newton", "born", "1643"),
        ("Isaac Newton", "died", "1727"),
        ("Isaac Newton", "known_for", "laws of motion and universal gravitation"),
        ("Isaac Newton", "nationality", "English"),
        ("Marie Curie", "born", "1867"),
        ("Marie Curie", "died", "1934"),
        ("Marie Curie", "known_for", "radioactivity, discovered Polonium and Radium"),
        ("Marie Curie", "Nobel Prize", "1903 (Physics) and 1911 (Chemistry)"),
        ("Nikola Tesla", "born", "1856"),
        ("Nikola Tesla", "died", "1943"),
        ("Nikola Tesla", "known_for", "alternating current (AC) electrical system"),
        ("Alan Turing", "born", "1912"),
        ("Alan Turing", "died", "1954"),
        ("Alan Turing", "known_for", "father of computer science, Enigma code-breaking"),
        ("Ada Lovelace", "born", "1815"),
        ("Ada Lovelace", "known_for", "first computer programmer"),
        ("Charles Darwin", "born", "1809"),
        ("Charles Darwin", "known_for", "theory of evolution by natural selection"),
        ("Leonardo da Vinci", "born", "1452"),
        ("Leonardo da Vinci", "known_for", "Mona Lisa, Vitruvian Man, polymath"),
        ("Aristotle", "born", "384 BC"),
        ("Aristotle", "known_for", "logic, philosophy, biology"),
        ("Plato", "born", "428 BC"),
        ("Plato", "known_for", "philosophy, The Republic, Academy"),
        ("Socrates", "born", "470 BC"),
        ("Socrates", "known_for", "Socratic method, Western philosophy"),
        ("Galileo Galilei", "born", "1564"),
        ("Galileo Galilei", "known_for", "father of modern physics, telescope observations"),
        ("Stephen Hawking", "born", "1942"),
        ("Stephen Hawking", "died", "2018"),
        ("Stephen Hawking", "known_for", "Hawking radiation, A Brief History of Time"),
        ("Elon Musk", "born", "1971"),
        ("Elon Musk", "known_for", "SpaceX, Tesla, Neuralink"),
        ("Steve Jobs", "born", "1955"),
        ("Steve Jobs", "died", "2011"),
        ("Steve Jobs", "known_for", "co-founder of Apple, iPhone revolution"),
        ("Linus Torvalds", "born", "1969"),
        ("Linus Torvalds", "known_for", "Linux kernel and Git"),
        ("Tim Berners-Lee", "born", "1955"),
        ("Tim Berners-Lee", "known_for", "inventor of the World Wide Web"),
        ("Grace Hopper", "born", "1906"),
        ("Grace Hopper", "known_for", "COBOL, first compiler, popularized 'debugging'"),
        ("Claude Shannon", "born", "1916"),
        ("Claude Shannon", "known_for", "father of information theory"),
        ("John von Neumann", "born", "1903"),
        ("John von Neumann", "known_for", "von Neumann architecture, game theory"),
    ])


def _inject_philosophy(ks) -> int:
    """Philosophy and concepts."""
    return _store(ks, [
        ("philosophy", "definition", "study of fundamental questions about existence, knowledge, values, and reason"),
        ("epistemology", "definition", "study of knowledge and justified belief"),
        ("ethics", "definition", "study of moral principles"),
        ("logic", "definition", "study of valid reasoning"),
        ("metaphysics", "definition", "study of the nature of reality"),
        ("empiricism", "definition", "knowledge comes from sensory experience"),
        ("rationalism", "definition", "knowledge comes from reason"),
        ("utilitarianism", "definition", "actions are right if they promote happiness"),
        ("utilitarianism", "founders", "Jeremy Bentham and John Stuart Mill"),
        ("existentialism", "definition", "existence precedes essence"),
        ("existentialism", "key_figures", "Kierkegaard, Sartre, Camus"),
        ("Occam's razor", "principle", "simpler explanations are generally better"),
        ("scientific method", "steps", "observation, hypothesis, experiment, analysis, conclusion"),
        ("falsifiability", "author", "Karl Popper"),
        ("falsifiability", "definition", "a theory must be testable and potentially disprovable"),
    ])


def _inject_music(ks) -> int:
    """Music."""
    return _store(ks, [
        ("Beethoven", "born", "1770"),
        ("Beethoven", "died", "1827"),
        ("Beethoven", "known_for", "symphonies, piano sonatas, became deaf"),
        ("Mozart", "born", "1756"),
        ("Mozart", "died", "1791"),
        ("Mozart", "known_for", "prodigy, operas, symphonies"),
        ("Bach", "born", "1685"),
        ("Bach", "died", "1750"),
        ("Bach", "known_for", "counterpoint, The Well-Tempered Clavier"),
        ("The Beatles", "formed", "1960"),
        ("The Beatles", "origin", "Liverpool, England"),
        ("The Beatles", "members", "John Lennon, Paul McCartney, George Harrison, Ringo Starr"),
        ("The Beatles", "known_for", "most influential band in popular music"),
    ])


def _inject_literature(ks) -> int:
    """Literature."""
    return _store(ks, [
        ("Shakespeare", "born", "1564"),
        ("Shakespeare", "died", "1616"),
        ("Shakespeare", "known_for", "Hamlet, Romeo and Juliet, Macbeth"),
        ("Shakespeare", "nationality", "English"),
        ("Hamlet", "author", "William Shakespeare"),
        ("Hamlet", "type", "tragedy"),
        ("1984", "author", "George Orwell"),
        ("1984", "published", "1949"),
        ("1984", "about", "totalitarian surveillance state"),
        ("The Lord of the Rings", "author", "J.R.R. Tolkien"),
        ("The Lord of the Rings", "published", "1954-1955"),
        ("Harry Potter", "author", "J.K. Rowling"),
        ("Harry Potter", "first_published", "1997"),
        ("Don Quixote", "author", "Miguel de Cervantes"),
        ("Don Quixote", "published", "1605"),
        ("Don Quixote", "known_for", "first modern novel"),
        ("The Art of War", "author", "Sun Tzu"),
        ("The Art of War", "about", "military strategy"),
    ])


def _inject_geography_extra(ks) -> int:
    """Extra geography beyond countries."""
    return _store(ks, [
        ("Mount Everest", "height", "8,849 meters"),
        ("Mount Everest", "location", "Nepal/Tibet border"),
        ("Mount Everest", "type", "highest mountain on Earth"),
        ("Mariana Trench", "depth", "10,994 meters"),
        ("Mariana Trench", "location", "Pacific Ocean"),
        ("Mariana Trench", "type", "deepest point on Earth"),
        ("Amazon River", "length", "6,400 km"),
        ("Amazon River", "location", "South America"),
        ("Nile River", "length", "6,650 km"),
        ("Nile River", "location", "Africa"),
        ("Sahara Desert", "area", "9.2 million km²"),
        ("Sahara Desert", "location", "North Africa"),
        ("Pacific Ocean", "area", "165.25 million km²"),
        ("Pacific Ocean", "type", "largest ocean"),
        ("Great Wall of China", "length", "21,196 km"),
        ("Great Wall of China", "type", "fortification"),
        ("Antarctica", "type", "continent"),
        ("Antarctica", "temperature", "lowest recorded: -89.2°C"),
        ("Antarctica", "population", "no permanent residents"),
    ])


def _inject_biology(ks) -> int:
    """Biology."""
    return _store(ks, [
        ("human genome", "base_pairs", "3.2 billion"),
        ("human genome", "genes", "approximately 20,000-25,000"),
        ("human body", "cells", "approximately 37.2 trillion"),
        ("human body", "bones", "206 in adults"),
        ("human brain", "neurons", "approximately 86 billion"),
        ("human brain", "weight", "approximately 1.4 kg"),
        ("heart", "beats_per_day", "about 100,000 times per day"),
        ("blood types", "types", "A, B, AB, O"),
        ("virus", "type", "microscopic infectious agent"),
        ("virus", "property", "replicates only inside living cells"),
        ("bacteria", "type", "single-celled microorganism"),
        ("bacteria", "size", "typically 0.5-5 micrometers"),
        ("antibiotics", "inventor", "Alexander Fleming (penicillin, 1928)"),
        ("vaccine", "inventor", "Edward Jenner (1796, smallpox)"),
        ("insulin", "discovered_by", "Banting and Best (1921)"),
        ("ecosystem", "definition", "community of living organisms and their environment"),
        ("biodiversity", "definition", "variety of life in a particular habitat"),
        ("extinction", "current_rate", "1,000 times higher than natural background rate"),
    ])


def _inject_physics(ks) -> int:
    """Physics."""
    return _store(ks, [
        ("E=mc²", "author", "Albert Einstein"),
        ("E=mc²", "meaning", "energy equals mass times speed of light squared"),
        ("general relativity", "author", "Albert Einstein (1915)"),
        ("general relativity", "about", "gravity as curvature of spacetime"),
        ("special relativity", "author", "Albert Einstein (1905)"),
        ("special relativity", "about", "physics of objects moving at constant speed"),
        ("quantum mechanics", "founders", "Planck, Bohr, Heisenberg, Schrödinger, Dirac"),
        ("quantum mechanics", "about", "behavior of matter at atomic and subatomic scales"),
        ("Heisenberg uncertainty principle", "states", "cannot simultaneously know exact position and momentum"),
        ("Schrödinger's cat", "type", "thought experiment"),
        ("Schrödinger's cat", "about", "superposition in quantum mechanics"),
        ("Standard Model", "type", "particle physics theory"),
        ("Standard Model", "particles", "17 fundamental particles"),
        ("Higgs boson", "discovered", "2012"),
        ("Higgs boson", "discovered_at", "CERN"),
        ("Higgs boson", "known_for", "gives particles mass"),
        ("thermodynamics", "first_law", "energy cannot be created or destroyed"),
        ("thermodynamics", "second_law", "entropy of an isolated system always increases"),
        ("entropy", "definition", "measure of disorder in a system"),
        ("electromagnetic spectrum", "includes", "radio, microwave, infrared, visible, UV, X-ray, gamma"),
        ("Planck constant", "value", "6.626 × 10^-34 J·s"),
        ("Boltzmann constant", "value", "1.381 × 10^-23 J/K"),
    ])


def _inject_chemistry(ks) -> int:
    """Chemistry."""
    return _store(ks, [
        ("hydrogen", "symbol", "H"),
        ("hydrogen", "atomic_number", "1"),
        ("hydrogen", "known_for", "most abundant element in universe"),
        ("helium", "symbol", "He"),
        ("helium", "atomic_number", "2"),
        ("carbon", "symbol", "C"),
        ("carbon", "atomic_number", "6"),
        ("carbon", "known_for", "basis of organic chemistry"),
        ("oxygen", "symbol", "O"),
        ("oxygen", "atomic_number", "8"),
        ("nitrogen", "symbol", "N"),
        ("nitrogen", "atomic_number", "7"),
        ("nitrogen", "known_for", "78% of Earth's atmosphere"),
        ("iron", "symbol", "Fe"),
        ("iron", "atomic_number", "26"),
        ("gold", "symbol", "Au"),
        ("gold", "atomic_number", "79"),
        ("silicon", "symbol", "Si"),
        ("silicon", "atomic_number", "14"),
        ("silicon", "known_for", "basis of semiconductor industry"),
        ("uranium", "symbol", "U"),
        ("uranium", "atomic_number", "92"),
        ("uranium", "known_for", "nuclear energy and weapons"),
        ("pH scale", "range", "0 (acidic) to 14 (basic)"),
        ("pH scale", "neutral", "7"),
        ("chemical bond", "types", "ionic, covalent, metallic, hydrogen"),
        ("Avogadro's number", "value", "6.022 × 10^23"),
        ("mole", "definition", "amount of substance containing 6.022 × 10^23 entities"),
    ])


def _inject_economics(ks) -> int:
    """Economics and finance."""
    return _store(ks, [
        ("GDP", "stands_for", "Gross Domestic Product"),
        ("GDP", "definition", "total value of goods and services produced in a country"),
        ("inflation", "definition", "general increase in prices and fall in purchasing power"),
        ("supply and demand", "definition", "economic model of price determination"),
        ("stock market", "definition", "marketplace for buying and selling shares of companies"),
        ("Federal Reserve", "type", "central bank of the United States"),
        ("Federal Reserve", "founded", "1913"),
        ("Bitcoin", "type", "cryptocurrency"),
        ("Bitcoin", "creator", "Satoshi Nakamoto"),
        ("Bitcoin", "created", "2009"),
        ("Adam Smith", "known_for", "The Wealth of Nations"),
        ("Adam Smith", "occupation", "economist"),
        ("Adam Smith", "born", "1723"),
        ("capitalism", "definition", "economic system based on private ownership and free markets"),
        ("socialism", "definition", "economic system based on social ownership of means of production"),
        ("recession", "definition", "period of economic decline, typically two consecutive quarters of falling GDP"),
        ("interest rate", "definition", "cost of borrowing money, expressed as a percentage"),
        ("currency", "types", "fiat currency, commodity money, cryptocurrency"),
        ("European Central Bank", "location", "Frankfurt"),
        ("World Bank", "founded", "1944"),
        ("IMF", "stands_for", "International Monetary Fund"),
    ])


def _inject_space(ks) -> int:
    """Space and astronomy."""
    return _store(ks, [
        ("Sun", "type", "star"),
        ("Sun", "spectral_class", "G2V"),
        ("Sun", "age", "4.6 billion years"),
        ("Sun", "distance_from_earth", "149.6 million km"),
        ("Moon", "type", "natural satellite"),
        ("Moon", "distance_from_earth", "384,400 km"),
        ("Moon", "first_landing", "Apollo 11 (1969)"),
        ("Moon landing", "date", "July 20, 1969"),
        ("Moon landing", "astronauts", "Neil Armstrong and Buzz Aldrin"),
        ("Moon landing", "spacecraft", "Apollo 11"),
        ("Jupiter", "type", "gas giant planet"),
        ("Jupiter", "known_for", "largest planet in the solar system"),
        ("Jupiter", "moons", "95 known moons including Ganymede, Europa, Io, Callisto"),
        ("Saturn", "type", "gas giant planet"),
        ("Saturn", "known_for", "prominent ring system"),
        ("Venus", "type", "terrestrial planet"),
        ("Venus", "known_as", "morning star and evening star"),
        ("Mercury", "type", "terrestrial planet"),
        ("Mercury", "known_for", "closest planet to the Sun"),
        ("Neptune", "type", "ice giant planet"),
        ("Uranus", "type", "ice giant planet"),
        ("Pluto", "type", "dwarf planet"),
        ("Pluto", "reclassified", "2006 by IAU"),
        ("black hole", "definition", "region of spacetime where gravity is so strong nothing can escape"),
        ("black hole", "first_image", "2019 by Event Horizon Telescope"),
        ("Milky Way", "type", "spiral galaxy"),
        ("Milky Way", "stars", "100-400 billion"),
        ("International Space Station", "type", "space station"),
        ("International Space Station", "launched", "1998"),
        ("International Space Station", "orbit", "408 km above Earth"),
        ("Hubble Space Telescope", "launched", "1990"),
        ("James Webb Space Telescope", "launched", "2021"),
        ("light year", "definition", "distance light travels in one year"),
        ("light year", "value", "9.461 trillion km"),
        ("Big Bang", "type", "cosmological theory"),
        ("Big Bang", "age", "13.8 billion years ago"),
        ("neutron star", "definition", "collapsed core of a massive star after supernova"),
        ("supernova", "definition", "explosive death of a massive star"),
        ("dark matter", "definition", "hypothetical matter that does not emit light"),
        ("dark matter", "percentage", "about 27% of the universe"),
        ("dark energy", "definition", "hypothetical energy causing accelerating expansion of universe"),
        ("dark energy", "percentage", "about 68% of the universe"),
    ])


def _inject_animals(ks) -> int:
    """Animals and zoology."""
    return _store(ks, [
        ("blue whale", "type", "marine mammal"),
        ("blue whale", "known_for", "largest animal ever lived"),
        ("blue whale", "length", "up to 30 meters"),
        ("cheetah", "type", "big cat"),
        ("cheetah", "known_for", "fastest land animal"),
        ("cheetah", "speed", "up to 120 km/h"),
        ("elephant", "type", "mammal"),
        ("elephant", "known_for", "largest land animal"),
        ("dolphin", "type", "marine mammal"),
        ("dolphin", "known_for", "high intelligence"),
        ("octopus", "type", "cephalopod"),
        ("octopus", "known_for", "eight arms and high intelligence"),
        ("eagle", "type", "bird of prey"),
        ("penguin", "type", "flightless bird"),
        ("penguin", "habitat", "Southern Hemisphere, mainly Antarctica"),
        ("dinosaur", "extinction", "66 million years ago"),
        ("dinosaur", "cause_of_extinction", "asteroid impact (Chicxulub)"),
        ("T-Rex", "type", "theropod dinosaur"),
        ("T-Rex", "lived", "68-66 million years ago"),
        ("honey bee", "known_for", "pollination and honey production"),
        ("ant", "known_for", "colony organization and teamwork"),
    ])


def _inject_sports(ks) -> int:
    """Sports."""
    return _store(ks, [
        ("FIFA World Cup", "type", "international football tournament"),
        ("FIFA World Cup", "frequency", "every 4 years"),
        ("Olympic Games", "type", "international multi-sport event"),
        ("Olympic Games", "origin", "ancient Greece, 776 BC"),
        ("Olympic Games", "modern_revival", "Athens 1896"),
        ("football", "known_as", "soccer in the US"),
        ("football", "players_per_team", "11"),
        ("basketball", "inventor", "James Naismith"),
        ("basketball", "invented", "1891"),
        ("tennis", "Grand Slams", "Australian Open, French Open, Wimbledon, US Open"),
        ("cricket", "origin", "England, 16th century"),
        ("baseball", "known_as", "America's pastime"),
        ("Formula 1", "type", "motorsport racing"),
        ("chess", "type", "strategy board game"),
        ("chess", "origin", "India, 6th century"),
        ("marathon", "distance", "42.195 km (26.2 miles)"),
        ("marathon", "origin", "Battle of Marathon, 490 BC"),
    ])


def _inject_food(ks) -> int:
    """Food and cuisine."""
    return _store(ks, [
        ("pizza", "origin", "Naples, Italy"),
        ("sushi", "origin", "Japan"),
        ("chocolate", "origin", "Mesoamerica"),
        ("chocolate", "made_from", "cacao beans"),
        ("coffee", "origin", "Ethiopia"),
        ("coffee", "active_ingredient", "caffeine"),
        ("tea", "origin", "China"),
        ("tea", "types", "green, black, white, oolong, herbal"),
        ("bread", "ingredients", "flour, water, yeast, salt"),
        ("rice", "known_for", "staple food for over half the world's population"),
        ("pasta", "origin", "Italy"),
        ("wine", "made_from", "fermented grapes"),
        ("beer", "ingredients", "water, malt, hops, yeast"),
        ("vitamin C", "found_in", "citrus fruits, peppers, broccoli"),
        ("vitamin C", "function", "supports immune system and collagen production"),
    ])


def _inject_art(ks) -> int:
    """Art and architecture."""
    return _store(ks, [
        ("Mona Lisa", "artist", "Leonardo da Vinci"),
        ("Mona Lisa", "location", "Louvre Museum, Paris"),
        ("Mona Lisa", "created", "1503-1519"),
        ("Starry Night", "artist", "Vincent van Gogh"),
        ("Starry Night", "created", "1889"),
        ("The Scream", "artist", "Edvard Munch"),
        ("David", "artist", "Michelangelo"),
        ("David", "created", "1501-1504"),
        ("Sistine Chapel ceiling", "artist", "Michelangelo"),
        ("Guernica", "artist", "Pablo Picasso"),
        ("Guernica", "about", "bombing of Guernica during Spanish Civil War"),
        ("Renaissance", "period", "14th to 17th century"),
        ("Renaissance", "origin", "Florence, Italy"),
        ("Impressionism", "period", "1860s-1880s"),
        ("Impressionism", "founders", "Monet, Renoir, Degas"),
        ("Eiffel Tower", "location", "Paris, France"),
        ("Eiffel Tower", "built", "1889"),
        ("Eiffel Tower", "height", "330 meters"),
        ("Great Wall of China", "length", "21,196 km"),
        ("Great Wall of China", "purpose", "defense against invasions"),
        ("Taj Mahal", "location", "Agra, India"),
        ("Taj Mahal", "built_by", "Mughal Emperor Shah Jahan"),
        ("Colosseum", "location", "Rome, Italy"),
        ("Colosseum", "built", "70-80 AD"),
        ("Pyramids of Giza", "location", "Egypt"),
        ("Pyramids of Giza", "built", "around 2560 BC"),
        ("Pyramids of Giza", "known_for", "one of the Seven Wonders of the Ancient World"),
    ])


def _inject_religion(ks) -> int:
    """World religions."""
    return _store(ks, [
        ("Christianity", "type", "monotheistic religion"),
        ("Christianity", "followers", "2.4 billion"),
        ("Christianity", "holy_book", "Bible"),
        ("Islam", "type", "monotheistic religion"),
        ("Islam", "followers", "1.9 billion"),
        ("Islam", "holy_book", "Quran"),
        ("Islam", "founder", "Prophet Muhammad"),
        ("Hinduism", "type", "polytheistic religion"),
        ("Hinduism", "followers", "1.2 billion"),
        ("Hinduism", "origin", "Indian subcontinent"),
        ("Buddhism", "type", "nontheistic religion"),
        ("Buddhism", "founder", "Siddhartha Gautama (Buddha)"),
        ("Buddhism", "origin", "India, 5th century BC"),
        ("Judaism", "type", "monotheistic religion"),
        ("Judaism", "holy_book", "Torah"),
    ])


def _inject_earth_science(ks) -> int:
    """Earth science and natural phenomena."""
    return _store(ks, [
        ("earthquake", "cause", "movement of tectonic plates"),
        ("earthquake", "measured_by", "Richter scale and moment magnitude scale"),
        ("earthquake", "type", "natural disaster"),
        ("volcano", "cause", "magma rising through Earth's crust"),
        ("volcano", "type", "geological formation"),
        ("tsunami", "cause", "underwater earthquake or volcanic eruption"),
        ("tsunami", "type", "natural disaster"),
        ("hurricane", "cause", "warm ocean water evaporating and forming storm systems"),
        ("hurricane", "known_as", "typhoon in the Pacific, cyclone in the Indian Ocean"),
        ("tornado", "definition", "violently rotating column of air"),
        ("climate change", "cause", "increased greenhouse gas emissions"),
        ("climate change", "effects", "rising temperatures, sea level rise, extreme weather"),
        ("greenhouse effect", "definition", "trapping of heat by atmospheric gases"),
        ("ozone layer", "function", "absorbs UV radiation from the Sun"),
        ("ozone layer", "threat", "CFCs and other ozone-depleting substances"),
        ("tectonic plates", "definition", "large segments of Earth's lithosphere that move"),
        ("tectonic plates", "number", "about 15 major plates"),
        ("Earth", "age", "4.54 billion years"),
        ("Earth", "layers", "crust, mantle, outer core, inner core"),
        ("Earth", "diameter", "12,742 km"),
        ("Earth", "distance_from_sun", "149.6 million km"),
        ("ocean", "percentage", "71% of Earth's surface"),
        ("ocean", "deepest_point", "Mariana Trench (10,994 m)"),
        ("atmosphere", "composition", "78% nitrogen, 21% oxygen, 1% other"),
        ("water cycle", "stages", "evaporation, condensation, precipitation, collection"),
        ("penguin", "can_fly", "no, penguins cannot fly"),
        ("penguin", "adaptation", "wings evolved into flippers for swimming"),
        # Geography
        ("Mount Everest", "type", "mountain"),
        ("Mount Everest", "height", "8,849 meters (29,032 feet)"),
        ("Mount Everest", "location", "Nepal-China border, Himalayas"),
        ("Mount Everest", "known_as", "tallest mountain on Earth"),
        ("Nile River", "type", "river"),
        ("Nile River", "length", "6,650 km (longest river in Africa)"),
        ("Amazon River", "type", "river"),
        ("Amazon River", "known_for", "largest river by water volume"),
        ("Sahara Desert", "type", "desert"),
        ("Sahara Desert", "location", "North Africa"),
        ("Sahara Desert", "known_for", "largest hot desert in the world"),
    ])


def _inject_medicine(ks) -> int:
    """Medicine and health."""
    return _store(ks, [
        ("penicillin", "discovered_by", "Alexander Fleming"),
        ("penicillin", "discovered", "1928"),
        ("penicillin", "type", "antibiotic"),
        ("vaccine", "inventor", "Edward Jenner"),
        ("vaccine", "first_used", "1796 (smallpox)"),
        ("vaccine", "function", "stimulates immune system to prevent disease"),
        ("DNA", "structure", "double helix"),
        ("heart", "function", "pump blood through the body"),
        ("brain", "function", "controls all body functions and consciousness"),
        ("brain", "neurons", "approximately 86 billion"),
        ("blood", "types", "A, B, AB, O"),
        ("blood", "function", "transports oxygen, nutrients, and waste"),
        ("cancer", "definition", "uncontrolled cell growth"),
        ("cancer", "treatments", "surgery, chemotherapy, radiation, immunotherapy"),
        ("diabetes", "type", "metabolic disease"),
        ("diabetes", "types", "Type 1 (autoimmune) and Type 2 (insulin resistance)"),
        ("antibiotic", "function", "kills or inhibits bacteria"),
        ("virus", "definition", "microscopic infectious agent that replicates inside cells"),
        ("bacteria", "definition", "single-celled microorganism"),
        ("aspirin", "discovered", "1897 by Felix Hoffmann at Bayer"),
        ("aspirin", "function", "pain relief and anti-inflammatory"),
        ("X-ray", "discovered_by", "Wilhelm Roentgen"),
        ("X-ray", "discovered", "1895"),
        ("stethoscope", "inventor", "Rene Laennec"),
        ("stethoscope", "invented", "1816"),
        ("Hippocrates", "known_as", "Father of Medicine"),
        ("Hippocrates", "origin", "ancient Greece"),
    ])


def _inject_law(ks) -> int:
    """Law and governance."""
    return _store(ks, [
        ("constitution", "definition", "fundamental law of a state or organization"),
        ("United States Constitution", "adopted", "1787"),
        ("United States Constitution", "amendments", "27 amendments including Bill of Rights"),
        ("Magna Carta", "signed", "1215"),
        ("Magna Carta", "significance", "limited the power of the English monarch"),
        ("human rights", "document", "Universal Declaration of Human Rights"),
        ("human rights", "adopted", "1948 by the United Nations"),
        ("United Nations", "founded", "1945"),
        ("United Nations", "headquarters", "New York City"),
        ("United Nations", "members", "193 member states"),
        ("democracy", "definition", "government by the people"),
        ("democracy", "origin", "Athens, ancient Greece, 5th century BC"),
        ("communism", "founder", "Karl Marx and Friedrich Engels"),
        ("communism", "key_work", "The Communist Manifesto (1848)"),
        ("habeas corpus", "definition", "right to challenge unlawful detention"),
        ("patent", "definition", "exclusive right to an invention for a limited time"),
        ("copyright", "definition", "legal right to control reproduction of creative work"),
        ("Geneva Convention", "purpose", "protection of war victims"),
        ("Geneva Convention", "adopted", "1949"),
        ("International Court of Justice", "location", "The Hague, Netherlands"),
        ("NATO", "founded", "1949"),
        ("NATO", "purpose", "collective defense alliance"),
        ("European Union", "founded", "1993 (Maastricht Treaty)"),
        ("European Union", "members", "27 member states"),
    ])


def _inject_languages(ks) -> int:
    """World languages."""
    return _store(ks, [
        ("Mandarin Chinese", "speakers", "over 1.1 billion native speakers"),
        ("Mandarin Chinese", "type", "tonal language"),
        ("Mandarin Chinese", "writing", "Chinese characters (Hanzi)"),
        ("English", "speakers", "about 1.5 billion total speakers"),
        ("English", "type", "Germanic language"),
        ("English", "origin", "Anglo-Saxon, influenced by Latin and French"),
        ("Spanish", "speakers", "about 560 million total speakers"),
        ("Spanish", "type", "Romance language"),
        ("Arabic", "speakers", "about 420 million native speakers"),
        ("Arabic", "type", "Semitic language"),
        ("Arabic", "writing", "right-to-left script"),
        ("Hindi", "speakers", "about 600 million total speakers"),
        ("Hindi", "writing", "Devanagari script"),
        ("French", "speakers", "about 320 million total speakers"),
        ("French", "type", "Romance language"),
        ("German", "speakers", "about 130 million total speakers"),
        ("German", "type", "Germanic language"),
        ("Japanese", "writing", "three scripts: Hiragana, Katakana, Kanji"),
        ("Korean", "writing", "Hangul alphabet, invented 1443"),
        ("Latin", "type", "dead language (no native speakers)"),
        ("Latin", "significance", "ancestor of Romance languages"),
        ("Esperanto", "creator", "L. L. Zamenhof"),
        ("Esperanto", "created", "1887"),
        ("Esperanto", "type", "constructed international language"),
        ("sign language", "type", "visual-manual language"),
        ("Braille", "inventor", "Louis Braille"),
        ("Braille", "invented", "1824"),
        ("Braille", "type", "tactile writing system for the blind"),
    ])


def _inject_tech_companies(ks) -> int:
    """Technology companies."""
    return _store(ks, [
        ("Apple", "founder", "Steve Jobs, Steve Wozniak, Ronald Wayne"),
        ("Apple", "founded", "1976"),
        ("Apple", "headquarters", "Cupertino, California"),
        ("Apple", "known_for", "iPhone, Mac, iPad"),
        ("Google", "founder", "Larry Page and Sergey Brin"),
        ("Google", "founded", "1998"),
        ("Google", "headquarters", "Mountain View, California"),
        ("Google", "known_for", "search engine and Android"),
        ("Microsoft", "founder", "Bill Gates and Paul Allen"),
        ("Microsoft", "founded", "1975"),
        ("Microsoft", "headquarters", "Redmond, Washington"),
        ("Microsoft", "known_for", "Windows and Office"),
        ("Amazon", "founder", "Jeff Bezos"),
        ("Amazon", "founded", "1994"),
        ("Amazon", "known_for", "e-commerce and cloud computing (AWS)"),
        ("Tesla", "founder", "Elon Musk, Martin Eberhard, Marc Tarpenning"),
        ("Tesla", "founded", "2003"),
        ("Tesla", "known_for", "electric vehicles and autonomous driving"),
        ("SpaceX", "founder", "Elon Musk"),
        ("SpaceX", "founded", "2002"),
        ("SpaceX", "known_for", "reusable rockets and space exploration"),
        ("Meta", "founder", "Mark Zuckerberg"),
        ("Meta", "founded", "2004 (as Facebook)"),
        ("Meta", "known_for", "Facebook, Instagram, WhatsApp"),
        ("Netflix", "founder", "Reed Hastings and Marc Randolph"),
        ("Netflix", "founded", "1997"),
        ("Netflix", "known_for", "streaming video and original content"),
    ])


def _inject_historical_events(ks) -> int:
    """Major historical events."""
    return _store(ks, [
        ("World War I", "started", "1914"),
        ("World War I", "ended", "1918"),
        ("World War I", "cause", "assassination of Archduke Franz Ferdinand"),
        ("World War II", "started", "1939"),
        ("World War II", "ended", "1945"),
        ("World War II", "cause", "Nazi Germany's invasion of Poland"),
        ("Cold War", "period", "1947-1991"),
        ("Cold War", "between", "United States and Soviet Union"),
        ("French Revolution", "started", "1789"),
        ("French Revolution", "significance", "overthrow of monarchy, birth of modern democracy"),
        ("American Revolution", "period", "1775-1783"),
        ("American Revolution", "result", "independence of United States from Britain"),
        ("Industrial Revolution", "period", "1760-1840"),
        ("Industrial Revolution", "origin", "Britain"),
        ("Industrial Revolution", "significance", "transition from hand production to machine manufacturing"),
        ("Moon landing", "date", "July 20, 1969"),
        ("Moon landing", "astronaut", "Neil Armstrong"),
        ("Moon landing", "mission", "Apollo 11"),
        ("Berlin Wall", "built", "1961"),
        ("Berlin Wall", "fell", "November 9, 1989"),
        ("Berlin Wall", "significance", "symbol of Cold War division"),
        ("Renaissance", "period", "14th-17th century"),
        ("Roman Empire", "period", "27 BC - 476 AD"),
        ("Roman Empire", "capital", "Rome"),
        ("Roman Empire", "known_for", "law, engineering, and governance"),
        ("Black Death", "period", "1347-1351"),
        ("Black Death", "killed", "approximately 75-200 million people"),
        ("printing press", "inventor", "Johannes Gutenberg"),
        ("printing press", "invented", "around 1440"),
        ("printing press", "significance", "revolutionized communication and knowledge sharing"),
    ])
