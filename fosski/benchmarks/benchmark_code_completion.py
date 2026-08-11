#!/usr/bin/env python3
"""
Code Completion Benchmark — Domain-matched test for tech-trained FLM.

100 code/tech completion tasks. Each: context + 4 endings, 1 correct.
Tests whether FLM learned programming/tech language patterns.

Usage:
    python3 benchmarks/benchmark_code_completion.py [--flm PATH]
"""

import sys
import os
import json
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# 100 code/tech completion tasks: (context, [4 endings], correct_index)
TASKS = [
    # Python basics
    ("def sort_list(arr):\n    return", [
        " sorted(arr)", " arr.bake()", " the cat sat on", " import printer"
    ], 0),
    ("import os\npath = os.path.", [
        "join('/tmp', 'file.txt')", "swim(upstream)", "cook(dinner)", "sing(song)"
    ], 0),
    ("for i in range(10):\n    print(", [
        "i)", "weather)", "mountain)", "banana)"
    ], 0),
    ("try:\n    result = func()\nexcept", [
        " Exception as e:", " the rain:", " a sandwich:", " my feelings:"
    ], 0),
    ("with open('data.txt', 'r') as f:\n    content = f.", [
        "read()", "fly()", "dance()", "dream()"
    ], 0),
    ("class Database:\n    def __init__(self):\n        self.", [
        "connection = None", "favorite_color = 'blue'", "breakfast = 'eggs'", "pet_name = 'fluffy'"
    ], 0),
    ("def fibonacci(n):\n    if n <= 1:\n        return", [
        " n", " 'hello world'", " os.remove('/')", " print('goodbye')"
    ], 0),
    ("import json\ndata = json.", [
        "loads(response.text)", "paint(canvas)", "drive(car)", "plant(tree)"
    ], 0),
    ("# Calculate the average\ntotal = sum(numbers)\navg =", [
        " total / len(numbers)", " total * weather", " total + 'hello'", " total.sing()"
    ], 0),
    ("if __name__ == '__main__':\n    ", [
        "main()", "go_swimming()", "bake_cake()", "pet_the_cat()"
    ], 0),

    # Error handling / debugging
    ("The error 'KeyError' means the", [
        " key was not found in the dictionary", " keyboard is broken", " car key is lost", " music was off key"
    ], 0),
    ("A segmentation fault occurs when a program tries to access", [
        " memory it doesn't have permission to access", " a restaurant after closing", " the internet without wifi", " a locked car"
    ], 0),
    ("To debug a Python program, you can use", [
        " pdb or breakpoint()", " a magnifying glass", " a crystal ball", " a hammer"
    ], 0),
    ("A stack overflow happens when", [
        " recursive calls exceed the stack size", " you stack too many books", " the river overflows", " the stack of pancakes falls"
    ], 0),
    ("The HTTP status code 404 means", [
        " the resource was not found", " the weather is nice", " the server is on fire", " the pizza is ready"
    ], 0),

    # Git / version control
    ("To create a new branch in git, run", [
        " git checkout -b feature", " git bake new-cake", " git fly away", " git dance --now"
    ], 0),
    ("After staging changes, commit them with", [
        " git commit -m 'description'", " git send --express", " git cook --recipe", " git paint --brush"
    ], 0),
    ("To see the diff of unstaged changes, use", [
        " git diff", " git look", " git taste", " git smell"
    ], 0),
    ("A merge conflict happens when two branches", [
        " modify the same lines in a file", " grow different flowers", " serve different food", " play different music"
    ], 0),
    ("To undo the last commit but keep changes staged, use", [
        " git reset --soft HEAD~1", " git forget --please", " git undo --magic", " git reverse --time"
    ], 0),

    # Data structures
    ("A hash table provides O(1) average time for", [
        " lookup and insertion", " cooking and eating", " running and jumping", " singing and dancing"
    ], 0),
    ("In a linked list, each node contains data and a", [
        " pointer to the next node", " recipe for cookies", " map of the city", " picture of a cat"
    ], 0),
    ("A binary search tree has the property that left children are", [
        " less than the parent", " taller than the parent", " older than the parent", " funnier than the parent"
    ], 0),
    ("A queue follows the FIFO principle, meaning", [
        " first in, first out", " fish in, fish out", " fun in, fun out", " fast in, fast out"
    ], 0),
    ("The time complexity of binary search is", [
        " O(log n)", " O(n!)", " O(infinity)", " O(never)"
    ], 0),

    # Networking
    ("TCP ensures reliable delivery by using", [
        " acknowledgments and retransmission", " prayer and hope", " loud shouting", " carrier pigeons"
    ], 0),
    ("DNS resolves domain names to", [
        " IP addresses", " phone numbers", " home addresses", " email passwords"
    ], 0),
    ("A firewall filters network traffic based on", [
        " rules and policies", " the weather", " personal preference", " random chance"
    ], 0),
    ("SSH provides secure remote access by", [
        " encrypting the connection", " locking the door", " hiding the computer", " whispering quietly"
    ], 0),
    ("Port 443 is the default port for", [
        " HTTPS", " pizza delivery", " radio stations", " telephone calls"
    ], 0),

    # Linux / system admin
    ("To list all running processes, use", [
        " ps aux", " ls friends", " cat pets", " rm everything"
    ], 0),
    ("The chmod command changes", [
        " file permissions", " the weather", " your mood", " the channel"
    ], 0),
    ("To find files by name in Linux, use", [
        " find / -name 'pattern'", " ask / -politely 'pattern'", " search / -google 'pattern'", " look / -carefully 'pattern'"
    ], 0),
    ("Cron jobs are used to", [
        " schedule recurring tasks", " order food delivery", " plan vacations", " make phone calls"
    ], 0),
    ("The /etc/passwd file contains", [
        " user account information", " all passwords in plain text", " restaurant menus", " song lyrics"
    ], 0),

    # Python specific
    ("A list comprehension in Python looks like", [
        " [x for x in items if condition]", " {please give me x from items}", " (x wants to be in items)", " <x dreams of items>"
    ], 0),
    ("The __init__ method in Python is", [
        " the constructor called when creating an instance", " the destructor called when deleting", " a magic spell", " a recipe for init-soup"
    ], 0),
    ("To install a Python package, use", [
        " pip install package-name", " yell install package-name", " pray install package-name", " wish install package-name"
    ], 0),
    ("A decorator in Python is applied with the", [
        " @ symbol above a function", " # symbol inside a function", " $ symbol below a function", " & symbol next to a function"
    ], 0),
    ("The GIL in Python stands for", [
        " Global Interpreter Lock", " Great Internet Library", " General Input Layer", " Good Idea License"
    ], 0),

    # Algorithms
    ("Quicksort works by selecting a pivot and", [
        " partitioning elements around it", " cooking elements in a pot", " painting elements on canvas", " singing elements a song"
    ], 0),
    ("Dynamic programming solves problems by", [
        " breaking them into overlapping subproblems", " using dynamite", " being dynamic and energetic", " programming dynamically"
    ], 0),
    ("BFS explores a graph level by level using a", [
        " queue", " telescope", " fishing rod", " magic wand"
    ], 0),
    ("The greedy algorithm makes the", [
        " locally optimal choice at each step", " most expensive choice always", " random choice every time", " worst possible choice"
    ], 0),
    ("Dijkstra's algorithm finds the", [
        " shortest path in a weighted graph", " tallest tree in a forest", " best restaurant in town", " fastest car on the road"
    ], 0),

    # Databases
    ("A SQL JOIN combines rows from two or more", [
        " tables based on a related column", " restaurants based on cuisine", " songs based on rhythm", " books based on color"
    ], 0),
    ("An index in a database speeds up", [
        " query performance", " the internet connection", " the computer's fan", " the user's typing"
    ], 0),
    ("ACID in databases stands for Atomicity, Consistency,", [
        " Isolation, and Durability", " Ice cream, and Donuts", " Intelligence, and Design", " Innovation, and Discovery"
    ], 0),
    ("A primary key uniquely identifies each", [
        " row in a table", " person in a city", " star in the sky", " word in a book"
    ], 0),
    ("NoSQL databases are designed for", [
        " flexible schemas and horizontal scaling", " storing music playlists", " organizing recipe collections", " managing personal diaries"
    ], 0),

    # Security
    ("SQL injection exploits applications that", [
        " don't sanitize user input in queries", " use too many tables", " have slow connections", " store data alphabetically"
    ], 0),
    ("A buffer overflow occurs when data exceeds the", [
        " allocated buffer size", " speed limit", " weight limit", " noise level"
    ], 0),
    ("Public key cryptography uses", [
        " a key pair: public for encryption, private for decryption", " a single master key for everything", " no keys at all", " physical door keys"
    ], 0),
    ("Cross-site scripting (XSS) allows attackers to", [
        " inject malicious scripts into web pages", " cross the street faster", " write better scripts", " improve site performance"
    ], 0),
    ("A hash function maps data to a fixed-size", [
        " output that cannot be reversed", " meal that tastes the same", " room that never changes", " song that always plays"
    ], 0),

    # Machine Learning / AI
    ("Overfitting occurs when a model", [
        " memorizes training data but fails on new data", " exercises too much", " eats too much data", " thinks too hard"
    ], 0),
    ("A neural network consists of layers of", [
        " interconnected nodes with weights", " interconnected pipes with water", " interconnected roads with cars", " interconnected wires with electricity"
    ], 0),
    ("Gradient descent optimizes by moving in the direction of", [
        " steepest decrease of the loss function", " the nearest restaurant", " the loudest sound", " the brightest light"
    ], 0),
    ("The learning rate controls", [
        " how large each optimization step is", " how fast students learn", " the speed of the internet", " the brightness of the screen"
    ], 0),
    ("Cross-validation helps assess model", [
        " generalization to unseen data", " fashion sense", " cooking ability", " athletic performance"
    ], 0),

    # Docker / DevOps
    ("A Docker container packages an application with its", [
        " dependencies and runtime environment", " lunch and snacks", " books and magazines", " tools and furniture"
    ], 0),
    ("CI/CD stands for Continuous Integration and", [
        " Continuous Deployment", " Continuous Dancing", " Continuous Dining", " Continuous Dreaming"
    ], 0),
    ("Kubernetes orchestrates", [
        " container deployment and scaling", " symphony concerts", " traffic lights", " wedding ceremonies"
    ], 0),
    ("A load balancer distributes incoming traffic across", [
        " multiple servers", " multiple restaurants", " multiple highways", " multiple classrooms"
    ], 0),
    ("Infrastructure as Code means managing infrastructure through", [
        " version-controlled configuration files", " verbal instructions to the team", " handwritten notes on paper", " telepathic communication"
    ], 0),

    # API / Web
    ("REST APIs use HTTP methods like GET, POST, PUT, and", [
        " DELETE", " SLEEP", " DREAM", " DANCE"
    ], 0),
    ("JSON stands for JavaScript Object", [
        " Notation", " Navigation", " Nutrition", " Negotiation"
    ], 0),
    ("An API rate limit restricts the number of", [
        " requests a client can make in a time period", " meals a person can eat", " songs a radio can play", " steps a person can walk"
    ], 0),
    ("WebSocket enables", [
        " full-duplex communication between client and server", " physical socket installation", " electrical socket management", " socket wrench operations"
    ], 0),
    ("CORS stands for Cross-Origin Resource", [
        " Sharing", " Shopping", " Swimming", " Sleeping"
    ], 0),

    # General tech knowledge
    ("Moore's Law predicts that transistor density", [
        " doubles approximately every two years", " stays the same forever", " decreases over time", " depends on the weather"
    ], 0),
    ("UTF-8 is a character encoding that supports", [
        " virtually all writing systems worldwide", " only English letters", " only numbers", " only emojis"
    ], 0),
    ("A compiler translates source code into", [
        " machine code or intermediate representation", " human language", " musical notes", " visual art"
    ], 0),
    ("Garbage collection automatically", [
        " reclaims memory no longer in use", " cleans the office", " deletes user files", " empties the recycle bin"
    ], 0),
    ("Big-O notation describes the", [
        " upper bound of an algorithm's time complexity", " size of the letter O", " biggest ocean", " largest number"
    ], 0),

    # Code patterns
    ("The singleton pattern ensures a class has", [
        " only one instance", " only one method", " no instances", " infinite instances"
    ], 0),
    ("In MVC architecture, the controller handles", [
        " user input and updates the model", " painting the view", " storing the database", " serving the food"
    ], 0),
    ("A callback function is passed as an argument and", [
        " executed later when an event occurs", " never executed", " immediately deleted", " printed to screen"
    ], 0),
    ("Dependency injection reduces coupling by", [
        " providing dependencies from outside", " injecting medicine", " reducing the budget", " removing all code"
    ], 0),
    ("The observer pattern notifies subscribers when", [
        " the subject's state changes", " the newspaper arrives", " the sun rises", " dinner is ready"
    ], 0),

    # Crypto / Math (David's domain)
    ("AES encryption uses a block size of", [
        " 128 bits", " 3 bytes", " 1 megabyte", " 42 gigabytes"
    ], 0),
    ("A Markov chain models transitions between", [
        " states with defined probabilities", " countries with defined borders", " songs with defined rhythms", " meals with defined recipes"
    ], 0),
    ("The eigenvalues of a matrix describe its", [
        " scaling factors along eigenvectors", " favorite colors", " cooking temperatures", " musical preferences"
    ], 0),
    ("RSA security is based on the difficulty of", [
        " factoring large prime numbers", " climbing large mountains", " eating large meals", " reading large books"
    ], 0),
    ("A Bayesian prior represents", [
        " belief before observing data", " the first person in line", " yesterday's weather", " an ancient civilization"
    ], 0),

    # System design
    ("Horizontal scaling adds more", [
        " machines to handle load", " floors to the building", " pages to the book", " ingredients to the recipe"
    ], 0),
    ("A message queue decouples", [
        " producers from consumers", " trains from tracks", " birds from nests", " fish from water"
    ], 0),
    ("Consistent hashing minimizes key redistribution when", [
        " nodes are added or removed", " locks are changed", " roads are repaired", " weather changes"
    ], 0),
    ("A CDN improves performance by serving content from", [
        " geographically closer servers", " louder speakers", " bigger screens", " faster keyboards"
    ], 0),
    ("Microservices architecture splits an application into", [
        " small independent services", " tiny physical pieces", " microscopic organisms", " miniature paintings"
    ], 0),

    # Testing
    ("Unit tests verify that individual", [
        " functions work correctly in isolation", " apartments are clean", " soldiers are healthy", " students pass exams"
    ], 0),
    ("Test-driven development writes tests", [
        " before the implementation code", " after deployment to production", " only on weekends", " never"
    ], 0),
    ("A mock object simulates", [
        " the behavior of a real dependency", " a fake smile", " artificial intelligence", " virtual reality"
    ], 0),
    ("Code coverage measures the percentage of", [
        " code executed during tests", " floor covered by carpet", " sky covered by clouds", " bread covered by butter"
    ], 0),
    ("Integration tests verify that", [
        " components work together correctly", " immigrants integrate well", " math integrals are correct", " ingredients integrate in recipes"
    ], 0),

    # Misc tech
    ("A regex pattern \\d+ matches one or more", [
        " digits", " dogs", " doors", " drums"
    ], 0),
    ("In functional programming, pure functions have no", [
        " side effects", " friends", " purpose", " color"
    ], 0),
    ("Idempotent operations produce the same result when", [
        " applied multiple times", " the moon is full", " you clap twice", " the wind blows"
    ], 0),
    ("A deadlock occurs when two processes each", [
        " wait for a resource held by the other", " cook the same meal", " sing the same song", " read the same book"
    ], 0),
    ("Race conditions happen when the outcome depends on", [
        " the timing of concurrent operations", " which horse runs fastest", " the weather forecast", " random number generation"
    ], 0),
]


def evaluate_without_flm(tasks):
    """Baseline: random choice (25% expected)."""
    import random
    random.seed(42)
    correct = sum(1 for _, endings, label in tasks if random.randint(0, 3) == label)
    return correct, len(tasks)


def evaluate_with_flm(flm, tasks):
    """Score each ending by FLM perplexity. Lower = more natural = chosen."""
    correct = 0
    total = 0
    for ctx, endings, label in tasks:
        scores = []
        for ending in endings:
            text = ctx + ending
            ppl = flm.perplexity(list(text))
            scores.append(-ppl)  # lower perplexity = better
        pred = max(range(len(scores)), key=lambda i: scores[i])
        if pred == label:
            correct += 1
        total += 1
    return correct, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--flm', help='Path to FLM pickle', default=None)
    args = parser.parse_args()

    print("=" * 65)
    print("  CODE COMPLETION BENCHMARK — 100 Tech Tasks")
    print("=" * 65)

    # Random baseline
    rc, rt = evaluate_without_flm(TASKS)
    print(f"\n  Random baseline: {rc}/{rt} ({rc*100/rt:.1f}%)")

    if args.flm:
        import pickle
        print(f"\n  Loading FLM from {args.flm}...")
        t0 = time.time()
        with open(args.flm, 'rb') as f:
            flm = pickle.load(f)
        print(f"  Loaded in {time.time()-t0:.1f}s")

        t0 = time.time()
        fc, ft = evaluate_with_flm(flm, TASKS)
        elapsed = time.time() - t0
        pct = fc * 100 / ft

        print(f"\n  FLM score: {fc}/{ft} ({pct:.1f}%)")
        print(f"  Time: {elapsed:.1f}s")
        print(f"\n  {'Metric':<35} {'Score':>10}")
        print(f"  {'─' * 47}")
        print(f"  {'Random baseline':<35} {'25.0%':>10}")
        print(f"  {'FLM ({args.flm})':<35} {f'{pct:.1f}%':>10}")
        print(f"  {'Delta vs random':<35} {f'+{pct-25:.1f}%':>10}")
    else:
        print("\n  No FLM provided. Use --flm PATH to test.")

    print(f"\n{'=' * 65}")


if __name__ == '__main__':
    main()
