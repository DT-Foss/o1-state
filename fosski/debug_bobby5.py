import re

sent = "Brian's friend Bobby has 5 fewer than 3 times as many video games as Brian does."

# The pattern from equation_graph
m = re.search(
    r'(\d+\.?\d*)\s+\w+\s+(less|more|fewer|greater)\s+than\s+'
    r'(\d+\.?\d*)\s+times\s+(?:what\s+|as\s+(?:many|much)\s+(?:as\s+)?)?'
    r'(?:\w+\s+)?(\b[A-Z][a-z]+)',
    sent)
print(f"Pattern match: {m}")
if m:
    print(f"  groups: {m.groups()}")

# The text is "5 fewer than 3 times as many video games as Brian"
# Let me check step by step
# (\d+\.?\d*) → 5
# \s+\w+\s+ → " fewer " (hmm, "fewer" is the next word)
# Wait: "5 fewer than" → (\d+)\s+\w+\s+(less|more|fewer|greater)\s+than
# 5 → digits OK
# \s+\w+\s+ → this expects a word between the number and less/more/fewer
# But "5 fewer" has "fewer" directly after 5, no word in between

# Let me check: "5 fewer than 3 times"
# regex: (\d+)\s+\w+\s+(less|more|fewer|greater)\s+than
# This requires: number SPACE word SPACE (less|more|fewer) SPACE than
# "5 fewer than" → \d+ = 5, then \s+ = space, then \w+ = "fewer"
# then \s+ = space, then (less|more|fewer|greater) = ???
# "fewer" was already consumed by \w+!
# So "than" would need to match (less|more|fewer|greater) which it doesn't

# The pattern expects: "5 pounds fewer than 3 times" (a unit word between number and direction)
# But the text is: "5 fewer than 3 times" (no unit word)

# Let me try a fixed pattern
m2 = re.search(
    r'(\d+\.?\d*)\s+(?:\w+\s+)?(less|more|fewer|greater)\s+than\s+'
    r'(\d+\.?\d*)\s+times\s+(?:what\s+|as\s+(?:many|much)\s+(?:as\s+)?)?'
    r'(?:\w+\s+)?(\b[A-Z][a-z]+)',
    sent)
print(f"Fixed pattern: {m2}")
if m2:
    print(f"  groups: {m2.groups()}")
