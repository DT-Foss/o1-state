import sys, re
sys.path.insert(0, '.')

# Directly test the matching logic
context_sents = ['Brian has 20 video games but lost 5 right before the comparison was made',
                 "Brian's friend Bobby has 5 fewer than 3 times as many video games as Brian does."]

for sent in context_sents:
    m = re.search(
        r'(\d+\.?\d*)\s+\w+\s+(less|more|fewer|greater)\s+than\s+'
        r'(\d+\.?\d*)\s+times\s+(?:what\s+|as\s+(?:many|much)\s+(?:as\s+)?)?'
        r'(?:\w+\s+)?(\b[A-Z][a-z]+)',
        sent)
    if m:
        print(f"Pattern match in: {sent[:80]}")
        diff_val = float(m.group(1))
        direction = m.group(2)
        mult = float(m.group(3))
        ref_name = m.group(4)
        print(f"  diff={diff_val}, dir={direction}, mult={mult}, ref={ref_name}")

        for s2 in context_sents:
            m2 = re.search(
                re.escape(ref_name) + r'\s+(?:has|had|have|is|was|weighs?|contains?|holds?|costs?|gets?)\s+'
                r'(?:a\s+total\s+of\s+|exactly\s+|about\s+|only\s+)?'
                r'\$?([\d,.]+)', s2)
            if m2:
                ref_val = float(m2.group(1).replace(',', ''))
                print(f"  Found ref_val={ref_val} in: {s2[:80]}")

                mod_m = re.search(
                    re.escape(ref_name) + r'.*?(?:but\s+)?(?:lost|lose|gave\s+away|spent|used|broke)\s+(\d+)',
                    s2)
                print(f"  mod_m in s2: {mod_m}")
                if mod_m:
                    print(f"    mod_val={mod_m.group(1)}")
