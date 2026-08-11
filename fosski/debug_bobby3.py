import re

s2 = "Brian has 20 video games but lost 5 right before the comparison was made"
ref_name = "Brian"

mod_m = re.search(
    re.escape(ref_name) + r'.*?(?:but\s+)?(?:lost|lose|gave\s+away|spent|used|broke)\s+(\d+)',
    s2)
print(f"mod_m: {mod_m}")
if mod_m:
    print(f"  mod_val: {mod_m.group(1)}")

# Also check the question
q = "If Brian has 20 video games but lost 5 right before the comparison was made, how many does Bobby have?"
mod_m2 = re.search(
    re.escape(ref_name) + r'.*?(?:but\s+)?(?:lost|lose|gave\s+away|spent|used|broke)\s+(\d+)',
    q)
print(f"mod_m2 from question: {mod_m2}")
