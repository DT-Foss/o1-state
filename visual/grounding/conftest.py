"""
Path shim for running this package's test suite from a parent repo root
(e.g. `python3 -m pytest visual/grounding -q` invoked from O1_juli/) without
an editable/site-packages install of `grounding_kernel`.

In the source repo (grounding_kernel_v0/, run from its own directory) pytest
implicitly puts the invocation directory on sys.path via rootdir-relative
conftest discovery, so `import grounding_kernel` resolves. Nested one level
deeper under visual/grounding/ inside a DIFFERENT repo, that implicit path
insertion no longer lands on THIS directory -- this file exists solely to
put visual/grounding/ (this file's own directory, containing the
grounding_kernel/ package) back on sys.path before test collection, same
effect as an editable pip install would have had, with zero changes to any
module under grounding_kernel/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
