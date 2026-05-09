"""
core/__init__.py

Bootstrap: insert the bundled fastQuot at the front of sys.path so that
all subsequent ``import quot`` calls resolve to the bundled copy rather
than any system-installed quot package.

fastQuot is bundled inside the repository:
    quot_saspt_workflow/
        fastQuot/
            quot/           ← the quot package lives here
        core/               ← this file lives here
"""

import os
import sys

_CORE_DIR = os.path.dirname(os.path.abspath(__file__))          # .../core/
_REPO_ROOT = os.path.dirname(_CORE_DIR)                          # .../quot_saspt_workflow/
_BUNDLED_FASTQUOT = os.path.join(_REPO_ROOT, "fastQuot")

# Prepend so the bundled copy always wins over any installed quot
if _BUNDLED_FASTQUOT not in sys.path:
    sys.path.insert(0, _BUNDLED_FASTQUOT)
