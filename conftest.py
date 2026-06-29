"""Pytest configuration.

Adds the project root to sys.path so `agent`, `utils`, and `config` import
cleanly when running tests.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
