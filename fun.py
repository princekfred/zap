"""Compatibility helpers / shared imports.

This repo originally used `fun.py` as a place to keep common imports. Make the
imports optional so importing this module doesn't crash when optional
dependencies aren't installed.
"""

import os

try:
    import pennylane as qml
    from pennylane import numpy as np
except ModuleNotFoundError:
    qml = None
    np = None
