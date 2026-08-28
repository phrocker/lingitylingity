"""Deterministic governed text analysis."""

from lingity.analyzer import analyze_text
from lingity.invariants import compare_protected
from lingity.profiles import Profile, load_profile

__all__ = ["Profile", "analyze_text", "compare_protected", "load_profile"]
__version__ = "0.1.0"
