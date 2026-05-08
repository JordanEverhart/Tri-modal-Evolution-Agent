"""Minimal ms-swift interface reference used by this repository.

This module intentionally mirrors only the public interfaces touched by
Train/. Use the official ModelScope ms-swift package for real training.
"""

from .rewards import AsyncORM, ORM, orms

__all__ = ["AsyncORM", "ORM", "orms"]
__version__ = "reference-only"
