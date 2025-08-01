"""
Embeddings-Similarity Rating (ESR) Package

A package for converting LLM textual responses to Likert scale probability distributions
using semantic similarity against reference statements.

This package implements the ESR methodology described in the paper:
"Measuring Synthetic Consumer Purchase Intent Using Embeddings-Similarity Ratings"
"""

from beartype.claw import beartype_this_package

from .compute import response_embeddings_to_pmf, scale_pmf
from .response_rater import ResponseRater

__version__ = "1.0.0"
__author__ = "Ben F. Maier, Ulf Aslak"

__all__ = [
    "ResponseRater",
    "response_embeddings_to_pmf",
    "scale_pmf",
]

beartype_this_package()
