"""
This module contains the main model for the package.

✨ For usage examples, see /tests/test_model.py.
"""

from embeddings_similarity_rating.types import DataClean


def my_model(data: DataClean) -> int:
    """
    This function is a placeholder for the main model.
    """
    return sum(data["series"])
