"""Top-level module for embeddings_similarity_rating."""

from beartype.claw import beartype_this_package

from .model import my_model

__all__ = ["my_model"]

beartype_this_package()
