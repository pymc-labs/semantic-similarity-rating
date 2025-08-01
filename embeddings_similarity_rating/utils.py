"""
Various utility functions, for use outside of the `embeddings_similarity_rating` module.

Do not add any functions here that gets imported into the `embeddings_similarity_rating` module.
If you need to add logic that gets imported into the `embeddings_similarity_rating` module,
add it to another file in this directory where it logically belongs. If none such file
exists, create a new file in this directory.

These are are utilities that can be used in runner scripts, tests, etc.
"""

from embeddings_similarity_rating.types import DataRaw, DataClean


def data_cleaner(data: DataRaw) -> DataClean:
    """Clean the data."""
    return {
        "series": [int(value) for value in data["series"]],
        "index": int(data["index"]),
    }
