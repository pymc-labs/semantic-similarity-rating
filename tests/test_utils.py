from embeddings_similarity_rating.utils import data_cleaner

data = {"series": ["1", "2", "3", "4", "5"], "index": "1"}


def test_data_cleaner():
    assert data_cleaner(data) == {"series": [1, 2, 3, 4, 5], "index": 1}
