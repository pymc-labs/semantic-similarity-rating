from package_name.utils import data_cleaner

data = {"series": ["1", "2", "3", "4", "5"], "index": "1"}


def test_data_cleaner():
    assert data_cleaner(data) == {"series": [1, 2, 3, 4, 5], "index": 1}
