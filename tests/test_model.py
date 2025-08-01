from package_name import my_model

data = {"series": [1, 2, 3, 4, 5], "index": 1}


def test_my_model():
    assert my_model(data) == 15
