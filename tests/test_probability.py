# Tests for probability.py

from data_ravers_utils.probability import *

def test_add_subset():
    ods = OneDimensionalSet("ball")
    ods.add_subset("red", 5)
    ods.add_subset("blue", 3)
    assert ods.subsets == {"red": 5, "blue": 3}

def test_get_subsets():
    ods = OneDimensionalSet("ball")
    ods.add_subset("red", 5)
    ods.add_subset("blue", 3)
    assert ods.get_subsets() == {"red": 5, "blue": 3}

def test_get_total_count():
    ods = OneDimensionalSet("ball")
    ods.add_subset("red", 5)
    ods.add_subset("blue", 3)
    assert ods.get_total_count() == 8

def test_get_item_count():
    ods = OneDimensionalSet("ball")
    ods.add_subset("red", 5)
    ods.add_subset("blue", 3)
    assert ods.get_item_count() == 2

def test_prob_all_unique():
    ods = OneDimensionalSet("ball")
    ods.add_subset("red", 5)
    ods.add_subset("blue", 3)
    ods.add_subset("green", 2)
    result = ods.prob_all_unique()
    expected = (5 * 3 * 2) / (10 * 9 * 8)  # (n_red * n_blue * n_green) / (total * (total-1) * (total-2))
    assert math.isclose(result, expected, rel_tol=1e-9)