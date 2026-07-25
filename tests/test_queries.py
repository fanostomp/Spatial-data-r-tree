import math

import pytest

from spatial_rtree.bulk_load import build_rtree
from spatial_rtree.models import LeafEntry
from spatial_rtree.queries import (
    distance_range_query,
    intersects_window,
    knn_query,
    minimum_distance,
    point_in_window,
    window_query,
)


def sample_tree():
    points = [
        LeafEntry(1, (0.0, 0.0)),
        LeafEntry(2, (1.0, 1.0)),
        LeafEntry(3, (2.0, 2.0)),
        LeafEntry(4, (5.0, 5.0)),
        LeafEntry(5, (-1.0, -1.0)),
    ]
    return build_rtree(
        points,
        leaf_max=2,
        leaf_min=1,
        internal_max=2,
        internal_min=1,
    )


def test_spatial_predicates_include_boundaries() -> None:
    assert point_in_window((2.0, 2.0), (0.0, 0.0, 2.0, 2.0))
    assert intersects_window((0.0, 0.0, 1.0, 1.0), (1.0, 1.0, 2.0, 2.0))
    assert minimum_distance((3.0, 1.0), (0.0, 0.0, 2.0, 2.0)) == 1.0


def test_window_query_returns_points_inside_window() -> None:
    assert set(window_query(sample_tree(), (0.0, 0.0, 2.0, 2.0))) == {1, 2, 3}


def test_distance_query_uses_inclusive_radius() -> None:
    results = distance_range_query(sample_tree(), (0.0, 0.0), math.sqrt(2))
    assert set(results) == {1, 2, 5}


def test_knn_query_uses_best_first_order() -> None:
    results = knn_query(sample_tree(), (0.0, 0.0), 3)
    assert results[0] == 1
    assert set(results[1:]) == {2, 5}


def test_invalid_query_arguments_are_rejected() -> None:
    tree = sample_tree()
    with pytest.raises(ValueError, match="non-negative"):
        distance_range_query(tree, (0.0, 0.0), -1.0)
    with pytest.raises(ValueError, match="positive"):
        knn_query(tree, (0.0, 0.0), 0)
