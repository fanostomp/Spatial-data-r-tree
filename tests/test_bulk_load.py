from pathlib import Path

import pytest

from spatial_rtree.bulk_load import (
    build_rtree,
    chunk_with_underflow,
    pack_leaf_entries_str,
    read_points,
)
from spatial_rtree.models import LeafEntry


def make_points(count: int) -> list[LeafEntry]:
    return [
        LeafEntry(index, (float(index), float(index % 3)))
        for index in range(1, count + 1)
    ]


def test_read_points_preserves_data_line_as_record_id(tmp_path: Path) -> None:
    source = tmp_path / "points.txt"
    source.write_text("3\n1.5 2.5\n3 4\n5 6\n", encoding="utf-8")

    points = read_points(source)

    assert points == [
        LeafEntry(1, (1.5, 2.5)),
        LeafEntry(2, (3.0, 4.0)),
        LeafEntry(3, (5.0, 6.0)),
    ]


def test_read_points_rejects_declared_count_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "points.txt"
    source.write_text("2\n1 2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Declared point count"):
        read_points(source)


def test_chunk_repairs_last_node_underflow() -> None:
    chunks = chunk_with_underflow(list(range(11)), max_capacity=6, min_capacity=4)

    assert [len(chunk) for chunk in chunks] == [6, 5]


def test_str_packing_does_not_mutate_input() -> None:
    points = make_points(10)
    original = list(points)

    groups = pack_leaf_entries_str(points, max_capacity=4, min_capacity=2)

    assert points == original
    assert sum(map(len, groups)) == 10
    assert all(2 <= len(group) <= 4 for group in groups)


def test_str_repairs_underflow_in_a_single_final_slice() -> None:
    groups = pack_leaf_entries_str(make_points(9), max_capacity=4, min_capacity=2)

    assert [len(group) for group in groups] == [4, 3, 2]


def test_build_rtree_creates_leaf_root_for_small_dataset() -> None:
    tree = build_rtree(
        make_points(3),
        leaf_max=4,
        leaf_min=2,
        internal_max=3,
        internal_min=1,
    )

    assert tree.root_id == 0
    assert tree.nodes[tree.root_id].is_leaf
    assert tree.level_statistics[0].node_count == 1
