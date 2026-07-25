from pathlib import Path

import pytest

from spatial_rtree.bulk_load import build_rtree
from spatial_rtree.models import LeafEntry
from spatial_rtree.storage import load_rtree, write_rtree


def test_tree_round_trip_preserves_nodes_and_root(tmp_path: Path) -> None:
    points = [LeafEntry(index, (float(index), float(index))) for index in range(1, 9)]
    original = build_rtree(
        points,
        leaf_max=3,
        leaf_min=2,
        internal_max=2,
        internal_min=1,
    )
    output = tmp_path / "rtree.csv"

    write_rtree(original, output)
    loaded = load_rtree(output)

    assert loaded.root_id == original.root_id
    assert set(loaded.nodes) == set(original.nodes)
    assert loaded.nodes[loaded.root_id].mbr == original.nodes[original.root_id].mbr


def test_loader_rejects_incorrect_declared_entry_count(tmp_path: Path) -> None:
    output = tmp_path / "invalid.csv"
    output.write_text("0 , 2 , 0 , (1,(0.0, 0.0))\n", encoding="utf-8")

    with pytest.raises(ValueError, match="declares 2 entries"):
        load_rtree(output)


def test_loader_rejects_missing_child_reference(tmp_path: Path) -> None:
    output = tmp_path / "invalid.csv"
    output.write_text(
        "0 , 1 , 1 , (99, [0.0, 0.0, 1.0, 1.0])\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing child 99"):
        load_rtree(output)
