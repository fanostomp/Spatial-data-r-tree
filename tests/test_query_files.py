from pathlib import Path

from spatial_rtree.bulk_load import build_rtree
from spatial_rtree.models import LeafEntry
from spatial_rtree.queries import execute_query_file, parse_query_line


def test_parse_query_line_accepts_brackets_and_commas() -> None:
    assert parse_query_line("[1, 2, 3]") == [1.0, 2.0, 3.0]


def test_execute_query_file_uses_assignment_output_format(tmp_path: Path) -> None:
    tree = build_rtree(
        [LeafEntry(1, (0.0, 0.0)), LeafEntry(2, (5.0, 5.0))],
        leaf_max=2,
        leaf_min=1,
        internal_max=2,
        internal_min=1,
    )
    queries = tmp_path / "queries.txt"
    queries.write_text("[0 0 1 1]\n", encoding="utf-8")

    assert execute_query_file(tree, "window", queries) == ["0 (1): 1"]
