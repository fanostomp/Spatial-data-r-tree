"""Command-line interfaces for building and querying the R-tree."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from .bulk_load import build_rtree, read_points
from .queries import execute_query_file
from .storage import load_rtree, write_rtree


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build an R-tree using STR bulk loading."
    )
    parser.add_argument("input_file", help="Point-data input file")
    parser.add_argument("output_file", help="Destination rtree.csv file")
    return parser


def query_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Execute spatial queries on an R-tree."
    )
    parser.add_argument("query_type", choices=("window", "distance", "knn"))
    parser.add_argument("tree_file", help="Path to rtree.csv")
    parser.add_argument("query_file", help="Path to the query file")
    parser.add_argument(
        "k",
        type=int,
        nargs="?",
        default=10,
        help="Number of neighbours for kNN queries (default: 10)",
    )
    return parser


def build_main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    points = read_points(args.input_file)
    tree = build_rtree(points)
    write_rtree(tree, args.output_file)

    for statistics in tree.level_statistics:
        print(
            f"{statistics.node_count} nodes at level {statistics.level} "
            f"with average MBR area {statistics.average_mbr_area}"
        )
    return 0


def query_main(argv: Sequence[str] | None = None) -> int:
    args = query_parser().parse_args(argv)
    tree = load_rtree(args.tree_file)
    for output_line in execute_query_file(
        tree,
        args.query_type,
        args.query_file,
        k=args.k,
    ):
        print(output_line)
    return 0
