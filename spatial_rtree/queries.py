"""Window, distance-range, and k-nearest-neighbour queries for an R-tree."""

from __future__ import annotations

import heapq
import math
from pathlib import Path
from typing import Literal

from .models import InternalEntry, LeafEntry, MBR, Point, RTree

QueryType = Literal["window", "distance", "knn"]


def validate_mbr(mbr: MBR) -> None:
    if mbr[0] > mbr[2] or mbr[1] > mbr[3]:
        raise ValueError("Invalid MBR: lower bounds must not exceed upper bounds.")


def point_in_window(point: Point, window: MBR) -> bool:
    """Return True when a point is inside or on the boundary of a window."""

    validate_mbr(window)
    return (
        window[0] <= point[0] <= window[2]
        and window[1] <= point[1] <= window[3]
    )


def intersects_window(mbr: MBR, window: MBR) -> bool:
    """Return True when two closed rectangles overlap or touch."""

    validate_mbr(mbr)
    validate_mbr(window)
    return not (
        mbr[2] < window[0]
        or mbr[0] > window[2]
        or mbr[3] < window[1]
        or mbr[1] > window[3]
    )


def euclidean_distance(first: Point, second: Point) -> float:
    """Calculate the Euclidean distance between two points."""

    return math.hypot(first[0] - second[0], first[1] - second[1])


def minimum_distance(point: Point, mbr: MBR) -> float:
    """Calculate the minimum Euclidean distance from a point to an MBR."""

    validate_mbr(mbr)
    dx = max(mbr[0] - point[0], 0.0, point[0] - mbr[2])
    dy = max(mbr[1] - point[1], 0.0, point[1] - mbr[3])
    return math.hypot(dx, dy)


def window_query(tree: RTree, window: MBR) -> list[int]:
    """Return record IDs whose points are inside the supplied window."""

    validate_mbr(window)
    results: list[int] = []

    def visit(node_id: int) -> None:
        node = tree.nodes[node_id]
        if node.is_leaf:
            for entry in node.entries:
                if not isinstance(entry, LeafEntry):
                    raise TypeError("Leaf node contains an internal entry.")
                if point_in_window(entry.point, window):
                    results.append(entry.record_id)
            return

        for entry in node.entries:
            if not isinstance(entry, InternalEntry):
                raise TypeError("Internal node contains a leaf entry.")
            if intersects_window(entry.mbr, window):
                visit(entry.child_id)

    visit(tree.root_id)
    return results


def distance_range_query(
    tree: RTree, query_point: Point, epsilon: float
) -> list[int]:
    """Return record IDs at Euclidean distance at most epsilon from a point."""

    if epsilon < 0:
        raise ValueError("epsilon must be non-negative.")

    results: list[int] = []

    def visit(node_id: int) -> None:
        node = tree.nodes[node_id]
        if node.is_leaf:
            for entry in node.entries:
                if not isinstance(entry, LeafEntry):
                    raise TypeError("Leaf node contains an internal entry.")
                if euclidean_distance(query_point, entry.point) <= epsilon:
                    results.append(entry.record_id)
            return

        for entry in node.entries:
            if not isinstance(entry, InternalEntry):
                raise TypeError("Internal node contains a leaf entry.")
            if minimum_distance(query_point, entry.mbr) <= epsilon:
                visit(entry.child_id)

    visit(tree.root_id)
    return results


def knn_query(tree: RTree, query_point: Point, k: int) -> list[int]:
    """Return the k nearest record IDs using best-first search."""

    if k <= 0:
        raise ValueError("k must be positive.")

    # Heap item: (distance, kind, identifier). Points use kind 0 so ties are
    # resolved deterministically before nodes with the same lower-bound distance.
    queue: list[tuple[float, int, int]] = [(0.0, 1, tree.root_id)]
    results: list[int] = []

    while queue and len(results) < k:
        _, kind, identifier = heapq.heappop(queue)

        if kind == 0:
            results.append(identifier)
            continue

        node = tree.nodes[identifier]
        if node.is_leaf:
            for entry in node.entries:
                if not isinstance(entry, LeafEntry):
                    raise TypeError("Leaf node contains an internal entry.")
                distance = euclidean_distance(query_point, entry.point)
                heapq.heappush(queue, (distance, 0, entry.record_id))
        else:
            for entry in node.entries:
                if not isinstance(entry, InternalEntry):
                    raise TypeError("Internal node contains a leaf entry.")
                distance = minimum_distance(query_point, entry.mbr)
                heapq.heappush(queue, (distance, 1, entry.child_id))

    return results


def parse_query_line(line: str) -> list[float]:
    """Parse one bracketed or plain whitespace-separated query line."""

    clean_line = line.strip().strip("[]").replace(",", " ")
    if not clean_line:
        raise ValueError("Query line is empty.")

    try:
        return [float(value) for value in clean_line.split()]
    except ValueError as exc:
        raise ValueError(f"Query line contains a non-numeric value: {line!r}") from exc


def read_query_file(path: str | Path) -> list[tuple[int, list[float]]]:
    """Read queries while preserving their zero-based physical line numbers."""

    queries: list[tuple[int, list[float]]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle):
            if line.strip():
                queries.append((line_number, parse_query_line(line)))
    return queries


def execute_query_file(
    tree: RTree,
    query_type: QueryType,
    query_path: str | Path,
    *,
    k: int = 10,
) -> list[str]:
    """Execute all queries from a file and return assignment-compatible output lines."""

    output: list[str] = []
    for line_number, parameters in read_query_file(query_path):
        if query_type == "window":
            if len(parameters) != 4:
                raise ValueError("Window queries require four values.")
            window: MBR = tuple(parameters)  # type: ignore[assignment]
            results = window_query(tree, window)
            output.append(
                f"{line_number} ({len(results)}): {','.join(map(str, results))}"
            )
        elif query_type == "distance":
            if len(parameters) != 3:
                raise ValueError("Distance queries require x, y, and epsilon.")
            results = distance_range_query(
                tree, (parameters[0], parameters[1]), parameters[2]
            )
            output.append(
                f"{line_number} ({len(results)}): {','.join(map(str, results))}"
            )
        elif query_type == "knn":
            if len(parameters) != 2:
                raise ValueError("kNN queries require x and y.")
            results = knn_query(tree, (parameters[0], parameters[1]), k)
            output.append(f"{line_number}: {','.join(map(str, results))}")
        else:
            raise ValueError(f"Unsupported query type: {query_type}")

    return output
