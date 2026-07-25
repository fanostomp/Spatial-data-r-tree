"""R-tree construction using STR leaf packing and bottom-up internal packing."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import TypeVar

from .constants import (
    INTERNAL_MAX_ENTRIES,
    INTERNAL_MIN_ENTRIES,
    LEAF_MAX_ENTRIES,
    LEAF_MIN_ENTRIES,
)
from .models import InternalEntry, LeafEntry, LevelStatistics, MBR, Node, RTree

T = TypeVar("T")


def read_points(path: str | Path) -> list[LeafEntry]:
    """Read point data where the first line declares the number of records.

    Each following non-empty line must contain exactly two floating-point values.
    The first point has record ID 1, matching its data-line position.
    """

    input_path = Path(path)
    with input_path.open("r", encoding="utf-8") as handle:
        lines = handle.readlines()

    if not lines:
        raise ValueError(f"Input file is empty: {input_path}")

    try:
        declared_count = int(lines[0].strip())
    except ValueError as exc:
        raise ValueError("The first line must contain the number of points.") from exc

    points: list[LeafEntry] = []
    for record_id, raw_line in enumerate(lines[1:], start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue

        parts = stripped.split()
        if len(parts) != 2:
            raise ValueError(
                f"Malformed point at data line {record_id}: expected two values."
            )

        try:
            x, y = (float(value) for value in parts)
        except ValueError as exc:
            raise ValueError(
                "Malformed point at data line "
                f"{record_id}: coordinates must be numeric."
            ) from exc

        points.append(LeafEntry(record_id=record_id, point=(x, y)))

    if not points:
        raise ValueError("No valid points were found in the input file.")

    if declared_count != len(points):
        raise ValueError(
            f"Declared point count is {declared_count}, "
            f"but {len(points)} points were read."
        )

    return points


def chunk_with_underflow(
    elements: Sequence[T], max_capacity: int, min_capacity: int
) -> list[list[T]]:
    """Split entries into nodes while repairing underflow in the final node."""

    if max_capacity <= 0:
        raise ValueError("max_capacity must be positive.")
    if min_capacity <= 0 or min_capacity > max_capacity:
        raise ValueError("min_capacity must be in the range 1..max_capacity.")
    if not elements:
        return []

    chunks = [
        list(elements[index : index + max_capacity])
        for index in range(0, len(elements), max_capacity)
    ]

    if len(chunks) > 1 and len(chunks[-1]) < min_capacity:
        needed = min_capacity - len(chunks[-1])
        donor_size_after_transfer = len(chunks[-2]) - needed
        if donor_size_after_transfer < min_capacity:
            raise ValueError(
                "The supplied capacities cannot satisfy the minimum fill requirement."
            )

        borrowed = chunks[-2][-needed:]
        chunks[-2] = chunks[-2][:-needed]
        chunks[-1] = borrowed + chunks[-1]

    return chunks


def calculate_leaf_mbr(entries: Sequence[LeafEntry]) -> MBR:
    """Return the minimum bounding rectangle containing all leaf points."""

    if not entries:
        raise ValueError("Cannot calculate an MBR for an empty leaf node.")

    xs = [entry.point[0] for entry in entries]
    ys = [entry.point[1] for entry in entries]
    return min(xs), min(ys), max(xs), max(ys)


def calculate_internal_mbr(entries: Sequence[InternalEntry]) -> MBR:
    """Return the minimum bounding rectangle containing all child MBRs."""

    if not entries:
        raise ValueError("Cannot calculate an MBR for an empty internal node.")

    return (
        min(entry.mbr[0] for entry in entries),
        min(entry.mbr[1] for entry in entries),
        max(entry.mbr[2] for entry in entries),
        max(entry.mbr[3] for entry in entries),
    )


def mbr_area(mbr: MBR) -> float:
    """Calculate the area of an MBR."""

    return (mbr[2] - mbr[0]) * (mbr[3] - mbr[1])


def pack_leaf_entries_str(
    entries: Sequence[LeafEntry],
    max_capacity: int = LEAF_MAX_ENTRIES,
    min_capacity: int = LEAF_MIN_ENTRIES,
) -> list[list[LeafEntry]]:
    """Pack point entries into leaf groups using Sort-Tile-Recursive."""

    if not entries:
        return []

    sorted_by_x = sorted(entries, key=lambda entry: entry.point[0])
    required_nodes = math.ceil(len(sorted_by_x) / max_capacity)
    slice_count = math.ceil(math.sqrt(required_nodes))
    slice_capacity = slice_count * max_capacity

    groups: list[list[LeafEntry]] = []
    for index in range(0, len(sorted_by_x), slice_capacity):
        current_slice = sorted(
            sorted_by_x[index : index + slice_capacity],
            key=lambda entry: entry.point[1],
        )
        groups.extend(
            chunk_with_underflow(current_slice, max_capacity, min_capacity)
        )

    # The final STR slice can contain a single underfilled group. In that case,
    # repair the last group using its predecessor from the previous slice.
    if len(groups) > 1 and len(groups[-1]) < min_capacity:
        needed = min_capacity - len(groups[-1])
        if len(groups[-2]) - needed < min_capacity:
            raise ValueError(
                "STR packing cannot satisfy the minimum fill requirement."
            )
        borrowed = groups[-2][-needed:]
        groups[-2] = groups[-2][:-needed]
        groups[-1] = borrowed + groups[-1]

    return groups


def build_rtree(
    points: Sequence[LeafEntry],
    *,
    leaf_max: int = LEAF_MAX_ENTRIES,
    leaf_min: int = LEAF_MIN_ENTRIES,
    internal_max: int = INTERNAL_MAX_ENTRIES,
    internal_min: int = INTERNAL_MIN_ENTRIES,
) -> RTree:
    """Build an R-tree using STR for leaves and ordered bottom-up packing above them."""

    if not points:
        raise ValueError("Cannot build an R-tree without points.")

    nodes: dict[int, Node] = {}
    level_statistics: list[LevelStatistics] = []
    next_node_id = 0

    leaf_groups = pack_leaf_entries_str(points, leaf_max, leaf_min)
    current_level: list[InternalEntry] = []

    for group in leaf_groups:
        mbr = calculate_leaf_mbr(group)
        node = Node(
            node_id=next_node_id,
            is_leaf=True,
            entries=list(group),
            mbr=mbr,
        )
        nodes[next_node_id] = node
        current_level.append(InternalEntry(child_id=next_node_id, mbr=mbr))
        next_node_id += 1

    # The assignment reports zero average area for the point-storage level.
    level_statistics.append(
        LevelStatistics(level=0, node_count=len(leaf_groups), average_mbr_area=0.0)
    )

    level = 1
    while len(current_level) > 1:
        groups = chunk_with_underflow(current_level, internal_max, internal_min)
        next_level: list[InternalEntry] = []
        total_area = 0.0

        for group in groups:
            mbr = calculate_internal_mbr(group)
            node = Node(
                node_id=next_node_id,
                is_leaf=False,
                entries=list(group),
                mbr=mbr,
            )
            nodes[next_node_id] = node
            next_level.append(InternalEntry(child_id=next_node_id, mbr=mbr))
            total_area += mbr_area(mbr)
            next_node_id += 1

        level_statistics.append(
            LevelStatistics(
                level=level,
                node_count=len(groups),
                average_mbr_area=total_area / len(groups),
            )
        )
        current_level = next_level
        level += 1

    root_id = current_level[0].child_id
    return RTree(
        nodes=nodes,
        root_id=root_id,
        level_statistics=level_statistics,
    )
