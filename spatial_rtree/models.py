"""Core data models used by the R-tree implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

Point: TypeAlias = tuple[float, float]
MBR: TypeAlias = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class LeafEntry:
    """A point stored in a leaf node."""

    record_id: int
    point: Point


@dataclass(frozen=True, slots=True)
class InternalEntry:
    """A reference from an internal node to a child node."""

    child_id: int
    mbr: MBR


NodeEntry: TypeAlias = LeafEntry | InternalEntry


@dataclass(slots=True)
class Node:
    """A leaf or internal R-tree node."""

    node_id: int
    is_leaf: bool
    entries: list[NodeEntry]
    mbr: MBR


@dataclass(frozen=True, slots=True)
class LevelStatistics:
    """Statistics produced for one level of the tree."""

    level: int
    node_count: int
    average_mbr_area: float


@dataclass(slots=True)
class RTree:
    """An in-memory R-tree and its root identifier."""

    nodes: dict[int, Node]
    root_id: int
    level_statistics: list[LevelStatistics]
