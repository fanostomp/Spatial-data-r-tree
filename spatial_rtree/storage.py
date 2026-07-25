"""Serialization and deserialization for the assignment's rtree.csv format."""

from __future__ import annotations

import ast
from pathlib import Path

from .bulk_load import calculate_internal_mbr, calculate_leaf_mbr
from .models import InternalEntry, LeafEntry, Node, RTree


def _format_number(value: float) -> str:
    return repr(float(value))


def write_rtree(tree: RTree, path: str | Path) -> None:
    """Write nodes bottom-up in the textual format required by the assignment."""

    output_path = Path(path)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        for node_id in sorted(tree.nodes):
            node = tree.nodes[node_id]
            flag = 0 if node.is_leaf else 1
            parts = [str(node.node_id), str(len(node.entries)), str(flag)]

            if node.is_leaf:
                for raw_entry in node.entries:
                    if not isinstance(raw_entry, LeafEntry):
                        raise TypeError("Leaf node contains an internal entry.")
                    x, y = raw_entry.point
                    parts.append(
                        f"({raw_entry.record_id},("
                        f"{_format_number(x)}, {_format_number(y)}))"
                    )
            else:
                for raw_entry in node.entries:
                    if not isinstance(raw_entry, InternalEntry):
                        raise TypeError("Internal node contains a leaf entry.")
                    xl, yl, xh, yh = raw_entry.mbr
                    parts.append(
                        f"({raw_entry.child_id}, [{_format_number(xl)}, "
                        f"{_format_number(yl)}, {_format_number(xh)}, "
                        f"{_format_number(yh)}])"
                    )

            handle.write(" , ".join(parts) + "\n")


def load_rtree(path: str | Path) -> RTree:
    """Load and validate an R-tree stored in the assignment format."""

    input_path = Path(path)
    nodes: dict[int, Node] = {}
    root_id: int | None = None

    with input_path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                parsed = ast.literal_eval(f"[{line}]")
            except (SyntaxError, ValueError) as exc:
                raise ValueError(f"Malformed R-tree line {line_number}.") from exc

            if len(parsed) < 3:
                raise ValueError(f"R-tree line {line_number} has too few fields.")

            node_id, declared_count, flag, *raw_entries = parsed
            if not isinstance(node_id, int) or node_id < 0:
                raise ValueError(f"Invalid node ID at line {line_number}.")
            if node_id in nodes:
                raise ValueError(f"Duplicate node ID {node_id} at line {line_number}.")
            if flag not in (0, 1):
                raise ValueError(f"Invalid node flag at line {line_number}.")
            if declared_count != len(raw_entries):
                raise ValueError(
                    f"Node {node_id} declares {declared_count} entries but contains "
                    f"{len(raw_entries)}."
                )

            is_leaf = flag == 0
            if is_leaf:
                entries = []
                for entry in raw_entries:
                    try:
                        record_id, point = entry
                        x, y = point
                        entries.append(
                            LeafEntry(int(record_id), (float(x), float(y)))
                        )
                    except (TypeError, ValueError) as exc:
                        raise ValueError(
                            f"Malformed leaf entry in node {node_id}."
                        ) from exc
                mbr = calculate_leaf_mbr(entries)
            else:
                entries = []
                for entry in raw_entries:
                    try:
                        child_id, raw_mbr = entry
                        xl, yl, xh, yh = raw_mbr
                        mbr = (float(xl), float(yl), float(xh), float(yh))
                        if xl > xh or yl > yh:
                            raise ValueError
                        entries.append(InternalEntry(int(child_id), mbr))
                    except (TypeError, ValueError) as exc:
                        raise ValueError(
                            f"Malformed internal entry in node {node_id}."
                        ) from exc
                mbr = calculate_internal_mbr(entries)

            nodes[node_id] = Node(
                node_id=node_id,
                is_leaf=is_leaf,
                entries=entries,
                mbr=mbr,
            )
            root_id = node_id

    if root_id is None:
        raise ValueError(f"No nodes were found in {input_path}.")

    for node in nodes.values():
        if node.is_leaf:
            continue
        for entry in node.entries:
            if not isinstance(entry, InternalEntry):
                raise TypeError("Internal node contains a leaf entry.")
            if entry.child_id not in nodes:
                raise ValueError(
                    f"Node {node.node_id} references missing child {entry.child_id}."
                )

    return RTree(nodes=nodes, root_id=root_id, level_statistics=[])
