from pathlib import Path

from spatial_rtree.cli import build_main, query_main


def test_build_cli_creates_tree_file(tmp_path: Path, capsys) -> None:
    points = tmp_path / "points.txt"
    tree_file = tmp_path / "rtree.csv"
    points.write_text("3\n0 0\n1 1\n2 2\n", encoding="utf-8")

    exit_code = build_main([str(points), str(tree_file)])

    assert exit_code == 0
    assert tree_file.exists()
    assert "1 nodes at level 0" in capsys.readouterr().out


def test_query_cli_prints_assignment_format(tmp_path: Path, capsys) -> None:
    points = tmp_path / "points.txt"
    tree_file = tmp_path / "rtree.csv"
    query_file = tmp_path / "window.txt"
    points.write_text("2\n0 0\n5 5\n", encoding="utf-8")
    query_file.write_text("[0 0 1 1]\n", encoding="utf-8")
    build_main([str(points), str(tree_file)])
    capsys.readouterr()

    exit_code = query_main(["window", str(tree_file), str(query_file)])

    assert exit_code == 0
    assert capsys.readouterr().out.strip() == "0 (1): 1"
