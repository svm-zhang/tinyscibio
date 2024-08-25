from __future__ import annotations

from pathlib import Path

PathLike = Path | str


def parse_path(path: PathLike) -> Path:
    """Parse a given path."""
    if isinstance(path, Path):
        return path
    return Path(path)


def make_dir(path: PathLike, parents: bool = False, exist_ok: bool = False) -> None:
    """Make new directory at the given path."""
    path = parse_path(path)
    if not path.exists():
        path.mkdir(parents=parents, exist_ok=exist_ok)


def check_existence(
    path: PathLike,
) -> None:
    path = parse_path(path)

    if not path.exists():
        raise FileNotFoundError("Found no file or directory at given path {path}")
