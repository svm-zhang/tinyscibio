from pathlib import Path

_PathLike = Path | str


def parse_path(path: _PathLike) -> Path:
    """Parse a given path."""

    if isinstance(path, str):
        # Although Path("") return an path object of ".",
        # here we dont allow it
        if not path:
            raise ValueError(
                "Failed to parse the give path, because it is an empty string"
            )
        else:
            return Path(path)
    return path


def make_dir(
    path: _PathLike, parents: bool = False, exist_ok: bool = False
) -> None:
    """Make new directory at the given path."""
    path = parse_path(path)
    try:
        path.mkdir(parents=parents, exist_ok=exist_ok)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Failed to create directory at the given path {path}, because of "
            "missing parents."
        )
    except FileExistsError:
        if path.is_dir():
            raise FileExistsError(
                f"Failed to create directory at the given path {path}, "
                "because it exists already."
            )
        else:
            raise FileExistsError(
                f"Failed to create directory at the given path {path}, "
                "because it is not a directory."
            )


def get_parent_dir(path: _PathLike, level: int = 0) -> Path:
    """Get parental directory at the specified level for a given path"""
    parents = parse_path(path).parents

    if level < 0:
        raise ValueError(
            f"level paramer cannot be negative. Value received level={level}"
        )

    if level > len(parents):
        raise ValueError(
            f"Cannot go back {level} levels along the give path "
            f"that has {len(parents)} logical parents."
        )

    return parents[level]
