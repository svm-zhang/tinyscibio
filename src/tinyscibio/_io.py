from pathlib import Path

_PathLike = Path | str


def parse_path(path: _PathLike) -> Path:
    """
    Parse a string literal or a Path object into a Path object.

    In the latter case, the function does not do anything but return
    the given Path object directly.

    I also use this function to directly parse file and directory input
    strings passed in from command line.

    Examples:
        >>> from tinyscibio import parse_path
        >>> p = "/home/user1/project"
        >>> parse_path(p)
        Path("/home/user1/project")

        >>> from pathlib import Path
        >>> from tinyscibio import parse_path
        >>> p_obj = Path("/home/user1/project")
        >>> parse_path(p_obj)
        Path("/home/user1/project")

        >>> from tinyscibio import parse_path
        >>> import argparse
        >>> parser = argparse.ArgumentParser()
        >>> parser.add_argument(
                "--bam",
                metavar="FILE",
                type=parse_path,
                help="Specify the path to the BAM file"
            )
        >>> args = parser.parse_args()
        >>> isinstance(args.bam, Path)
        true

    Parameters:
        path: Either a string literal or a Path object.

    Returns:
        A Path object.

    Raises:
        ValueError: When input path parameter is an empty string
    """

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
    """
    Create a directory at the specified location.

    Examples:
        >>> from tinyscibio import make_dir
        >>> p = "/home/user1/projects"
        >>> make_dir(p, parents=True, exist_ok=True)

    Parameters:
        path: Either a string literal or a Path object.
        parents: Whether or not creating parental folders leading
                 to the given path when not existed.
        exist_ok: Whether or not allowing the directory trying to be
                  created already existed.

    Returns:
        None.

    Raises:
        FileNotFoundError: When parental folders leading to the given path
                           does not exists, and parents parameter is False
        FileExistsError: When trying to create a directory that already
                         existed, and exist_ok parameter is False.
                         Or when trying to create a directory at the given
                         location leading to a file.
    """
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
    """
    Get parent directory at the specified level for a given path.

    Setting a level=0 means return the given path.

    Examples:
        >>> from tinyscibio import get_parent_dir
        >>> p = "/home/user1/projects"
        >>> get_parent_dir(p, level=1)
        Path("/home/user1/")

    Parameters:
        path: Either a string literal or a Path object.
        level: Num of levels to walk back along a givne path

    Returns:
        A Path object.

    Raises:
        ValueError: When the value of level is negative or larger than
                    the number of parent folders in the given path
    """
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
