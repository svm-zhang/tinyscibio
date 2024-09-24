from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import pytest

from libpybio import make_dir, parse_path


def test_parse_path_empty_string():
    """Test parse_path() with given path is an empty string"""
    with pytest.raises(ValueError):
        parse_path("")


def test_parse_path_return_path_obj():
    with NamedTemporaryFile() as tmp:
        assert isinstance(parse_path(tmp.name), Path)


def test_mkdir_file_not_found_error():
    """
    Test make_dir(parents=False), when a given path misses parents
    """
    with TemporaryDirectory() as tmp:
        p = Path(tmp) / "missing" / "destination"
        print(p)
        with pytest.raises(FileNotFoundError) as exc:
            make_dir(p, parents=False)
        assert str(exc.value) == (
            f"Failed to create directory at the given path {p}, because of "
            "missing parents."
        )


def test_mkdir_file_exists_error():
    """
    Test make_dir(exist=False), when a given path already exists
    """
    with TemporaryDirectory() as tmp:
        p = Path(tmp) / "existed_destination"
        p.mkdir()
        print(p)
        with pytest.raises(FileExistsError) as exc:
            make_dir(p, exist_ok=False)
        assert str(exc.value) == (
            f"Failed to create directory at the given path {p}, because it "
            "exists already."
        )


def test_mkdir_given_path_is_file_and_existed():
    """
    Test make_dir(exist=True), when a given path is not a directory
    """
    with NamedTemporaryFile() as tmp:
        p = Path(tmp.name)
        p.touch()
        print(p)
        with pytest.raises(FileExistsError) as exc:
            make_dir(p, exist_ok=True)
        assert str(exc.value) == (
            f"Failed to create directory at the given path {p}, because it "
            "is not a directory."
        )
