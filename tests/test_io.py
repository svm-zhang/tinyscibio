from pathlib import Path

import pytest

from tinyscibio import get_parent_dir, make_dir, parse_path


def test_parse_path_empty_string():
    """Test parse_path() with given path is an empty string"""
    with pytest.raises(ValueError):
        parse_path("")


def test_parse_path_return_path_obj(temp_file):
    assert isinstance(parse_path(temp_file.name), Path)


def test_mkdir_file_not_found_error(temp_dir):
    """
    Test make_dir(parents=False), when a given path misses parents
    """
    p = Path(temp_dir.name) / "missing" / "destination"
    with pytest.raises(FileNotFoundError):
        make_dir(p, parents=False)


def test_mkdir_file_exists_error(temp_dir):
    """
    Test make_dir(exist=False), when a given path already exists
    """
    p = Path(temp_dir.name) / "existed_destination"
    p.mkdir()
    with pytest.raises(FileExistsError):
        make_dir(p, exist_ok=False)


def test_mkdir_given_path_is_file_and_existed(temp_file):
    """
    Test make_dir(exist=True), when a given path is not a directory
    """
    p = Path(temp_file.name)
    p.touch()
    with pytest.raises(FileExistsError):
        make_dir(p, exist_ok=True)


def test_parent_dir_return_type(temp_dir):
    """Test get_parent_dir() function returns a Path object"""
    p = Path(temp_dir.name)
    assert isinstance(get_parent_dir(p), Path)


def test_level_beyond_num_logic_parents(temp_dir):
    """Test get_parent_dir() function to capture ValueError when level is
    beyond number of logical ancestors
    """
    p = Path(temp_dir.name)
    level = len(p.parents) + 1
    with pytest.raises(ValueError):
        get_parent_dir(p, level)


def test_negative_level_value(temp_dir):
    """Test get_parent_dir() function to capture ValueError
    when level is negative
    """
    p = Path(temp_dir.name)
    level = -1
    with pytest.raises(ValueError):
        get_parent_dir(p, level)
