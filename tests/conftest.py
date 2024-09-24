from tempfile import NamedTemporaryFile, TemporaryDirectory

import pytest


@pytest.fixture(autouse=True, scope="module")
def temp_dir():
    tmp_dir = TemporaryDirectory()
    yield tmp_dir
    tmp_dir.cleanup()


@pytest.fixture(autouse=True, scope="module")
def temp_file():
    tmp_file = NamedTemporaryFile()
    yield tmp_file
    tmp_file.close()
