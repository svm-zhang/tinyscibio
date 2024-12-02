import numpy as np
import polars as pl
import pytest

from tinyscibio.bam import _MAX_CHUNK_SIZE, _BamArrays


@pytest.fixture
def default_fields():
    return [
        "rnames",
        "rends",
        "rstarts",
        "mqs",
        "propers",
        "primarys",
        "sc_bps",
    ]


@pytest.fixture
def optional_fields():
    return ["qnames", "mm_ecnt", "indel_ecnt", "bqs", "mds"]


@pytest.fixture
def chunk_size(request):
    return request.param


@pytest.fixture
def default_bamarray_df(request):
    chunk_size = request.param
    return pl.DataFrame(
        {
            "rnames": np.zeros(chunk_size, dtype=np.uint16),
            "rstarts": np.zeros(chunk_size, dtype=np.int32),
            "rends": np.zeros(chunk_size, dtype=np.int32),
            "mqs": np.zeros(chunk_size, dtype=np.uint8),
            "propers": np.full(chunk_size, False, dtype=bool),
            "primarys": np.full(chunk_size, False, dtype=bool),
            "sc_bps": np.zeros(chunk_size, dtype=np.int16),
        }
    )


@pytest.mark.parametrize("chunk_size", [1, 10, 100], indirect=True)
def test_bamarray_create_with_default(
    default_fields, optional_fields, chunk_size
):
    bamarray = _BamArrays.create(chunk_size)
    for f in default_fields:
        assert bamarray.__dict__[f].size == chunk_size

    for f in optional_fields:
        assert bamarray.__dict__[f].size == 0


@pytest.mark.parametrize("chunk_size", [1, 10, 100], indirect=True)
def test_bamarray_create_with_optional(
    default_fields, optional_fields, chunk_size
):
    chunk_size = 10
    bamarray = _BamArrays.create(
        chunk_size, with_ecnt=True, with_bq=True, with_md=True, with_qname=True
    )

    for f in default_fields + optional_fields:
        assert bamarray.__dict__[f].size == chunk_size

    bamarray = _BamArrays.create(chunk_size, with_ecnt=True)
    assert bamarray.mm_ecnt.size == chunk_size
    assert bamarray.indel_ecnt.size == chunk_size


def test_bamarray_create_with_zero_chunk_size():
    with pytest.raises(ValueError):
        _BamArrays.create(chunk_size=0)


def test_bamarray_create_with_chunk_size_larger_than_max(default_fields):
    chunk_size = _MAX_CHUNK_SIZE + 1
    bamarray = _BamArrays.create(chunk_size)
    for f in default_fields:
        assert bamarray.__dict__[f].size == _MAX_CHUNK_SIZE


@pytest.mark.parametrize(
    "default_bamarray_df, chunk_size",
    [(1, 1), (10, 10), (100, 100)],
    indirect=True,
)
def test_bamarray_to_df_without_content(default_bamarray_df, chunk_size):
    bamarray = _BamArrays.create(chunk_size)
    for c in default_bamarray_df.columns:
        print(f"column {c=}")
        assert (bamarray.df(idx=chunk_size)[c] == default_bamarray_df[c]).all()
    assert bamarray.df(idx=chunk_size).shape[0] == chunk_size


@pytest.mark.parametrize(
    "chunk_size",
    [1, 10, 100],
    indirect=True,
)
def test_bamarray_to_df_with_negative_idx(chunk_size):
    with pytest.raises(ValueError):
        bamarray = _BamArrays.create(chunk_size)
        bamarray.df(idx=-2)


@pytest.mark.parametrize(
    "chunk_size",
    [1, 10, 100],
    indirect=True,
)
def test_bamarray_to_df_with_idx_larger_than_alloc(chunk_size):
    with pytest.raises(IndexError):
        bamarray = _BamArrays.create(chunk_size)
        bamarray.df(chunk_size + 1)
