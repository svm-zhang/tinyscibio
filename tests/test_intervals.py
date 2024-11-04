from pathlib import Path
from tempfile import NamedTemporaryFile

import polars as pl
import pytest

from tinyscibio import (
    NoMatchingChr,
    NotInt64StartEnd,
    NotSatisfyMinColReq,
    bed_to_df,
    find_overlaps,
)


@pytest.fixture
def _minimum_interval_record():
    return [["chr1", "1", "51"], ["chr2", "120", "240"]]


@pytest.fixture
def _not_meet_min_req_interval_record():
    return [["chr1", "51"], ["chr2", "240"]]


def _write_record_to_bed(fspath, record):
    with open(fspath, "w") as fp:
        record_str = "\n".join(["\t".join(itv) for itv in record])
        fp.write(f"{record_str}")


@pytest.fixture
def bed3(_minimum_interval_record):
    with NamedTemporaryFile() as tmp:
        p = Path(tmp.name).with_suffix(".bed")
        p.touch()
        _write_record_to_bed(p, _minimum_interval_record)
        yield p


@pytest.fixture
def bad_bed(_not_meet_min_req_interval_record):
    with NamedTemporaryFile() as tmp:
        p = Path(tmp.name).with_suffix(".bed")
        p.touch()
        _write_record_to_bed(p, _not_meet_min_req_interval_record)
        yield p


@pytest.fixture
def query_interval():
    return pl.DataFrame(
        {
            "column_1": ["chr1", "chr1", "chr2"],
            "column_2": [1, 20, 200],
            "column_3": [51, 37, 250],
        }
    )


@pytest.fixture
def query_interval_short_cols():
    return pl.DataFrame(
        {
            "column_1": ["chr1", "chr1", "chr2"],
            "column_2": [51, 37, 250],
        }
    )


@pytest.fixture
def query_interval_wrong_type():
    return pl.DataFrame(
        {
            "column_1": ["chr1", "chr1", "chr2"],
            "column_2": [1.0, 20.0, 200],
            "column_3": [51, 37, 250],
        }
    )


@pytest.fixture
def subject_interval_has_overlap():
    return pl.DataFrame(
        {
            "column_1": ["chr1", "chr2", "chr2"],
            "column_2": [38, 120, 35],
            "column_3": [76, 240, 95],
        }
    )


@pytest.fixture
def subject_interval_no_overlap():
    return pl.DataFrame(
        {
            "column_1": ["chr1", "chr2"],
            "column_2": [52, 35],
            "column_3": [67, 95],
        }
    )


@pytest.fixture
def subject_interval_no_matching_chr():
    return pl.DataFrame(
        {
            "column_1": ["1", "2", "3"],
            "column_2": [38, 120, 35],
            "column_3": [76, 240, 95],
        }
    )


def test_find_overlaps(query_interval, subject_interval_has_overlap):
    overlaps = find_overlaps(query_interval, subject_interval_has_overlap)
    assert overlaps.shape[0] == 2
    assert overlaps.equals(
        pl.DataFrame(
            {
                "column_1_q": ["chr1", "chr2"],
                "column_2_q": [1, 200],
                "column_3_q": [51, 250],
                "column_1_s": ["chr1", "chr2"],
                "column_2_s": [38, 120],
                "column_3_s": [76, 240],
            }
        )
    )


def test_find_no_overlaps(query_interval, subject_interval_no_overlap):
    overlaps = find_overlaps(query_interval, subject_interval_no_overlap)
    assert overlaps.shape[0] == 0
    assert overlaps.equals(
        pl.DataFrame(
            {
                "column_1_q": [],
                "column_2_q": [],
                "column_3_q": [],
                "column_1_s": [],
                "column_2_s": [],
                "column_3_s": [],
            }
        )
    )


def test_find_overlap_on_no_matching_chr_intervals(
    query_interval, subject_interval_no_matching_chr
):
    with pytest.raises(NoMatchingChr):
        find_overlaps(query_interval, subject_interval_no_matching_chr)


def test_find_overlap_on_miss_col_intervals(
    query_interval_short_cols, subject_interval_has_overlap
):
    with pytest.raises(NotSatisfyMinColReq):
        find_overlaps(query_interval_short_cols, subject_interval_has_overlap)


def test_find_overlap_on_wrong_type_intervals(
    query_interval_wrong_type, subject_interval_has_overlap
):
    with pytest.raises(NotInt64StartEnd):
        find_overlaps(query_interval_wrong_type, subject_interval_has_overlap)


def test_bed_to_df(bed3):
    assert bed_to_df(bed3).shape[0] == 2


def test_read_bad_bed(bad_bed):
    with pytest.raises(NotSatisfyMinColReq):
        bed_to_df(bad_bed)


def test_read_non_exist_bed():
    p = "non_exist.bed"
    with pytest.raises(FileNotFoundError):
        bed_to_df(p)


def test_bed_to_df_on_one_based_input(bed3):
    assert bed_to_df(bed3, one_based=True).equals(
        pl.DataFrame(
            {
                "column_1": ["chr1", "chr2"],
                "column_2": [0, 119],
                "column_3": [51, 240],
            }
        )
    )
