from __future__ import annotations

import numpy as np
import polars as pl
from ncls import NCLS

from ._io import _PathLike, parse_path


class NoMatchingChr(Exception):
    """When no matching chromosome found between query and subject"""


class NotSatisfyMinColReq(Exception):
    """When input does not meet the requirement of minimum 3 columns"""


class NotInt64StartEnd(Exception):
    """When columns for start and end coords are not of type Int64"""


def _check_if_matching_chrom(qry_chrs: pl.Series, subj_chrs: pl.Series):
    """
    Check if any matching chromosomes found between query and subject
    input chromosome values.
    """
    m = [k for k in qry_chrs if k in subj_chrs]
    if not m:
        raise NoMatchingChr(
            (
                "Found no matching chromosomes between query and subject.\n"
                f"Chromosome in query: {qry_chrs}\n"
                f"Chromosome in subject: {subj_chrs}\n"
            )
        )


def _check_input_interval(df: pl.DataFrame) -> None:
    def _check_minimun_col_req(df: pl.DataFrame) -> None:
        if df.shape[1] < 3:
            raise NotSatisfyMinColReq(
                "A minimum of 3 columns (chrom, start, end) required for "
                "interval dataframe to do overlap operation.\n"
                f"Column received: {df.columns}"
            )

    def _check_start_end_cols(df: pl.DataFrame) -> None:
        if df[:, 1].dtype != pl.Int64 or df[:, 2].dtype != pl.Int64:
            raise NotInt64StartEnd("2nd and 3rd columns must be of int64 type")

    _check_minimun_col_req(df)
    _check_start_end_cols(df)


def _overlaps(qry: pl.DataFrame, subj: pl.DataFrame) -> pl.DataFrame:
    """
    Find overlapping itv between two set of ranges.
    """
    ncls = NCLS(
        subj[:, 0].__array__(),
        subj[:, 1].__array__(),
        np.array(range(0, subj.shape[0]), dtype=np.int64),
    )
    idx_q = idx_s = np.empty(0, dtype=np.int64)
    idx_q, idx_s = ncls.all_overlaps_both(
        qry[:, 0].__array__(),
        qry[:, 1].__array__(),
        np.array(range(0, qry.shape[0]), dtype=np.int64),
    )

    return pl.DataFrame({"idx_q": idx_q, "idx_s": idx_s})


def find_overlaps(qry: pl.DataFrame, subj: pl.DataFrame) -> pl.DataFrame:
    """
    Find overlaps between query and subject intervals.
    """
    _check_input_interval(qry)
    _check_input_interval(subj)

    _check_if_matching_chrom(
        qry[:, 0].unique(),
        subj[:, 0].unique(),
    )

    # append _q and _s to columns of each dataframe
    qry.columns = [f"{c}_q" for c in qry.columns]
    subj.columns = [f"{c}_s" for c in subj.columns]

    res = _overlaps(qry[:, [1, 2]], subj[:, [1, 2]])
    ovls = pl.concat(
        [qry[res.get_column("idx_q")], subj[res.get_column("idx_s")]],
        how="horizontal",
    )
    ovls = ovls.filter(pl.col(ovls.columns[0]) == pl.col(ovls.columns[3]))

    return ovls


def bed_to_df(
    bed_file: _PathLike,
    one_based: bool = False,
) -> pl.DataFrame:
    """
    Retrieve intervals from given BED file.
    """
    bed_file = parse_path(bed_file)

    intervals = pl.read_csv(bed_file, has_header=False, separator="\t")
    _check_input_interval(intervals)

    # Make start coord zero-based
    if one_based:
        intervals = intervals.with_columns(pl.col("column_2") - 1)

    # Sort intervals by start and end coord per chromosome
    return intervals.select(
        pl.all().sort_by(["column_2", "column_3"]).over("column_1")
    )
