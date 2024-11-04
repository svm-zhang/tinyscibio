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


def _check_if_matching_chrom(
    qry_chrs: pl.Series, subj_chrs: pl.Series
) -> None:
    """
    Check if any matching chromosomes found between query and subject
    input chromosome values. If there is none, no need to perform find_overlap
    operation; return error instead.
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
        """
        Check if a minimum 3 columns requirment is met.
        """
        if df.shape[1] < 3:
            raise NotSatisfyMinColReq(
                "A minimum of 3 columns (chrom, start, end) required for "
                "interval dataframe to do overlap operation.\n"
                f"Column received: {df.columns}"
            )

    def _check_start_end_cols(df: pl.DataFrame) -> None:
        """
        Check if start (2nd) and end (3rd) columns are of int64 type
        """
        if df[:, 1].dtype != pl.Int64 or df[:, 2].dtype != pl.Int64:
            raise NotInt64StartEnd("2nd and 3rd columns must be of int64 type")

    _check_minimun_col_req(df)
    _check_start_end_cols(df)


def _overlaps(qry: pl.DataFrame, subj: pl.DataFrame) -> pl.DataFrame:
    """
    Find overlappings between query and subject intervals using ncls package.

    Parameters:
        qry: query intervals in dataframe.
        subj: subject intervals from which qry querys.

    Returns:
        polars dataframe with indices of the query and subject intervals
        where overlaps are found.

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
    Find overlaps between query and subject intervals. Strand is not taken into
    consideration at the moment.

    The function returns an empty (not None) polars dataframe if no overlapping
    intervals are found.

    Also, "_q" and "_s" are automatically appended to column names of
    the query and subject dataframes, respectively, to avoid column name
    conflict error when combining overlapping intervals from query and subject.

    Examples:

        >>> import polars as pl
        >>> from tinyscibio import find_overlaps
        >>> qry = pl.DataFrame(
        ...    {
        ...        "column_1": ["chr1", "chr2", "chr3"],
        ...        "column_2": [1, 200, 800],
        ...        "column_3": [51, 240, 850]
        ...    }
        ... )
        >>> subj = pl.DataFrame(
        ...    {
        ...        "column_1": ["chr1", "chr2"],
        ...        "column_2": [38, 120],
        ...        "column_3": [95, 330]
        ...    }
        ... )
        >>> find_overlaps(qry, subj)
        shape: (2, 6)
        ┌────────────┬────────────┬────────────┬────────────┬────────────┬────────────┐
        │ column_1_q ┆ column_2_q ┆ column_3_q ┆ column_1_s ┆ column_2_s ┆ column_3_s │
        │ ---        ┆ ---        ┆ ---        ┆ ---        ┆ ---        ┆ ---        │
        │ str        ┆ i64        ┆ i64        ┆ str        ┆ i64        ┆ i64        │
        ╞════════════╪════════════╪════════════╪════════════╪════════════╪════════════╡
        │ chr1       ┆ 1          ┆ 51         ┆ chr1       ┆ 38         ┆ 95         │
        │ chr2       ┆ 200        ┆ 240        ┆ chr2       ┆ 120        ┆ 330        │
        └────────────┴────────────┴────────────┴────────────┴────────────┴────────────┘


    Parameters:
        qry: query intervals in dataframe.
        subj: subject intervals from which qry querys.

    Returns:
        polars dataframe with overlapping intervals between
        query and subject inputs.

    Raises:
        NotSatisfyMinColReq: when input qry and/or subj dataframes fail to
                             have a minimum of 3 columns.
        NotInt64StartEnd: when 2nd and 3rd columns of qry and/or subj
                          dataframes have non-int64 types.
        NoMatchingChr: when no matching chromosomes (1st column) found between
                       qry and subj dataframes.

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

    Examples:
        Assuming we have a BED file with the following intervals:

        │ chr6 ┆ 29909037 ┆ 29913661 │\n
        │ chr6 ┆ 31236526 ┆ 31239869 │\n
        │ chr6 ┆ 31321649 ┆ 31324964 │\n

        >>> import polars as pl
        >>> from tinyscbio import bed_to_df
        >>> bed_file = "hla.bed"
        >>> bed_to_df(bed_file)
        shape: (3, 3)
        ┌──────────┬──────────┬──────────┐
        │ column_1 ┆ column_2 ┆ column_3 │
        │ ---      ┆ ---      ┆ ---      │
        │ str      ┆ i64      ┆ i64      │
        ╞══════════╪══════════╪══════════╡
        │ chr6     ┆ 29909037 ┆ 29913661 │
        │ chr6     ┆ 31236526 ┆ 31239869 │
        │ chr6     ┆ 31321649 ┆ 31324964 │
        └──────────┴──────────┴──────────┘

    Parameters:
        bed_file: string or path object pointing to a BED file.
        one_based: whether or not the given BED file is 1-based

    Returns:
        polars dataframe with input columns.

    Raises:
        FileNotFoundError: when the given path to the BED file does not exist.
        NotSatisfyMinColReq: when input qry and/or subj dataframes fail to
                             have a minimum of 3 columns.
        NotInt64StartEnd: when 2nd and 3rd columns of qry and/or subj
                          dataframes have non-int64 types.

    """
    bed_file = parse_path(bed_file)
    if not bed_file.exists():
        raise FileNotFoundError(
            f"Failed to find the given BED file at path: {bed_file}"
        )

    intervals = pl.read_csv(bed_file, has_header=False, separator="\t")
    _check_input_interval(intervals)

    # Make start coord zero-based
    if one_based:
        intervals = intervals.with_columns(pl.col("column_2") - 1)

    # Sort intervals by start and end coord per chromosome
    return intervals.select(
        pl.all().sort_by(["column_2", "column_3"]).over("column_1")
    )
