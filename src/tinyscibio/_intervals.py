from __future__ import annotations

import numpy as np
import polars as pl
from ncls import NCLS

from tinyscibio import _PathLike, parse_path

# TODO: define customized interval error


# TODO: fix start and end coord if latter is smaller than the former
def extract_intervals_from_bed(
    bed_file: _PathLike,
    stranded: bool = False,
    one_based: bool = False,
) -> pl.DataFrame:
    """
    Retrieve intervals from given BED file.
    """
    bed_file = parse_path(bed_file)

    intervals = pl.read_csv(bed_file, has_header=False, separator="\t")
    n_cols = intervals.shape[1]
    if n_cols < 3:
        raise ValueError("Expect at least 3 columns in a BED file.")

    if stranded and n_cols < 6:
        raise ValueError(
            "Provided BED file does not have the 6th column for "
            "strand information."
        )

    # Make start coord zero-based
    if one_based:
        intervals = intervals.with_columns(pl.col("column_2") - 1)

    # Sort intervals by start and end coord per chromosome
    return intervals.select(
        pl.all().sort_by(["column_2", "column_3"]).over("column_1")
    )


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

    if len(idx_q) != len(idx_s):
        raise ValueError

    return pl.DataFrame({"idx_q": idx_q, "idx_s": idx_s})


def _check_if_matching_chrom(qry_chrs: pl.Series, subj_chrs: pl.Series):
    """
    Check if any matching chromosomes found between query and subject
    input chromosome values.
    """
    m = [k for k in qry_chrs if k in subj_chrs]
    if not m:
        raise ValueError("No matching chromosomes found. No need to continue")


def find_overlaps(qry: pl.DataFrame, subj: pl.DataFrame) -> pl.DataFrame:
    """
    Find overlaps between query and subject intervals.
    """
    _check_if_matching_chrom(
        qry[:, 0].unique(),
        subj[:, 0].unique(),
    )

    res = _overlaps(qry[:, [1, 2]], subj[:, [1, 2]])
    print(res)
    ovls = pl.concat(
        [qry[res.get_column("idx_q")], subj[res.get_column("idx_s")]],
        how="horizontal",
    ).filter(pl.col("column_1_q") == pl.col("column_1_s"))

    print(ovls)
    return ovls


if __name__ == "__main__":
    import sys

    hla_bed = sys.argv[1]
    wes_bed = sys.argv[2]

    qry = extract_intervals_from_bed(hla_bed).with_columns(
        pl.col("column_1").cast(pl.String)
    )

    subj = extract_intervals_from_bed(wes_bed).with_columns(
        pl.col("column_1").cast(pl.String)
    )
    qry.columns = [f"{c}_q" for c in qry.columns]
    subj.columns = [f"{c}_s" for c in subj.columns]

    find_overlaps(qry, subj)
