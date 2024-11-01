from __future__ import annotations

import numpy as np
import polars as pl
from ncls import NCLS

from tinyscibio import _PathLike, parse_path


def extract_intervals_from_bed(bed_file: _PathLike) -> pl.DataFrame:
    """
    Retrieve intervals from given BED file.
    """
    bed_file = parse_path(bed_file)

    intervals = pl.read_csv(bed_file, has_header=False, separator="\t")
    n_cols = intervals.shape[1]
    if n_cols < 3:
        raise ValueError("Expect at least 3 columns in a BED file.")
    return intervals


def _overlaps(
    intervals_a: pl.DataFrame, intervals_b: pl.DataFrame
) -> tuple[int, int]:
    """
    Find overlapping intervals between two set of ranges.
    """
    ncls = NCLS(
        intervals_b[:, 1].__array__(),
        intervals_b[:, 2].__array__(),
        np.array(range(0, intervals_b.shape[1]), dtype=np.int64),
    )
    print(ncls)

    return ncls.all_overlaps_both(
        intervals_a[:, 1].__array__(),
        intervals_a[:, 2].__array__(),
        np.array(range(0, intervals_a.shape[1]), dtype=np.int64),
    )


if __name__ == "__main__":
    intervals_a = pl.DataFrame(
        {"chr": ["chr1", "chr2"], "start": [1, 200], "end": [51, 250]}
    )
    intervals_b = pl.DataFrame(
        {"chr": ["chr1", "chr2"], "start": [38, 120], "end": [76, 240]}
    )

    # TODO: split intervals by chr
    l_idx, r_idx = _overlaps(intervals_a, intervals_b)
    # FIXME: check if no overlaps found, return None
    # FIXME: also need to assert l and r idx have the same length
    # FIXME: Need to make sure chr are the same between l and r
    print(l_idx)
    print(r_idx)
