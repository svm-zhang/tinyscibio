from __future__ import annotations

import numpy as np
import polars as pl
from ncls import NCLS


def find_overlaps(
    ranges_a: pl.DataFrame, ranges_b: pl.DataFrame
) -> pl.DataFrame:
    """
    Find overlapping intervals between two set of ranges.
    """
    starts = pl.Series(np.array([-100, -200]))
    ends = starts + 50

    subject_df = pl.DataFrame({"sstart": starts, "send": ends}).with_row_index(
        "sindex"
    )
    print(subject_df)
    ncls = NCLS(
        ranges_b.get_column("sstart").__array__(),
        ranges_b.get_column("send").__array__(),
        ranges_b.get_column("sindex").__array__(dtype=np.int64),
    )
    print(ncls)

    qstarts = pl.Series(np.array([-52, -180]))
    qends = pl.Series([-2, -120])
    query_df = pl.DataFrame({"qstart": qstarts, "qend": qends}).with_row_index(
        "qindex"
    )
    print(query_df)

    l_idx, r_idx = ncls.all_overlaps_both(
        ranges_a.get_column("qstart").__array__(),
        ranges_a.get_column("qend").__array__(),
        ranges_a.get_column("qindex").__array__(dtype=np.int64),
    )
    print(l_idx)
    print(r_idx)
    print(subject_df[r_idx])
    print(query_df[l_idx])
    return pl.concat([query_df[l_idx], subject_df[r_idx]], how="horizontal")
