from dataclasses import dataclass
from typing import Optional

import numpy as np
import pytest

from tinyscibio import walk_bam


@pytest.fixture
def bam_header_sorted():
    return {
        "HD": {"SO": "coordinate"},
        "RG": [{"ID": "RG1", "SM": "Sample1"}],
        "SQ": [{"SN": "chr1", "LN": "1000"}, {"SN": "chr2", "LN": "2000"}],
    }


class MockAlignmentHeader:
    def __init__(self, header=None):
        self.header: dict = header or {
            "HD": {"SO": "unsorted"},
            "RG": [
                {"ID": "RG1", "SM": "Sample1"},
                {"ID": "RG2", "SM": "Sample2"},
            ],
            "SQ": [{"SN": "chr1", "LN": "120"}, {"SN": "chr2", "LN": "2000"}],
        }

    def __getitem__(self, k):
        return self.header[k]


class MockAlignmentFile:
    def __init__(self, fspath: str, mode: str, header=None):
        self.fspath = fspath
        self.mode = mode
        self.header = MockAlignmentHeader(header)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb): ...

    def seqnames(self):
        return [s["SN"] for s in self.header["SQ"]]

    def seqname2idx(self):
        return {s: i for i, s in enumerate(self.seqnames())}

    def seqmap(self):
        return {s["SN"]: s["LN"] for s in self.header["SQ"]}

    def fetch(
        self,
        contig: str,
        start: Optional[int] = None,
        stop: Optional[int] = None,
    ):
        # FIXME: this is better defined in the class for each available contig
        # FIXME: becuase bam_start and bam_end defines what inside the BAM
        bam_start = 90  # the very first mapped position on rname in the BAM
        # the very last mapped position on rname in the BAM
        # here simply set to length of the given contig
        bam_end = int(self.seqmap()[contig])

        rid = self.seqname2idx()[contig]
        # This is the requested start position from user
        # Do not confuse this with bam_start below
        start = start if start is not None else 0
        end = stop if stop is not None else self.seqmap()[contig]
        flags = [99, 163, 403, 3153]
        cigarstrings = ["75M", "75M", "25M1D49M"]
        mqs = [20, 1, 30, 17]
        propers = [True, True, True, False]
        secondaries = [False, False, True, False]
        qnames = ["r1", "r2", "r3"]
        rgs = ["grp1", "grp2", "grp1", "grp1"]
        mds = [
            "75",
            "37A37",
            "25^C49",
        ]
        if end < bam_start:
            return
        if start > bam_end:
            return
        for pos in range(bam_start, bam_end):
            # mock the logic when pos is larger than the requested end position
            if pos >= end:
                break
            i = pos % 4
            match i:
                case 0:
                    yield MockAlignmentSegment(
                        flag=flags[i],
                        mapping_quality=mqs[i],
                        is_proper_pair=propers[i],
                        is_secondary=secondaries[i],
                        reference_id=rid,
                        reference_start=pos,
                        reference_end=pos + 75,
                        cigarstring=cigarstrings[i],
                        tags={"MD": mds[i], "RG": rgs[i]},
                        query_name=qnames[i],
                        query_qualities=list(range(75)),
                    )
                case 1:
                    yield MockAlignmentSegment(
                        flag=flags[i],
                        mapping_quality=mqs[i],
                        is_proper_pair=propers[i],
                        is_secondary=secondaries[i],
                        reference_id=rid,
                        reference_start=pos,
                        reference_end=pos + 75,
                        cigarstring=cigarstrings[i],
                        tags={"MD": mds[i], "RG": rgs[i]},
                        query_name=qnames[i],
                        query_qualities=list(range(75)),
                    )
                case 2:
                    yield MockAlignmentSegment(
                        flag=flags[i],
                        mapping_quality=mqs[i],
                        is_proper_pair=propers[i],
                        is_secondary=secondaries[i],
                        reference_id=rid,
                        reference_start=pos,
                        reference_end=None,
                        cigarstring=None,
                        tags={"RG": rgs[i]},
                        query_name=qnames[i],
                        query_qualities=None,
                    )
                case 3:
                    yield MockAlignmentSegment(
                        flag=flags[i],
                        mapping_quality=mqs[i],
                        is_proper_pair=propers[i],
                        is_secondary=secondaries[i],
                        reference_id=rid,
                        reference_start=pos,
                        reference_end=None,
                        cigarstring=None,
                        tags={"RG": rgs[i]},
                        query_name=None,
                        query_qualities=list(range(75)),
                    )


@dataclass
class MockAlignmentSegment:
    # remember to have record with
    # missing MD tag, cigarstring, query_qualities, reference_end
    flag: int
    mapping_quality: int
    is_proper_pair: bool
    is_secondary: bool
    reference_id: int
    reference_start: int
    tags: dict[str, str]
    reference_end: Optional[int] = None
    cigarstring: Optional[str] = None
    query_name: Optional[str] = None
    query_qualities: Optional[list[int]] = None

    def get_tag(self, tag: str) -> Optional[str]:
        return self.tags[tag] if self.has_tag(tag) else None

    def has_tag(self, tag: str) -> bool:
        return True if tag in self.tags else False


def init_mock_alignment_file(monkeypatch):
    def _mock_alignment_file(fspath, mode):
        return MockAlignmentFile(fspath, mode)

    monkeypatch.setattr("pysam.AlignmentFile", _mock_alignment_file)
    return _mock_alignment_file


def test_walk_bam(monkeypatch):
    init_mock_alignment_file(monkeypatch)
    region = "chr1:100-104"
    res_df = walk_bam(
        "test.bam",
        region,
        chunk_size=2,
        exclude=3584,
        return_ecnt=True,
        return_bq=True,
        return_md=True,
        return_qname=True,
    )
    assert res_df.shape[0] == 3
    assert (
        res_df["rnames"].to_numpy() == np.array([0, 0, 0], dtype=np.uint16)
    ).all()
    assert (
        res_df["rstarts"].to_numpy()
        == np.array([100, 101, 102], dtype=np.int32)
    ).all()
    assert (
        res_df["rends"].to_numpy() == np.array([175, 176, -1], dtype=np.int32)
    ).all()
    assert (
        res_df["mqs"].to_numpy() == np.array([20, 1, 30], dtype=np.int32)
    ).all()
    assert (
        res_df["propers"].to_numpy()
        == np.array([True, True, True], dtype=bool)
    ).all()
    assert (
        res_df["primarys"].to_numpy()
        == np.array([True, True, False], dtype=bool)
    ).all()
    assert (
        res_df["sc_bps"].to_numpy() == np.array([0, 0, -1], dtype=np.int16)
    ).all()
    assert (
        res_df["mm_ecnt"].to_numpy() == np.array([0, 1, -1], dtype=np.int16)
    ).all()
    assert (
        res_df["indel_ecnt"].to_numpy() == np.array([0, 0, -1], dtype=np.int16)
    ).all()
    assert (
        res_df["qnames"].to_numpy()
        == np.array(["r1", "r2", "r3"], dtype="object")
    ).all()
    assert [
        np.array_equal(arr1, arr2)
        for arr1, arr2 in zip(
            res_df["mds"].to_numpy(),
            np.array([["75"], ["37", "A", "37"], []], dtype=np.ndarray),
        )
    ] == [True, True, True]
    assert [
        np.array_equal(arr1, arr2)
        for arr1, arr2 in zip(
            res_df["bqs"].to_numpy(),
            np.array([list(range(75)), list(range(75)), []], dtype=np.ndarray),
        )
    ] == [True, True, True]


# case where bam file does not have the read group walk_bam function asks
# empty dataframe to return
def test_walk_bam_filtered_by_rg(monkeypatch):
    init_mock_alignment_file(monkeypatch)
    region = "chr1:100-103"
    res_df = walk_bam(
        "test.bam",
        region,
        exclude=3584,
        read_groups={"grp1"},
        return_qname=True,
    )

    assert res_df.shape[0] == 2
    assert (
        res_df["qnames"].to_numpy() == np.array(["r1", "r3"], dtype="object")
    ).all()


def test_walk_bam_return_no_qname(monkeypatch):
    init_mock_alignment_file(monkeypatch)
    region = "chr1:100-103"
    res_df = walk_bam(
        "test.bam",
        region,
        return_qname=False,
    )

    assert "qname" not in res_df.columns


def test_walk_bam_skip_none_qname(monkeypatch):
    init_mock_alignment_file(monkeypatch)
    region = "chr1:100-104"
    res_df = walk_bam(
        "test.bam",
        region,
        exclude=3584,
    )

    assert res_df.shape[0] == 3


def test_walk_bam_return_empty_df_because_rg(monkeypatch):
    init_mock_alignment_file(monkeypatch)
    region = "chr1:100-103"
    res_df = walk_bam(
        "test.bam",
        region,
        exclude=2,
        read_groups={"grp1", "grp2"},
        return_qname=False,
    )

    assert res_df.shape[0] == 0


def test_walk_bam_filtered_by_exclude(monkeypatch):
    init_mock_alignment_file(monkeypatch)
    region = "chr1:100-103"
    res_df = walk_bam(
        "test.bam",
        region,
        exclude=3840,
        return_qname=True,
    )

    assert res_df.shape[0] == 2
    assert (
        res_df["qnames"].to_numpy() == np.array(["r1", "r2"], dtype="object")
    ).all()


def test_walk_bam_grabs_at_and_after_start_position(monkeypatch):
    init_mock_alignment_file(monkeypatch)
    # one-based, the first fetched position should be 101
    region = "chr1:102-103"
    res_df = walk_bam(
        "test.bam",
        region,
        exclude=3840,
        return_qname=True,
    )

    assert res_df.shape[0] == 1
    assert res_df[0, "rstarts"] == 101
