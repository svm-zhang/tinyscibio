from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest

from tinyscibio import (
    BAMetadata,
    count_indel_bases,
    count_indel_events,
    count_mismatch_events,
    count_soft_clip_bases,
    count_unaligned_events,
    parse_cigar,
    parse_md,
)


class MockAlignmentFile:
    def __init__(self, fspath, mode, header=None):
        self.fspath = fspath
        self.mode = mode
        self.header = MockAlignmentHeader(header)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class MockAlignmentHeader:
    def __init__(self, header=None):
        self.header = header or {
            "HD": {"SO": "unsorted"},
            "RG": [
                {"ID": "RG1", "SM": "Sample1"},
                {"ID": "RG2", "SM": "Sample2"},
            ],
            "SQ": [{"SN": "chr1", "LN": "1000"}, {"SN": "chr2", "LN": "2000"}],
        }

    def to_dict(self):
        return self.header


@pytest.fixture
def tmp_bam_file():
    with NamedTemporaryFile() as tmp:
        yield Path(tmp.name).with_suffix(".bam")


@pytest.fixture
def bam_header_no_sq():
    return {
        "HD": {"SO": "unsorted"},
        "RG": [{"ID": "RG1", "SM": "Sample1"}, {"ID": "RG2", "SM": "Sample2"}],
    }


@pytest.fixture
def bam_header_no_rg():
    return {
        "HD": {"SO": "unsorted"},
        "SQ": [{"SN": "chr1", "LN": "1000"}, {"SN": "chr2", "LN": "2000"}],
    }


@pytest.fixture
def bam_header_sorted():
    return {
        "HD": {"SO": "coordinate"},
        "RG": [{"ID": "RG1", "SM": "Sample1"}],
        "SQ": [{"SN": "chr1", "LN": "1000"}, {"SN": "chr2", "LN": "2000"}],
    }


@pytest.fixture
def bam_header_zero_sq():
    return {
        "HD": {"SO": "coordinate"},
        "RG": [{"ID": "RG1", "SM": "Sample1"}],
        "SQ": [],
    }


@pytest.fixture
def bam_header_missing_sn_or_ln():
    return {
        "HD": {"SO": "unsorted"},
        "RG": [{"ID": "RG1", "SM": "Sample1"}],
        "SQ": [
            {
                "SN": "chr1",
            },
            {"LN": "2000"},
        ],
    }


def init_mock_alignment_file(monkeypatch, header=None):
    def _mock_alignment_file(fspath, mode, header=header):
        return MockAlignmentFile(fspath, mode, header)

    monkeypatch.setattr("pysam.AlignmentFile", _mock_alignment_file)

    return _mock_alignment_file


def test_bametadata_init(monkeypatch):
    init_mock_alignment_file(monkeypatch)
    metadata = BAMetadata("test.bam")

    assert metadata.fspath == "test.bam"
    assert metadata.sort_by == "unsorted"
    assert metadata.read_groups == [
        {"ID": "RG1", "SM": "Sample1"},
        {"ID": "RG2", "SM": "Sample2"},
    ]
    assert metadata.references == [{"chr1": 1000}, {"chr2": 2000}]


def test_bametadata_on_header_without_sq(monkeypatch, bam_header_no_sq):
    init_mock_alignment_file(monkeypatch, bam_header_no_sq)
    with pytest.raises(IndexError):
        BAMetadata("test.bam")


def test_bametadata_on_header_without_rg(monkeypatch, bam_header_no_rg):
    init_mock_alignment_file(monkeypatch, bam_header_no_rg)
    assert BAMetadata("test.bam").read_groups == []
    assert len(BAMetadata("test.bam").read_groups) == 0


def test_bametadata_on_header_missing_sn_or_ln(
    monkeypatch, bam_header_missing_sn_or_ln
):
    init_mock_alignment_file(monkeypatch, bam_header_missing_sn_or_ln)
    with pytest.raises(IndexError):
        BAMetadata("test.bam")


def test_bametadata_coordinate_sorted(
    monkeypatch, bam_header_sorted, tmp_bam_file
):
    init_mock_alignment_file(monkeypatch, bam_header_sorted)
    # no bai file here
    with pytest.raises(FileNotFoundError):
        BAMetadata(str(tmp_bam_file))

    # make name.bai
    tmp_bai_file = tmp_bam_file.with_suffix(".bai")
    tmp_bai_file.touch()
    assert BAMetadata(str(tmp_bam_file)).sort_by == "coordinate"
    # del name.bai
    tmp_bai_file.unlink()

    # no bai file here again
    with pytest.raises(FileNotFoundError):
        BAMetadata(str(tmp_bam_file))

    # make name.bam.bai
    tmp_bai_file = tmp_bam_file.parent / (tmp_bam_file.name + ".bai")
    tmp_bai_file.touch()
    # should pass here
    assert BAMetadata(str(tmp_bam_file)).sort_by == "coordinate"
    tmp_bai_file.unlink()


def test_bamedata_seqnames(monkeypatch):
    init_mock_alignment_file(monkeypatch)
    metadata = BAMetadata("test.bam")
    assert metadata.seqnames() == ["chr1", "chr2"]


def test_bametadata_repr(monkeypatch, bam_header_no_rg):
    init_mock_alignment_file(monkeypatch, bam_header_no_rg)
    metadata = BAMetadata("test.bam")
    assert repr(metadata) == (
        f"BAM file: {metadata.fspath}\n"
        f"Sort by: {metadata.sort_by}\n"
        f"# references: {len(metadata.references)}\n"
        f"# read groups: {len(metadata.read_groups)}\n"
    )


# https://vincebuffalo.com/notes/2014/01/17/md-tags-in-bam-files.html
@pytest.mark.parametrize(
    "md, expect",
    [
        ("10A3T0T10", ["10", "A", "3", "T", "0", "T", "10"]),
        ("85^A16", ["85", "^A", "16"]),
        ("100", ["100"]),
        ("0A", ["0", "A"]),
        ("10^AC41G49", ["10", "^AC", "41", "G", "49"]),
        (
            "6G4C20G1A5C5A1^C3A15G",
            [
                "6",
                "G",
                "4",
                "C",
                "20",
                "G",
                "1",
                "A",
                "5",
                "C",
                "5",
                "A",
                "1",
                "^C",
                "3",
                "A",
                "15",
                "G",
            ],
        ),
    ],
)
def test_parse_md(md, expect):
    assert parse_md(md) == expect


def test_parse_md_empty_str():
    with pytest.raises(ValueError):
        parse_md("")


@pytest.mark.parametrize("bad_md", ["A", "^A", "10AA31"])
def test_parse_md_bad_str(bad_md):
    with pytest.raises(ValueError):
        parse_md(bad_md)


@pytest.mark.parametrize(
    "cigar, expect",
    [
        ("89M1I11M", 1),
        ("31M1I17M1D37M", 2),
        ("27S73M", 1),
        ("5S73M22S", 2),
        ("5S45M1D27M1I20M", 3),
        ("45M55H", 1),
        ("45H35M1D23M1I30S", 4),
    ],
)
def test_count_unaligned_events(cigar, expect):
    assert count_unaligned_events(cigar) == expect
    assert count_unaligned_events(parse_cigar(cigar)) == expect


@pytest.mark.parametrize(
    "cigar, expect",
    [
        ("89M1I11M", [("89", "M"), ("1", "I"), ("11", "M")]),
        ("27S73M", [("27", "S"), ("73", "M")]),
        (
            "45M1D27M1I20M5H",
            [
                ("45", "M"),
                ("1", "D"),
                ("27", "M"),
                ("1", "I"),
                ("20", "M"),
                ("5", "H"),
            ],
        ),
        ("49=2D5X6=", [("49", "="), ("2", "D"), ("5", "X"), ("6", "=")]),
        ("27M351N11M", [("27", "M"), ("351", "N"), ("11", "M")]),
    ],
)
def test_parse_cigar(cigar, expect):
    assert parse_cigar(cigar) == expect


def test_parse_cigar_empty_str():
    with pytest.raises(ValueError):
        parse_cigar("")


@pytest.mark.parametrize(
    "cigar",
    ["SDIMMM", "=101", "101", "SDI18MM2H"],
)
def test_parse_bad_cigar_str(cigar):
    with pytest.raises(ValueError):
        parse_cigar(cigar)


@pytest.mark.parametrize(
    "cigar, expect",
    [
        ("89M1I11M", 0),
        ("27S73M", 27),
        ("5S73M22S", 27),
        ("45M1D27M1I20M5S", 5),
    ],
)
def test_count_sc_bases(cigar, expect):
    assert count_soft_clip_bases(cigar) == expect
    assert count_soft_clip_bases(parse_cigar(cigar)) == expect


@pytest.mark.parametrize(
    "cigar, expect",
    [
        ("89M1I11M", 1),
        ("27S73M", 0),
        ("45M1D27M1I20M5S", 2),
    ],
)
def test_count_indel_events(cigar, expect):
    assert count_indel_events(cigar) == expect
    assert count_indel_events(parse_cigar(cigar)) == expect


@pytest.mark.parametrize(
    "cigar, expect",
    [
        ("89M1I11M", 1),
        ("27S73M", 0),
        ("45M9D27M1I20M5S", 10),
    ],
)
def test_count_indel_bases(cigar, expect):
    assert count_indel_bases(cigar) == expect
    assert count_indel_bases(parse_cigar(cigar)) == expect


@pytest.mark.parametrize(
    "md, expect",
    [
        ("10A3T0T10", 3),
        ("85^A16", 0),
        ("100", 0),
        ("6G4C20G1A5C5A1^C3A15G", 8),
    ],
)
def test_count_mm_events(md, expect):
    assert count_mismatch_events(md) == expect
    assert count_mismatch_events(parse_md(md)) == expect
