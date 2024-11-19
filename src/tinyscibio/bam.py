import inspect
import itertools
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Optional, Union

import numpy as np
import polars as pl
import pysam

from tinyscibio import parse_path


@dataclass
class BAMetadata:
    """
    A BAMetadata object holds metadata about a given BAM file.

    Examples:
        Let us use a hypothetical coordiante-sorted BAM file with one read
        group and two references, as an example:

        >>> bametadata = BAMetadata("test.bam")
        >>> print(bametadata.sort_by)
        coordinate
        >>> print(bametadata.read_groups)
        [{"ID": "test", "SM": "test"}]
        >>> print(bametadata.references)
        [{"r1": 1000}, {"r2": 2000}]
        >>> print(bametadata)
        BAM file: test.bam
        Sort by: coordinate
        # references: 2
        # read groups: 1

    Attributes:
        fspath: path to the BAM file
        sort_by: sort state, e.g. unknown, unsorted, queryname, and coordinate.
        references: list of mappings of reference name to its length.
        read_groups: list of read group dictionaries
    """

    fspath: str
    sort_by: str = field(init=False, default="")
    references: dict[str, int] = field(init=False, default_factory=dict)
    read_groups: list[dict[str, str]] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        """Parse metadata out of given BAM file"""
        with pysam.AlignmentFile(self.fspath, "rb") as bamf:
            header = bamf.header.to_dict()
            self._parse_read_groups(header)
            self._parse_sort_by(header)
            self._parse_references(header)

    def _parse_read_groups(self, header: dict[str, Any]) -> None:
        """Parse read groups from the header"""
        self.read_groups = header.get("RG", [])

    def _parse_sort_by(self, header: dict[str, Any]) -> None:
        """Parse sort_by information from the header"""
        hd = header.get("HD", {})
        self.sort_by = hd.get("SO", "")
        if self.sort_by == "coordinate":
            self._check_bai()

    def _parse_references(self, header: dict[str, Any]) -> None:
        """Parse references from the header"""
        sqs = header.get("SQ", [])

        if not sqs:
            raise IndexError("No sequence information found in the header")

        self.references = {
            s["SN"]: int(s["LN"]) for s in sqs if "SN" in s and "LN" in s
        }

        # if some references does not have SN and LN info,
        # they will not be read in the self.references
        # the check below fails if some references are missing due to this
        if len(self.references) != len(sqs):
            raise IndexError(
                "Missing reference sequences in 'self.references', "
                "likely due to missing SN or LN keys"
            )

    def _check_bai(self) -> None:
        """Check if index file exists for the coordinate-sorted BAM"""
        bam = parse_path(self.fspath)
        bai = bam.parent / f"{bam.name}.bai"
        if not bai.exists():
            bai = bam.with_suffix(".bai")
            if not bai.exists():
                raise FileNotFoundError(
                    f"Cannot find the index file for the given BAM {bam}"
                )

    def __repr__(self) -> str:
        """Return BAMetadata object representation"""
        return (
            f"BAM file: {self.fspath}\n"
            f"Sort by: {self.sort_by}\n"
            f"# references: {len(self.references)}\n"
            f"# read groups: {len(self.read_groups)}\n"
        )

    # def seqnames(self) -> list[str]:
    #     return [k for r in self.references for k in r.keys()]


def parse_cigar(cigar: str) -> list[tuple[str, str]]:
    """
    Parse a given CIGAR string into a list of tuple items of
    (length, operation).

    Valid operations in a CIGAR string are: M, I, D, N, S, H, P, =, and X.

    Examples:
        >>> cigar_str = "27S89M1I11M"
        >>> assert parse_cigar(cigar_str) == [
            ("27", "S"),
            ("89", "M"),
            ("1", "I"),
            ("11", "M")
        ]
        true

    Parameters:
        cigar: a CIGAR string

    Returns:
        A list of tuple items, each of which consists of length and operation.

    Raises:
        ValueError: when the given CIGAR string is empty, or when the CIGAR string
                    parsed into an empty list, or when failure of reconstructing
                    parsed list back to the original CIGAR string.
    """
    if not cigar:
        raise ValueError("Cannot parse empty CIGAR string")

    cigar_parsed = re.findall("([0-9]+)([MIDNSHP=X])", cigar)

    if not cigar_parsed:
        raise ValueError(
            f"CIGAR string {cigar} failed to be parsed. Empty list returned"
        )
    if "".join(["".join(k) for k in cigar_parsed]) != cigar:
        raise ValueError(
            f"Failed to reconstruct {cigar_parsed=} back to original {cigar=}"
        )
    return cigar_parsed


def parse_md(md: str) -> list[str]:
    """
    Parse a given MD string into a list.

    Examples:
        >>> md_str = "10A3T0T10"
        >>> assert parse_cigar(cigar_str) == [
                "10", "A", "3", "T", "0", "T", "10"
            ]
        true

    Parameters:
        md: a MD string

    Returns:
        A list of strings.

    Raises:
        ValueError: when the given MD string is empty, or when the MD string
                    parsed into an empty list, or when failure of reconstructing
                    parsed list back to the original MD string.
    """
    if not md:
        raise ValueError("Cannot parse empty MD string")

    # this to make sure MD string start with digit
    if not md[:1].isdigit():
        raise ValueError(
            f"Invalid MD string {md=}. MD string should always start "
            "with digit"
        )

    md_iter = itertools.groupby(md, lambda k: k.isalpha() or not k.isalnum())
    md_parsed = ["".join(group) for c, group in md_iter if not c or group]
    # below is to handle bad CIGAR such as 10AA90.
    # Such string can mean either 10^AA90 or 10A0A90
    # I artificially added the ^ to make it become ^AA
    # I then fail the function by comparing reconstructed to original
    # Will update if I have better solution
    md_parsed = [
        f"^{k}" if k.isalpha() and len(k) > 1 else k for k in md_parsed
    ]
    if "".join(md_parsed) != md:
        raise ValueError(
            f"Failed to reconstruct {md_parsed=} back to original {md=}"
        )
    return md_parsed


def count_soft_clip_bases(cigar: Union[str, Sequence[tuple[str, str]]]) -> int:
    """
    Count the number of soft-clipped bases from a given CIGAR string.

    Soft-clipped bases are represented as "S" in the CIGAR string.

    Examples:
        >>> cigar_str = "27S89M1I11M"
        >>> assert count_soft_clip_bases(cigar_str) == 27
        true

        >>> cigar_list = [("89", "M"), ("1", "I"), ("11", "M")]
        >>> assert count_soft_clip_bases(cigar_list) == 0
        true

        >>> cigar_str = "27S89M1I11M"
        >>> assert count_soft_clip_bases(parse_cigar(cigar_str)) == 27 # use result from parse_cigar function
        true


    Parameters:
        cigar: a CIGAR string or a list of tuple items parsed by parse_cigar() function.

    Returns:
        The number of soft-clipped bases.
    """
    if isinstance(cigar, str):
        cigar = parse_cigar(cigar)
    return sum([int(c) for c, op in cigar if op == "S"])


def count_unaligned_events(
    cigar: Union[str, Sequence[tuple[str, str]]],
) -> int:
    """
    Count the number of unaligned events in a given CIGAR string.

    Unaligned events in this context include: insertion(I), deletion(D),
    soft-clipping(S), and hard-clipping(H).

    The function counts the number of events, rather than unaligned bases.

    Examples:
        >>> cigar_str = "45H35M1D23M1I30S"
        >>> assert count_unaligned_events(cigar_str) == 4
        true

        >>> cigar_list = [("89", "M"), ("1", "I"), ("11", "M")]
        >>> assert count_unaligned_events(cigar_list) == 1
        true

        >>> cigar_str = "45H35M1D23M1I30S"
        >>> assert count_unaligned_events(parse_cigar(cigar_str)) == 4
        true


    Parameters:
        cigar: a CIGAR string or a list of tuple items parsed by parse_cigar() function.

    Returns:
        The number of unaligned events.
    """
    if isinstance(cigar, str):
        cigar = parse_cigar(cigar)
    return len([op for _, op in cigar if op in ["I", "D", "S", "H"]])


def count_indel_events(cigar: Union[str, Sequence[tuple[str, str]]]) -> int:
    """
    Count the number of Is and Ds event in a given CIGAR string.

    Indel events include: insertion(I) and deletion(D).

    The function counts the number of events, rather than inserted and deleted
    bases.

    Examples:
        >>> cigar_str = "45H35M1D23M1I30S"
        >>> assert count_indel_events(cigar_str) == 2
        true

        >>> cigar_list = [("89", "M"), ("1", "I"), ("11", "M")]
        >>> assert count_indel_events(cigar_list) == 1
        true


    Parameters:
        cigar: a CIGAR string or a list of tuple items parsed by parse_cigar() function.

    Returns:
        The number of insertion and deletion events.
    """
    if isinstance(cigar, str):
        cigar = parse_cigar(cigar)
    return len([op for _, op in cigar if op in ["I", "D"]])


def count_indel_bases(cigar: Union[str, Sequence[tuple[str, str]]]) -> int:
    """
    Count the length of indels in the given CIGAR string.

    The function counts the number of bases, rather than events.

    Examples:
        >>> cigar_str = "45M9D27M1I20M5S"
        >>> assert count_indel_bases(cigar_str) == 10
        true

        >>> cigar_list = [("89", "M"), ("1", "I"), ("11", "M")]
        >>> assert count_indel_bases(cigar_list) == 1
        true


    Parameters:
        cigar: a CIGAR string or a list of tuple items parsed by parse_cigar() function.

    Returns:
        The number of inserted and deleted bases.
    """
    if isinstance(cigar, str):
        cigar = parse_cigar(cigar)
    return sum([int(c) for c, op in cigar if op in ["I", "D"]])


def count_mismatch_events(md: Union[str, Sequence[str]]) -> int:
    """
    Count the number of mismatch events in a given MD string.

    Mismatches are represented by any non-digit characters, e.g. A, C, G, ^T.

    Examples:
        >>> md_str = "10A3T0T10"
        >>> assert count_mismatch_events(md_str) == 3
        true

        >>> md_list = ["85", "^A", "16"] # an inserted base A on the read
        >>> assert count_mismatch_events(md_list) == 0
        true


    Parameters:
        md: a CIGAR string or a list strings parsed from parse_md() function.

    Returns:
        The number of mismatch events.
    """
    if isinstance(md, str):
        md = parse_md(md)
    return len([e for e in md if e.isalpha()])


@dataclass
class BamArrays:
    rnames: np.ndarray
    rstarts: np.ndarray
    rends: np.ndarray
    mqs: np.ndarray
    propers: np.ndarray
    primarys: np.ndarray
    sc_bps: np.ndarray
    mm_ecnt: np.ndarray
    indel_ecnt: np.ndarray
    qnames: np.ndarray

    @classmethod
    def create(cls, chunk_size: int) -> "BamArrays":
        attrs = {}
        for k in inspect.get_annotations(cls).keys():
            match k:
                case "rnames":
                    attrs[k] = np.empty(chunk_size, dtype="object")
                case "rstarts":
                    attrs[k] = np.empty(chunk_size, dtype=np.int32)
                case "rends":
                    attrs[k] = np.empty(chunk_size, dtype=np.int32)
                case "mqs":
                    attrs[k] = np.empty(chunk_size, dtype=np.uint8)
                case "propers":
                    attrs[k] = np.empty(chunk_size, dtype=bool)
                case "primarys":
                    attrs[k] = np.empty(chunk_size, dtype=bool)
                case "mm_ecnt":
                    attrs[k] = np.empty(chunk_size, dtype=np.int16)
                case "indel_ecnt":
                    attrs[k] = np.empty(chunk_size, dtype=np.int16)
                case "sc_bps":
                    attrs[k] = np.empty(chunk_size, dtype=np.int16)
                case "qnames":
                    attrs[k] = np.empty(chunk_size, dtype="object")
                case _:
                    pass
        return cls(**attrs)


# TODO: return base qualities as ndarray
# TODO: use rname index not name
# TODO: filter by read_group
def walk_bam(
    bametadata: BAMetadata,
    contig: Optional[str] = None,
    start: Optional[int] = None,
    stop: Optional[int] = None,
    exclude: int = 3840,
    chunk_size: int = 100_000,
) -> pl.DataFrame:
    # FIXME: write a function to check first
    if contig is not None and contig not in bametadata.references.keys():
        raise KeyError(f"Given {contig=} is not found in the BAM header.")

    if start is not None and start < 0:
        start = 0

    # FIXME: contig can be NoneType
    if stop is not None and stop > bametadata.references[contig]:
        stop = bametadata.references[contig]

    bam_arrays = BamArrays.create(chunk_size)

    chunks: list[pl.DataFrame] = []
    with pysam.AlignmentFile(bametadata.fspath, "rb") as bamf:
        idx = 0
        for aln in bamf.fetch(contig=contig, start=start, stop=stop):
            # Skip alignments if any of the exclude bit is set
            if bool(aln.flag & exclude):
                continue
            if aln.query_qualities is None:
                continue
            if aln.query_name is None:
                continue

            bam_arrays.qnames[idx] = aln.query_name
            bam_arrays.mqs[idx] = aln.mapping_quality
            bam_arrays.propers[idx] = aln.is_proper_pair
            bam_arrays.primarys[idx] = not aln.is_secondary
            bam_arrays.rnames[idx] = aln.reference_name
            bam_arrays.rstarts[idx] = aln.reference_start
            bam_arrays.rends[idx] = (
                aln.reference_end if aln.reference_end else -1
            )
            bam_arrays.sc_bps[idx] = (
                count_soft_clip_bases(aln.cigarstring)
                if aln.cigarstring is not None
                else -1
            )
            bam_arrays.mm_ecnt[idx] = (
                count_mismatch_events(str(aln.get_tag("MD")))
                if aln.has_tag("MD")
                else -1
            )
            bam_arrays.mm_ecnt[idx] = (
                count_indel_events(aln.cigarstring)
                if aln.cigarstring is not None
                else -1
            )
            idx += 1

            if idx == chunk_size:
                chunk = pl.DataFrame(
                    {
                        k: getattr(bam_arrays, k)
                        for k in bam_arrays.__dict__.keys()
                    }
                )
                chunks.append(chunk)
                idx = 0

        if idx > 0:
            chunk = pl.DataFrame(
                {
                    k: getattr(bam_arrays, k)[:idx]
                    for k in bam_arrays.__dict__.keys()
                }
            )
            chunks.append(chunk)

    return pl.concat(chunks)


if __name__ == "__main__":
    bam = "/Users/simo/work/bio/resource/hla/1kg/NA18740/class1/realigner/NA18740.hla.realn.so.bam"
    bametadata = BAMetadata(bam)
    contig = "hla_a_01_01_01_01"
    df = walk_bam(bametadata, exclude=3584, chunk_size=100_000)
    print(df.shape)
    print(df.head())
    print(df.tail())
