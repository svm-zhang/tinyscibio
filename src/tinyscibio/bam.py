import itertools
import re
from collections.abc import Sequence, Set
from dataclasses import dataclass, field, fields
from typing import Any, Optional, Union

import numpy as np
import numpy.typing as npt
import polars as pl
import pysam

from ._io import parse_path

_MAX_CHUNK_SIZE = 1_000_000


@dataclass
class BAMetadata:
    """
    A BAMetadata object holds metadata about a given BAM file.

    Examples:
        Let us use a hypothetical coordinate-sorted BAM file with one read
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
    references: list[dict[str, int]] = field(init=False, default_factory=list)
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

        self.references = [
            {s["SN"]: int(s["LN"])} for s in sqs if "SN" in s and "LN" in s
        ]

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

    def seqmap(self) -> dict[str, int]:
        return {k: v for r in self.references for k, v in r.items()}

    def seqnames(self) -> list[str]:
        return [k for r in self.references for k in r.keys()]

    def idx2seqname(self) -> dict[int, str]:
        return {i: k for i, r in enumerate(self.references) for k in r.keys()}

    def seqname2idx(self) -> dict[str, int]:
        return {k: i for i, r in enumerate(self.references) for k in r.keys()}


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
class _BamArrays:
    """
    Arrays of a pre-defined size to store various aspects of alignments.

    walk_bam() function uses _BamArrays to store relevant alignment data
    during its traverse.

    Attributes:
        rnames: Array of indices of reference that a read is mapped to.
        rstarts: Array of 0-based positions marking the beginning of an
                 alignment on the reference.
        rends: Array of positions on the reference marking the end of alignments.
        mqs: Array of mapping qualities.
        propers: Array of whether alignments are marked as properly mapped.
        primarys: Array of whether alignments are marked as primary.
        sc_bps: Array of number of soft-clipped bases.
        qnames: Array of query/read names (not initiated by default).
        mm_ecnt: Array of number of mismatch events (not initiated by default).
        indel_ecnt: Array of number of indel events (not initiated by default).
        bqs: Array of array of base qualities (not initiated by default).
        mds: Array of parsed MDs (not initiated by default).
    """

    rnames: npt.NDArray[np.uint16]
    rstarts: npt.NDArray[np.int32]
    rends: npt.NDArray[np.int32]
    mqs: npt.NDArray[np.uint8]
    propers: npt.NDArray[np.bool]
    primarys: npt.NDArray[np.bool]
    sc_bps: npt.NDArray[np.int16]
    qnames: npt.NDArray[np.object_]
    mm_ecnt: npt.NDArray[np.int16]
    indel_ecnt: npt.NDArray[np.int16]
    bqs: npt.NDArray[np.uint8]
    mds: npt.NDArray[np.object_]

    @classmethod
    def create(
        cls,
        chunk_size: int,
        with_ecnt: bool = False,
        with_bq: bool = False,
        with_md: bool = False,
        with_qname: bool = False,
    ) -> "_BamArrays":
        """
        Create a _BamArrays object of pre-defined size.

        By default, not all fields of _BamArrays object will be initialized.
        The optional ones can be additionally initialized by using the
        parameters prefixed with 'with_'.

        The maximum array size allowed is defined by _MAX_CHUNK_SIZE, which
        is 1_000_000.

        Examples:

            Create _BamArrays object with pre-defined size of 1000
            >>> from tinyscibio.bam import _BamArrays
            >>> chunk_size = 1_000
            >>> bamarrays = _BamArrays.create(chunk_size=chunk_size)

            Create _BamArrays object with pre-defined size of 1000, and also
            initialize _BamArrays.mm_ecnt and _BamArrays.indel_ecnt arrays.
            >>> from tinyscibio.bam import _BamArrays
            >>> chunk_size = 1_000
            >>> bamarrays = _BamArrays.create(chunk_size=chunk_size, with_ecnt=True)

        Parameters:
            chunk_size: Pre-defined size of arrays.
            with_ecnt: Initialize _BamArrays.mm_ecnt and _BamArrays.indel_ecnt arrays.
            with_bq: Initialize _BamArrays.bqs.
            with_md: Initialize _BamArrays.mds.
            with_qname: Initialize _BamArrays.qnames.

        Raises:
            ValueError: When given chunk_size is not a positive number.
        """
        if chunk_size <= 0:
            raise ValueError(f"Given {chunk_size=} must be positive number.")
        # Put a upper cap on maximum chunk_size allowed
        chunk_size = (
            _MAX_CHUNK_SIZE if chunk_size > _MAX_CHUNK_SIZE else chunk_size
        )

        attrs: dict[str, Any] = {}
        for k in fields(cls):
            f_name = k.name
            match f_name:
                case "rnames":
                    attrs[f_name] = np.zeros(chunk_size, dtype=np.uint16)
                case "rstarts":
                    attrs[f_name] = np.zeros(chunk_size, dtype=np.int32)
                case "rends":
                    attrs[f_name] = np.zeros(chunk_size, dtype=np.int32)
                case "mqs":
                    attrs[f_name] = np.zeros(chunk_size, dtype=np.uint8)
                case "propers":
                    attrs[f_name] = np.full(chunk_size, False, dtype=bool)
                case "primarys":
                    attrs[f_name] = np.full(chunk_size, False, dtype=bool)
                case "sc_bps":
                    attrs[f_name] = np.zeros(chunk_size, dtype=np.int16)
                case "qnames":
                    attrs[f_name] = (
                        np.empty(chunk_size, dtype="object")
                        if with_qname
                        else np.array([])
                    )
                case "mm_ecnt" | "indel_ecnt":
                    attrs[f_name] = (
                        np.zeros(chunk_size, dtype=np.int16)
                        if with_ecnt
                        else np.array([])
                    )
                case "bqs":
                    attrs[f_name] = (
                        np.empty(chunk_size, dtype=np.ndarray)
                        if with_bq
                        else np.array([])
                    )
                case "mds":
                    attrs[f_name] = (
                        np.empty(chunk_size, dtype=np.ndarray)
                        if with_md
                        else np.array([])
                    )
                case _:  # pragma: no cover
                    continue
        return cls(**attrs)

    def df(self, idx: int) -> pl.DataFrame:
        """
        Collect arrays of _BamArrays object into a polars DataFrame.

        See doc of walk_bam for example dataframes returned by this function.

        Parameters:
            idx: Values of arrays up to this index will be collected.

        Returns:
            polars.DataFrame

        Raises:
            ValueError: When the given idx parameter is negatvie.
            IndexError: When the given idx parameter is larger than
                        the size of arrays.
        """
        if idx < 0:
            raise ValueError(f"Given {idx=} must be positive")
        if idx > self.rnames.size:
            raise IndexError(
                f"Given {idx=} is larger than the size of "
                f"bamarray {self.qnames.size}. "
                "Try to create BamArrays object with bigger chunk_size."
            )
        return pl.DataFrame(
            {
                k: getattr(self, k)[:idx].copy()
                for k in self.__dict__.keys()
                if getattr(self, k).size > 0
            }
        )


def parse_region(
    region_string: str, one_based: bool = True
) -> tuple[str, int | None, int | None]:
    """
    Parse a given samtools-style region string into a tuple of rname,
    start, and end.

    Examples:
        >>> from tinyscibio import parse_region
        >>> region_str = "chr1"
        >>> assert parse_region(region_str) == ("chr1", None, None)
        true

        >>> region_str = "chr1:100"
        >>> assert parse_region(region_str) == ("chr1", 100, None)
        true

        >>> region_str = "chr1:100-1000"
        >>> assert parse_region(region_str) == ("chr1", 100, 1000)
        true

        >>> region_str = "chr1:100-1000"
        >>> assert parse_region(region_str, one_based=True) == ("chr1", 99, 1000)
        true

    Parameters:
        region_string: user-provide string following samtools-style
                       region definition.
        one_based: whether or not coordinate in the given string is one-based.

    Returns:
        A tuple of (rname, start, end).

    Raises:
        ValueError: When parser finds no matching pattern, or when start
                    is larger than end coordinate in the given region string.
    """
    # The minimal presentation should be region/contig/chrom name
    m = re.match(r"^([\w\.]+)?(?::(\d+)-?)?(\d+)?$", region_string)

    if not m:
        raise ValueError(f"Invalid region string format: {region_string}")

    rname, start, end = m.groups()
    # I do not need to worry about if start and end position to be negative
    # If user provides such string, the re parser above will complain
    if start is not None:
        # if one-based, start = 1 => start = max(0, start-1) => start = 0
        # if one-based, start = 0 => start = max(0, start-1) => start = 0
        # if not one-based, start = 1 => start = max(0, start) => start = 1
        # if not one-based, start = 0 => start = max(0, start) => start = 0
        start = int(start) if not one_based else int(start) - 1
        start = max(0, start)
    end = int(end) if end is not None else end

    if start is not None and end is not None:
        if start > end:
            raise ValueError(
                f"{start=} cannot be larger than {end=}. "
                f"Please check input region string {region_string=}."
            )

    return (rname, start, end)


def walk_bam(
    fspath: str,
    region: str,
    exclude: int = 3840,
    chunk_size: int = 100_000,
    read_groups: Optional[Set[str]] = None,
    return_ecnt: bool = False,
    return_bq: bool = False,
    return_md: bool = False,
    return_qname: bool = False,
) -> pl.DataFrame:
    """
    Traverse BAM file across given region and collect read-level alignment
    summary into a dataframe.

    Input BAM file must be sorted and indexed to make the walk feasible. Value
    to region parameter follows the samtools format, and therefore positions
    are 1-based.

    By default, walk_bam returns a dataframe with 7 columns:

        - rnames: reference index (easy to map back to sequence name)
        - rstarts: 0-based start position on mapped reference
        - rends: same as above but marking the end of an alignment
        - mqs: mapping qualities
        - propers: whether or not a given alignment is properly aligned
        - primarys: whether or not a given alignment is primary
        - sc_bps: # of soft-clipped base pairs

    When return_ecnt is set,

        - mm_ecnt: # of mismatch events per alignment
        - mm_ecnt: # of indel events per alignment

    When return_qname is set,

        - qnames: read name

    When return_bq is set,

        - bqs: base qualities in np.ndarray per alignment

    When return_md is set,

        - mds: parsed MD in np.ndarray per alignment

    Examples:
        >>> from tinyscibio import BAMetadata, walk_bam
        >>> bam_fspath = "example.bam"
        >>> bametadata = BAMetadata(bam_fspath)

        Traverse entire region of HLA_A_01_01_01_01 allele:
        >>> region_str = "hla_a_01_01_01_01"
        >>> df = walk_bam(bam_fspath, region_str)
        >>> df = df.with_columns(
            pl.col("rnames").replace_strict(bametadata.idx2seqnames())
        )
        >>> df.head()
        shape: (5, 7)
        ┌───────────────────┬─────────┬───────┬─────┬─────────┬──────────┬────────┐
        │ rnames            ┆ rstarts ┆ rends ┆ mqs ┆ propers ┆ primarys ┆ sc_bps │
        │ ---               ┆ ---     ┆ ---   ┆ --- ┆ ---     ┆ ---      ┆ ---    │
        │ str               ┆ i32     ┆ i32   ┆ u8  ┆ bool    ┆ bool     ┆ i16    │
        ╞═══════════════════╪═════════╪═══════╪═════╪═════════╪══════════╪════════╡
        │ hla_a_01_01_01_01 ┆ 4       ┆ 104   ┆ 0   ┆ true    ┆ false    ┆ 0      │
        │ hla_a_01_01_01_01 ┆ 4       ┆ 102   ┆ 0   ┆ true    ┆ false    ┆ 0      │
        │ hla_a_01_01_01_01 ┆ 128     ┆ 229   ┆ 0   ┆ true    ┆ false    ┆ 0      │
        │ hla_a_01_01_01_01 ┆ 287     ┆ 388   ┆ 0   ┆ true    ┆ false    ┆ 0      │
        │ hla_a_01_01_01_01 ┆ 294     ┆ 395   ┆ 0   ┆ true    ┆ false    ┆ 0      │
        └───────────────────┴─────────┴───────┴─────┴─────────┴──────────┴────────┘

        Traverse the same region and collect more alignment summaries:
        >>> df = walk_bam(bam_fspath, region_str, return_ecnt=True, return_qname=True)
        >>> df = df.with_columns(
            pl.col("rnames").replace_strict(bametadata.idx2seqnames())
        )
        >>> df.head()
        shape: (5, 10)
        ┌───────────────────┬─────────┬───────┬─────┬───┬────────┬────────────────────┬─────────┬────────────┐
        │ rnames            ┆ rstarts ┆ rends ┆ mqs ┆ … ┆ sc_bps ┆ qnames             ┆ mm_ecnt ┆ indel_ecnt │
        │ ---               ┆ ---     ┆ ---   ┆ --- ┆   ┆ ---    ┆ ---                ┆ ---     ┆ ---        │
        │ str               ┆ i32     ┆ i32   ┆ u8  ┆   ┆ i16    ┆ str                ┆ i16     ┆ i16        │
        ╞═══════════════════╪═════════╪═══════╪═════╪═══╪════════╪════════════════════╪═════════╪════════════╡
        │ hla_a_01_01_01_01 ┆ 4       ┆ 104   ┆ 0   ┆ … ┆ 0      ┆ SRR702076.16980922 ┆ 0       ┆ 1          │
        │ hla_a_01_01_01_01 ┆ 4       ┆ 102   ┆ 0   ┆ … ┆ 0      ┆ SRR702076.16980922 ┆ 0       ┆ 1          │
        │ hla_a_01_01_01_01 ┆ 128     ┆ 229   ┆ 0   ┆ … ┆ 0      ┆ SRR702076.10444434 ┆ 6       ┆ 0          │
        │ hla_a_01_01_01_01 ┆ 287     ┆ 388   ┆ 0   ┆ … ┆ 0      ┆ SRR702076.10444434 ┆ 10      ┆ 0          │
        │ hla_a_01_01_01_01 ┆ 294     ┆ 395   ┆ 0   ┆ … ┆ 0      ┆ SRR702076.2819225  ┆ 0       ┆ 0          │
        └───────────────────┴─────────┴───────┴─────┴───┴────────┴────────────────────┴─────────┴────────────┘

        Traverse the HLA_A_01_01_01_01 allele starting at position 100. Note
        that walk_bam has different behavior from pysam.AlignmentFile.fetch()
        function. It only grabs alignments from postion 100 and onwards.
        >>> region_str = "hla_a_01_01_01_01:100"
        >>> df = walk_bam(bam_fspath, region_str, return_ecnt=True)
        >>> df = df.with_columns(
            pl.col("rnames").replace_strict(bametadata.idx2seqnames())
        )
        >>> df.head()
        shape: (5, 9)
        ┌───────────────────┬─────────┬───────┬─────┬───┬──────────┬────────┬─────────┬────────────┐
        │ rnames            ┆ rstarts ┆ rends ┆ mqs ┆ … ┆ primarys ┆ sc_bps ┆ mm_ecnt ┆ indel_ecnt │
        │ ---               ┆ ---     ┆ ---   ┆ --- ┆   ┆ ---      ┆ ---    ┆ ---     ┆ ---        │
        │ str               ┆ i32     ┆ i32   ┆ u8  ┆   ┆ bool     ┆ i16    ┆ i16     ┆ i16        │
        ╞═══════════════════╪═════════╪═══════╪═════╪═══╪══════════╪════════╪═════════╪════════════╡
        │ hla_a_01_01_01_01 ┆ 128     ┆ 229   ┆ 0   ┆ … ┆ false    ┆ 0      ┆ 6       ┆ 0          │
        │ hla_a_01_01_01_01 ┆ 287     ┆ 388   ┆ 0   ┆ … ┆ false    ┆ 0      ┆ 10      ┆ 0          │
        │ hla_a_01_01_01_01 ┆ 294     ┆ 395   ┆ 0   ┆ … ┆ false    ┆ 0      ┆ 0       ┆ 0          │
        │ hla_a_01_01_01_01 ┆ 341     ┆ 442   ┆ 0   ┆ … ┆ false    ┆ 0      ┆ 0       ┆ 0          │
        │ hla_a_01_01_01_01 ┆ 638     ┆ 724   ┆ 0   ┆ … ┆ false    ┆ 0      ┆ 9       ┆ 1          │
        └───────────────────┴─────────┴───────┴─────┴───┴──────────┴────────┴─────────┴────────────┘

    Parameters:
        fspath: Path pointing to the BAM file.
        region: Region string in samtools style, e.g. chr1:100-1000.
        exclude: Skip alignments whose flags are set with any bits defined.
        chunk_size: Maximum size of arrays to be held in memory at one time.
        read_groups: Set of read groups whose alignments are retrieved.
        return_ecnt: Whether or not returning number of mismatch+indel events.
        return_bq: Whether or not returning base qualities.
        return_md: Whether or not returning parsed MD tuples.
        return_qname: Whether or not returning query/read names.

    Returns:
        Collection of summaries of read alignments in dataframe.

    """
    bam_arrays = _BamArrays.create(
        chunk_size,
        with_ecnt=return_ecnt,
        with_bq=return_bq,
        with_md=return_md,
        with_qname=return_qname,
    )

    rname, start, end = parse_region(region)
    # if only rname is given, starting from beginning
    start = 0 if start is None else start

    chunks: list[pl.DataFrame] = []
    with pysam.AlignmentFile(fspath, "rb") as bamf:
        idx = 0
        for aln in bamf.fetch(contig=rname, start=start, stop=end):
            # Make sure to only grab alignments at and after given start
            # position and before given end position
            if aln.reference_start < start:
                continue
            if aln.query_name is None:
                continue
            # Skip records whose read group is not defined in read_groups
            if read_groups is not None:
                rg = aln.get_tag("RG") if aln.has_tag("RG") else ""
                if rg not in read_groups:
                    continue
            # Skip alignments if any of the exclude bit is set
            if bool(aln.flag & exclude):
                continue

            bam_arrays.mqs[idx] = aln.mapping_quality
            bam_arrays.propers[idx] = aln.is_proper_pair
            bam_arrays.primarys[idx] = not aln.is_secondary
            bam_arrays.rnames[idx] = aln.reference_id
            bam_arrays.rstarts[idx] = aln.reference_start
            bam_arrays.rends[idx] = (
                aln.reference_end if aln.reference_end is not None else -1
            )
            bam_arrays.sc_bps[idx] = (
                count_soft_clip_bases(aln.cigarstring)
                if aln.cigarstring is not None
                else -1
            )
            if return_ecnt:
                bam_arrays.mm_ecnt[idx] = (
                    count_mismatch_events(str(aln.get_tag("MD")))
                    if aln.has_tag("MD")
                    else -1
                )
                bam_arrays.indel_ecnt[idx] = (
                    count_indel_events(aln.cigarstring)
                    if aln.cigarstring is not None
                    else -1
                )
            if return_md:
                bam_arrays.mds[idx] = (
                    parse_md(str(aln.get_tag("MD")))
                    if aln.has_tag("MD")
                    else np.array([])
                )
            if return_bq:
                bam_arrays.bqs[idx] = (
                    np.array(aln.query_qualities, dtype=np.uint8)
                    if aln.query_qualities is not None
                    else np.array([])
                )
            if return_qname:
                bam_arrays.qnames[idx] = aln.query_name
            idx += 1

            if idx == chunk_size:
                chunks.append(bam_arrays.df(idx))
                idx = 0

        if idx > 0:
            chunks.append(bam_arrays.df(idx))

    # handle cases where no df returned, e.g. when no records have rg in
    # the defined read_groups
    if not chunks:
        return bam_arrays.df(idx=0)

    return pl.concat(chunks, rechunk=True)
