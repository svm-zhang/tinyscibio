import itertools
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Union

import pysam

from libscibio import parse_path


@dataclass
class BAMetadata:
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

    def seqnames(self) -> list[str]:
        return [k for r in self.references for k in r.keys()]


def parse_cigar(cigar: str) -> list[tuple[str, str]]:
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
    if isinstance(cigar, str):
        cigar = parse_cigar(cigar)
    return sum([int(c) for c, op in cigar if op == "S"])


def count_unaligned_events(
    cigar: Union[str, Sequence[tuple[str, str]]],
) -> int:
    """
    Count the number of unaligned events in the given CIGAR string.
    Unaligned events include: I, D, S, and H
    """
    if isinstance(cigar, str):
        cigar = parse_cigar(cigar)
    return len([op for _, op in cigar if op in ["I", "D", "S", "H"]])


def count_indel_events(cigar: Union[str, Sequence[tuple[str, str]]]) -> int:
    """Count the number of Is and Ds event in the given cigar string"""
    if isinstance(cigar, str):
        cigar = parse_cigar(cigar)
    return len([op for _, op in cigar if op in ["I", "D"]])


def count_indel_bases(cigar: Union[str, Sequence[tuple[str, str]]]) -> int:
    """Count the length of indels in the given CIGAR string"""
    if isinstance(cigar, str):
        cigar = parse_cigar(cigar)
    return sum([int(c) for c, op in cigar if op in ["I", "D"]])


def count_mismatch_events(md: Union[str, Sequence[str]]) -> int:
    """Count the number of mismatch events in the given MD string"""
    if isinstance(md, str):
        md = parse_md(md)
    return len([e for e in md if e.isalpha()])
