from __future__ import annotations

from ._io import _PathLike, get_parent_dir, make_dir, parse_path
from ._version import version as __version__
from .api_server import request_api_server
from .bam import (
    BAMetadata,
    count_indel_bases,
    count_indel_events,
    count_mismatch_events,
    count_soft_clip_bases,
    count_unaligned_events,
    parse_cigar,
    parse_md,
)

__all__ = [
    "_PathLike",
    "BAMetadata",
    "make_dir",
    "parse_path",
    "get_parent_dir",
    "request_api_server",
    "parse_md",
    "parse_cigar",
    "count_unaligned_events",
    "count_indel_events",
    "count_mismatch_events",
    "count_soft_clip_bases",
    "count_indel_bases",
    "__version__",
]
