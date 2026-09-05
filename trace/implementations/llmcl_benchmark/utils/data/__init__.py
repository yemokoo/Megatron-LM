"""TRACE data helpers plus the local SLoRA-compatible collators.

The vendored TRACE dataset implementation remains the source of truth for
``data_utils`` and ``raw_datasets``.  Extending this package path avoids a
second, drifting copy while allowing this implementation to provide its
training-format-specific collators.
"""

from pathlib import Path


_TRACE_DATA_PACKAGE = (
    Path(__file__).resolve().parents[4] / "upstream" / "TRACE" / "utils" /
    "data"
)
if not _TRACE_DATA_PACKAGE.is_dir():
    raise ImportError(
        f"vendored TRACE data package is missing: {_TRACE_DATA_PACKAGE}")
__path__.append(str(_TRACE_DATA_PACKAGE))
