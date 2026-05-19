"""
Convert LENA .its files to .rttm files compatible with the BabAR pipeline.

LENA speaker codes are mapped to BabAR/VTC labels:
    CHN              -> KCHI   (key child)
    CXN              -> OCH    (other child)
    FAN              -> FEM    (female adult)
    MAN              -> MAL    (male adult)
    CHF, CXF, FAF, MAF, SIL, NON, NOF, OLN, OLF, TVN, TVF -> skipped (silence/far)

ITS timing: each <Segment> has startTime / endTime attributes in ISO 8601
duration format (e.g. PT12.34S) that are absolute (file-relative), even
across multiple <Recording> blocks. Times are used directly as-is.
"""

import logging
import re
from pathlib import Path
from xml.etree import ElementTree as ET

logger = logging.getLogger("babar.its_to_rttm")

# Minimum segment duration in seconds. Segments shorter than this are dropped
# to avoid HuBERT conv kernel errors on near-empty audio tensors.
MIN_SEGMENT_DURATION: float = 0.1

# ---------------------------------------------------------------------------
# Speaker label mapping: LENA code -> BabAR/VTC label
# "Near" variants (xN) are mapped to speech labels.
# "Far" variants (xF) and non-speech labels are mapped to None (skipped).
# ---------------------------------------------------------------------------
LENA_TO_BABAR: dict[str, str | None] = {
    # Key child
    "CHN": "KCHI",
    # Other child
    "CXN": "OCH",
    # Female adult
    "FAN": "FEM",
    # Male adult
    "MAN": "MAL",
    # Far variants — treated as silence, skip
    "CHF": None,
    "CXF": None,
    "FAF": None,
    "MAF": None,
    # Non-speech — skip
    "SIL": None,
    "NON": None,
    "NOF": None,
    "OLN": None,
    "OLF": None,
    "TVN": None,
    "TVF": None,
}

# Matches ISO 8601 durations like PT0.00S, PT1M2.34S, PT1H2M3.45S
_ISO8601_RE = re.compile(
    r"PT(?:(\d+)H)?(?:(\d+)M)?(?:([\d.]+)S)?$"
)


def _parse_time(value: str) -> float:
    """Parse an ISO 8601 duration string (PTxxx) or plain float string to seconds."""
    value = value.strip()
    m = _ISO8601_RE.match(value)
    if m:
        hours = float(m.group(1) or 0)
        minutes = float(m.group(2) or 0)
        seconds = float(m.group(3) or 0)
        return hours * 3600 + minutes * 60 + seconds
    # Fallback: plain float (older LENA versions)
    return float(value)


def _warn_unknown(code: str, seen: set[str]) -> None:
    if code not in seen:
        logger.warning(f"Unknown LENA speaker code '{code}' – skipping.")
        seen.add(code)


def its_to_rttm_lines(its_path: Path) -> list[str]:
    """
    Parse a LENA .its file and return RTTM-formatted lines.

    The RTTM format used by pyannote / VTC is:
        SPEAKER <file_id> 1 <onset> <duration> <NA> <NA> <label> <NA> <NA>

    Times in the output are in seconds from the start of the audio file.
    """
    try:
        tree = ET.parse(its_path)
    except ET.ParseError as exc:
        raise ValueError(f"Could not parse {its_path}: {exc}") from exc

    root = tree.getroot()
    file_id = its_path.stem
    rttm_lines: list[str] = []
    unknown_codes: set[str] = set()

    # Segment startTime/endTime are absolute (file-relative) in LENA ITS files,
    # even across multiple <Recording> blocks. Iterate all segments directly.
    for segment in root.iter("Segment"):
        lena_code = segment.get("spkr", "")
        if not lena_code:
            continue

        babar_label = LENA_TO_BABAR.get(lena_code, "UNKNOWN")
        if babar_label == "UNKNOWN":
            _warn_unknown(lena_code, unknown_codes)
            continue
        if babar_label is None:
            continue

        seg_start = _parse_time(segment.get("startTime", "PT0S"))
        seg_end = _parse_time(segment.get("endTime", segment.get("startTime", "PT0S")))
        duration = seg_end - seg_start

        if duration < MIN_SEGMENT_DURATION:
            continue

        rttm_lines.append(
            f"SPEAKER {file_id} 1 {seg_start:.3f} {duration:.3f} "
            f"<NA> <NA> {babar_label} <NA> <NA>"
        )

    return rttm_lines


def convert_file(its_path: Path, output_dir: Path) -> Path:
    """Convert a single .its file to a .rttm file."""
    rttm_lines = its_to_rttm_lines(its_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    rttm_path = output_dir / f"{its_path.stem}.rttm"

    with open(rttm_path, "w", encoding="utf-8") as f:
        f.write("\n".join(rttm_lines))
        if rttm_lines:
            f.write("\n")

    n_speech = len(rttm_lines)
    logger.info(f"  {its_path.name} -> {rttm_path.name}  ({n_speech} speech segments)")
    return rttm_path