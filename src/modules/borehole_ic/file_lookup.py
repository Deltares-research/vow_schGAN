"""Find CPT / borehole data files by name inside a folder tree.

Names are matched case-insensitively against the full file stem
(e.g. "CPT000000061801_IMBRO_A") and against its leading BRO id
("CPT000000061801" / "BHR000000397320"), so a name with or without the
"_IMBRO_A"-style suffix finds the same file. When several files match, the
first one in sorted path order is used and the number of candidates is reported.
"""

import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

_ID_PATTERN = re.compile(r"^((?:CPT|BHR)\d+)", re.IGNORECASE)


def index_files(root: Path, suffixes: Iterable[str]) -> Dict[str, List[Path]]:
    """Map lookup keys (full stem, and leading CPT/BHR id) -> sorted list of matching files under `root`."""
    suffixes = {s.lower() for s in suffixes}
    index: Dict[str, List[Path]] = {}
    for path in sorted(Path(root).rglob("*")):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        keys = {path.stem.upper()}
        match = _ID_PATTERN.match(path.stem)
        if match:
            keys.add(match.group(1).upper())
        for key in keys:
            index.setdefault(key, []).append(path)
    return index


def find_file(index: Dict[str, List[Path]], name: str) -> Tuple[Optional[Path], int]:
    """First file whose stem or leading id matches `name`; returns (path or None, number of candidates)."""
    key = str(name).strip().upper()
    candidates = index.get(key)
    if not candidates:
        match = _ID_PATTERN.match(key)
        candidates = index.get(match.group(1).upper()) if match else None
    return (candidates[0], len(candidates)) if candidates else (None, 0)


def leading_id(name: str) -> Optional[str]:
    match = _ID_PATTERN.match(str(name).strip())
    return match.group(1).upper() if match else None
