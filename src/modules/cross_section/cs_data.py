"""Cross-section data: read the guide and profile line, place each test on the line, build its Ic profile.

Conventions:
- The line runs from A (first point of the profile CSV) to B (last point).
  Each test is projected perpendicularly onto the straight line A-B:
  `distance_along_m` is measured from A, `offset_m` is the perpendicular
  distance (positive = left of the direction A->B). All tests are kept, also
  those projecting before A or beyond B (flagged in `beyond_line_ends`).
- Vertical axis: elevation in m NAP, each test at its own ground level.
- Real CPTs: geolib_plus Ic (same interpretation as the pair evaluation).
- Boreholes (names starting with "BHR"): synthetic Ic from the borehole-to-Ic
  workflow, single realization with the default seed (42). If the borehole XML
  has no surface elevation, the profile's ground elevation at the projected
  position is used and flagged in `surface_from_profile`.
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # add src/ to sys.path

from modules.borehole_ic.file_lookup import find_file, leading_id  # noqa: E402
from modules.borehole_ic.lengkeek_reference import build_reference_lengkeek_profile  # noqa: E402
from modules.borehole_ic.profile_builder import build_synthetic_profile  # noqa: E402
from modules.borehole_ic.xml_readers import read_borehole_xml  # noqa: E402

COORDINATE_CONFLICT_TOLERANCE_M = 0.5  # distinct coordinates for one name farther apart than this are reported

# --- SchemaGAN frame (design decisions, applied to every cross-section) ---------------
DESIGN_BOTTOM_M_NAP = -5.0  # bottom row of the SchemaGAN rectangle
SCHEMAGAN_N_COLS = 512
SCHEMAGAN_N_ROWS = 32


@dataclass
class SchemaganFrame:
    """The SchemaGAN rectangle in cross-section coordinates (distance along the line, m NAP).

    left/right = start (A, 0 m) and end (B) of the profile line; top = highest
    ground level of the drawn tests (CPTs and boreholes); bottom = DESIGN_BOTTOM_M_NAP.
    """
    left_m: float
    right_m: float
    top_m_nap: float
    bottom_m_nap: float
    n_cols: int = SCHEMAGAN_N_COLS
    n_rows: int = SCHEMAGAN_N_ROWS
    top_source: str = ""

    @property
    def col_width_m(self) -> float:
        return (self.right_m - self.left_m) / self.n_cols

    @property
    def row_height_m(self) -> float:
        return (self.top_m_nap - self.bottom_m_nap) / self.n_rows


def schemagan_frame(line, tests: pd.DataFrame) -> SchemaganFrame:
    drawn = tests[(tests["status"] == "ok") & tests["surface_elevation_m_nap"].notna()] if "surface_elevation_m_nap" in tests else tests.iloc[0:0]
    if not len(drawn):
        raise ValueError("no test with a ground level - cannot define the top of the SchemaGAN frame")
    top_row = drawn.loc[drawn["surface_elevation_m_nap"].idxmax()]
    top = float(top_row["surface_elevation_m_nap"])
    if top <= DESIGN_BOTTOM_M_NAP:
        raise ValueError(f"highest test ground level {top:.2f} m NAP is not above the design bottom {DESIGN_BOTTOM_M_NAP} m NAP")
    return SchemaganFrame(0.0, line.length_m, top, DESIGN_BOTTOM_M_NAP, top_source=str(top_row["name"]))


# --- guide and profile -------------------------------------------------------

def read_guide(path: Path) -> pd.DataFrame:
    guide = pd.read_csv(path, dtype=str, skipinitialspace=True)
    guide.columns = [c.strip().lower() for c in guide.columns]
    missing = {"cs_id", "cs_name", "cs_profile", "cpts_in_cs"} - set(guide.columns)
    if missing:
        raise ValueError(f"{path} is missing column(s) {sorted(missing)}; found {list(guide.columns)}")
    return guide


def split_names(cell: str) -> List[str]:
    return [n.strip().strip('"') for n in str(cell).split(",") if n.strip().strip('"')]


def find_profile(cs_root: Path, filename: str) -> Path:
    matches = sorted(Path(cs_root).rglob(filename.strip()))
    if not matches:
        raise FileNotFoundError(f"profile file {filename!r} not found under {cs_root}")
    if len(matches) > 1:
        raise ValueError(f"profile file {filename!r} found more than once: {[str(m) for m in matches]}")
    return matches[0]


def fill_elevation_gaps(profile: pd.DataFrame) -> Tuple[pd.DataFrame, List[Tuple[float, float]]]:
    """Fill missing ground elevations by linear interpolation along the line.

    Every internal gap is bridged with a straight line between the last
    measured point before it and the first measured point after it; a gap at
    the start or end of the profile takes the nearest measured value (flat).
    Keeps the original values in `elev_nap_raw` and flags filled rows in
    `elev_filled`. Returns the profile and the list of gaps as (start, end) distances.
    """
    p = profile.copy()
    raw = pd.to_numeric(p["elev_nap"], errors="coerce")
    p["elev_nap_raw"] = raw
    p["elev_filled"] = raw.isna()
    measured = ~p["elev_filled"]
    if measured.sum() == 0:
        raise ValueError("profile has no ground elevations at all")
    p["elev_nap"] = np.interp(p["dist"], p.loc[measured, "dist"], raw[measured])

    gaps = []
    runs = (p["elev_filled"] != p["elev_filled"].shift()).cumsum()
    for _, run in p[p["elev_filled"]].groupby(runs[p["elev_filled"]]):
        gaps.append((float(run["dist"].min()), float(run["dist"].max())))
    return p, gaps


@dataclass
class ProfileLine:
    profile: pd.DataFrame  # columns dist, x, y[, elev_nap (gaps filled), elev_nap_raw, elev_filled]
    a: np.ndarray
    b: np.ndarray
    unit: np.ndarray
    normal: np.ndarray  # unit normal pointing left of A->B
    length_m: float
    max_deviation_m: float  # largest distance of a profile point from the straight line A-B
    elevation_gaps: List[Tuple[float, float]]  # (start, end) distances of filled stretches

    def project(self, x: float, y: float) -> Tuple[float, float]:
        v = np.array([x, y]) - self.a
        return float(v @ self.unit), float(v @ self.normal)

    def ground_at(self, distance_along_m: float) -> float:
        """Profile ground elevation (m NAP) at a distance along the line; NaN outside the profile."""
        p = self.profile
        if "elev_nap" not in p or not (p["dist"].min() <= distance_along_m <= p["dist"].max()):
            return np.nan
        return float(np.interp(distance_along_m, p["dist"], p["elev_nap"]))


def read_profile_line(path: Path) -> ProfileLine:
    p = pd.read_csv(path, sep=None, engine="python")
    p.columns = [c.strip().lower() for c in p.columns]
    if not {"x", "y"} <= set(p.columns):
        raise ValueError(f"{path} needs x and y columns; found {list(p.columns)}")
    xy = p[["x", "y"]].to_numpy(dtype=float)
    if "dist" not in p:
        p["dist"] = np.r_[0.0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
    gaps = []
    if "elev_nap" in p:
        p, gaps = fill_elevation_gaps(p)
    a, b = xy[0], xy[-1]
    length = float(np.linalg.norm(b - a))
    if length == 0:
        raise ValueError(f"{path}: first and last profile points coincide")
    unit = (b - a) / length
    normal = np.array([-unit[1], unit[0]])
    deviation = float(np.abs((xy - a) @ normal).max())
    return ProfileLine(p, a, b, unit, normal, length, deviation, gaps)


# --- coordinates ---------------------------------------------------------------

def load_coordinate_index(coords_dir: Path) -> Dict[str, List[Tuple[float, float, str, bool]]]:
    """Key -> [(x, y, source file, exact_name_match)] from every CSV in `coords_dir` with name/x/y columns."""
    index: Dict[str, List[Tuple[float, float, str, bool]]] = {}
    for path in sorted(Path(coords_dir).glob("*.csv")):
        try:
            df = pd.read_csv(path, sep=None, engine="python")
        except Exception:
            continue
        cols = {c.strip().lower(): c for c in df.columns}
        if not {"name", "x", "y"} <= set(cols):
            continue
        for name, x, y in zip(df[cols["name"]], pd.to_numeric(df[cols["x"]], errors="coerce"),
                              pd.to_numeric(df[cols["y"]], errors="coerce")):
            if pd.isna(name) or pd.isna(x) or pd.isna(y):
                continue
            full = str(name).strip().upper()
            index.setdefault(full, []).append((float(x), float(y), path.name, True))
            lid = leading_id(full)
            if lid and lid != full:
                index.setdefault(lid, []).append((float(x), float(y), path.name, False))
    return index


def lookup_coordinates(index, name: str) -> Tuple[Optional[float], Optional[float], str]:
    """Coordinates for `name` (exact-name matches preferred over id matches); the note reports conflicts."""
    key = name.strip().upper()
    hits = list(index.get(key, []))
    lid = leading_id(key)
    if lid:
        hits += [h for h in index.get(lid, []) if h not in hits]
    if not hits:
        return None, None, "not found in coordinate CSVs"
    hits.sort(key=lambda h: (not h[3], h[2]))
    x, y, source, _ = hits[0]
    spread = max(np.hypot(h[0] - x, h[1] - y) for h in hits)
    note = source
    if spread > COORDINATE_CONFLICT_TOLERANCE_M:
        note += f" (WARNING: other coordinate CSVs differ by up to {spread:.1f} m)"
    return x, y, note


# --- Ic profiles -------------------------------------------------------------------

def test_kind(name: str) -> str:
    return "borehole" if name.strip().upper().startswith("BHR") else "cpt"


def build_ic_profile(kind: str, path: Path, fallback_surface_m_nap: float) -> Tuple[pd.DataFrame, dict]:
    """Ic profile (depth_m, elevation_m_nap, ic, class_code) for one test, plus metadata."""
    if kind == "cpt":
        ref = build_reference_lengkeek_profile(path)
        df = pd.DataFrame({"depth_m": ref["depth_m"], "elevation_m_nap": ref["elevation_m_nap"],
                           "ic": ref["reference_ic_geolib_plus"], "class_code": None})
        surface = float(ref["elevation_m_nap"].iloc[0] + ref["depth_m"].iloc[0]) if len(ref) else np.nan
        return df, {"surface_elevation_m_nap": surface, "surface_from_profile": False, "note": ""}

    borehole = read_borehole_xml(path)
    surface_from_profile = borehole.get("surface_elevation_m_nap") is None
    if surface_from_profile:
        if np.isnan(fallback_surface_m_nap):
            raise ValueError("borehole XML has no surface elevation and the test lies outside the profile")
        borehole = dict(borehole, surface_elevation_m_nap=fallback_surface_m_nap)
    layers_df, samples, _ = build_synthetic_profile(borehole)
    df = pd.DataFrame({"depth_m": samples["depth_m"], "elevation_m_nap": samples["elevation_m_nap"],
                       "ic": samples["synthetic_ic"], "class_code": samples["source_class_code"]})
    notes = []
    if (~layers_df["supported"]).any():
        unsupported = layers_df[~layers_df["supported"]]
        notes.append("unsupported layers (Ic interpolated where marked ~): " + "; ".join(
            f"{'~' if r.ic_interpolated else '?'}{r.soil_name_nen5104}" for r in unsupported.itertuples()))
    if layers_df["is_guess"].any():
        notes.append("guessed layers: " + "; ".join(
            f"{r.soil_name_nen5104}->{r.class_code}" for r in layers_df[layers_df["is_guess"]].itertuples()))
    return df, {"surface_elevation_m_nap": float(borehole["surface_elevation_m_nap"]),
                "surface_from_profile": surface_from_profile, "note": " | ".join(notes)}


# --- one cross-section --------------------------------------------------------------

def collect_cross_section(
    names: List[str], line: ProfileLine, coord_index, cpt_index, borehole_index,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Place every named test on the line and build its Ic profile.

    Returns (tests, ic_profiles): one row per test (numbered in order along
    the line), and the long-format Ic table
    (name, kind, distance_along_m, depth_m, elevation_m_nap, ic, class_code).
    """
    rows, profiles = [], []
    for name in names:
        kind = test_kind(name)
        row = {"name": name, "kind": kind, "status": "ok", "note": ""}
        x, y, coord_note = lookup_coordinates(coord_index, name)
        row.update(x=x, y=y, coord_source=coord_note)
        path, n_candidates = find_file(cpt_index if kind == "cpt" else borehole_index, name)
        row.update(file=str(path) if path else None, n_file_candidates=n_candidates)
        if x is None:
            row["status"] = "no_coordinates"
        else:
            along, offset = line.project(x, y)
            row.update(distance_along_m=along, offset_m=offset,
                       beyond_line_ends=bool(along < 0 or along > line.length_m),
                       profile_ground_m_nap=line.ground_at(along))
            if path is None:
                row["status"] = "file_not_found"
        if row["status"] == "ok":
            try:
                df, meta = build_ic_profile(kind, path, row["profile_ground_m_nap"])
                row.update(meta)
                valid = df.dropna(subset=["ic"])
                row.update(n_ic_samples=int(len(valid)),
                           top_ic_elevation_m_nap=float(valid["elevation_m_nap"].max()) if len(valid) else np.nan,
                           bottom_elevation_m_nap=float(df["elevation_m_nap"].min()) if len(df) else np.nan)
                df.insert(0, "name", name)
                df.insert(1, "kind", kind)
                df.insert(2, "distance_along_m", row["distance_along_m"])
                profiles.append(df)
            except Exception as exc:
                row.update(status="error", note=f"{type(exc).__name__}: {exc}")
        rows.append(row)

    tests = pd.DataFrame(rows)
    if "distance_along_m" in tests:
        tests = tests.sort_values("distance_along_m", na_position="last").reset_index(drop=True)
    tests.insert(0, "number", range(1, len(tests) + 1))
    ic_profiles = pd.concat(profiles, ignore_index=True) if profiles else pd.DataFrame()
    return tests, ic_profiles
