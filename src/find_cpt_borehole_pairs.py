"""Find CPT-borehole pairs that lie close together, from two coordinate CSVs.

Inputs are CSVs with a name column and RD New (EPSG:28992) X/Y columns, e.g.
the output of `extract_bro_xml_coords.py` (name, X, Y) or the older CPT
coordinate CSV from `core/extract_coords.py` (name, x, y, fixed). Column names
are matched case-insensitively.

Every CPT-borehole combination closer than --max-distance (metres) is written,
sorted by distance, with columns: name_cpt, name_borehole, distance_m.
One CPT may appear with several boreholes (and vice versa); use
--nearest-only to keep only the closest CPT for each borehole.

Run with:
    .venv/Scripts/python.exe src/find_cpt_borehole_pairs.py ^
        --cpt-csv <cpt_coords.csv> --borehole-csv <borehole_coords.csv> ^
        --output-csv <pairs.csv> [--max-distance 3] [--nearest-only]
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from scipy.spatial import cKDTree


def _read_coords(path: Path, label: str) -> pd.DataFrame:
    """Read a coordinate CSV into columns name, x, y (case-insensitive column matching)."""
    df = pd.read_csv(path, sep=None, engine="python")  # sniff the delimiter (',' or ';' from Excel)
    columns = {c.strip().lower(): c for c in df.columns}
    missing = [c for c in ("name", "x", "y") if c not in columns]
    if missing:
        sys.exit(f"{label} CSV {path} is missing column(s) {missing}; found {list(df.columns)}")
    df = df[[columns["name"], columns["x"], columns["y"]]]
    df.columns = ["name", "x", "y"]
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")

    n_bad = int(df[["x", "y"]].isna().any(axis=1).sum())
    if n_bad:
        print(f"{label}: dropped {n_bad} row(s) without numeric coordinates")
    df = df.dropna(subset=["x", "y"]).reset_index(drop=True)

    # RD New covers roughly X 0-300 km, Y 300-625 km; lat/lon or scaled values point to a wrong input.
    outside = df[(df["x"] < 0) | (df["x"] > 300_000) | (df["y"] < 300_000) | (df["y"] > 625_000)]
    if len(outside):
        print(f"{label}: WARNING {len(outside)} row(s) outside the RD New range, e.g. "
              f"{outside.head(3).to_dict('records')}")

    n_dup = int(df["name"].duplicated().sum())
    if n_dup:
        print(f"{label}: WARNING {n_dup} duplicated name(s); each row is paired separately")
    return df


def find_pairs(cpts: pd.DataFrame, boreholes: pd.DataFrame, max_distance: float, nearest_only: bool) -> pd.DataFrame:
    tree = cKDTree(cpts[["x", "y"]].to_numpy())
    rows = []
    for bh in boreholes.itertuples():
        if nearest_only:
            dist, idx = tree.query([bh.x, bh.y])
            matches = [(dist, idx)] if dist <= max_distance else []
        else:
            idxs = tree.query_ball_point([bh.x, bh.y], r=max_distance)
            matches = [(((cpts.at[i, "x"] - bh.x) ** 2 + (cpts.at[i, "y"] - bh.y) ** 2) ** 0.5, i) for i in idxs]
        for dist, i in matches:
            rows.append({"name_cpt": cpts.at[i, "name"], "name_borehole": bh.name, "distance_m": round(float(dist), 2)})
    pairs = pd.DataFrame(rows, columns=["name_cpt", "name_borehole", "distance_m"])
    return pairs.sort_values(["distance_m", "name_borehole", "name_cpt"]).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cpt-csv", type=Path, required=True, help="CSV with CPT name, X, Y.")
    parser.add_argument("--borehole-csv", type=Path, required=True, help="CSV with borehole name, X, Y.")
    parser.add_argument("--output-csv", type=Path, required=True, help="Output pairs CSV.")
    parser.add_argument("--max-distance", type=float, default=3.0, help="Maximum CPT-borehole distance in metres (default 3).")
    parser.add_argument("--nearest-only", action="store_true", help="Keep only the closest CPT per borehole.")
    args = parser.parse_args()
    print(f"Output will be written to: {args.output_csv.resolve()}")

    cpts = _read_coords(args.cpt_csv, "CPT")
    boreholes = _read_coords(args.borehole_csv, "Borehole")
    pairs = find_pairs(cpts, boreholes, args.max_distance, args.nearest_only)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(args.output_csv, index=False)

    print(f"{len(cpts)} CPTs, {len(boreholes)} boreholes -> {len(pairs)} pair(s) within {args.max_distance:g} m")
    if len(pairs):
        print(f"  boreholes with a CPT: {pairs['name_borehole'].nunique()}, CPTs used: {pairs['name_cpt'].nunique()}")
        print(f"  distance (m): min {pairs['distance_m'].min():.2f}, median {pairs['distance_m'].median():.2f}, "
              f"max {pairs['distance_m'].max():.2f}")
    print(f"Written to: {args.output_csv}")


if __name__ == "__main__":
    main()
