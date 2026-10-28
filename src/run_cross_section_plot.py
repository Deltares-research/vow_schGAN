"""Entry point: plot cross-sections defined in the cross-section guide CSV.

For each selected row of the guide (columns cs_id, cs_name, cs_profile,
cpts_in_cs) this finds the profile CSV (searched under the guide's folder),
places every listed test on the straight line from the profile's first to its
last point, builds its Ic profile (real CPT: geolib_plus Ic; borehole "BHR...":
synthetic Ic, seed 42), adds augmented copies near the line ends (see
`modules/cross_section/cs_augment.py`: 5 m from the start, 10 m and 50 m from
the end) and writes, next to the profile CSV:

    <profile stem>_cross_section.png   distance along the line vs elevation (m NAP), SchemaGAN frame drawn
    <profile stem>_tests.csv           one row per test incl. copies (augmented, copy_of): position, offset, ground level, status
    <profile stem>_ic_profiles.csv     every Ic sample: name, kind, distance_along_m, depth_m, elevation_m_nap, ic, augmented
    <profile stem>_profile_filled.csv  only if the profile has missing elevations: the profile with gaps
                                       filled by linear interpolation (elev_nap), original values in
                                       elev_nap_raw and filled rows flagged in elev_filled
    <profile stem>_schemagan_input.csv       the SchemaGAN input: 32 lines x 512 comma-separated values,
                                             no header, no index; Ic or 0 (see cs_matrix.py for the rules)
    <profile stem>_schemagan_input_info.txt  frame, rules, which test fills which column, row elevations
    <profile stem>_schemagan_input.png       image of the matrix (white = 0)

(<profile stem> drops a trailing "_profile", e.g. cs_O62,7_profile.csv -> cs_O62,7.)

Run with (defaults point at the repository's data/ folder):
    .venv/Scripts/python.exe src/run_cross_section_plot.py [--cs-id 1] [--cs-name "62,7"]
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))  # add src/ to sys.path

from modules.borehole_ic.file_lookup import index_files  # noqa: E402
from modules.cross_section.cs_augment import augment_cross_section, occupied_columns  # noqa: E402
from modules.cross_section.cs_data import (  # noqa: E402
    collect_cross_section, find_profile, load_coordinate_index, read_guide, read_profile_line, schemagan_frame, split_names,
)
from modules.cross_section.cs_matrix import build_matrix, write_matrix_csv, write_metadata  # noqa: E402
from modules.cross_section.cs_plot import plot_cross_section, plot_matrix_preview  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
STRAIGHTNESS_WARNING_M = 1.0


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cs-guide", type=Path, default=DATA_DIR / "cs" / "cs_guide.csv", help="Cross-section guide CSV.")
    parser.add_argument("--cs-id", action="append", help="Only this cs_id (repeatable). Default: all rows.")
    parser.add_argument("--cs-name", action="append", help="Only this cs_name (repeatable). Default: all rows.")
    parser.add_argument("--coords-dir", type=Path, default=DATA_DIR / "coords", help="Folder with coordinate CSVs (name, x, y).")
    parser.add_argument("--cpt-dir", type=Path, default=DATA_DIR / "cpts", help="Folder with CPT .gef/.xml files.")
    parser.add_argument("--borehole-dir", type=Path, default=DATA_DIR / "boreholes", help="Folder with borehole BRO XMLs.")
    return parser.parse_args()


def main():
    args = _parse_args()
    guide = read_guide(args.cs_guide)
    if args.cs_id:
        guide = guide[guide["cs_id"].str.strip().isin([s.strip() for s in args.cs_id])]
    if args.cs_name:
        guide = guide[guide["cs_name"].str.strip().isin([s.strip() for s in args.cs_name])]
    if not len(guide):
        sys.exit("No cross-section selected from the guide.")

    coord_index = load_coordinate_index(args.coords_dir)
    cpt_index = index_files(args.cpt_dir, {".gef", ".xml"})
    borehole_index = index_files(args.borehole_dir, {".xml"})

    for row in guide.itertuples():
        print(f"\n=== cs_id {row.cs_id} '{row.cs_name}' ===")
        profile_path = find_profile(args.cs_guide.parent, row.cs_profile)
        line = read_profile_line(profile_path)
        if line.max_deviation_m > STRAIGHTNESS_WARNING_M:
            print(f"WARNING: profile points deviate up to {line.max_deviation_m:.1f} m from the straight line A-B")

        names = split_names(row.cpts_in_cs)
        tests, ic_profiles = collect_cross_section(names, line, coord_index, cpt_index, borehole_index)

        stem = profile_path.stem[: -len("_profile")] if profile_path.stem.endswith("_profile") else profile_path.stem
        out_dir = profile_path.parent
        if line.elevation_gaps:
            print("Ground elevation interpolated over gaps: "
                  + ", ".join(f"{s:.1f}-{e:.1f} m" for s, e in line.elevation_gaps))
            line.profile.to_csv(out_dir / f"{stem}_profile_filled.csv", index=False)
        frame = schemagan_frame(line, tests)
        tests, ic_profiles = augment_cross_section(tests, ic_profiles, line, frame.col_width_m)
        occupied = occupied_columns(tests, frame)
        tests.to_csv(out_dir / f"{stem}_tests.csv", index=False)
        if len(ic_profiles):
            ic_profiles.to_csv(out_dir / f"{stem}_ic_profiles.csv", index=False)
        print(f"SchemaGAN frame {frame.n_cols} x {frame.n_rows}: x {frame.left_m:.1f}-{frame.right_m:.1f} m, "
              f"top {frame.top_m_nap:.2f} m NAP ({frame.top_source}), bottom {frame.bottom_m_nap:g} m NAP, "
              f"cell {frame.col_width_m:.3f} x {frame.row_height_m:.3f} m, occupied columns {occupied}")
        fig = plot_cross_section(line, tests, ic_profiles,
                                 title=f"Cross-section {row.cs_name} (cs_id {row.cs_id}) - "
                                       f"{int((~tests['augmented']).sum())} tests + {int(tests['augmented'].sum())} augmented copies",
                                 output_path=out_dir / f"{stem}_cross_section.png", frame=frame, occupied=occupied)
        plt.close(fig)

        matrix, assignments = build_matrix(tests, ic_profiles, frame)
        matrix_file = out_dir / f"{stem}_schemagan_input.csv"
        write_matrix_csv(matrix, matrix_file)
        write_metadata(out_dir / f"{stem}_schemagan_input_info.txt", f"{row.cs_name} (cs_id {row.cs_id})",
                       frame, assignments, matrix, matrix_file.name)
        fig = plot_matrix_preview(matrix, frame, assignments,
                                  title=f"SchemaGAN input {frame.n_cols} x {frame.n_rows} - cross-section {row.cs_name}",
                                  output_path=out_dir / f"{stem}_schemagan_input.png")
        plt.close(fig)
        dropped = assignments[~assignments["used"]]
        print(f"SchemaGAN matrix: {matrix.shape[0]} x {matrix.shape[1]}, {int((matrix != 0).any(axis=0).sum())} columns "
              f"with data, {int(np.count_nonzero(matrix))} non-zero cells"
              + (f"; dropped (same column): {', '.join(dropped['name'])}" if len(dropped) else ""))
        print(assignments[["number", "name", "column", "used", "first_row", "last_row", "n_rows_filled"]].to_string(index=False))
        print(f"Line length {line.length_m:.1f} m. Outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
