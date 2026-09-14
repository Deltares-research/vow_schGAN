"""Entry point: BRO borehole XML -> composition sampling -> synthetic Ic depth profile.

Prototype script (no pytest) - CLI arguments select the input BRO XML files
and the output folder; no personal data paths are hard-coded here.

Run with:
    .venv/Scripts/python.exe src/run_borehole_to_ic_profile.py ^
        --cpt-xml <path/to/CPT....xml> ^
        --borehole-xml <path/to/BHR....xml> [<path/to/BHR....xml> ...] ^
        [--output-dir <folder>]

If several --borehole-xml candidates are given, the one closest (RD
coordinates) to the CPT is selected automatically.
"""

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # add src/ to sys.path

from modules.borehole_ic.lengkeek_reference import LENGKEEK_IMPLEMENTATION_NOTE, build_reference_lengkeek_profile
from modules.borehole_ic.plotting import plot_three_panel, plot_triangle_samples
from modules.borehole_ic.profile_builder import (
    CORRELATION_LENGTH_M, DEPTH_STEP_M, TRANSITION_THICKNESS_M,
    build_synthetic_profile,
)
from modules.borehole_ic.composition_sampling import RANDOM_SEED
from modules.borehole_ic.verification import format_report, run_all_checks
from modules.borehole_ic.xml_readers import read_borehole_xml, read_cpt_xml_metadata

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "res" / "borehole_ic"


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cpt-xml", type=Path, required=True, help="Path to the BRO CPT XML file.")
    parser.add_argument(
        "--borehole-xml", type=Path, required=True, nargs="+",
        help="Path(s) to one or more candidate BRO borehole XML files. "
             "If more than one is given, the closest to the CPT (RD coordinates) is selected.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"Base output folder; a '{{borehole_id}}_{{cpt_id}}' subfolder is created inside it "
             f"(default: {DEFAULT_OUTPUT_DIR}).",
    )
    return parser.parse_args()


def _select_associated_borehole(cpt_meta: dict, candidate_paths):
    """Pick the borehole XML closest (RD coordinates) to the CPT; report all distances."""
    candidates = []
    for path in candidate_paths:
        meta = read_cpt_xml_metadata(path)  # same reader works for the id+coords of any BRO document
        dist = math.hypot(meta["x"] - cpt_meta["x"], meta["y"] - cpt_meta["y"])
        candidates.append((path, meta, dist))
    candidates.sort(key=lambda c: c[2])
    return candidates


def main():
    args = _parse_args()
    cpt_meta = read_cpt_xml_metadata(args.cpt_xml)
    candidates = _select_associated_borehole(cpt_meta, args.borehole_xml)

    print("CPT:", cpt_meta)
    print("Borehole candidates by distance to CPT (m):")
    for path, meta, dist in candidates:
        print(f"  {meta['bro_id']}: ({meta['x']}, {meta['y']}) -> {dist:.2f} m")

    chosen_path, chosen_meta, chosen_dist = candidates[0]
    print(f"Selected borehole: {chosen_meta['bro_id']} at {chosen_dist:.2f} m from the CPT")

    borehole = read_borehole_xml(chosen_path)
    layers_df, samples_df, run_info = build_synthetic_profile(borehole)

    def _rebuild():
        _, resampled, _ = build_synthetic_profile(borehole)
        return resampled

    reference_df = build_reference_lengkeek_profile(args.cpt_xml)

    out_dir = args.output_dir / f"{borehole['bro_id']}_{cpt_meta['bro_id']}"
    out_dir.mkdir(parents=True, exist_ok=True)

    layers_df.to_csv(out_dir / "layer_table.csv", index=False)
    samples_df.to_csv(out_dir / "samples.csv", index=False)
    reference_df.to_csv(out_dir / "reference_lengkeek_cpt.csv", index=False)

    unsupported = layers_df[~layers_df["supported"]]
    metadata = {
        "cpt": cpt_meta,
        "borehole": {k: v for k, v in borehole.items() if k != "layers"},
        "borehole_cpt_horizontal_separation_m": chosen_dist,
        "all_candidate_distances_m": {meta["bro_id"]: dist for _, meta, dist in candidates},
        "sampling_parameters": {
            "depth_step_m": DEPTH_STEP_M,
            "transition_thickness_m_requested": TRANSITION_THICKNESS_M,
            "correlation_length_m (uncalibrated assumption)": CORRELATION_LENGTH_M,
            "random_seed": RANDOM_SEED,
        },
        "transition_half_thicknesses_m": run_info["transition_half_thicknesses"],
        "sampler_fallback_counts": run_info["fallback_counts"],
        "unsupported_layers": unsupported[["layer_index", "soil_name_nen5104", "unsupported_reason"]].to_dict("records"),
        "lengkeek_reference_implementation": LENGKEEK_IMPLEMENTATION_NOTE,
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)

    plot_three_panel(
        layers_df, samples_df, reference_df, borehole["surface_elevation_m_nap"],
        title=f"{borehole['bro_id']} vs {cpt_meta['bro_id']}",
        output_path=out_dir / "three_panel_profile.png",
    )
    plot_triangle_samples(samples_df, output_path=out_dir / "triangle_samples_interactive.html")

    checks = run_all_checks(layers_df, samples_df, borehole["surface_elevation_m_nap"], _rebuild)
    report = format_report(checks)
    print(report)
    (out_dir / "verification_report.txt").write_text(report, encoding="utf-8")

    print(f"\nOutputs written to: {out_dir}")


if __name__ == "__main__":
    main()
