"""Batch entry point: folder of BRO borehole XMLs -> one synthetic Ic profile per borehole.

For every `*.xml` borehole in the input folder this runs the same workflow as
`run_borehole_to_ic_profile.py` (NEN 5104 class -> composition sampling ->
representative Ic), but without a reference CPT. Outputs:

    <output-dir>/<BRO_ID>_synthetic_ic.csv   per-depth synthetic Ic profile
    <output-dir>/_plots/<BRO_ID>.png         borehole NEN classes | synthetic Ic
    <output-dir>/_batch_summary.csv          one row per borehole (status, coords, counts)

Layers are routed by their main soil noun: klei/zand/leem -> mineral
lutum-silt-zand triangle, veen -> peat triangle, grind -> gravel triangle.
Names that match no exact NEN class but contain a main noun (e.g. "Zand",
"zandNietGespecificeerd", "matigKleiigVeen") get a generic class ("Zand*",
"Veen*", ...) sampled over that noun's whole main-name region; the plots show
the raw name in the borehole column and the generic code in the Ic panel.
Only names with no recognisable main noun are left as NaN gaps. A failure on
one borehole is recorded in the summary and does not stop the batch.

Run with:
    .venv/Scripts/python.exe src/run_batch_boreholes_to_ic.py ^
        --input-dir <folder with BHR*.xml> ^
        --output-dir <folder for synthetic profiles>
"""

import argparse
import sys
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))  # add src/ to sys.path

from modules.borehole_ic.composition_sampling import RANDOM_SEED  # noqa: E402
from modules.borehole_ic.plotting import plot_borehole_synthetic_ic  # noqa: E402
from modules.borehole_ic.profile_builder import build_synthetic_profile  # noqa: E402
from modules.borehole_ic.triangles import all_composition_columns  # noqa: E402
from modules.borehole_ic.xml_readers import read_borehole_xml  # noqa: E402

PROFILE_COLUMNS = [
    "depth_m", "elevation_m_nap", "synthetic_ic", "triangle", *all_composition_columns(),
    "source_class_code", "evaluated_class_code", "is_transition", "is_fallback", "is_interpolated", "note",
]


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, required=True, help="Folder containing BRO borehole XML files.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Folder for the synthetic Ic profiles.")
    parser.add_argument("--no-plots", action="store_true", help="Skip writing the per-borehole PNGs.")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N XML files (trial runs).")
    return parser.parse_args()


def process_borehole(xml_path: Path, output_dir: Path, plot_dir: Path, make_plot: bool) -> dict:
    borehole = read_borehole_xml(xml_path)
    bro_id = borehole["bro_id"] or xml_path.stem
    record = {
        "file": xml_path.name, "bro_id": bro_id, "x": borehole["x"], "y": borehole["y"],
        "surface_elevation_m_nap": borehole["surface_elevation_m_nap"],
        "final_depth_m": borehole["final_depth_m"], "n_layers": len(borehole["layers"]),
    }
    if not borehole["layers"] or borehole["final_depth_m"] is None:
        record.update(status="skipped", message="no layers or no final depth in XML")
        return record

    layers_df, samples_df, _ = build_synthetic_profile(borehole)
    samples_df[PROFILE_COLUMNS].to_csv(output_dir / f"{bro_id}_synthetic_ic.csv", index=False)

    unsupported = layers_df[~layers_df["supported"]]
    guessed = layers_df[layers_df["is_guess"]]
    record.update(
        triangles_used="; ".join(sorted(layers_df.loc[layers_df["supported"], "triangle"].dropna().unique())),
        n_guessed_layers=len(guessed),
        guessed_names="; ".join(sorted({f"{r.soil_name_nen5104} -> {r.class_code}" for r in guessed.itertuples()})),
        n_unsupported_layers=len(unsupported),
        unsupported_names="; ".join(sorted({str(n) for n in unsupported["soil_name_nen5104"]})),
        supported_fraction_of_samples=round(float(samples_df["synthetic_ic"].notna().mean()), 4),
        status="ok", message="",
    )

    if make_plot:
        fig = plot_borehole_synthetic_ic(
            layers_df, samples_df, borehole["surface_elevation_m_nap"],
            title=f"{bro_id} - synthetic Ic (seed {RANDOM_SEED})",
            output_path=plot_dir / f"{bro_id}.png",
        )
        plt.close(fig)
    return record


def main():
    args = _parse_args()
    xml_files = sorted(args.input_dir.glob("*.xml"))[: args.limit]
    if not xml_files:
        sys.exit(f"No .xml files found in {args.input_dir}")

    plot_dir = args.output_dir / "_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for i, xml_path in enumerate(xml_files, start=1):
        try:
            record = process_borehole(xml_path, args.output_dir, plot_dir, not args.no_plots)
        except Exception as exc:  # keep the batch going; the failure is recorded in the summary
            plt.close("all")
            record = {"file": xml_path.name, "status": "error", "message": f"{type(exc).__name__}: {exc}"}
            traceback.print_exc()
        records.append(record)
        print(f"[{i}/{len(xml_files)}] {xml_path.name}: {record['status']} {record.get('message', '')}")

    summary = pd.DataFrame(records)
    summary.to_csv(args.output_dir / "_batch_summary.csv", index=False)
    print("\nStatus counts:\n" + summary["status"].value_counts().to_string())
    print(f"Outputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()
