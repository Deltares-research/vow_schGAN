"""Evaluate borehole-derived synthetic Ic against nearby measured CPTs, for a list of pairs.

Input: a pairs CSV (columns name_cpt, name_borehole[, distance_m]) such as the
output of `find_cpt_borehole_pairs.py`, plus the folders holding the CPT files
(.gef or BRO .xml, searched recursively) and the borehole BRO XMLs.

For every pair: build the synthetic Ic ensemble from the borehole, interpret
the CPT with geolib_plus (Lengkeek 2024), align the measured Ic onto the
borehole grid by elevation (m NAP) and compare. Outputs in --output-dir:

    pair_metrics.csv               one row per pair (status, distance, bias, MAE, RMSE, r, ...)
    class_metrics.csv              metrics per borehole class, all pairs pooled
    layer_metrics.csv              mean predicted vs mean measured Ic per borehole layer
    all_compared_samples.csv       every compared depth sample of every pair
    robertson_zone_confusion.csv   measured vs predicted Robertson zone counts
    summary.json                   settings, overall metrics, file lookup notes
    per_pair/<CPT>_<BHR>.png       borehole | predicted vs measured Ic | error | Lengkeek
    per_pair/<CPT>_<BHR>_profile.csv
    plots/*.png                    pooled scatter, error per class / per pair, zone confusion, histogram

Run with:
    .venv/Scripts/python.exe src/run_pair_evaluation.py ^
        --pairs-csv <pairs.csv> --cpt-dir <CPT folder> --borehole-dir <borehole folder> ^
        --output-dir <folder> [--n-realizations 20] [--base-seed 42] [--limit N]
"""

import argparse
import json
import sys
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))  # add src/ to sys.path

from modules.borehole_ic import evaluation_plots  # noqa: E402
from modules.borehole_ic.file_lookup import find_file, index_files  # noqa: E402
from modules.borehole_ic.lengkeek_reference import LENGKEEK_IMPLEMENTATION_NOTE, build_reference_lengkeek_profile  # noqa: E402
from modules.borehole_ic.pair_evaluation import (  # noqa: E402
    class_metrics, evaluate_pair, flatten_pair_metrics, overall_summary, zone_confusion,
)
from modules.borehole_ic.plotting import plot_pair_evaluation  # noqa: E402
from modules.borehole_ic.xml_readers import read_borehole_xml  # noqa: E402

PROFILE_COLUMNS = [
    "depth_m", "elevation_m_nap", "source_class_code", "sample_category", "is_transition",
    "ic_median", "ic_p05", "ic_p95", "reference_ic", "error", "zone_synthetic", "zone_measured",
]


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs-csv", type=Path, required=True, help="CSV with name_cpt, name_borehole[, distance_m].")
    parser.add_argument("--cpt-dir", type=Path, required=True, help="Folder with CPT .gef/.xml files (searched recursively).")
    parser.add_argument("--borehole-dir", type=Path, required=True, help="Folder with borehole BRO XMLs (searched recursively).")
    parser.add_argument("--output-dir", type=Path, required=True, help="Folder for the evaluation results.")
    parser.add_argument("--n-realizations", type=int, default=20, help="Synthetic realizations per borehole (default 20).")
    parser.add_argument("--base-seed", type=int, default=42, help="Base seed for the realizations (default 42).")
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate the first N pairs (trial runs).")
    return parser.parse_args()


def main():
    args = _parse_args()
    pairs = pd.read_csv(args.pairs_csv, sep=None, engine="python")
    pairs.columns = [c.strip().lower() for c in pairs.columns]
    missing = {"name_cpt", "name_borehole"} - set(pairs.columns)
    if missing:
        sys.exit(f"Pairs CSV is missing column(s) {sorted(missing)}; found {list(pairs.columns)}")
    if "distance_m" not in pairs:
        pairs["distance_m"] = np.nan
    pairs = pairs.head(args.limit) if args.limit else pairs

    print(f"Indexing CPT files in {args.cpt_dir} ...")
    cpt_index = index_files(args.cpt_dir, {".gef", ".xml"})
    print(f"Indexing borehole files in {args.borehole_dir} ...")
    borehole_index = index_files(args.borehole_dir, {".xml"})

    per_pair_dir = args.output_dir / "per_pair"
    plots_dir = args.output_dir / "plots"
    per_pair_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    pair_rows, sample_frames, layer_frames = [], [], []
    for i, pair in enumerate(pairs.itertuples(), start=1):
        pair_id = f"{pair.name_cpt}_{pair.name_borehole}"
        row = {"pair_id": pair_id, "name_cpt": pair.name_cpt, "name_borehole": pair.name_borehole,
               "distance_m": pair.distance_m}
        try:
            cpt_path, n_cpt = find_file(cpt_index, pair.name_cpt)
            bh_path, n_bh = find_file(borehole_index, pair.name_borehole)
            row.update(cpt_file=str(cpt_path) if cpt_path else None, n_cpt_file_candidates=n_cpt,
                       borehole_file=str(bh_path) if bh_path else None)
            if cpt_path is None:
                row.update(status="cpt_not_found")
            elif bh_path is None:
                row.update(status="borehole_not_found")
            else:
                borehole = read_borehole_xml(bh_path)
                if borehole.get("surface_elevation_m_nap") is None or not borehole["layers"]:
                    row.update(status="no_surface_elevation_or_layers")
                else:
                    reference_df = build_reference_lengkeek_profile(cpt_path)
                    result = evaluate_pair(borehole, reference_df, args.n_realizations, args.base_seed)
                    info = result["info"]
                    row.update({k: v for k, v in info.items() if k != "realization_seeds"})
                    row["surface_elevation_diff_m"] = info["cpt_top_elevation_m_nap"] - info["borehole_surface_elevation_m_nap"]
                    row.update(flatten_pair_metrics(result["metrics"]))
                    row["status"] = "ok" if result["metrics"]["overall"].get("n_samples", 0) else "no_overlap"

                    ensemble_df = result["ensemble_df"]
                    ensemble_df[PROFILE_COLUMNS].to_csv(per_pair_dir / f"{pair_id}_profile.csv", index=False)
                    fig = plot_pair_evaluation(
                        result["layers_df"], ensemble_df, reference_df, borehole["surface_elevation_m_nap"],
                        args.n_realizations, result["metrics"],
                        title=f"{pair.name_cpt} (measured) vs {pair.name_borehole} (predicted) - "
                              f"distance {pair.distance_m:.2f} m" if pd.notna(pair.distance_m)
                        else f"{pair.name_cpt} (measured) vs {pair.name_borehole} (predicted)",
                        output_path=per_pair_dir / f"{pair_id}.png",
                    )
                    plt.close(fig)

                    if len(result["samples"]):
                        samples = result["samples"][PROFILE_COLUMNS].copy()
                        samples.insert(0, "pair_id", pair_id)
                        samples.insert(1, "distance_m", pair.distance_m)
                        sample_frames.append(samples)
                    if len(result["layer_table"]):
                        layers = result["layer_table"].copy()
                        layers.insert(0, "pair_id", pair_id)
                        layer_frames.append(layers)
        except Exception as exc:  # keep going; the failure is recorded per pair
            plt.close("all")
            traceback.print_exc()
            row.update(status="error", message=f"{type(exc).__name__}: {exc}")
        pair_rows.append(row)
        extra = f"MAE {row['mae']:.2f}, bias {row['bias']:+.2f}" if row.get("status") == "ok" else row.get("message", "")
        print(f"[{i}/{len(pairs)}] {pair_id}: {row['status']} {extra}")

    pair_metrics = pd.DataFrame(pair_rows)
    pair_metrics.to_csv(args.output_dir / "pair_metrics.csv", index=False)
    print("\nStatus counts:\n" + pair_metrics["status"].value_counts().to_string())

    if not sample_frames:
        print("No pair produced comparable samples; no pooled statistics or plots written.")
        return

    pooled = pd.concat(sample_frames, ignore_index=True)
    layer_table = pd.concat(layer_frames, ignore_index=True) if layer_frames else pd.DataFrame()
    pooled.to_csv(args.output_dir / "all_compared_samples.csv", index=False)
    layer_table.to_csv(args.output_dir / "layer_metrics.csv", index=False)
    classes = class_metrics(pooled)
    classes.to_csv(args.output_dir / "class_metrics.csv", index=False)
    confusion = zone_confusion(pooled)
    confusion.to_csv(args.output_dir / "robertson_zone_confusion.csv")

    evaluation_plots.plot_scatter(pooled, layer_table, plots_dir / "predicted_vs_measured.png")
    evaluation_plots.plot_error_by_class(pooled, plots_dir / "error_by_class.png")
    evaluation_plots.plot_error_per_pair(pair_metrics, plots_dir / "error_per_pair.png")
    evaluation_plots.plot_zone_confusion(confusion, plots_dir / "robertson_zone_confusion.png")
    evaluation_plots.plot_error_histogram(pooled, plots_dir / "error_histogram.png")

    summary = overall_summary(pooled, layer_table)
    report = {
        "settings": {
            "pairs_csv": str(args.pairs_csv), "cpt_dir": str(args.cpt_dir), "borehole_dir": str(args.borehole_dir),
            "n_realizations": args.n_realizations, "base_seed": args.base_seed,
            "error_definition": "synthetic ensemble median Ic - measured CPT Ic",
            "alignment": "measured Ic linearly interpolated onto the borehole grid by elevation (m NAP); "
                         "no extrapolation, no bridging of CPT gaps",
            "measured_ic": LENGKEEK_IMPLEMENTATION_NOTE + " Groundwater: geolib_plus default (NetCDF lookup by coordinates).",
        },
        "status_counts": pair_metrics["status"].value_counts().to_dict(),
        "n_pairs_evaluated": int((pair_metrics["status"] == "ok").sum()),
        "overall": summary,
        "per_class": classes.to_dict("records"),
    }
    with open(args.output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=lambda v: None if isinstance(v, float) and np.isnan(v) else str(v))

    s = summary["all_samples"]
    print(f"\nAll compared samples (n={s['n_samples']}): bias {s['bias']:+.3f}, MAE {s['mae']:.3f}, "
          f"RMSE {s['rmse']:.3f}, r {s['pearson_r']:.2f}, inside band {s['fraction_inside_band']:.0%}, "
          f"same Robertson zone {s['robertson_zone_agreement']:.0%}")
    if "layer_means" in summary:
        lm = summary["layer_means"]
        print(f"Per-layer means (n={lm['n_layers']}): bias {lm['bias']:+.3f}, MAE {lm['mae']:.3f}, r {lm['pearson_r']:.2f}")
    print(f"Outputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()
