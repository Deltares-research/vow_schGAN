"""Entry point: ensemble exercise (20 realizations) for BHR000000397320 / CPT000000061801.

Reuses the existing single-realization workflow (XML readers, sampling,
mapping, Ic evaluation, Lengkeek reference) unchanged; only the ensembling,
comparison, and plotting are new. No pytest/framework - plain checks below.

Run with:
    .venv/Scripts/python.exe src/run_ensemble_borehole_to_ic_profile.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # add src/ to sys.path

from modules.borehole_ic.ensemble import align_reference_to_grid, build_ensemble, summarize_comparison
from modules.borehole_ic.lengkeek_reference import LENGKEEK_IMPLEMENTATION_NOTE, build_reference_lengkeek_profile
from modules.borehole_ic.plotting import plot_ensemble_three_panel
from modules.borehole_ic.xml_readers import read_borehole_xml, read_cpt_xml_metadata

DATA_DIR = Path(r"c:\Users\camposmo\OneDrive - Stichting Deltares\Documents\Projects\VOW\data\O_62_64\O_62,7")
CPT_XML_PATH = DATA_DIR / "CPT000000061801.xml"
BOREHOLE_XML_PATH = DATA_DIR / "BHR000000397320.xml"

OUTPUT_DIR = Path(r"d:\codes\vow_schGAN\res\borehole_ic")

N_REALIZATIONS = 20
DEPTH_STEP_M = 0.02
BASE_SEED = 42


def _run_checks(layers_df, ensemble_df, seeds, run_info, rebuild_fn) -> list:
    checks = []

    a = rebuild_fn()
    b = rebuild_fn()
    ic_cols = [f"ic_realization_{r}" for r in range(N_REALIZATIONS)]
    max_dev = np.nanmax(np.abs(a[ic_cols].to_numpy() - b[ic_cols].to_numpy()))
    checks.append(("PASS" if max_dev < 1e-12 else "FAIL",
                    f"reproducibility (same base seed): max abs diff = {max_dev:.2e}"))

    distinct = sum(
        1 for i in range(N_REALIZATIONS) for j in range(i + 1, N_REALIZATIONS)
        if not np.allclose(ensemble_df[f"ic_realization_{i}"].dropna(), ensemble_df[f"ic_realization_{j}"].dropna())
    )
    n_pairs = N_REALIZATIONS * (N_REALIZATIONS - 1) // 2
    checks.append(("PASS" if distinct == n_pairs else "FAIL",
                    f"distinct streams: {distinct}/{n_pairs} realization pairs differ"))

    valid = ensemble_df.dropna(subset=["ic_p05", "ic_median", "ic_p95"])
    ordering_ok = ((valid["ic_p05"] <= valid["ic_median"] + 1e-9) & (valid["ic_median"] <= valid["ic_p95"] + 1e-9)).all()
    checks.append(("PASS" if ordering_ok else "FAIL", "percentile ordering p05 <= median <= p95 holds everywhere"))

    supported_rows = ensemble_df[~ensemble_df["source_class_code"].isna() & (ensemble_df["source_class_code"] != "")]
    non_transition_supported = supported_rows[~supported_rows["is_transition"]]
    n_nan_supported = non_transition_supported["ic_median"].isna().sum()
    unsupported_rows = ensemble_df[ensemble_df["source_class_code"].isna()]
    n_finite_unsupported = unsupported_rows["ic_median"].notna().sum() if len(unsupported_rows) else 0
    checks.append(("PASS" if n_nan_supported == 0 and n_finite_unsupported == 0 else "FAIL",
                    f"finite where supported ({n_nan_supported} unexpected NaN), gaps where unsupported "
                    f"({n_finite_unsupported} unexpected finite values)"))

    same_class_boundaries = 0
    for i in range(len(layers_df) - 1):
        if layers_df.iloc[i]["class_code"] == layers_df.iloc[i + 1]["class_code"] and layers_df.iloc[i]["supported"]:
            b_depth = layers_df.iloc[i]["lower_boundary_m"]
            near = ensemble_df[(ensemble_df["depth_m"] - b_depth).abs() < 0.001]
            if len(near) and not near["is_transition"].any():
                same_class_boundaries += 1
    n_same_class = sum(
        1 for i in range(len(layers_df) - 1)
        if layers_df.iloc[i]["class_code"] == layers_df.iloc[i + 1]["class_code"] and layers_df.iloc[i]["supported"]
    )
    checks.append(("PASS" if same_class_boundaries == n_same_class else "FAIL",
                    f"same-class boundaries without artificial transition: {same_class_boundaries}/{n_same_class}"))

    return checks


def main():
    cpt_meta = read_cpt_xml_metadata(CPT_XML_PATH)
    borehole = read_borehole_xml(BOREHOLE_XML_PATH)

    layers_df, ensemble_df, seeds, run_info = build_ensemble(
        borehole, n_realizations=N_REALIZATIONS, base_seed=BASE_SEED, depth_step_m=DEPTH_STEP_M,
    )
    reference_df = build_reference_lengkeek_profile(CPT_XML_PATH)

    aligned_reference_ic = align_reference_to_grid(reference_df, ensemble_df["depth_m"].to_numpy())
    ensemble_df["reference_ic_aligned"] = aligned_reference_ic
    comparison_df = summarize_comparison(ensemble_df, aligned_reference_ic)

    out_dir = OUTPUT_DIR / f"{borehole['bro_id']}_{cpt_meta['bro_id']}_ensemble"
    out_dir.mkdir(parents=True, exist_ok=True)

    ensemble_df.to_csv(out_dir / "ensemble_profile.csv", index=False)
    comparison_df.to_csv(out_dir / "comparison_summary.csv", index=False)

    plot_ensemble_three_panel(
        layers_df, ensemble_df, reference_df, borehole["surface_elevation_m_nap"], N_REALIZATIONS,
        title=f"{borehole['bro_id']} vs {cpt_meta['bro_id']} - {N_REALIZATIONS}-realization ensemble",
        output_path=out_dir / "ensemble_three_panel.png",
    )

    def _rebuild():
        _, e_df, _, _ = build_ensemble(borehole, n_realizations=N_REALIZATIONS, base_seed=BASE_SEED, depth_step_m=DEPTH_STEP_M)
        return e_df

    checks = _run_checks(layers_df, ensemble_df, seeds, run_info, _rebuild)

    summary = {
        "cpt": cpt_meta,
        "borehole_bro_id": borehole["bro_id"],
        "n_realizations": N_REALIZATIONS,
        "depth_step_m": DEPTH_STEP_M,
        "base_seed": BASE_SEED,
        "realization_seeds": seeds,
        "seed_derivation": run_info["seed_derivation"],
        "correlation_length_m (uncalibrated assumption)": run_info["correlation_length_m"],
        "transition_thickness_m_requested": run_info["transition_thickness_m_requested"],
        "groups": run_info["groups"],
        "lengkeek_reference_implementation": LENGKEEK_IMPLEMENTATION_NOTE,
        "band_label": "Central 90% interval across 20 synthetic realizations "
                       "(sampling-assumption variability only, not a calibrated predictive interval; "
                       "tail estimates from n=20 are approximate)",
        "comparison_summary": comparison_df.to_dict("records"),
        "checks": [{"status": s, "detail": d} for s, d in checks],
    }
    with open(out_dir / "ensemble_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"Realization seeds (derived from base_seed={BASE_SEED} via SeedSequence.spawn): {seeds}")
    print()
    print("Checks:")
    for status, detail in checks:
        print(f"  [{status}] {detail}")
    print()
    print("Comparison summary:")
    print(comparison_df.to_string(index=False))
    print(f"\nOutputs written to: {out_dir}")


if __name__ == "__main__":
    main()
