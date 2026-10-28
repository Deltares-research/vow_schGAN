"""Run SchemaGAN on a cross-section input matrix and mask the result above the ground surface.

Input (from `run_cross_section_plot.py`, in the cross-section folder):
    <stem>_schemagan_input.csv    32 x 512 matrix, Ic or 0
    <stem>_schemagan_frame.json   frame (top/bottom/cell size) and the profile CSV used for the ground

Steps:
    1. normalise Ic 0..4.3 -> -1..1 (same as the existing pipeline, core.utils.IC_normalization)
    2. SchemaGAN prediction (one pass; the model contains no random seed dependence in inference)
    3. de-normalise back to Ic
    4. mask: keep a cell when its centre is at or below the ground surface (masked = empty / NaN)

Outputs (same folder):
    <stem>_schemagan_output_raw.csv      32 x 512 Ic straight from the model (no mask)
    <stem>_schemagan_output_masked.csv   32 x 512 Ic, empty fields above the ground surface
    <stem>_schemagan_output.png          input | raw output | masked output, plus the masked result in m NAP
    <stem>_schemagan_classes_output.png  the same output in six soil-behaviour classes
                                         (modules/visualization.IC_SOIL_CLASSES: Robertson zones, all Ic < 2.05
                                         as sand, clay split at Ic 3.30 into 3a clay / 3b potential organic clay)

Run from the repository root (NOT inside the sandbox; TensorFlow needs C:\\Users\\<you>\\.keras):
    .venv\\Scripts\\python.exe src\\run_schemagan_generation.py --cs-folder "data\\cs\\cs_O62,7"
Redraw only the class figure from existing outputs (no TensorFlow needed):
    .venv\\Scripts\\python.exe src\\run_schemagan_generation.py --cs-folder "data\\cs\\cs_O62,7" --classes-plot-only
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))  # add src/ to sys.path

from modules.cross_section.cs_mask import apply_mask, ground_at_columns, read_frame, surface_mask, write_matrix  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = REPO_ROOT / "data" / "models" / "schemaGAN.h5"
IC_NORMALISATION_MAX = 4.3  # must match core.utils.IC_normalization / reverse_IC_normalization


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cs-folder", type=Path, required=True, help="Cross-section folder, e.g. data\\cs\\cs_O62,7")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help=f"SchemaGAN .h5 model (default {DEFAULT_MODEL}).")
    parser.add_argument("--classes-plot-only", action="store_true",
                        help="Skip the model; redraw only <stem>_schemagan_classes_output.png from the existing output CSVs.")
    return parser.parse_args()


def find_one(folder: Path, pattern: str) -> Path:
    matches = sorted(folder.glob(pattern))
    if len(matches) != 1:
        sys.exit(f"Expected exactly one '{pattern}' in {folder}, found {[m.name for m in matches]}. "
                 "Run src\\run_cross_section_plot.py first.")
    return matches[0]


def plot_result(input_m, raw, masked, frame, ground, title, path):
    from modules.cross_section.cs_plot import IC_VMAX, IC_VMIN, ROBERTSON_THRESHOLDS, save_png
    from matplotlib.colors import Normalize

    norm = Normalize(IC_VMIN, IC_VMAX)
    fig, axes = plt.subplots(4, 1, figsize=(16, 15), gridspec_kw={"height_ratios": [1, 1, 1, 1.6]})
    extent_idx = (-0.5, frame["n_cols"] - 0.5, frame["n_rows"] - 0.5, -0.5)
    panels = [
        (np.ma.masked_equal(input_m, 0.0), "Input (white = 0, to be filled)"),
        (raw, "SchemaGAN output (raw, no mask)"),
        (np.ma.masked_invalid(masked), "SchemaGAN output, masked above the ground surface (white = masked)"),
    ]
    for ax, (data, label) in zip(axes[:3], panels):
        mesh = ax.imshow(data, cmap="viridis", norm=norm, aspect="auto", interpolation="nearest", extent=extent_idx)
        ax.set_title(label, fontsize=10)
        ax.set_ylabel("row")
        ax.set_xticks(np.arange(0, frame["n_cols"] + 1, 64))
    for col in np.flatnonzero((input_m != 0).any(axis=0)):
        axes[1].plot([col], [-0.5], marker="v", markersize=5, color="black", clip_on=False)

    ax = axes[3]
    left, right = frame["left_m"], frame["right_m"]
    mesh = ax.imshow(np.ma.masked_invalid(masked), cmap="viridis", norm=norm, aspect="auto", interpolation="nearest",
                     extent=(left, right, frame["bottom_m_nap"], frame["top_m_nap"]))
    centres = left + (np.arange(frame["n_cols"]) + 0.5) * frame["col_width_m"]
    ax.plot(centres, ground, color="black", linewidth=1.2, label="ground surface")
    for c in frame["data_columns"]:
        x = left + (c["column"] + 0.5) * frame["col_width_m"]
        ax.annotate("copy" if c["augmented"] else "data", xy=(x, frame["top_m_nap"]), xytext=(0, 2),
                    textcoords="offset points", ha="center", va="bottom", fontsize=6.5,
                    color="dimgray" if c["augmented"] else "black")
        ax.plot([x], [frame["top_m_nap"]], marker="v", markersize=5, clip_on=False,
                color="dimgray" if c["augmented"] else "black")
    ax.set_xlabel("Distance along line from A (m)   (triangles: input data columns; 'copy' = augmented)")
    ax.set_ylabel("Elevation (m NAP)")
    ax.set_title("Masked result in real coordinates", fontsize=10)
    ax.legend(loc="lower right", fontsize=8)
    cax = fig.add_axes([0.93, 0.25, 0.012, 0.5])
    fig.colorbar(mesh, cax=cax, extend="both", ticks=ROBERTSON_THRESHOLDS).set_label("Ic (-)")
    fig.suptitle(title, y=0.995)
    fig.subplots_adjust(top=0.95, hspace=0.35, right=0.91)
    save_png(fig, path)
    plt.close(fig)


def plot_classes(raw, masked, frame, ground, title, path):
    """Secondary figure: SchemaGAN output in six soil-behaviour classes.

    Classes and colours: `modules.visualization.IC_SOIL_CLASSES` (Robertson zones,
    everything below 2.05 is sand, clay split at Ic 3.30 into 3a clay / 3b
    potential organic clay), one light-to-dark ramp per class. Masked cells are white.
    """
    from matplotlib.colors import Normalize
    from modules.cross_section.cs_plot import save_png
    from modules.visualization import IC_SOIL_CLASSES, create_ic_class_colormap, ic_class_boundaries

    inner = ic_class_boundaries()
    names = [f"{code} {label}" for code, label, *_ in IC_SOIL_CLASSES]
    cmap, vmin, vmax = create_ic_class_colormap()
    norm = Normalize(vmin, vmax)

    kept = masked[np.isfinite(masked)]
    shares = np.histogram(kept, bins=[-np.inf, *inner, np.inf])[0] / max(kept.size, 1)
    share_text = ", ".join(f"{code} {s:.1%}" if 0 < s < 0.01 else f"{code} {s:.0%}"
                           for (code, *_), s in zip(IC_SOIL_CLASSES, shares))

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={"height_ratios": [1, 1.6]})
    ax = axes[0]
    ax.imshow(raw, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest",
              extent=(-0.5, frame["n_cols"] - 0.5, frame["n_rows"] - 0.5, -0.5))
    ax.set_title("SchemaGAN output (raw, no mask)", fontsize=10, pad=14)
    ax.set_ylabel("row")
    ax.set_xticks(np.arange(0, frame["n_cols"] + 1, 64))
    for c in frame["data_columns"]:
        ax.plot([c["column"]], [-0.5], marker="v", markersize=5, clip_on=False,
                color="dimgray" if c["augmented"] else "black")

    ax = axes[1]
    left, right = frame["left_m"], frame["right_m"]
    mesh = ax.imshow(np.ma.masked_invalid(masked), cmap=cmap, norm=norm, aspect="auto", interpolation="nearest",
                     extent=(left, right, frame["bottom_m_nap"], frame["top_m_nap"]))
    centres = left + (np.arange(frame["n_cols"]) + 0.5) * frame["col_width_m"]
    ax.plot(centres, ground, color="black", linewidth=1.2, label="ground surface")
    for c in frame["data_columns"]:
        x = left + (c["column"] + 0.5) * frame["col_width_m"]
        ax.annotate("copy" if c["augmented"] else "data", xy=(x, frame["top_m_nap"]), xytext=(0, 2),
                    textcoords="offset points", ha="center", va="bottom", fontsize=6.5,
                    color="dimgray" if c["augmented"] else "black")
        ax.plot([x], [frame["top_m_nap"]], marker="v", markersize=5, clip_on=False,
                color="dimgray" if c["augmented"] else "black")
    ax.set_title(f"Masked result in real coordinates - share of cells below ground per class: {share_text}",
                 fontsize=10, pad=14)
    ax.set_xlabel("Distance along line from A (m)   (triangles: input data columns; 'copy' = augmented)")
    ax.set_ylabel("Elevation (m NAP)")
    ax.legend(loc="lower right", fontsize=8)

    cax = fig.add_axes([0.93, 0.15, 0.012, 0.7])
    cbar = fig.colorbar(mesh, cax=cax, extend="both", ticks=inner)
    cbar.set_label("Ic (-)")
    for name, (_, _, lower, upper, _, _) in zip(names, IC_SOIL_CLASSES):
        cax.text(-1.2, (lower + upper) / 2.0, name, transform=cax.get_yaxis_transform(),
                 ha="right", va="center", fontsize=8)
    fig.suptitle(title, y=0.995)
    fig.subplots_adjust(top=0.93, hspace=0.3, right=0.80)
    save_png(fig, path)
    plt.close(fig)


def main():
    args = _parse_args()
    folder = args.cs_folder
    input_csv = find_one(folder, "*_schemagan_input.csv")
    frame_json = find_one(folder, "*_schemagan_frame.json")
    stem = input_csv.name[: -len("_schemagan_input.csv")]
    frame = read_frame(frame_json)

    if args.classes_plot_only:
        raw = np.genfromtxt(folder / f"{stem}_schemagan_output_raw.csv", delimiter=",")
        masked = np.genfromtxt(folder / f"{stem}_schemagan_output_masked.csv", delimiter=",")
        ground = ground_at_columns(folder / frame["profile_csv"], frame)
        out = folder / f"{stem}_schemagan_classes_output.png"
        plot_classes(raw, masked, frame, ground, f"SchemaGAN generation - {stem} - Ic classes", out)
        print(f"Written: {out}")
        return

    if not args.model.exists():
        sys.exit(f"Model not found: {args.model}")

    input_m = np.loadtxt(input_csv, delimiter=",")
    if input_m.shape != (frame["n_rows"], frame["n_cols"]):
        sys.exit(f"Input matrix shape {input_m.shape} does not match the frame {(frame['n_rows'], frame['n_cols'])}")

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    from tensorflow.keras.models import load_model  # imported late: slow and only needed here

    model = load_model(args.model, compile=False)
    x = 2.0 * (input_m / IC_NORMALISATION_MAX) - 1.0
    y = model.predict(x.reshape(1, frame["n_rows"], frame["n_cols"], 1), verbose=0)
    raw = (np.squeeze(y) + 1.0) * (IC_NORMALISATION_MAX / 2.0)

    ground = ground_at_columns(folder / frame["profile_csv"], frame)
    keep = surface_mask(frame, ground)
    masked = apply_mask(raw, keep)

    write_matrix(raw, folder / f"{stem}_schemagan_output_raw.csv")
    write_matrix(masked, folder / f"{stem}_schemagan_output_masked.csv")
    plot_result(input_m, raw, masked, frame, ground, f"SchemaGAN generation - {stem}",
                folder / f"{stem}_schemagan_output.png")
    plot_classes(raw, masked, frame, ground, f"SchemaGAN generation - {stem} - Ic classes",
                 folder / f"{stem}_schemagan_classes_output.png")

    data = input_m != 0
    diff = np.abs(raw[data] - input_m[data])
    print(f"Model: {args.model}")
    print(f"Output Ic range: {raw.min():.3f} - {raw.max():.3f}; masked cells: {int((~keep).sum())} of {keep.size}")
    print(f"At the input data cells: mean |output - input| = {diff.mean():.3f}, max = {diff.max():.3f}")
    print(f"Outputs written to: {folder}")


if __name__ == "__main__":
    main()
