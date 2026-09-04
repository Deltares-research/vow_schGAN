"""
Synthetic CPT generator from a lithology table.

Converts a table of soil layers (defined by NAP elevations and material names)
into a CPT-like Ic profile sampled at 2 cm depth intervals, compatible with
the project's Robertson Ic classification scheme.

Robertson (1990) Ic boundaries used (from config):
    Sand        Ic in [1.00, 2.05]  → representative 1.80
    Sand mix    Ic in (2.05, 2.60]  → representative 2.30
    Silt mix    Ic in (2.60, 2.95]  → representative 2.75
    Clay        Ic in (2.95, 3.40]  → representative 3.20
    Peat        Ic in (3.40, 4.50]  → representative 4.00

Usage (standalone):
    python src/modules/preprocessing/synthetic_cpt.py

Usage (as module):
    from modules.preprocessing.synthetic_cpt import generate_synthetic_cpt, MATERIAL_IC
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage import gaussian_filter1d

# ---------------------------------------------------------------------------
# Material → representative Ic mapping (Robertson 1990 category midpoints)
# ---------------------------------------------------------------------------
MATERIAL_IC: dict[str, float] = {
    "sand":      1.80,
    "gravel":    1.40,
    "sand mix":  2.30,
    "sandmix":   2.30,
    "sandy":     2.10,
    "silt mix":  2.75,
    "siltmix":   2.75,
    "silt":      2.75,
    "clay":      3.20,
    "peat":      4.00,
    "organic":   4.00,
    "humus":     4.00,
}

# Per-material autocorrelated noise parameters (std in Ic units, corr_m = correlation length in metres)
# Sand is noisier and shorter-correlated; clay is smoother and longer-correlated.
MATERIAL_NOISE: dict[str, dict] = {
    "sand":     {"std": 0.18, "corr_m": 0.10},
    "gravel":   {"std": 0.22, "corr_m": 0.08},
    "sand mix": {"std": 0.14, "corr_m": 0.12},
    "sandmix":  {"std": 0.14, "corr_m": 0.12},
    "sandy":    {"std": 0.16, "corr_m": 0.10},
    "silt mix": {"std": 0.12, "corr_m": 0.15},
    "siltmix":  {"std": 0.12, "corr_m": 0.15},
    "silt":     {"std": 0.12, "corr_m": 0.15},
    "clay":     {"std": 0.09, "corr_m": 0.20},
    "peat":     {"std": 0.16, "corr_m": 0.15},
    "organic":  {"std": 0.16, "corr_m": 0.15},
    "humus":    {"std": 0.16, "corr_m": 0.15},
}

# Ic range that noise is allowed to stay within, per material (Robertson 1990 category bounds)
MATERIAL_IC_BOUNDS: dict[str, tuple[float, float]] = {
    "sand":     (1.00, 2.05),
    "gravel":   (1.00, 2.05),
    "sandy":    (1.00, 2.60),
    "sand mix": (2.05, 2.60),
    "sandmix":  (2.05, 2.60),
    "silt mix": (2.60, 2.95),
    "siltmix":  (2.60, 2.95),
    "silt":     (2.60, 2.95),
    "clay":     (2.95, 3.40),
    "peat":     (3.40, 4.50),
    "organic":  (3.40, 4.50),
    "humus":    (3.40, 4.50),
}

# Category boundaries from config (for colormap and legend)
_BOUNDARIES = [1.00, 2.05, 2.60, 2.95, 3.40, 4.50]
_COLORS = ["#FFD700", "#FFA500", "#ADD8E6", "#228B22", "#CC0000"]
_LABELS = ["Sand", "Sand mix", "Silt mix", "Clay", "Peat/Organic"]

CPT_INTERVAL_M = 0.02  # 2 cm


def material_to_ic(name: str) -> float:
    """Return representative Ic for a material name (case-insensitive).

    Tries exact match first, then substring fallback.
    Raises ValueError for completely unknown materials.
    """
    key = name.strip().lower()
    if key in MATERIAL_IC:
        return MATERIAL_IC[key]
    for k, v in MATERIAL_IC.items():
        if k in key or key in k:
            return v
    raise ValueError(
        f"Unknown material: '{name}'. Add it to MATERIAL_IC in synthetic_cpt.py."
    )


def _noise_params(name: str) -> dict:
    """Return noise parameters for a material name, with fuzzy fallback."""
    key = name.strip().lower()
    if key in MATERIAL_NOISE:
        return MATERIAL_NOISE[key]
    for k, v in MATERIAL_NOISE.items():
        if k in key or key in k:
            return v
    return {"std": 0.10, "corr_m": 0.15}


def _ic_bounds(name: str) -> tuple[float, float]:
    """Return the Robertson Ic category bounds for a material name, with fuzzy fallback."""
    key = name.strip().lower()
    if key in MATERIAL_IC_BOUNDS:
        return MATERIAL_IC_BOUNDS[key]
    for k, v in MATERIAL_IC_BOUNDS.items():
        if k in key or key in k:
            return v
    return (0.5, 4.5)  # fallback: no category restriction


def generate_synthetic_cpt(
    layers: pd.DataFrame,
    interval_m: float = CPT_INTERVAL_M,
    noise_std: float | None = None,
    random_seed: int | None = 42,
) -> pd.DataFrame:
    """Generate a synthetic CPT Ic profile from a lithology layer table.

    Args:
        layers: DataFrame with columns:
            ``Top_Elevation_m_NAP``, ``Bottom_Elevation_m_NAP``, ``Main_Material``.
            Extra columns (e.g. Thickness_m) are silently ignored.
        interval_m: Depth sampling interval in metres (default 0.02 = 2 cm).
        noise_std: Per-layer Gaussian noise std-dev (Ic units).
            ``None`` (default) uses realistic material-specific noise from MATERIAL_NOISE.
            ``0.0`` produces a clean step profile with no variability.
        random_seed: Seed for reproducible noise.

    Returns:
        DataFrame with columns:
            ``depth_m`` – depth from surface (0 = top of first layer),
            ``IC``       – Robertson Ic value,
            ``elevation_m_NAP`` – corresponding NAP elevation.
    """
    layers = (
        layers[["Top_Elevation_m_NAP", "Bottom_Elevation_m_NAP", "Main_Material"]]
        .copy()
        .sort_values("Top_Elevation_m_NAP", ascending=False)
        .reset_index(drop=True)
    )

    surface_nap: float = float(layers["Top_Elevation_m_NAP"].max())
    base_nap: float = float(layers["Bottom_Elevation_m_NAP"].min())
    total_depth: float = surface_nap - base_nap

    depths = np.round(np.arange(0.0, total_depth + interval_m * 0.5, interval_m), 6)
    elevations = np.round(surface_nap - depths, 6)

    ic_values = np.zeros(len(depths))
    rng = np.random.default_rng(random_seed)

    for _, row in layers.iterrows():
        top = float(row["Top_Elevation_m_NAP"])
        bot = float(row["Bottom_Elevation_m_NAP"])
        mat = str(row["Main_Material"])
        ic = material_to_ic(mat)
        mask = (elevations <= top + 1e-9) & (elevations >= bot - 1e-9)
        ic_values[mask] = ic  # base value (overwritten below if noise is on)

        n_pts = int(mask.sum())
        if n_pts == 0 or noise_std == 0.0:
            continue

        params = _noise_params(mat) if noise_std is None else {"std": noise_std, "corr_m": 0.10}
        std = params["std"]
        # sigma in samples: correlation length / interval, minimum 1 sample
        sigma = max(1.0, params["corr_m"] / interval_m)
        raw = rng.normal(0.0, std, n_pts)
        if n_pts >= 4 * sigma:
            # Enough samples for meaningful autocorrelation; smooth then rescale
            smooth = gaussian_filter1d(raw, sigma=sigma)
            if smooth.std() > 0:
                smooth *= std / smooth.std()
        else:
            # Layer too thin relative to correlation length; keep uncorrelated noise
            smooth = raw

        # Clip immediately to the Robertson category bounds of this material
        lo, hi = _ic_bounds(mat)
        ic_values[mask] = np.clip(ic + smooth, lo, hi)

    return pd.DataFrame({
        "depth_m": depths,
        "IC": np.round(ic_values, 4),
        "elevation_m_NAP": elevations,
    })


def plot_synthetic_cpt(cpt_df: pd.DataFrame, title: str = "Synthetic CPT") -> plt.Figure:
    """Plot a synthetic CPT profile coloured by Robertson Ic category."""
    fig, ax = plt.subplots(figsize=(4, 8))

    nap = cpt_df["elevation_m_NAP"].values
    ic = cpt_df["IC"].values
    surface_nap = float(nap.max())

    # Fill and line plotted against NAP elevation (primary left y-axis)
    # Robertson convention: lower boundary is exclusive, upper is inclusive (boundary → lower category)
    for i, (lo, hi) in enumerate(zip(_BOUNDARIES[:-1], _BOUNDARIES[1:])):
        mask = (ic >= lo) & (ic <= hi) if i == 0 else (ic > lo) & (ic <= hi)
        if mask.any():
            ax.fill_betweenx(nap, 0, ic, where=mask, color=_COLORS[i], alpha=0.6)

    ax.plot(ic, nap, color="black", linewidth=0.8)
    for bnd in [2.05, 2.60, 2.95, 3.40]:
        ax.axvline(x=bnd, color="gray", linewidth=0.5, linestyle="--", alpha=0.7)

    ax.set_xlabel("Ic (Robertson 1990)")
    ax.set_ylabel("Elevation (m NAP)")
    ax.set_xlim(0.5, 4.5)
    # Surface (highest NAP) at top, base at bottom
    ax.set_ylim(float(nap.min()), float(nap.max()))
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)

    # Secondary right axis: depth from surface (linear transform of NAP elevation)
    sec = ax.secondary_yaxis(
        "right",
        functions=(
            lambda z: surface_nap - z,   # NAP elevation → depth from surface
            lambda d: surface_nap - d,   # depth from surface → NAP elevation
        ),
    )
    sec.set_ylabel("Depth from surface (m)")

    patches = [
        mpatches.Patch(color=_COLORS[i], alpha=0.6, label=_LABELS[i])
        for i in range(len(_LABELS))
    ]
    ax.legend(handles=patches, loc="lower right", fontsize=7)
    fig.tight_layout()
    return fig


def process_borehole_csv(
    borehole_csv: Path,
    output_dir: Path,
    interval_m: float = CPT_INTERVAL_M,
) -> None:
    """Generate and save a synthetic CPT for every borehole in a CSV file.

    Args:
        borehole_csv: Path to CSV with columns Borehole_ID, Top_Elevation_m_NAP,
            Bottom_Elevation_m_NAP, Main_Material (plus any extra columns).
        output_dir: Folder where {Borehole_ID}.csv and {Borehole_ID}.png are saved.
        interval_m: Depth sampling interval in metres.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    all_layers = pd.read_csv(borehole_csv)

    for borehole_id, layers in all_layers.groupby("Borehole_ID"):
        print(f"Processing {borehole_id}  ({len(layers)} layers) …")
        cpt = generate_synthetic_cpt(layers, interval_m=interval_m)

        csv_path = output_dir / f"{borehole_id}.csv"
        cpt.to_csv(csv_path, index=False)

        fig = plot_synthetic_cpt(cpt, title=f"Synthetic CPT – {borehole_id}")
        plot_path = output_dir / f"{borehole_id}.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f"  → {len(cpt)} samples  |  CSV: {csv_path.name}  |  Plot: {plot_path.name}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    BOREHOLE_CSV = Path(
        r"c:\Users\camposmo\OneDrive - Stichting Deltares"
        r"\Documents\Projects\VOW\data\O_62_64\O_62,7\verified_borehole_layers.csv"
    )
    OUTPUT_DIR = Path(r"D:\codes\vow_schGAN\res\cpt-like")

    process_borehole_csv(BOREHOLE_CSV, OUTPUT_DIR)
