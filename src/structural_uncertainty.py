import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import logging


def compute_structural_uncertainty(
    original_csv: Path,
    validation_csvs: list,
    output_folder: Path,
    png_colormap: str = "hot",
    png_title: str = "Structural Uncertainty (RMSE)",
    save_png: bool = True,
    save_csv: bool = True,
):
    """
    Compute pixel-wise RMSE between the original mosaic and all validation mosaics.
    Save the result as CSV and PNG in output_folder.
    """
    logger = logging.getLogger(__name__)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # Load original mosaic
    original = pd.read_csv(original_csv).values
    H, W = original.shape

    # Load all validation mosaics
    val_preds = [pd.read_csv(csv).values for csv in validation_csvs]
    val_preds = np.stack(val_preds, axis=0)  # (K, H, W)

    # Compute squared differences
    sq_diff = (val_preds - original) ** 2  # (K, H, W)
    mse = np.mean(sq_diff, axis=0)  # (H, W)
    rmse = np.sqrt(mse)  # (H, W)

    # Save CSV
    if save_csv:
        csv_path = output_folder / "structural_uncertainty_rmse.csv"
        pd.DataFrame(rmse).to_csv(csv_path, index=False)
        logger.info(f"Structural uncertainty CSV saved: {csv_path}")

    # Save PNG
    if save_png:
        png_path = output_folder / "structural_uncertainty_rmse.png"
        plt.figure(figsize=(16, 4))
        plt.imshow(rmse, cmap=png_colormap, aspect="auto")
        plt.colorbar(label="RMSE (IC)")
        plt.title(png_title)
        plt.xlabel("Column (pixel)")
        plt.ylabel("Row (pixel)")
        plt.tight_layout()
        plt.savefig(png_path, dpi=300)
        plt.close()
        logger.info(f"Structural uncertainty PNG saved: {png_path}")

    return rmse
