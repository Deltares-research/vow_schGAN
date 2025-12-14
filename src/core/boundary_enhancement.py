"""
Boundary Enhancement for SchemaGAN outputs.

This module provides methods to sharpen layer boundaries in generated schemas,
making transitions less blurry and more geologically realistic.

Primary method: Dense CRF (Conditional Random Field) - fully connected, edge-aware
refinement that pulls transitions to where the signal actually changes while
preserving interior texture.

Apply these methods after individual schema generation and before mosaic creation.
"""

import numpy as np
import logging
import sys
from pathlib import Path

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Try to import pydensecrf - it's optional but recommended
try:
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import (
        unary_from_softmax,
        create_pairwise_bilateral,
        create_pairwise_gaussian,
    )

    DENSECRF_AVAILABLE = True
except ImportError:
    DENSECRF_AVAILABLE = False
    logger.warning("pydensecrf not available. Install with: pip install pydensecrf")


# =============================================================================
# CONFIG - Default parameters for boundary enhancement
# =============================================================================

# Dense CRF parameters - TUNED FOR SHARPER BOUNDARIES
DCRF_ITERATIONS = 15  # Number of CRF iterations (more iterations = sharper boundaries)
DCRF_SPATIAL_SIGMA_X = 3  # Horizontal spatial std dev in pixels (smaller = sharper)
DCRF_SPATIAL_SIGMA_Y = (
    0.8  # Vertical spatial std dev in rows (smaller = respects thin layers better)
)
DCRF_BILATERAL_SIGMA_VAL = 0.1  # Value-space std dev in Ic units (SMALLER = sharper boundaries, less texture smoothing)
DCRF_SPATIAL_WEIGHT = (
    5  # Weight for spatial smoothness term (higher = stronger smoothing)
)
DCRF_BILATERAL_WEIGHT = (
    10  # Weight for bilateral (edge-aware) term (HIGHER = stronger edge preservation)
)

# Value discretization for CRF (convert continuous Ic to discrete classes)
DCRF_N_CLASSES = 46  # Number of discrete classes (0.0 to 4.5 in 0.1 steps)
DCRF_MIN_VAL = 0.0
DCRF_MAX_VAL = 4.5

# Unsharp Masking parameters
UNSHARP_ALPHA = 3.0  # Sharpening strength (1.5-5.0, higher = sharper boundaries)
UNSHARP_SIGMA = 1.0  # Gaussian blur sigma (0.5-2.0, smaller = sharpen finer details)
UNSHARP_ANISOTROPIC = True  # Use different sigma for vertical (depth) direction

# Laplacian Sharpening parameters
LAPLACIAN_ALPHA = 0.8  # Sharpening strength (0.3-2.0, higher = more aggressive)
LAPLACIAN_KSIZE = 3  # Kernel size: 1, 3, or 5 (larger = coarser features)

# Guided Filter parameters
GUIDED_RADIUS = (
    4  # Filter radius in pixels (2-8, larger = smoother but preserves edges)
)
GUIDED_EPS = (
    0.01  # Regularization (1e-4 to 1e-1, smaller = sharper edges, larger = smoother)
)


def apply_dense_crf(
    image: np.ndarray,
    iterations: int = DCRF_ITERATIONS,
    spatial_sigma_x: float = DCRF_SPATIAL_SIGMA_X,
    spatial_sigma_y: float = DCRF_SPATIAL_SIGMA_Y,
    bilateral_sigma_val: float = DCRF_BILATERAL_SIGMA_VAL,
    spatial_weight: float = DCRF_SPATIAL_WEIGHT,
    bilateral_weight: float = DCRF_BILATERAL_WEIGHT,
    n_classes: int = DCRF_N_CLASSES,
    min_val: float = DCRF_MIN_VAL,
    max_val: float = DCRF_MAX_VAL,
) -> np.ndarray:
    """
    Apply Dense CRF to sharpen boundaries in a generated schema image.

    Dense CRF performs edge-aware smoothing that:
    - Sharpens boundaries where values change significantly
    - Reduces noise/pepper artifacts
    - Preserves texture within homogeneous regions (controlled by bilateral_sigma_val)

    Args:
        image: 2D numpy array (H, W) with Ic values, typically in range [0, 4.5]
        iterations: Number of mean-field inference iterations (5-10 typical)
        spatial_sigma_x: Horizontal spatial std dev in pixels (3-6 for smoothing across space)
        spatial_sigma_y: Vertical spatial std dev in rows (1-2 for respecting thin layers)
        bilateral_sigma_val: Value-space std dev in Ic units (0.15-0.25 to preserve texture)
        spatial_weight: Weight for spatial smoothness term (3-5 typical)
        bilateral_weight: Weight for bilateral edge-aware term (3-5 typical)
        n_classes: Number of discrete classes for CRF
        min_val: Minimum Ic value
        max_val: Maximum Ic value

    Returns:
        Enhanced image (H, W) with sharper boundaries, same value range as input

    Raises:
        ImportError: If pydensecrf is not installed

    Example:
        >>> schema = np.load("schema.npy")  # (32, 512)
        >>> enhanced = apply_dense_crf(schema, iterations=10, spatial_sigma_x=5, spatial_sigma_y=1.5)
    """
    if not DENSECRF_AVAILABLE:
        raise ImportError(
            "pydensecrf is required for Dense CRF. Install with: pip install pydensecrf\n"
            "Note: On Windows, you may need: pip install git+https://github.com/lucasb-eyer/pydensecrf.git"
        )

    if image.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {image.shape}")

    H, W = image.shape

    # 1. Discretize continuous Ic values to class labels
    # Map [min_val, max_val] to [0, n_classes-1]
    image_clipped = np.clip(image, min_val, max_val)
    class_indices = (
        (image_clipped - min_val) / (max_val - min_val) * (n_classes - 1)
    ).astype(np.int32)
    class_indices = np.clip(class_indices, 0, n_classes - 1)

    # 2. Create unary potentials (negative log-probabilities)
    # Convert hard labels to soft probabilities with high confidence
    unary = np.zeros((n_classes, H * W), dtype=np.float32)
    flat_labels = class_indices.flatten()
    for i in range(H * W):
        unary[:, i] = 1.0  # Small uniform probability for all classes
        unary[flat_labels[i], i] = 0.01  # High confidence for observed class

    # Negative log probability
    unary = -np.log(unary)

    # 3. Setup Dense CRF
    d = dcrf.DenseCRF2D(W, H, n_classes)
    d.setUnaryEnergy(unary)

    # 4. Add pairwise potentials

    # Spatial smoothness term (Gaussian): encourages nearby pixels to have similar labels
    # Uses anisotropic smoothing (different sigma for x and y) to respect layer geometry
    pairwise_energy = create_pairwise_gaussian(
        sdims=(spatial_sigma_x, spatial_sigma_y), shape=(H, W)  # (sx, sy)
    )
    d.addPairwiseEnergy(pairwise_energy, compat=spatial_weight)

    # Bilateral term (edge-aware): preserves boundaries where color/value changes
    # Combines spatial proximity with value similarity
    # Small bilateral_sigma_val = only similar values get smoothed together (preserves texture)
    # Large bilateral_sigma_val = more aggressive smoothing (may flatten interiors)

    # For bilateral, we need RGB-like input. Use image values in all 3 channels
    # Scale to 0-255 range for bilateral term
    image_rgb = np.stack([image_clipped] * 3, axis=-1)  # (H, W, 3)
    image_rgb = ((image_rgb - min_val) / (max_val - min_val) * 255).astype(np.uint8)

    pairwise_bilateral = create_pairwise_bilateral(
        sdims=(spatial_sigma_x, spatial_sigma_y),  # Spatial sigmas
        schan=(bilateral_sigma_val * 255 / (max_val - min_val),)
        * 3,  # Value-space sigma (scaled to 0-255)
        img=image_rgb,
        chdim=2,  # Channel dimension
    )
    d.addPairwiseEnergy(pairwise_bilateral, compat=bilateral_weight)

    # 5. Inference
    Q = d.inference(iterations)

    # 6. Get most probable class for each pixel
    Q = np.array(Q).reshape((n_classes, H, W))
    map_labels = np.argmax(Q, axis=0)  # (H, W)

    # 7. Convert back to continuous Ic values
    enhanced = (map_labels.astype(np.float32) / (n_classes - 1)) * (
        max_val - min_val
    ) + min_val

    return enhanced


def apply_unsharp_mask(
    image: np.ndarray,
    alpha: float = UNSHARP_ALPHA,
    sigma: float = UNSHARP_SIGMA,
    anisotropic: bool = UNSHARP_ANISOTROPIC,
    min_val: float = DCRF_MIN_VAL,
    max_val: float = DCRF_MAX_VAL,
) -> np.ndarray:
    """
    Apply unsharp masking to sharpen layer boundaries.

    Unsharp masking is the classic image sharpening technique:
    1. Create a blurred version of the image
    2. Subtract blur from original to get edge detail
    3. Add amplified edge detail back to original

    This makes boundaries crisper by enhancing local contrast at transitions.

    Args:
        image: 2D numpy array (H, W) with Ic values, typically in range [0, 4.5]
        alpha: Sharpening strength (1.5-5.0 typical)
            - 1.5-2.5: Subtle sharpening
            - 2.5-4.0: Moderate sharpening (recommended)
            - 4.0+: Aggressive sharpening (may create artifacts)
        sigma: Gaussian blur kernel size (0.5-2.0)
            - Smaller values sharpen finer details
            - Larger values sharpen coarser features
        anisotropic: If True, use smaller sigma in vertical (depth) direction
            to respect thin horizontal layers
        min_val: Minimum Ic value for clipping
        max_val: Maximum Ic value for clipping

    Returns:
        Sharpened image (H, W) with crisper boundaries

    Example:
        >>> schema = np.load("schema.npy")  # (32, 512)
        >>> sharpened = apply_unsharp_mask(schema, alpha=3.0, sigma=1.0)
    """
    from scipy.ndimage import gaussian_filter

    if image.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {image.shape}")

    # Apply Gaussian blur
    if anisotropic:
        # Use smaller sigma in vertical (row/depth) direction to preserve thin layers
        # Typical geotechnical data has horizontal layers, so vertical smoothing should be minimal
        sigma_y = sigma * 0.5  # Half the vertical sigma
        sigma_x = sigma
        blurred = gaussian_filter(image, sigma=(sigma_y, sigma_x))
    else:
        blurred = gaussian_filter(image, sigma=sigma)

    # Extract edge detail (high-frequency components lost in blurring)
    edge_detail = image - blurred

    # Amplify edges and add back to original
    sharpened = image + alpha * edge_detail

    # Clip to valid range
    sharpened = np.clip(sharpened, min_val, max_val)

    return sharpened


def apply_laplacian_sharpen(
    image: np.ndarray,
    alpha: float = LAPLACIAN_ALPHA,
    ksize: int = LAPLACIAN_KSIZE,
    min_val: float = DCRF_MIN_VAL,
    max_val: float = DCRF_MAX_VAL,
) -> np.ndarray:
    """
    Apply Laplacian-based sharpening to enhance boundaries.

    The Laplacian operator detects edges by computing the second derivative.
    Subtracting the Laplacian from the original image sharpens edges.
    This is more aggressive than unsharp masking and can create strong contrast.

    Formula: sharpened = original - alpha × Laplacian(original)

    Args:
        image: 2D numpy array (H, W) with Ic values, typically in range [0, 4.5]
        alpha: Sharpening strength (0.3-2.0 typical)
            - 0.3-0.7: Subtle sharpening
            - 0.7-1.2: Moderate sharpening (recommended)
            - 1.2+: Aggressive sharpening (may overshoot/ring)
        ksize: Laplacian kernel size (1, 3, or 5)
            - 1: Fine detail sharpening
            - 3: Standard (recommended)
            - 5: Coarser features
        min_val: Minimum Ic value for clipping
        max_val: Maximum Ic value for clipping

    Returns:
        Sharpened image (H, W) with enhanced boundaries

    Warning:
        Can create "ringing" artifacts (overshoot/undershoot) at strong edges.
        Use lower alpha values if artifacts appear.

    Example:
        >>> schema = np.load("schema.npy")  # (32, 512)
        >>> sharpened = apply_laplacian_sharpen(schema, alpha=0.8, ksize=3)
    """
    from scipy.ndimage import laplace

    if image.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {image.shape}")

    # Compute Laplacian (detects edges/boundaries)
    # Note: scipy's laplace uses a default kernel that approximates second derivative
    laplacian = laplace(image.astype(np.float64))

    # Sharpen by subtracting Laplacian
    # (negative of second derivative enhances edges)
    sharpened = image - alpha * laplacian

    # Clip to valid range
    sharpened = np.clip(sharpened, min_val, max_val)

    return sharpened.astype(np.float32)


def apply_guided_filter(
    image: np.ndarray,
    radius: int = GUIDED_RADIUS,
    eps: float = GUIDED_EPS,
    min_val: float = DCRF_MIN_VAL,
    max_val: float = DCRF_MAX_VAL,
) -> np.ndarray:
    """
    Apply guided filter for edge-preserving smoothing and boundary sharpening.

    The guided filter is an edge-aware filter that uses the image itself as a guide
    to preserve edges while smoothing regions. Unlike bilateral filter, it:
    - Has better edge preservation with no halos
    - Is computationally efficient (linear time)
    - Preserves texture within layers while sharpening boundaries

    This is particularly effective for images that are already smooth but need
    edge recovery (like GAN outputs).

    Args:
        image: 2D numpy array (H, W) with Ic values, typically in range [0, 4.5]
        radius: Filter radius in pixels (2-8 typical)
            - Smaller values: local filtering, sharper details
            - Larger values: more smoothing, but edges still preserved
            Recommended: 3-5 for 32-row images
        eps: Regularization parameter (1e-4 to 1e-1)
            - Smaller values: sharper edges, more contrast
            - Larger values: smoother transitions
            Recommended: 0.001-0.01 (as fraction of Ic range)
        min_val: Minimum Ic value for clipping
        max_val: Maximum Ic value for clipping

    Returns:
        Filtered image (H, W) with enhanced boundaries and preserved texture

    Note:
        The guided filter uses the image itself as both input and guide.
        This creates an edge-preserving smoothing effect that can sharpen
        boundaries without creating halos or artifacts.

    Example:
        >>> schema = np.load("schema.npy")  # (32, 512)
        >>> enhanced = apply_guided_filter(schema, radius=4, eps=0.01)
    """
    try:
        import cv2
    except ImportError:
        raise ImportError(
            "OpenCV is required for guided filter. Install with: pip install opencv-python"
        )

    if image.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {image.shape}")

    # Convert to float32 if needed
    img = image.astype(np.float32)

    # Normalize to 0-1 range for better numerical stability
    img_norm = (img - min_val) / (max_val - min_val)

    # Scale eps to normalized range
    eps_scaled = eps / (max_val - min_val)

    # Apply guided filter (using image as both guide and input)
    # This creates edge-preserving smoothing
    filtered_norm = cv2.ximgproc.guidedFilter(
        guide=img_norm.astype(np.float32),
        src=img_norm.astype(np.float32),
        radius=radius,
        eps=eps_scaled,
    )

    # Scale back to original range
    filtered = filtered_norm * (max_val - min_val) + min_val

    # Clip to valid range
    filtered = np.clip(filtered, min_val, max_val)

    return filtered.astype(np.float32)


def enhance_schema(
    schema: np.ndarray, method: str = "guided_filter", **kwargs
) -> Tuple[np.ndarray, str]:
    """
    Apply boundary enhancement to a schema image.

    Args:
        schema: 2D numpy array (H, W) with Ic values
        method: Enhancement method to use. Options:
            - "guided_filter": Guided filter edge-preserving (RECOMMENDED - best for GAN outputs)
            - "unsharp_mask": Unsharp masking (classic sharpening)
            - "laplacian": Laplacian-based sharpening (more aggressive)
            - "dense_crf": Dense CRF edge-aware smoothing (experimental)
            - "none": No enhancement (returns original)
        **kwargs: Method-specific parameters (passed to enhancement function)

    Returns:
        (enhanced_schema, method_name) tuple

    Example:
        >>> schema = np.load("schema.npy")
        >>> enhanced, method = enhance_schema(schema, method="guided_filter", radius=4, eps=0.01)
    """
    if method == "none" or method is None:
        return schema.copy(), "none"

    elif method == "guided_filter":
        try:
            enhanced = apply_guided_filter(schema, **kwargs)
            return enhanced, "guided_filter"
        except ImportError as e:
            logger.warning(f"Guided filter not available: {e}")
            logger.warning(
                "Returning original image. Install opencv-contrib-python for guided filter."
            )
            return schema.copy(), "none"

    elif method == "unsharp_mask":
        enhanced = apply_unsharp_mask(schema, **kwargs)
        return enhanced, "unsharp_mask"

    elif method == "laplacian":
        enhanced = apply_laplacian_sharpen(schema, **kwargs)
        return enhanced, "laplacian"

    elif method == "dense_crf":
        if not DENSECRF_AVAILABLE:
            logger.warning("Dense CRF not available, returning original image")
            return schema.copy(), "none"

        enhanced = apply_dense_crf(schema, **kwargs)
        return enhanced, "dense_crf"

    else:
        raise ValueError(
            f"Unknown enhancement method: {method}. Options: 'guided_filter', 'unsharp_mask', 'laplacian', 'dense_crf', 'none'"
        )


def enhance_schema_from_file(
    input_path: Path,
    output_path: Optional[Path] = None,
    method: str = "guided_filter",
    **kwargs,
) -> Tuple[Path, str]:
    """
    Load a schema CSV, apply enhancement, and save the result.

    Args:
        input_path: Path to input CSV file (generated schema)
        output_path: Path for output CSV (if None, creates _enhanced.csv variant)
        method: Enhancement method to use ("guided_filter", "unsharp_mask", "laplacian", "dense_crf", or "none")
        **kwargs: Method-specific parameters

    Returns:
        (output_path, method_name) tuple

    Example:
        >>> from pathlib import Path
        >>> input_csv = Path("section_001_seed123_gan.csv")
        >>> output_csv, method = enhance_schema_from_file(input_csv, method="guided_filter")
    """
    import pandas as pd

    # Load schema
    df = pd.read_csv(input_path, header=None)
    schema = df.to_numpy(dtype=np.float32)

    # Apply enhancement
    enhanced, method_used = enhance_schema(schema, method=method, **kwargs)

    # Determine output path
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_enhanced.csv"

    # Save enhanced schema
    pd.DataFrame(enhanced).to_csv(output_path, index=False, header=False)

    logger.info(f"Enhanced {input_path.name} -> {output_path.name} using {method_used}")

    return output_path, method_used


def create_enhanced_png(
    enhanced_csv: Path,
    output_png: Path,
    manifest_csv: Path,
    coords_csv: Path,
    y_top_m: float,
    y_bottom_m: float,
    show_cpt_locations: bool = True,
    top_axis_0_to_32: bool = False,
):
    """
    Create a PNG visualization of an enhanced schema.

    This matches the style used for original GAN images with dual axes,
    CPT position markers, and proper depth/distance labeling.

    Args:
        enhanced_csv: Path to enhanced schema CSV
        output_png: Path for output PNG file
        manifest_csv: Path to manifest CSV (to get section metadata)
        coords_csv: Path to coordinates CSV (to get CPT positions)
        y_top_m: Top depth in meters
        y_bottom_m: Bottom depth in meters
        show_cpt_locations: Whether to show CPT position markers
        top_axis_0_to_32: Use normalized 0-32 axis instead of pixel indices
    """
    import pandas as pd
    import matplotlib.pyplot as plt

    # Load enhanced data
    df = pd.read_csv(enhanced_csv, header=None)
    enhanced = df.to_numpy(dtype=np.float32)
    SIZE_Y, SIZE_X = enhanced.shape

    # Load manifest and coords
    manifest = pd.read_csv(manifest_csv)
    coords = pd.read_csv(coords_csv)

    # Find manifest row for this file
    csv_name = enhanced_csv.name
    matches = manifest[
        manifest["csv_path"].apply(lambda p: Path(p).name)
        == csv_name.replace("_gan.csv", "_cpts_*.csv")
    ]

    # Try different matching patterns
    if matches.empty:
        # Extract section index from filename (e.g., section_001_z_00_seed123_gan.csv)
        import re

        match = re.search(r"section_(\d+)", csv_name)
        if match:
            sec_index = int(match.group(1))
            matches = manifest[manifest["section_index"] == sec_index]
            if "depth_window" in manifest.columns:
                # Also try to match depth window
                z_match = re.search(r"z_(\d+)", csv_name)
                if z_match:
                    depth_win = int(z_match.group(1))
                    matches = matches[matches["depth_window"] == depth_win]

    if matches.empty:
        # Fallback: use first manifest row
        mrow = manifest.iloc[0]
    else:
        mrow = matches.iloc[0]

    # Calculate x-axis extent
    total_span = float(mrow["span_m"] + mrow["left_pad_m"] + mrow["right_pad_m"])
    start_idx = int(mrow["start_idx"])
    m0 = float(coords.loc[start_idx, "cum_along_m"])
    x0 = m0 - float(mrow["left_pad_m"])
    dx = 1.0 if total_span <= 0 else total_span / (SIZE_X - 1)
    x1 = x0 + (SIZE_X - 1) * dx

    # Get CPT positions for this section
    end_idx = start_idx + int(mrow.get("n_cpts", 1)) - 1
    cpt_positions = coords.loc[start_idx:end_idx, "cum_along_m"].values

    # Extract section info
    sec_index = int(mrow["section_index"])
    depth_win = int(mrow["depth_window"]) if "depth_window" in mrow.index else None

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 2.4))

    im = ax.imshow(
        enhanced,
        cmap="viridis",
        vmin=0,
        vmax=4.5,
        aspect="auto",
        extent=[x0, x1, SIZE_Y - 1, 0],
    )
    plt.colorbar(im, label="Value")

    ax.set_xlabel("Distance along line (m)")
    ax.set_ylabel("Depth Index")

    # Top x-axis
    if not top_axis_0_to_32:

        def m_to_px(x):
            return (x - x0) / dx

        def px_to_m(p):
            return x0 + p * dx

        top = ax.secondary_xaxis("top", functions=(m_to_px, px_to_m))
        top.set_xlabel(f"Pixel index (0…{SIZE_X-1})")
    else:

        def m_to_u32(x):
            return 32.0 * (x - x0) / (x1 - x0 + 1e-12)

        def u32_to_m(u):
            return x0 + (u / 32.0) * (x1 - x0)

        top = ax.secondary_xaxis("top", functions=(m_to_u32, u32_to_m))
        top.set_xlabel("Normalized distance (0…32)")

    # Right y-axis: real depth
    def idx_to_meters(y_idx):
        return y_top_m + (y_idx / (SIZE_Y - 1)) * (y_bottom_m - y_top_m)

    def meters_to_idx(y_m):
        denom = y_bottom_m - y_top_m
        return 0.0 if abs(denom) < 1e-12 else (y_m - y_top_m) * (SIZE_Y - 1) / denom

    right = ax.secondary_yaxis("right", functions=(idx_to_meters, meters_to_idx))
    right.set_ylabel("Depth (m)")

    # Add CPT position markers
    if show_cpt_locations:
        for cpt_x in cpt_positions:
            ax.axvline(
                x=cpt_x, color="black", linewidth=1, linestyle="-", alpha=0.5, zorder=10
            )

    # Set x-limits
    ax.set_xlim(x0, x1)

    # Title
    if depth_win is not None:
        title = f"Enhanced Schema (Section {sec_index:03d}, z_{depth_win:02d})"
    else:
        title = f"Enhanced Schema (Section {sec_index:03d})"

    plt.title(title, fontsize=config.PLOT_FONT_SIZE)
    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Created enhanced PNG: {output_png.name}")

    # Create interactive HTML
    try:
        from utils import create_interactive_html

        html_path = output_png.with_suffix(".html")
        create_interactive_html(output_png, html_path, title=title)
    except Exception:
        pass  # Silently skip if fails


# =============================================================================
# Utility functions for parameter tuning
# =============================================================================


def compare_enhancement_params(
    schema: np.ndarray, param_grid: dict, output_dir: Optional[Path] = None
) -> list:
    """
    Test multiple parameter combinations for Dense CRF and optionally save results.

    Useful for finding optimal parameters for your specific data.

    Args:
        schema: Input schema to enhance
        param_grid: Dictionary of parameter lists to test, e.g.:
            {
                'iterations': [5, 10],
                'spatial_sigma_x': [3, 5],
                'bilateral_sigma_val': [0.15, 0.2, 0.25]
            }
        output_dir: If provided, save comparison images

    Returns:
        List of (params_dict, enhanced_schema) tuples

    Example:
        >>> schema = np.load("schema.npy")
        >>> param_grid = {
        ...     'iterations': [5, 10],
        ...     'spatial_sigma_x': [3, 5, 7],
        ...     'bilateral_sigma_val': [0.15, 0.2, 0.25]
        ... }
        >>> results = compare_enhancement_params(schema, param_grid)
    """
    from itertools import product

    if not DENSECRF_AVAILABLE:
        logger.error("Dense CRF not available for parameter comparison")
        return []

    # Generate all parameter combinations
    keys = list(param_grid.keys())
    values = [
        param_grid[k] if isinstance(param_grid[k], (list, tuple)) else [param_grid[k]]
        for k in keys
    ]

    results = []
    for combo in product(*values):
        params = dict(zip(keys, combo))
        logger.info(f"Testing params: {params}")

        try:
            enhanced = apply_dense_crf(schema, **params)
            results.append((params, enhanced))

            if output_dir is not None:
                output_dir.mkdir(parents=True, exist_ok=True)
                param_str = "_".join(f"{k}{v}" for k, v in params.items())
                output_path = output_dir / f"enhanced_{param_str}.npy"
                np.save(output_path, enhanced)
                logger.info(f"Saved to {output_path}")

        except Exception as e:
            logger.error(f"Failed with params {params}: {e}")

    return results


if __name__ == "__main__":
    """
    Example usage and testing.
    """
    print("Boundary Enhancement Module")
    print(f"Dense CRF available: {DENSECRF_AVAILABLE}")

    if DENSECRF_AVAILABLE:
        # Create a synthetic test image with layer boundaries
        print("\nCreating test image with synthetic layers...")
        test_img = np.zeros((32, 512))
        test_img[0:10, :] = 1.5  # Layer 1
        test_img[10:20, :] = 2.8  # Layer 2
        test_img[20:32, :] = 3.5  # Layer 3

        # Add some noise to make it more realistic
        noise = np.random.normal(0, 0.2, test_img.shape)
        test_img += noise
        test_img = np.clip(test_img, 0, 4.5)

        print(f"Test image shape: {test_img.shape}")
        print(f"Test image range: [{test_img.min():.2f}, {test_img.max():.2f}]")

        # Apply enhancement
        print("\nApplying Dense CRF enhancement...")
        enhanced = apply_dense_crf(test_img, iterations=10)

        print(f"Enhanced image shape: {enhanced.shape}")
        print(f"Enhanced image range: [{enhanced.min():.2f}, {enhanced.max():.2f}]")
        print("\nEnhancement complete! Boundaries should be sharper.")
    else:
        print("\nTo use Dense CRF, install pydensecrf:")
        print("  pip install pydensecrf")
        print(
            "  or on Windows: pip install git+https://github.com/lucasb-eyer/pydensecrf.git"
        )
