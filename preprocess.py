import sys
from pathlib import Path
from typing import Tuple, Optional

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from rembg import remove
from custom_logger import setup_logging


"""
Preprocessing pipeline for CIAK. We select an image and perform the following steps:
1) Load the image
2) Remove the background
3) Normalize the histogram for colors
"""

BACKGROUND_REMOVAL_TOOLS = {"rembg", "dis-bg-remover"}
logger = setup_logging()

# ----------------------------- Logging setup -------------------------------- #


# ----------------------------- Utility funcs -------------------------------- #

def load_image(path: str) -> Image.Image:
    """Load an image from disk using PIL. Returns a PIL Image object."""
    logger.debug(f"Attempting to load image from: {path}")
    try:
        pil_img = Image.open(path)
        logger.info(f"Loaded image: size={pil_img.size}, mode={pil_img.mode}, path={path}")
        return pil_img
    except Exception as e:
        logger.error(f"Failed to load image from {path}: {e}")
        raise FileNotFoundError(f"Image not found or couldn't be opened at {path}")


def save_image(path: str, image) -> None:
    """Save an image to disk, creating parent dirs if needed.
    Handles both PIL Image objects and numpy arrays."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    # Handle PIL Image objects
    if isinstance(image, Image.Image):
        try:
            image.save(path)
            logger.info(f"Saved PIL Image to {path}")
            return
        except Exception as e:
            logger.error(f"Failed to save PIL Image to {path}: {e}")
            raise
    
    # Handle numpy arrays (OpenCV format)
    ok = cv2.imwrite(path, image)
    if not ok:
        logger.error(f"Failed to save image to {path}")
        raise IOError(f"Could not write image to {path}")
    logger.info(f"Saved image to {path}")


def remove_background(input_path: str, output_path: str, mode: str = "rembg") -> None:
    """Remove the background from the image using the specified mode."""
    if mode not in BACKGROUND_REMOVAL_TOOLS:
        logger.error(f"Unknown background removal mode: {mode}")
        raise ValueError(f"mode must be one of {BACKGROUND_REMOVAL_TOOLS}")

    logger.info(f"Background removal: mode={mode}, input={input_path}, output={output_path}")

    if mode == "rembg":
        # Use PIL for rembg; output is usually RGBA with transparent background
        pil_in = Image.open(input_path).convert("RGBA")
        logger.debug(f"Input image for rembg opened via PIL: size={pil_in.size}, mode={pil_in.mode}")
        pil_out = remove(pil_in)  # type: ignore
        logger.debug("Background removed (rembg).")
        pil_out.save(output_path)
        logger.info(f"Background-removed image saved to {output_path}")
    elif mode == "dis-bg-remover":
        # Placeholder for an alternative remover
        logger.error("dis-bg-remover mode is not implemented yet.")
        raise NotImplementedError("dis-bg-remover mode not implemented.")


def _ensure_bgr(image: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Ensure we have a BGR image for processing.
    If input has alpha (BGRA), returns (BGR, alpha). If grayscale, converts to BGR with alpha=None.
    """
    if image.ndim == 2:
        logger.debug("Image is grayscale; converting to BGR.")
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), None

    if image.shape[2] == 4:
        bgr = image[:, :, :3]
        alpha = image[:, :, 3]
        logger.debug("Image has alpha channel; splitting alpha for safe processing.")
        return bgr, alpha

    if image.shape[2] == 3:
        return image, None

    logger.error(f"Unsupported number of channels: {image.shape}")
    raise ValueError("Unsupported image format.")


def normalize_histogram(image: np.ndarray) -> np.ndarray:
    """
    Equalize luminance while preserving color and (if present) alpha.
    Works with BGR, BGRA, or grayscale.
    """
    logger.info("Starting histogram normalization.")
    bgr, alpha = _ensure_bgr(image)

    # Convert to YCrCb and equalize Y channel (luminance).
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y_eq = cv2.equalizeHist(y)
    ycrcb_eq = cv2.merge((y_eq, cr, cb))
    bgr_eq = cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)
    logger.debug("Histogram equalization complete on luminance channel.")

    if alpha is not None:
        # Re-attach alpha channel
        out = np.dstack([bgr_eq, alpha])
        logger.debug("Reattached alpha channel after normalization.")
        return out

    return bgr_eq


def _to_rgb_for_matplotlib(image):
    """Convert OpenCV image (BGR/BGRA/GRAY) or PIL Image to RGB/RGBA for matplotlib display."""
    # Handle PIL Images
    if hasattr(image, 'mode'):  # Check if it's a PIL Image
        if image.mode == 'RGB' or image.mode == 'RGBA':
            return np.array(image)  # PIL RGB/RGBA is already in the right order for matplotlib
        elif image.mode == 'L':
            return np.array(image)  # Grayscale
        else:
            # Convert to RGB or RGBA
            return np.array(image.convert('RGBA' if 'A' in image.mode else 'RGB'))
    
    # Handle NumPy arrays (OpenCV images)
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            return image  # grayscale
        if image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    
    raise ValueError("Unsupported image format for display.")


# # ---------------------------------- Main ------------------------------------ #
# def main():
#     setup_logging(verbosity="DEBUG")  # change to INFO to reduce verbosity

#     input_path = "test.webp"
#     bg_removed_path = "./bg_removed_image.png"
#     logger.info(f"Pipeline starting. input_path={input_path}")

#     try:
#         # Step 1: Load the image
#         image = load_image(input_path)

#         # Step 2: Remove the background
#         remove_background(input_path, bg_removed_path, mode="rembg")
#         bg_removed_image = load_image(bg_removed_path)

#         # Step 3: Normalize the histogram
#         normalized_image = normalize_histogram(bg_removed_image)

#         # Optionally save the normalized image
#         normalized_out_path = "./normalized_image.png"
#         save_image(normalized_out_path, normalized_image)

#         # Display the results
#         logger.info("Rendering comparison figure with matplotlib.")
#         plt.figure(figsize=(15, 6))

#         plt.subplot(1, 3, 1)
#         plt.title("Original Image")
#         plt.imshow(_to_rgb_for_matplotlib(image))
#         plt.axis("off")

#         plt.subplot(1, 3, 2)
#         plt.title("Background Removed")
#         plt.imshow(_to_rgb_for_matplotlib(bg_removed_image))
#         plt.axis("off")

#         plt.subplot(1, 3, 3)
#         plt.title("Normalized Image")
#         plt.imshow(_to_rgb_for_matplotlib(normalized_image))
#         plt.axis("off")

#         plt.tight_layout()
#         plt.show()
#         logger.success("Pipeline completed successfully.")

#     except Exception as e:
#         logger.exception(f"Pipeline failed: {e}")
#         raise


def visualization_comparison(original, pasted_image, normalized_image=None):
        logger.info("Rendering comparison figure with matplotlib.")
        plt.figure(figsize=(15, 6))

        plt.subplot(1, 3, 1)
        plt.title("Original Image")
        plt.imshow(_to_rgb_for_matplotlib(original))
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.title("Background Removed")
        plt.imshow(_to_rgb_for_matplotlib(pasted_image))
        plt.axis("off")

        # plt.subplot(1, 3, 3)
        # plt.title("Normalized Image")
        # plt.imshow(_to_rgb_for_matplotlib(normalized_image))
        # plt.axis("off")

        plt.tight_layout()
        plt.show()
        logger.success("Pipeline completed successfully.")


