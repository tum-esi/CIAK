"""
    Attacker class that implements the pipeline of CIAK.
"""

from pyparsing import Optional, Path
from preprocess import load_image, save_image, remove_background, normalize_histogram, visualization_comparison
from custom_logger import setup_logging
import cv2
import matplotlib.pyplot as plt
import os
from PIL import Image, ImageChops, ImageFilter
import numpy as np
from typing import Union, Tuple
from collections import namedtuple
BBox = namedtuple('BBox', ['x', 'y', 'w', 'h'])




class Attacker:
    def __init__(self) -> None:
        self.project_root = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
        self.assets_dir = os.path.join(os.path.dirname(__file__), "./assets")
        self.logger = setup_logging()

    def attacker_image(self, input_path: str):
        return load_image(input_path)

    @staticmethod
    def resize_to_bbox(model_img: Image.Image, bbox_w: int, bbox_h: int, fit: str = "contain") -> Image.Image:
        if not isinstance(model_img, Image.Image):
            raise TypeError(f"model_img must be PIL.Image.Image, got {type(model_img)}")
        if bbox_w <= 0 or bbox_h <= 0:
            raise ValueError("bbox_w and bbox_h must be > 0")

        r_w = bbox_w / model_img.width
        r_h = bbox_h / model_img.height
        scale = min(r_w, r_h) if fit == "contain" else max(r_w, r_h)

        new_w = max(1, int(round(model_img.width * scale)))
        new_h = max(1, int(round(model_img.height * scale)))

        # Return a NEW image; do not mutate inputs
        return model_img.resize((new_w, new_h), Image.LANCZOS)

    def load_model_image(self, model_path: str):
        self.model_image = load_image(model_path)
        return self.model_image

    @staticmethod
    def trim_transparent_borders(img: Image.Image) -> Image.Image:
        """
        Crop the image to the smallest box containing all non-transparent pixels (alpha > 0).
        """
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        alpha = img.split()[-1]
        bbox = alpha.getbbox()
        if bbox:
            return img.crop(bbox)
        return img  # No non-transparent pixels, return as is

    def run_pipeline(self, model_image_path: str, input_path_attacker: str, output_dir: str, preprocess: bool, injection_coords: BBox) -> None:
        """
        Executes the image tampering pipeline.

        Parameters:
            model_image_path (str): Path to the model (injection) image file. Should be a string path to an image (e.g., PNG with alpha).
            input_path_attacker (str): Path to the attacker (background) image file. Should be a string path to an image.
            output_dir (str): Directory where the processed (tampered) image will be saved. Should be a string path to a directory.
            preprocess (bool): If True, only preprocess the model image (remove background and trim transparent borders), then exit.
            injection_coords (BBox): Bounding box specifying where and how large to inject the model image. 
                Should be a BBox namedtuple with fields (x, y, w, h), where:
                    x (int): X-coordinate (left) of the injection location in the attacker image.
                    y (int): Y-coordinate (top) of the injection location in the attacker image.
                    w (int): Width to which the model image should be resized before injection.
                    h (int): Height to which the model image should be resized before injection.

        Returns:
            None. The processed image is saved to the specified output directory.
        """
        self.logger.info(f"Pipeline starting.")

        try:
            if preprocess:
                # Preprocess the model image (remove bg + normalize)
                preprocessed_image_save_path = os.path.join(self.assets_dir, "garage", os.path.basename(model_image_path).split(".")[0] + "_no_bg.png")
                remove_background(model_image_path, preprocessed_image_save_path, mode="rembg")
                # Trim transparent borders after background removal
                img = load_image(preprocessed_image_save_path)
                img = self.trim_transparent_borders(img)
                save_image(preprocessed_image_save_path, img)
                self.logger.info(f"Saved trimmed image to {preprocessed_image_save_path}")
                model_image_path = preprocessed_image_save_path  # Use the preprocessed image for further steps
               
            # Load
            model_image = self.load_model_image(model_image_path)     # RGBA per your logs
            attacker_image = self.attacker_image(input_path_attacker)   # RGBA per your logs

            # Clean the matte first (fixes white edges)
            model_image = dematte_white(model_image)

            # Optional: tighten the alpha to reduce any residual halo by 1px
            model_image = tighten_alpha_border(model_image, shrink_px=1)

            # Resize with proper alpha handling (no halos)
            model_image = resize_to_bbox_antihalo(model_image, injection_coords.w, injection_coords.h, fit="contain")

            # Paste using the alpha as mask (critical!)
            back_image = attacker_image.copy()
            paste_rgba(back_image, model_image, (injection_coords.x, injection_coords.y))

            # Save the processed image
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, os.path.basename(input_path_attacker))
            save_image(output_path, back_image)
            self.logger.info(f"Saved processed image to {output_path}")

            # visualization_comparison(attacker_image, back_image)



        except Exception as e:
            self.logger.exception(f"Pipeline failed: {e}")
            raise

    @staticmethod
    def _to_rgb_for_matplotlib(image):
        if image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        raise ValueError("Unsupported image format for display.")



def paste_rgba(dst: Image.Image, src: Image.Image, xy: Tuple[int, int]) -> None:
    """Paste using the source's alpha channel explicitly."""
    if src.mode != "RGBA":
        src = src.convert("RGBA")
    dst.paste(src, xy, mask=src.split()[-1])

def dematte_white(img: Image.Image) -> Image.Image:
    """
    Remove white matte from semi-transparent edges:
    C_unmatted = (C_matted - (1-a)) / a, for a in (0,1]; white matte = 1.
    """
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    arr = np.array(img).astype(np.float32) / 255.0
    rgb = arr[..., :3]
    a   = arr[..., 3:4]  # keep dims

    # avoid division by zero
    eps = 1e-6
    a_safe = np.maximum(a, eps)

    # dematte from white (1.0)
    rgb_unmatted = (rgb - (1.0 - a)) / a_safe

    # where alpha==0, keep zeros
    rgb_unmatted = np.where(a > eps, np.clip(rgb_unmatted, 0.0, 1.0), 0.0)

    out = np.dstack((rgb_unmatted, a))
    out = (out * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(out, mode="RGBA")

def resize_rgba_antihalo(img: Image.Image, new_w: int, new_h: int) -> Image.Image:
    """
    Resize an RGBA image with alpha-premultiplication to avoid light/dark halos.
    """
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    r, g, b, a = img.split()

    # premultiply: channel = channel * alpha
    r = ImageChops.multiply(r, a)
    g = ImageChops.multiply(g, a)
    b = ImageChops.multiply(b, a)

    # resample all bands
    a2 = a.resize((new_w, new_h), Image.LANCZOS)
    r2 = r.resize((new_w, new_h), Image.LANCZOS)
    g2 = g.resize((new_w, new_h), Image.LANCZOS)
    b2 = b.resize((new_w, new_h), Image.LANCZOS)

    # unpremultiply safely
    arr_r = np.array(r2, dtype=np.float32)
    arr_g = np.array(g2, dtype=np.float32)
    arr_b = np.array(b2, dtype=np.float32)
    arr_a = np.array(a2, dtype=np.float32)

    eps = 1e-6
    denom = np.maximum(arr_a, eps) / 255.0
    rr = np.where(arr_a > 0, arr_r / denom, 0)
    gg = np.where(arr_a > 0, arr_g / denom, 0)
    bb = np.where(arr_a > 0, arr_b / denom, 0)

    rr = np.clip(rr, 0, 255).astype(np.uint8)
    gg = np.clip(gg, 0, 255).astype(np.uint8)
    bb = np.clip(bb, 0, 255).astype(np.uint8)

    return Image.merge("RGBA", (Image.fromarray(rr), Image.fromarray(gg), Image.fromarray(bb), a2))

def resize_to_bbox_antihalo(model_img: Image.Image, bbox_w: int, bbox_h: int, fit: str = "contain") -> Image.Image:
    """bbox resize that uses the anti-halo resizer above."""
    r_w = bbox_w / model_img.width
    r_h = bbox_h / model_img.height
    scale = min(r_w, r_h) if fit == "contain" else max(r_w, r_h)
    new_w = max(1, int(round(model_img.width * scale)))
    new_h = max(1, int(round(model_img.height * scale)))
    return resize_rgba_antihalo(model_img, new_w, new_h)

def tighten_alpha_border(img: Image.Image, shrink_px: int = 1) -> Image.Image:
    """
    Optional: slightly erode the alpha to kill stubborn fringes by 1-2 px.
    """
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    r, g, b, a = img.split()
    if shrink_px > 0:
        a = a.filter(ImageFilter.MinFilter(size=shrink_px * 2 + 1))
    return Image.merge("RGBA", (r, g, b, a))


if __name__ == "__main__":
    """
    Main entry point for the script.
    Put custom initialization code here.
    """

    # Set up paths
    project_root = Path(os.path.abspath(os.path.dirname(__file__)))
    assets_dir = os.path.join(project_root, "assets")
    attacker_base_dir = os.path.join(project_root, "dataset_modifications/opv2v/benign/")
    output_dir = os.path.join(assets_dir, "tampered_feeds/")

    attacker = Attacker()

    # Loop over images 000068 to 000414 and process each
    for i in range(68, 415, 2):
        attacker.run_pipeline(
            model_image_path=os.path.join(attacker.assets_dir, "garage", "seat_yellow_back_no_bg.png"),
            input_path_attacker=os.path.join(attacker_base_dir, f"test/2021_08_24_20_49_54/216/{str(i).zfill(6)}_camera0.png"),
            output_dir=os.path.join(output_dir, "test/2021_08_24_20_49_54/216/"),
            preprocess=False,
        injection_coords=BBox(x=300, y=280, w=270, h=115)
    )


    # Loop over images 000068 to 000126 and process each
    for i in range(68, 127, 2):
        attacker.run_pipeline(
            model_image_path=os.path.join(attacker.assets_dir, "garage", "audi_silver_back_no_bg.png"),
            input_path_attacker=os.path.join(attacker_base_dir, f"test/2021_08_23_12_58_19/366/{str(i).zfill(6)}_camera0.png"),
            output_dir=os.path.join(output_dir, "test/2021_08_23_12_58_19/366/"),
            preprocess=False,
        injection_coords=BBox(x=330, y=280, w=250, h=105)
    )
    