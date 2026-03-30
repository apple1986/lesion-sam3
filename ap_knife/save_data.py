import os
import numpy as np
from PIL import Image

def save_rgb_image(img, save_path):
    """
    img: (H, W, 3) or (3, H, W)
    """
    if img.ndim == 3 and img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))  # CHW → HWC

    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    else:
        img = img.astype(np.uint8)

    Image.fromarray(img, mode="RGB").save(save_path)

def save_mask(mask, save_path):
    """
    mask: (H, W)
    """
    if mask.ndim == 3:
        mask = mask.squeeze()

    if mask.max() <= 1:
        mask = (mask * 255).astype(np.uint8)
    else:
        mask = mask.astype(np.uint8)

    Image.fromarray(mask, mode="L").save(save_path)
