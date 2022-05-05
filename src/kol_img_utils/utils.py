# Image processing functions package
# Utility routines
# (c) kol, 2019-2022

import cv2
import numpy as np
from typing import Iterable, Optional

from .colors import rgb_to_rgba

def img1_to_img3(img: np.ndarray) -> np.ndarray:
    """ Converts 1-channel (BW) image to 3-channel (color) image.  
    This function complements OpenCV's `cvtColor` allowing to transform pure BW-image
    to color one, making it suitable for further manipulations.

    Args:
        img: An OpenCV image. Must be one-channel ([X,Y]) image of type `np.ndarray` or compatible

    Returns:
        3-channel OpenCV image ([X, Y, 3]) 
    """
    if img is None:
       return None

    if not isinstance(img, np.ndarray):
        img = np.array(img)

    if len(img.shape) > 2:
       raise ValueError('Image is not 1-channel')

    img3 = np.empty((img.shape[0], img.shape[1], 3), dtype=img.dtype)
    for i in range(3): img3[:,:,i] = img
    return img3

def get_image_area(img: np.ndarray, r: Iterable) -> np.ndarray:
    """Get part of an image defined by rectangular area.

    Args:
        img      An OpenCV BW or color image
        r        Area to extract (list or tuple [x1,y1,x2,y2])

    Returns:
        Extracted area as OpenCV image
    """
    if not isinstance(r, Iterable) or len(r) < 4:
       raise ValueError(f'4-element iterable is expected, {type(r)} found')
    if r[0] < 0 or r[1] < 0:
       raise ValueError(f'Invalid area origin: {r}')
    dx = r[2] - r[0]
    dy = r[3] - r[1]
    if dx <= 0 or dy <= 0:
       raise ValueError(f'Invalid area length: {r}')

    im = None
    if len(img.shape) > 2:
       im = np.empty((dy, dx, img.shape[2]), dtype=img.dtype)
    else:
       im = np.empty((dy, dx), dtype=img.dtype)

    im[:] = img[r[1]:r[3], r[0]:r[2]]
    return im

def apply_patch(
    src: np.ndarray, 
    x: int, 
    y: int, 
    patch: np.ndarray, 
    mask: Optional[np.ndarray] = None, 
    clip: Optional[bool] = False, 
    alpha: Optional[float] = None) -> np.ndarray:

    """ Applies a patch at given coordinates with with optional masking and alpha-channel blending.

    Args:
        img:    A 3-channel OpenCV image
        x:      Patch X offset
        y:      Patch Y offset
        patch:  Patch OpenCV image
        mask:   Optional boolean mask to apply over patch
        clip:   Whether to clip patch if it's bigger than image
        alpha:  Alpha channel value

    Returns:
        An OpenCV image
    """

    if y >= src.shape[0] or x >= src.shape[1] or y < 0 or x < 0:
        raise ValueError("Invalid coordinates")

    h, w = patch.shape[:2]
    if (y + h > src.shape[0] or x + w > src.shape[1]) and not clip:
        raise ValueError("Patch is outside image area and clipping not specified")
    if y + h >= src.shape[0] and clip:
        h = src.shape[0] - y
    if x + w >= src.shape[1] and clip:
        w = src.shape[1] - x
        patch = patch[0:h, 0:w]
        mask = mask[0:h, 0:w] if mask is not None else None

    if patch.shape[-1] == 3 and src.shape[-1] == 4:
        patch = rgb_to_rgba(patch)

    dst = src.copy()
    area = dst[y:(y+h), x:(x+w)]
    if mask is None:
        if alpha is None:
            area = patch
        else:
            cv2.addWeighted(area, 1 - alpha, patch, alpha, 0, area)
    else:
        if alpha is None:
            area[mask > 0] = patch[mask > 0]
        else:
            dtyp = area.dtype
            a = area[mask > 0] * (1-alpha)
            p = patch[mask > 0] * alpha
            area[mask > 0] = a.astype(dtyp) + p.astype(dtyp)

    dst[y:(y+h), x:(x+w)] = area
    return dst
