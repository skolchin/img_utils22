# Image processing functions package
# Misc routines
# (c) kol, 2019-2022

import cv2
import numpy as np
from skimage.metrics import structural_similarity
from typing import Iterable, Optional, Tuple

from .align_images import align_images

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
        img:     An OpenCV BW or color image
        r:       Area to extract (list or tuple [x1,y1,x2,y2])

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

def get_image_diff(
   im1: np.ndarray, 
   im2: np.ndarray, 
   align_mode: Optional[str] = None,
   multichannel: Optional[bool] = False) -> Tuple[float, np.ndarray]:

    """ Get the difference of two images. 
    The function calculates structure similarity index (SSIM) and difference matrix
    of two images with optional alignment.

    Args:
        im1:      First OpenCV image
        im2:      Second OpenCV image
        align_mode:     One of alignment modes (see `align_images` for details). If None,
                        images would not be aligned
        multichannel:   If False (default), comparison will be performed on grayed images
                        improving speed, otherwise - on each color channel separatelly

    Returns:
        2-element tuple with SSIM score (more similarity - higher score) and difference matrix.

    See also:
        align_images
    """

    # Algin images if requested
    if align_mode:
        im2 = align_images(im1, im2, mode=align_mode)

    if multichannel:
        # Check number of channels
        if len(im1.shape) < 3 or len(im2.shape) < 3:
            raise ValueError(f'At least one image is not multichannel: {im1.shape}, {im2.shape}')
        channel_axis = 2
    else:
        # No multichannel, downsample to gray images
        channel_axis = None
        im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Calculate SSIM and difference map
    (score, diff) = structural_similarity(im1, im2, channel_axis=channel_axis, 
                                            full=True, gradient=False, gaussian_weights=True)
    return score, diff
