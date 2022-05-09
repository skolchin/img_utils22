# Image processing functions package
# (c) kol, 2022

""" Extra image processing functions (filters) """

import numpy as np
import cv2
from functools import lru_cache
from typing import Tuple

from .colors import COLOR_BLACK

def pyramid_filter(img: np.ndarray, sp: int = 20, sr: int = 50) -> np.ndarray:
    """ Pyramid (aka mean shift) filtering.
    Produces some kind of "posterized" image with color gradients and fine-grain texture flattened.

    Args:
        img:    An OpenCV 3-channel image
        sp:	    The spatial window (area in texture filtering) radius
        sr:	    The color window (range of colors) radius

    Returns:
        An OpenCV image
    """
    return cv2.pyrMeanShiftFiltering(img, sp, sr)

def gray(img: np.ndarray) -> np.ndarray:
    """ Convert an image to gray scale.

    Args:
        img:    An OpenCV 3-channel image

    Returns:
        An OpenCV 1-channel (BW) image
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

_CHANNELS = ['B', 'G', 'R', 'A']

def extract_channel(img: np.ndarray, channel: str) -> np.ndarray:
    """ Extract channel """
    if channel not in _CHANNELS:
        raise ValueError(f'Unknown channel: {channel}')
    parts = cv2.split(img)
    return parts[_CHANNELS.index(channel)]

_THRESH_METHODS = {
    'binary': cv2.THRESH_BINARY,
    'binary_inv': cv2.THRESH_BINARY_INV,
    'truncate': cv2.THRESH_TRUNC,
    'to_zero': cv2.THRESH_TOZERO,
    'to_zero_inv': cv2.THRESH_TOZERO_INV
}
_THRESH_EXTRA = {
    'otsu': cv2.THRESH_OTSU,
    'triangle': cv2.THRESH_TRIANGLE,
}

def threshold(
    img: np.ndarray, 
    method: str = 'binary+otsu', 
    threshold: int = 0, 
    maxval: int = 127) -> np.ndarray:

    """ Thresholding """

    # Determine method
    tr_comp = method.split('+')
    if tr_comp[0] in _THRESH_METHODS:
        tr_method = _THRESH_METHODS[tr_comp[0]]
        if len(tr_comp) > 1:
            if tr_comp[1] not in _THRESH_EXTRA:
                raise ValueError(f'Invalid thresholding extra: {tr_comp[1]}')
            tr_method += _THRESH_EXTRA[tr_comp[1]]
    elif tr_comp[0] in _THRESH_EXTRA:            
        tr_method = cv2.THRESH_BINARY + _THRESH_EXTRA[tr_comp[0]]
    else:
        raise ValueError(f'Invalid thresholding method: {method}')

    _, thresh = cv2.threshold(img, threshold, maxval, tr_method)
    return thresh

@lru_cache
def _get_kernel(sz):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(100,100))
    return cv2.resize(kernel, (sz, sz))

def dilate(
    img: np.ndarray, 
    num_iter: int = 1,
    kernel_size: int = 10,
    pad_color: Tuple = COLOR_BLACK) -> np.ndarray:
    """ Dilation """

    kernel = _get_kernel(kernel_size)
    return cv2.dilate(img, kernel,
                        iterations=num_iter,
                        borderType = cv2.BORDER_CONSTANT,
                        borderValue = pad_color)

def erode(
    img: np.ndarray, 
    num_iter: int = 1,
    kernel_size: int = 10,
    pad_color: Tuple = COLOR_BLACK) -> np.ndarray:
    """ Erosion """

    kernel = _get_kernel(kernel_size)
    return cv2.erode(img, kernel,
                        iterations=num_iter,
                        borderType = cv2.BORDER_CONSTANT,
                        borderValue = pad_color)


def blur(img: np.ndarray, mask_size: int = 3) -> np.ndarray:
    """ Blur """
    return cv2.blur(img, (mask_size, mask_size))

def equalize_luminosity(img: np.ndarray, kernel_size: int = 8, clip_limit: float = 3.0) -> np.ndarray:
    """ Luminosity equaliztion (CLAHE) """

    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # Split channels
    l, a, b = cv2.split(lab)

    # Apply CLAHE to l_channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(kernel_size,kernel_size))
    cl = clahe.apply(l)

    # Merge back and convert to RGB color space
    merged = cv2.merge((cl,a,b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

def increase_brightness(img: np.ndarray, value: int = 10) -> np.ndarray:
    """ Increase image brightness

    Args:
        img:    An OpenCV image
        value:  Brightness change value (> or < 0)

    Returns:
        An OpenCV image
    """
    lim = 255 - int(value)
    if lim < 0 or lim > 255:
        raise ValueError(f'Invalid increment value {value}')

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    v[v > lim] = 255
    v[v <= lim] += np.uint8(value)

    final_hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
