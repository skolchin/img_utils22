# Image processing functions package
# (c) kol, 2019-2022

""" Misc routines """

import cv2
import numpy as np
from functools import lru_cache
from typing import Iterable

def _assert_1ch(img):
    assert len(img.shape) == 2, 'Image is not 1-channel'

def _assert_3ch(img):
    assert len(img.shape) == 3, 'Image is not 3-channel'

def imshow(img: np.ndarray, title: str = None):
    """ Shows the image and waits for keypress """
    cv2.imshow(title, img)
    cv2.waitKey(0)

def img1_to_img3(img: np.ndarray) -> np.ndarray:
    """ Converts 1-channel (BW) image to 3-channel (color) image.  
    This function complements OpenCV's `cvtColor` allowing to transform pure BW-image
    to color one, making it suitable for further manipulations.

    Args:
        img: An OpenCV image. Must be 1-channel ([X,Y]) image of type `np.ndarray` or compatible

    Returns:
        3-channel OpenCV image ([X, Y, 3]) 
    """
    if img is None:
       return None

    if not isinstance(img, np.ndarray):
        img = np.array(img)

    _assert_1ch(img)

    img3 = np.empty((img.shape[0], img.shape[1], 3), dtype=img.dtype)
    for i in range(3): img3[:,:,i] = img
    return img3

def get_image_area(img: np.ndarray, area: Iterable) -> np.ndarray:
    """Get part of an image defined by rectangular area.

    Args:
        img:    An OpenCV image
        area:   Area to extract (list or tuple [x1,y1,x2,y2])

    Returns:
        Extracted area as OpenCV image
    """
    if not isinstance(area, Iterable) or len(area) < 4:
       raise ValueError(f'4-element iterable is expected, {type(area)} found')
    if area[0] < 0 or area[1] < 0:
       raise ValueError(f'Invalid area origin: {area}')
    dx = area[2] - area[0]
    dy = area[3] - area[1]
    if dx <= 0 or dy <= 0:
       raise ValueError(f'Invalid area length: {area}')

    im = None
    if len(img.shape) > 2:
       im = np.empty((dy, dx, img.shape[2]), dtype=img.dtype)
    else:
       im = np.empty((dy, dx), dtype=img.dtype)

    im[:] = img[area[1]:area[3], area[0]:area[2]]
    return im

@lru_cache
def get_kernel(sz):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(sz,sz))

