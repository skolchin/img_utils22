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

def imshow(img: np.ndarray, title: str = 'imshow', max_size: Iterable = None):
    """ Shows the image and waits for keypress.

    Args:
        img: An OpenCV image
        title: Window title. If set to one's previously used, then
            it will replace content of that window, otherwise a new window will be displayed
        max_size:   Maximum image size (height, width). 
            If actual image is bigger and may not fit to screen, it will be downsized to given one

    Returns:
        None
    """
    from .transform import resize
    if img is None:
        return
    if max_size is not None and (img.shape[0] > max_size[0] or img.shape[1] > max_size[1]):
        scale_x, scale_y = max_size[1] / img.shape[1], max_size[0] / img.shape[0]
        img = resize(img, scale=min(scale_x, scale_y))
        
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
        Extracted area copy as OpenCV image
    """
    if not isinstance(area, Iterable) or len(area) < 4:
       raise ValueError(f'4-element iterable is expected, {type(area)} found')
    if any([a < 0 for a in area]):
       raise ValueError(f'Invalid area: {area}')
    dx = area[2] - area[0]
    dy = area[3] - area[1]
    if dx <= 0 or dy <= 0:
       raise ValueError(f'Invalid area length or width: {area}')

    if len(img.shape) > 2:
       im = np.empty((dy, dx, img.shape[2]), dtype=img.dtype)
    else:
       im = np.empty((dy, dx), dtype=img.dtype)

    im[:] = img[area[1]:area[3], area[0]:area[2]]
    return im

def get_center_area(img: np.ndarray, area: Iterable) -> np.ndarray:
    """Get central part of an image defined by rectangular area.

    Args:
        img:    An OpenCV image
        area:   Size of area to extract (list or tuple [heigth, width])

    Returns:
        Extracted area copy as OpenCV image
    """
    if not isinstance(area, Iterable) or len(area) != 2:
       raise ValueError(f'2-element iterable is expected, {type(area)} found')
    if any([a < 0 for a in area]):
       raise ValueError(f'Invalid area: {area}')

    dx = area[1]
    dy = area[0]
    # if len(img.shape) > 2:
    #    im = np.empty((dy, dx, img.shape[2]), dtype=img.dtype)
    # else:
    #    im = np.empty((dy, dx), dtype=img.dtype)

    x = int(img.shape[1] / 2 - dx / 2) 
    y = int(img.shape[0] / 2 - dy / 2) 
    w = int(img.shape[1] / 2 + dx / 2) 
    h = int(img.shape[0] / 2 + dy / 2) 
    return img[y:h, x:w].copy()

@lru_cache
def get_kernel(sz):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(sz,sz))

