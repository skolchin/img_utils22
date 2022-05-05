# Color management routines
# (c) kol, 2019-2022

import cv2
import numpy as np
from random import randint
from typing import Optional, Iterable, Union, Tuple

COLOR_BLACK = (0, 0, 0)
COLOR_WHITE = (255, 255, 255)

def random_colors(num_colors: int, with_alpha: bool = False):
    """ Iterator which returns a sequence of random colors.

    Args:
        num_colors: number of colors to return
        with_alpha: True to add alpha channel

    Returns:
        Iterator which yields 3- or 4-element tuples of random colors
    """
    for _ in num_colors:
        if not with_alpha:
            yield randint(0,255), randint(0,255), randint(0,255)
        else:
            yield randint(0,255), randint(0,255), randint(0,255), randint(0,255) / 255.0

def gradient_colors(colors: Iterable, num_colors: int) -> np.ndarray:
    """ Gradient color generator

    Args:
        colors: 2-element tuple or list containing lower and upper boundary colors
        num_colors: number of colors to return

    Returns:
        Array of 3-element gradient colors
    """
    if len(colors) < 2:
        raise ValueError("Two colors required to compute gradient")
    if num_colors < 2:
        raise ValueError("Gradient length must be greater than 1")

    c = np.linspace(0, 1, num_colors)[:, None, None]
    x = np.array([colors[0]])
    y = np.array([colors[1]])
    g = y + (x - y) * c

    return g.astype(x.dtype)

def rgba_to_rgb(rgba: Union[np.ndarray, Iterable]) -> Tuple[np.ndarray, np.ndarray]:
    """ Convert RGBA color to RGB color and mask.

    Args:
        rgba:   4-element numpy array or compatible type specifying an RBA color to convert

    Returns:
        2-element tuple containing converted RGB color (3-element array) and alpha channel mask
    """
    if not isinstance(rgba, np.ndarray):
        rgba = np.array(rgba)
    assert rgba.shape[-1] == 4

    rgb = np.empty((rgba.shape[0], rgba.shape[1], 3), dtype=rgba.dtype)
    r, g, b, m = rgba[:,:,0], rgba[:,:,1], rgba[:,:,2], rgba[:,:,3]

    rgb[:,:,0] = r
    rgb[:,:,1] = g
    rgb[:,:,2] = b

    return [rgb, m]

def rgb_to_rgba(rgb: Union[np.ndarray, Iterable], alpha: Optional[float] = 1.0) -> np.ndarray:
    """ Convert RGB color to RGB color and mask.

    Args:
        rgb:    3-element numpy array or compatible type specifying a color to convert
        alpha:  Alpha channel value

    Returns:
        4-element numpy array with RGBA color
    """
    if not isinstance(rgb, np.ndarray):
        rgb = np.array(rgb)
    assert rgb.shape[-1] == 3

    rgba = np.full((rgb.shape[0], rgb.shape[1], 4), alpha, dtype=rgb.dtype)
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]

    rgba[:,:,0] = r
    rgba[:,:,1] = g
    rgba[:,:,2] = b

    return rgba

def increase_brightness(img: np.ndarray, value: int = 30) -> np.ndarray:
    """ Increase image brightness

    Args:
        img:    An OpenCV image
        value:  Brightness increment value

    Returns:
        An OpenCV image
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += np.uint8(value)

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img