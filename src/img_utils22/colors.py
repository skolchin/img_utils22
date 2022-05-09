# Image processing functions package
# (c) kol, 2019-2022

""" Functions dealing with colors """

import cv2
import numpy as np
from random import randint
from typing import Tuple

COLOR_BLACK = (0, 0, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_BLUE = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_GRAY = (127, 127, 127)

def random_colors(num_colors: int):
    """ Iterator which returns a sequence of random colors.

    Args:
        num_colors: number of colors to return

    Returns:
        Iterator which yields 3-element tuples of random colors
    """
    for _ in range(num_colors):
        yield randint(0,255), randint(0,255), randint(0,255)

def gradient_colors(from_color: Tuple, to_color: Tuple, num_colors: int) -> np.ndarray:
    """ Gradient color generator

    Args:
        from_color: starting color
        to_color: ending colors
        num_colors: number of colors to generate

    Returns:
        Array of 3-element color tuples
    """
    if num_colors < 2:
        raise ValueError("Gradient length must be greater than 1")

    c = np.linspace(0, 1, num_colors)[:, None, None]
    x = np.array([from_color])
    y = np.array([to_color])
    g = y + (x - y) * c

    return g.astype(x.dtype)

def rgba_to_rgb(rgba: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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

def rgb_to_rgba(rgb: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """ Convert RGB color to RGBA color

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

