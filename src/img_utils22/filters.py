# Image processing functions package
# (c) kol, 2022

""" Stacked image processing functions (filters) """

import numpy as np
import cv2
from functools import lru_cache
from typing import Tuple, Union, Optional, Iterable

from .colors import COLOR_BLACK
from .misc import img1_to_img3, _assert_1ch, _assert_3ch
from .transform import resize, rescale, rotate

class LoadFile:
    """ Load an image from file 
    
    Args:
        filename    File name

    Returns
        img:    An OpenCV image
    """

    def __init__(self, filename: str):
        self.filename = filename

    def __call__(self, nothing) -> np.ndarray:
        return cv2.imread(self.filename)

class Resize:
    """ Resize image 
    
    Args:
        img:    An OpenCV image
        new_size:   New size
        scale:      Scaling ratio

    See `transform.resize` for details

    Returns
        img:    An OpenCV image
    """

    def __init__(self, new_size: Union[Iterable[int], int] = None, scale: Union[Iterable[float], float] = None):
        self.new_size, self.scale = new_size, scale

    def __call__(self, img: np.ndarray) -> np.ndarray:
        return resize(img, self.new_size, self.scale, return_extra=False)

class Gray:
    """ Convert an image to gray scale

    Args:
        img:    An OpenCV 3-channel image

    Returns:
        An OpenCV 1-channel image
    """

    def __call__(self, img: np.ndarray) -> np.ndarray:
        _assert_3ch(img)
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

class Edges:
    """ Find edges using Canny algorithm.

    Args:
        img:    An OpenCV 1-channel image

    Returns:
        An OpenCV 1-channel image
    """
    def __init__(self, thresh1: Optional[int] = 100, thresh2: Optional[int] = 200):
        self.thresh1, self.thresh2 = thresh1, thresh2

    def __call__(self, img: np.ndarray) -> np.ndarray:
        _assert_1ch(img)
        return cv2.Canny(img, self.thresh1, self.thresh2)

class Ensure3:
    """ Ensure the image has 3 channels.

    Args:
        img:    An OpenCV image (1- or 3-channel)

    Returns:
        An OpenCV 3-channel image
    """
    def __call__(self, img: np.ndarray) -> np.ndarray:
        return img if not isinstance(len(img), int) and len(img).shape == 3 else img1_to_img3(img)

class PyramidFilter:
    """ Pyramid (aka mean shift) filtering.
    Produces some kind of "posterized" image with color gradients and fine-grain texture flattened.

    Args:
        img:    An OpenCV 3-channel image
        sp:	    The spatial window (area in texture filtering) radius
        sr:	    The color window (range of colors) radius

    Returns:
        An OpenCV 3-channel image
    """
    def __init__(self, sp: Optional[int] = 10, sr: Optional[int] = 10):
        self.sp, self.sr = sp, sr

    def __call__(self, img: np.ndarray) -> np.ndarray:
        _assert_3ch(img)
        return cv2.pyrMeanShiftFiltering(img, self.sp, self.sr)


class Channel:
    """ Extract specified channel from image.
    
    Args:
        img:    An OpenCV 3-channel image
        channel: Channel index (0-3) or abbreviation (see `CHANNELS`)

    Returns:
        An OpenCV 1-channel (BW) image or None if channel is not available for the image
    """

    CHANNELS = ('B', 'G', 'R', 'A')

    def __init__(self, channel: Union[str, int]):
        if isinstance(channel, str) and channel not in self.CHANNELS:
            raise ValueError('Unknown channel %s', channel)
        if isinstance(channel, int) and (channel < 0 or channel >= len(self.CHANNELS)):
            raise ValueError('Invalid channel index %d', channel)

        self.channel = channel
        self.channel_index = self.CHANNELS.index(channel) if isinstance(channel, str) else int(channel)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        _assert_3ch(img)
        parts = cv2.split(img)
        return None if self.channel_index >= len(parts) else parts[self.channel_index]

class Threshold:
    """ Apply a threshold transformation.
    
    Args:
        img:    An OpenCV 1- or 3-channel image
        method: Either cv2.THRESH_xxx constant or appropriate string (binary, binary_inv, binary+otsu, ...).
            Note that OTSU thresholding works only on 1-channel images
        threshold: Threshold level
        maxval: Threshold upper boundary (for binary thresholding)

    Returns:
        An OpenCV 1- or 3-channel image
    """
    METHODS = {
        'binary': cv2.THRESH_BINARY,
        'binary_inv': cv2.THRESH_BINARY_INV,
        'trunc': cv2.THRESH_TRUNC,
        'tozero': cv2.THRESH_TOZERO,
        'tozero_inv': cv2.THRESH_TOZERO_INV,
    }
    EXTRA_METHODS = {
        'otsu': cv2.THRESH_OTSU
    }
    def __init__(self, method: Union[int, str] = cv2.THRESH_BINARY, threshold: int = 127, maxval: int = 255):
        self.threshold = threshold
        self.maxval = maxval

        if isinstance(method, str) and method.split('+')[0] not in self.METHODS:
            raise ValueError('Unknown method %s', method)
        if isinstance(method, int) and method not in self.METHODS.values():
            raise ValueError('Invalid method value %d', method)

        self.method = method
        if isinstance(method, int):
            self.method_value = self.method
        else:
            parts = method.split('+')
            self.method_value = self.METHODS[parts[0]]
            if len(parts) > 1:
                self.method_value += self.EXTRA_METHODS[parts[1]]

    def __call__(self, img: np.ndarray) -> np.ndarray:
        self._thesh_ret, thresh = cv2.threshold(img, self.threshold, self.maxval, self.method_value)
        return thresh

@lru_cache
def _get_kernel(sz):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(sz,sz))

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

