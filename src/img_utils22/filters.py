# Image processing functions package
# (c) kol, 2022

""" Stacked image processing functions (filters) """

import cv2
import numpy as np
from typing import Tuple, Union, Optional, Iterable

from .colors import COLOR_BLACK
from .misc import img1_to_img3, get_image_area, _assert_1ch, _assert_3ch, get_kernel
from .transform import resize, rescale, rotate
from .mask import get_bgsub_mask, apply_image_mask
from .pipe import PipedMixin

class LoadFile(PipedMixin):
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

class ShowImage(PipedMixin):
    """ Shows the image 
    
    Args:
        img:    An OpenCV image
        title:  Optional window title

    Returns
        img:    An OpenCV image
    """

    def __init__(self, title: str = 'image'):
        self.title = title

    def __call__(self, img: np.ndarray, title: str = None) -> np.ndarray:
        cv2.imshow(title or self.title, img)
        cv2.waitKey(0)
        return img
        
class Resize(PipedMixin):
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

class Area(PipedMixin):
    """ Take an area from image 
    
    Args:
        img:    An OpenCV image
        area:   Area to extract (list or tuple [x1,y1,x2,y2])

    Returns
        img:    An OpenCV image
    """

    def __init__(self, area: Iterable):
        self.area = area

    def __call__(self, img: np.ndarray) -> np.ndarray:
        return get_image_area(img, self.area)


class Gray(PipedMixin):
    """ Convert an image to gray scale

    Args:
        img:    An OpenCV 3-channel image

    Returns:
        An OpenCV 1-channel image
    """

    def __call__(self, img: np.ndarray) -> np.ndarray:
        _assert_3ch(img)
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

class Edges(PipedMixin):
    """ Find edges using Canny algorithm.

    Args:
        img:    An OpenCV 1-channel image

    Returns:
        An OpenCV 1-channel image
    """
    def __init__(self, thresh1: Optional[int] = 100, thresh2: Optional[int] = 200, apertureSize: Optional[int] = 3):
        self.thresh1, self.thresh2, self.apertureSize = thresh1, thresh2, apertureSize

    def __call__(self, img: np.ndarray) -> np.ndarray:
        _assert_1ch(img)
        return cv2.Canny(img, self.thresh1, self.thresh2, apertureSize=self.apertureSize)

class Ensure3(PipedMixin):
    """ Ensure the image has 3 channels.

    Args:
        img:    An OpenCV image (1- or 3-channel)

    Returns:
        An OpenCV 3-channel image
    """
    def __call__(self, img: np.ndarray) -> np.ndarray:
        return img if not isinstance(len(img), int) and len(img).shape == 3 else img1_to_img3(img)

class PyramidFilter(PipedMixin):
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


class Channel(PipedMixin):
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

class Threshold(PipedMixin):
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

class Dilate(PipedMixin):
    """ Dilates an image.
    
    Args:
        img:    An OpenCV 1- or 3-channel image
        num_iter:   Number of iterations (default 1)
        kernel_size: Kernel size
        pad_color: Padding color

    Returns:
        An OpenCV 1- or 3-channel image
    """
    def __init__(self,
        num_iter: int = 1,
        kernel_size: int = 10,
        pad_color: Tuple = COLOR_BLACK):

        self.num_iter = num_iter
        self.kernel_size = kernel_size
        self.pad_color = pad_color

    def __call__(self, img: np.ndarray) -> np.ndarray:
        kernel = get_kernel(self.kernel_size)
        return cv2.dilate(img, kernel,
                            iterations=self.num_iter,
                            borderType = cv2.BORDER_CONSTANT,
                            borderValue = self.pad_color)

class Erode(PipedMixin):
    """ Erodes an image.
    
    Args:
        img:    An OpenCV 1- or 3-channel image
        num_iter:   Number of iterations (default 1)
        kernel_size: Kernel size
        pad_color: Padding color

    Returns:
        An OpenCV 1- or 3-channel image
    """
    def __init__(self,
        num_iter: int = 1,
        kernel_size: int = 10,
        pad_color: Tuple = COLOR_BLACK):

        self.num_iter = num_iter
        self.kernel_size = kernel_size
        self.pad_color = pad_color

    def __call__(self, img: np.ndarray) -> np.ndarray:
        kernel = get_kernel(self.kernel_size)
        return cv2.erode(img, kernel,
                            iterations=self.num_iter,
                            borderType = cv2.BORDER_CONSTANT,
                            borderValue = self.pad_color)

class Blur(PipedMixin):
    """ Blurs an image.
    
    Args:
        img:    An OpenCV 1- or 3-channel image
        mask_size: Mask size

    Returns:
        An OpenCV 1- or 3-channel image
    """
    def __init__(self, mask_size: int = 3):
        self.mask_size = mask_size

    def __call__(self, img: np.ndarray) -> np.ndarray:
        return cv2.blur(img, (self.mask_size, self.mask_size))

class EqualizeLuminosity(PipedMixin):
    """ Luminosity equalization filter (CLAHE).
    
    Args:
        img:    An OpenCV 1- or 3-channel image
        kernel_size: Kernel size
        clip_limit: Clip limit

    Returns:
        An OpenCV 1- or 3-channel image
    """
    def __init__(self,
        kernel_size: int = 10,
        clip_limit: float = 3.0):

        self.kernel_size = kernel_size
        self.clip_limit = clip_limit

    def __call__(self, img: np.ndarray) -> np.ndarray:
        # Convert to LAB color space
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

        # Split channels
        l, a, b = cv2.split(lab)

        # Apply CLAHE to l_channel
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=(self.kernel_size, self.kernel_size))
        cl = clahe.apply(l)

        # Merge back and convert to RGB color space
        merged = cv2.merge((cl,a,b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

class IncreaseBrightness(PipedMixin):
    """ Increase image brightness.
    
    Args:
        img:    An OpenCV 1- or 3-channel image
        value:  Brightness change value (> or < 0)

    Returns:
        An OpenCV 1- or 3-channel image
    """
    def __init__(self, value: int = 10):
        self.value = value

    def __call__(self, img: np.ndarray) -> np.ndarray:
        lim = 255 - int(self.value)
        if lim < 0 or lim > 255:
            raise ValueError(f'Invalid increment value {self.value}')

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        v[v > lim] = 255
        v[v <= lim] += np.uint8(self.value)

        final_hsv = cv2.merge((h, s, v))
        return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

class ExtractForeground(PipedMixin):
    """ Extracts foreground from single-colored background 
    Args:
        img:    An OpenCV 3-channel image
        bgcolor:    Background color, 3-element tuple or integer

    Returns:
        An OpenCV 3-channel image with background removed
    """
    def __init__(self, bgcolor: Union[Tuple, int] = COLOR_BLACK):
        self.bgcolor = bgcolor
        self.mask = None

    def __call__(self, img: np.ndarray) -> np.ndarray:
        _assert_3ch(img)
        bgclr = self.bgcolor[0] if isinstance(self.bgcolor, tuple) else self.bgcolor
        self.mask = get_bgsub_mask(img, np.full(img.shape, bgclr, img.dtype))
        return apply_image_mask(img, self.mask)

class ExtractFgObjectsArea(PipedMixin):
    """ Extracts an area containing foreground objects from single-colored background 
    Args:
        img:    An OpenCV 3-channel image
        bgcolor:  Background color, 3-element tuple or integer
        relax_rect: How much is to shift away object's bounding box

    Returns:
        An OpenCV 3-channel image containing area which encompasses all objects found
    """
    def __init__(self, bgcolor: Union[Tuple, int] = COLOR_BLACK, relax_rect: int = 5):
        self.bgcolor = bgcolor
        self.relax_rect = relax_rect
        self.mask = None
        self.fg_rect = None

    def __call__(self, img: np.ndarray) -> np.ndarray:
        _assert_3ch(img)

        bgclr = self.bgcolor[0] if isinstance(self.bgcolor, tuple) else self.bgcolor
        self.mask = get_bgsub_mask(img, np.full(img.shape, bgclr, img.dtype))
        nonzero = np.nonzero(self.mask)
        if not nonzero[0].any() or not nonzero[1].any():
            # only background color
            return np.full(img.shape, bgclr, img.dtype)

        self.fg_rect = [
            max(min(nonzero[1]) - self.relax_rect, 0),
            max(min(nonzero[0]) - self.relax_rect, 0),
            min(max(nonzero[1]) + self.relax_rect, img.shape[1]),
            min(max(nonzero[0]) + self.relax_rect, img.shape[0])
        ]
        return get_image_area(img, self.fg_rect)
