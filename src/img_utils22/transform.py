# Image processing functions package
# (c) kol, 2019-2022

""" Image transformation routines """

import cv2
import numpy as np
from typing import Iterable, Union, Tuple

from .misc import get_center_area
from .colors import COLOR_BLACK, rgb_to_rgba

def center_image(
    img: np.ndarray, 
    new_size: Iterable[int], 
    pad_color: Tuple[int] = COLOR_BLACK) -> Tuple[np.ndarray, Tuple]:

    """ Make a bigger image and center initial image on it.

    Args:
        img:        An OpenCV image
        new_size:   2-element tuple or list with new image size (height, width)
        pad_color:  Padding color

    Returns:
        A 2-element tuple containing bigger image and (offset_x, offset_y) of original one
    """
    if len(img.shape) > 2:
        im = np.full((new_size[0], new_size[1], img.shape[2]), pad_color, dtype=img.dtype)
    else:
        c = pad_color[0] if type(pad_color) is tuple else pad_color
        im = np.full((new_size[0], new_size[1]), c, dtype=img.dtype)

    h = min(img.shape[0], new_size[0])
    w = min(img.shape[1], new_size[1])
    dy, dx = int((new_size[0] - h)/2), int((new_size[1] - w)/2)

    im[dy:dy+h, dx:dx+w] = img[0:h,0:w]
    return im, (dx, dy)

def extend_image(
    img: np.ndarray, 
    new_size: Iterable[int], 
    pad_color: Tuple[int] = COLOR_BLACK) -> np.ndarray:

    """ Extends or shrinks image dimensions to new size

    Args:
        img:        An OpenCV image
        new_size:   2-element tuple or list with new image size (height, width)
        pad_color:  Padding color

    Returns:
        OpenCV image with target dimensions
    """

    if len(img.shape) > 2:
        im = np.full((new_size[0], new_size[1], img.shape[2]), pad_color, dtype=img.dtype)
    else:
        c = pad_color[0] if type(pad_color) is tuple else pad_color
        im = np.full((new_size[0], new_size[1]), c, dtype=img.dtype)

    dx = min(new_size[1], img.shape[1])
    dy = min(new_size[0], img.shape[0])
        
    im[0:dy,0:dx] = img[0:dy,0:dx]
    return im

def rescale(
    img: np.ndarray, 
    new_size: Union[Iterable[int], int] = None,
    scale: Union[Iterable[float], float] = None, 
    center: bool = False, 
    pad_color: Tuple[int] = COLOR_BLACK,
    return_extra: bool = False,
    ) -> Union[np.ndarray, Tuple]:

    """ Proportionally extends an image either by specified scale or to specified size.  

    Differs from `resize` in that this function leaves original image intact
    while simply extending or shrinking its dimensions. In case of upsizing, original
    image might be also placed in the center of the new one.

    Args:
        img:        An OpenCV image
        new_size:   New size. If is an iterable with 2 elements, 
                    specifies precise image size (height, width). Otherwise, 
                    specfies maximum size of any image side after rescaling - in this case
                    image is resized proportionally to that size
        scale:      Scaling ratio. If is an iterable with 2 elements, 
                    specifies image scaling ratio on (height, width). Otherwise,
                    specifies single scale for both sides
        center:     If True, smaller images will be centered on bigger image with padding when extending
        pad_color:  Padding color used when extending images to bigger size.
                    Note that in actual resizing (`rescale==True`) `pad_color` is ignored.
        return_extra: If True, returns additional information (scale and offset)

    Returns:
        If `return_extra` is False, returns only rescaled OpenCV image. 
        Else, returns 3-element tuple containing resized image, scaling factor (tuple of 2 floats) 
        and offset of original image in scaled one (tuple of 2 ints).

    See Also:
        resize
    """

    def _inner_rescale(img, new_size, scale, center, pad_color):
        if scale is not None:
            # Resizing by scale
            if not isinstance(scale, Iterable):
                im_scale = (scale, scale)
            else:
                if len(scale) < 2:
                    raise ValueError(f'Scale must be either scalar or 2-element vector')
                im_scale = tuple(scale[:2])

            new_size = (int(img.shape[1] * im_scale[1]), int(img.shape[0] * im_scale[0]))
        else:
            # Resizing to new_size
            if isinstance(new_size, Iterable):
                # Size vector provided
                if len(new_size) < 2:
                    raise ValueError(f'New_size must be either scalar or 2-element vector')
                h, w = img.shape[:2]
                im_scale = (new_size[0] / h, new_size[1] / w)
            else:
                # Only max size given
                im_size_max = np.max(img.shape[:2])
                im_size_min = np.min(img.shape[:2])
                im_scale = float(new_size) / float(im_size_min)

                if np.round(im_scale * im_size_max) > new_size:
                    im_scale = float(new_size) / float(im_size_max)

                new_size = (int(img.shape[0]*im_scale), int(img.shape[1]*im_scale))
                im_scale = (im_scale, im_scale)

        if not center:
            return extend_image(img, new_size, pad_color), im_scale, (0,0)
        else:
            if img.shape[0] > new_size[0] or img.shape[1] > new_size[1]:
                img = get_center_area(img, new_size)
            im, offs = center_image(img, new_size, pad_color)
            return im, im_scale, offs

    if new_size is None and scale is None:
        raise ValueError('Either new_size or scale must be provided')

    result = _inner_rescale(img, new_size, scale, center, pad_color)
    return result if return_extra else result[0]

def resize(
    img: np.ndarray, 
    new_size: Union[Iterable[int], int] = None,
    scale: Union[Iterable[float], float] = None, 
    return_extra: bool = False,
) -> Union[np.ndarray, Tuple]:

    """ Proportionally resizes an image either to specified scale or to specified size.

    Args:
        img:        An OpenCV image
        new_size:   New size. If is an iterable with 2 elements, 
                    specifies precise target size (height, width). Otherwise, 
                    specfies maximum size of any image side after resizing - in this case
                    image is resized proportionally
        scale:      Scaling ratio. If is an iterable with 2 elements, 
                    specifies image scaling ratio on (height, width). Otherwise,
                    specifies single scale for both sides
        return_extra: If True, returns actual scale along with the image

    Returns:
        If `return_extra` is False, returns only resized OpenCV image. 
        Else, returns 2-element tuple containing resized image and actuals scaling factor (tuple of 2 floats)
    """

    def _inner_resize(img, new_size, scale):
        if scale is not None:
            # Resizing by scale
            if not isinstance(scale, Iterable):
                im_scale = (scale, scale)
            else:
                if len(scale) < 2:
                    raise ValueError(f'Scale must be either scalar or 2-element vector')
                im_scale = tuple(scale[:2])

            im = cv2.resize(img, dsize=None, fx=float(im_scale[1]), fy=float(im_scale[0]))
            return im, im_scale

        else:
            # Resizing to new_size
            if isinstance(new_size, Iterable):
                # Size vector provided
                if len(new_size) < 2:
                    raise ValueError(f'New_size must be either scalar or 2-element vector')
                h, w = img.shape[:2]
                im_scale = (new_size[0] / h, new_size[1] / w)
            else:
                # Only max size given
                im_size_max = np.max(img.shape[:2])
                im_size_min = np.min(img.shape[:2])
                im_scale = float(new_size) / float(im_size_min)

                if np.round(im_scale * im_size_max) > new_size:
                    im_scale = float(new_size) / float(im_size_max)

                new_size = (new_size, new_size)
                im_scale = (im_scale, im_scale)

            im = cv2.resize(img, dsize=None, fx=im_scale[1], fy=im_scale[0])
            return im, im_scale

    if new_size is None and scale is None:
        raise ValueError('Either new_size or scale must be provided')

    result = _inner_resize(img, new_size, scale)
    return result if return_extra else result[0]

def rotate(img: np.ndarray, 
    angle: float, 
    pad_color: Tuple = COLOR_BLACK,
    avoid_clipping: bool = True) -> np.ndarray:

    """ Rotate given image to specified angle.

    Args:
        img:        An OpenCV image
        angle:      An angle to rotate to (>0 - clockwise)
        pad_color:  Padding color
        avoid_clipping: If True, image will be downsized to keep it intact.
                        Otherwise, image will be rotated as is
    Returns:
        Rotated OpenCV image
    """

    h, w = img.shape[:2]
    cx, cy = w / 2, h / 2

    M = cv2.getRotationMatrix2D((cx, cy), -angle, 1.0)
    cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])

    if avoid_clipping:
        nw = int((h * sin) + (w * cos))
        nh = int((h * cos) + (w * sin))
        w, h = nw, nh

        M[0, 2] += (w / 2) - cx
        M[1, 2] += (h / 2) - cy

    return cv2.warpAffine(img, M, (w, h),
                          borderMode = cv2.BORDER_CONSTANT,
                          borderValue = pad_color)

def patch(
    img: np.ndarray, 
    x: int, 
    y: int, 
    patch_img: np.ndarray, 
    patch_mask: np.ndarray = None, 
    clip: bool = False, 
    alpha: float = None) -> np.ndarray:

    """ Applies a patch at given coordinates with with optional masking and alpha-channel blending.

    Args:
        img:        A 3-channel OpenCV image
        x:          Patch X offset
        y:          Patch Y offset
        patch_img:  Patch OpenCV image
        patch_mask: Optional boolean mask to apply over patch
        clip:       Whether to clip patch if it's bigger than image
        alpha:      Alpha channel value

    Returns:
        An OpenCV image
    """

    if y >= img.shape[0] or x >= img.shape[1] or y < 0 or x < 0:
        raise ValueError("Invalid coordinates")

    h, w = patch_img.shape[:2]
    if (y + h > img.shape[0] or x + w > img.shape[1]) and not clip:
        raise ValueError("Patch is outside image area and clipping not specified")
    if y + h >= img.shape[0] and clip:
        h = img.shape[0] - y
    if x + w >= img.shape[1] and clip:
        w = img.shape[1] - x
        patch_img = patch_img[0:h, 0:w]
        patch_mask = patch_mask[0:h, 0:w] if patch_mask is not None else None

    if patch_img.shape[-1] == 3 and img.shape[-1] == 4:
        # Patch contains alpha channel
        patch_img = rgb_to_rgba(patch_img)

    dst = img.copy()
    area = dst[y:(y+h), x:(x+w)]
    if patch_mask is None:
        if alpha is None:
            area = patch_img
        else:
            cv2.addWeighted(area, 1 - alpha, patch_img, alpha, 0, area)
    else:
        if alpha is None:
            area[patch_mask > 0] = patch_img[patch_mask > 0]
        else:
            dtyp = area.dtype
            a = area[patch_mask > 0] * (1-alpha)
            p = patch_img[patch_mask > 0] * alpha
            area[patch_mask > 0] = a.astype(dtyp) + p.astype(dtyp)

    dst[y:(y+h), x:(x+w)] = area
    return dst

# https://stackoverflow.com/questions/69050464/zoom-into-image-with-opencv
def zoom_at(img: np.ndarray, zoom: float, pad_color: Tuple[int] = COLOR_BLACK) -> np.ndarray:
    """ Zooms image

    Args:
        img:    An 1- or 3-channel OpenCV image
        zoom:   Zoom factor, float value greater than 0. 
            If it is greater than 1, then image is zoomed in (become larger), 
            and if less - zoomed out (become smaller).
        pad_color:  padding color

    Returns:
        An OpenCV image
    """
    cy, cx = [ i // 2 for i in img.shape[:-1] ]

    rot_mat = cv2.getRotationMatrix2D((cx,cy), 0, zoom)
    return cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_NEAREST, 
        borderMode=cv2.BORDER_CONSTANT, borderValue=pad_color)
