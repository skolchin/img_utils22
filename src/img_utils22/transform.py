# Image processing functions package
# Image transformation routines
# (c) kol, 2019-2022

import cv2
import numpy as np
from typing import Optional, Iterable, Union, Tuple

from .colors import COLOR_WHITE, rgb_to_rgba

def center_image(
    img: np.ndarray, 
    new_size: Iterable, 
    pad_color: Optional[Tuple] = COLOR_WHITE) -> Tuple[np.ndarray, Tuple]:

    """ Make a bigger image and center initial image on it.

    Args:
        img:        A 3-channel OpenCV image
        new_size:   2-element tuple or list with new image size (width, height)
        pad_color:  Padding color

    Returns:
        A 2-element tuple containing bigger image and (offset_x, offset_y) of original one
    """
    if len(img.shape) > 2:
        im = np.full((new_size[0], new_size[1], img.shape[2]), pad_color, dtype=img.dtype)
    else:
        c = pad_color[0] if type(pad_color) is tuple else pad_color
        im = np.full((new_size[0], new_size[1]), c, dtype=img.dtype)

    h, w = img.shape[:2]
    dx, dy = int((new_size[1] - w)/2), int((new_size[0] - h)/2)
    if dx < 0 or dy < 0:
        raise ValueError(f'New image dimensions {new_size} are less than original one {img.shape[:2]}')

    im[dy:dy + h, dx:dx + w] = img
    return im, (dx, dy)

def resize(
    img: np.ndarray, 
    new_size: Optional[Union[Iterable, int]] = None,
    scale: Optional[Union[Iterable, float]] = None, 
    upsize: Optional[bool] = True, 
    center: Optional[bool] = False, 
    pad_color: Optional[Tuple] = COLOR_WHITE,
    extra_info: Optional[bool] = False,
    ) -> Union[np.ndarray, Tuple]:

    """Wrapper over `cv2.resize()` allowing to proportionally resize an image 
    either to specified scale or to specified size.

    Args:
        img:        An OpenCV image
        new_size:   New size. If is an iterable with 2 elements, 
                    specifies precise image size (width, height). Otherwise, 
                    specfies maximum size of any image side after resizing - in this case
                    image is resized proportionally to that size
        scale:      Scaling ratio. If is an iterable with 2 elements, 
                    specifies image scaling ratio on (width, height). Otherwise,
                    specifies single scale for both sides
        upsize:     If True, images with size less than `max_size` will be upsized
        center:     If True, smaller images will be centered on bigger image with padding
        pad_color:  Padding color
        extra_info: If True, returns additional information (scale and offset)

    Returns:
        If `extra_info` is False, returns only resized OpenCV image. 
        Else, returns 3-element tuple containing resized image, scaling factor (tuple of 2 floats) 
        and offset of original image in scaled one (tuple of 2 ints).  
    """

    def _inner_resize():
        if scale is not None:
            # Resizing by scale
            if not isinstance(scale, Iterable):
                im_scale = (scale, scale)
            else:
                if len(scale) < 2:
                    raise ValueError(f'Scale must be either scalar or 2-element vector')
                im_scale = tuple(scale[:2])

            im = cv2.resize(img, dsize=None, fx=im_scale[0], fy=im_scale[1])
            img_size = int(np.max(im.shape[:2]))
            if new_size is not None and img_size < int(np.max(new_size)) and center:
                return center_image(im, new_size, pad_color, im_scale)
            else:
                return im, im_scale, (0, 0)

        else:
            # Resizing to new_size
            if isinstance(new_size, Iterable):
                # Size vector provided
                if len(new_size) < 2:
                    raise ValueError(f'New_size must be either scalar or 2-element vector')
                h, w = img.shape[:2]
                im_scale = (new_size[0] / w, new_size[1] / h)
            else:
                # Only max size given
                im_size_max = np.max(img.shape[:2])
                im_size_min = np.min(img.shape[:2])
                im_scale = float(new_size) / float(im_size_min)

                if np.round(im_scale * im_size_max) > new_size:
                    im_scale = float(new_size) / float(im_size_max)

                new_size = (new_size, new_size)
                im_scale = (im_scale, im_scale)

            if not upsize and min(im_scale) > 1.0:
                # Image size is less than new_size and upsize not allowed
                if not center:
                    # Nothing to do!
                    return img, (1.0, 1.0), (0, 0)
                else:
                    # Make a bigger image and center source image on it
                    im, ofs = center_image(img, new_size, pad_color)
                    return im, im_scale, ofs
            else:
                # Perform normal resize
                im = cv2.resize(img, dsize=None, fx=im_scale[0], fy=im_scale[1])
                return im, im_scale, (0, 0)

    if new_size is None and scale is None:
        raise ValueError('Either new_size or scale must be provided')

    result = _inner_resize()
    return result if extra_info else result[0]

def rotate(img: np.ndarray, 
    angle: float, 
    pad_color: Optional[Tuple] = COLOR_WHITE, 
    avoid_clipping: Optional[bool] = True) -> np.ndarray:

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
    c1, c2 = np.abs(M[0, 0]), np.abs(M[0, 1])

    if avoid_clipping:
        nw = int((h * c2) + (w * c1))
        nh = int((h * c1) + (w * c2))
        w, h = nw, nh

        M[0, 2] += (w / 2) - cx
        M[1, 2] += (h / 2) - cy

    return cv2.warpAffine(img, M, (w, h),
                          borderMode = cv2.BORDER_CONSTANT,
                          borderValue = pad_color)

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
        # Patch contains alpha channel
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
