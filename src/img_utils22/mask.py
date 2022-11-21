# Image processing functions package
# (c) kol, 2019-2022

""" Mask-related routines """

import cv2
import numpy as np
from skimage.metrics import structural_similarity
from typing import Iterable, Tuple, Optional

from .align_images import align_images
from .filters import _get_kernel
from .colors import COLOR_BLACK

def get_image_diff(
   im1: np.ndarray, 
   im2: np.ndarray, 
   align_mode: Optional[str] = None,
   multichannel: Optional[bool] = False,
   pad_color: Optional[Tuple] = COLOR_BLACK) -> Tuple[float, np.ndarray]:

    """ Get the difference of two images. 
    The function calculates structure similarity index (SSIM) and difference matrix
    of two images with optional alignment.

    Args:
        im1:      First OpenCV image
        im2:      Second OpenCV image
        align_mode:     One of alignment modes (see `align_images` for details). If None,
            images will not be aligned
        multichannel:   If False (default), comparison will be performed on grayed images
            improving speed, otherwise - on each color channel separatelly
        pad_color: Padding color for `align_images`

    Returns:
        2-element tuple with SSIM score (more similarity - higher score) and difference matrix.

    See also:
        align_images
    """

    # Algin images if requested
    if align_mode:
        im2 = align_images(im1, im2, mode=align_mode, pad_color=pad_color)

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

def get_bgsub_mask(
    img: np.ndarray,
    img_bg: np.ndarray,
    kernel_size: Optional[int] = 10
) -> np.ndarray:
    """ Calculate background subtraction mask using static background and foreground images.

    Based on https://stackoverflow.com/questions/25617252/opencv-background-segmentation-subtraction-from-single-image

    Args:
        img:    An OpenCV image with background and probably some foreground objects
        img_bg: An OpenCV image with pure background

    Returns:
        A mask to subtract the background from foreground image. Use `apply_image_mask` to actually 
        extract the foreground

    Examples:

            img = cv2.imread('image.png')
            img_bg = cv2.imread('background.png')
            mask = get_bgsub_mask(img_fg, img_bg, kernel_size=21)
            masked_img = apply_image_mask(img, mask)
            cv2.imshow('Masked image', masked_img)
    """

    backSub = cv2.createBackgroundSubtractorMOG2()
    _ = backSub.apply(img_bg)
    mask = backSub.apply(img)

    kernel = _get_kernel(kernel_size)
    mask_morph = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask_morph = cv2.morphologyEx(mask_morph, cv2.MORPH_CLOSE, kernel)

    return mask_morph

def apply_image_mask(
    img: np.ndarray, 
    mask: np.ndarray) -> np.ndarray:

    """ Applies a mask to image 

    Args:
        img:    An OpenCV image
        mask:   A mask to apply to image. Must be 2D uint8 array with the same shape as image

    Returns:
        An OpenCV image
    """

    if len(mask.shape) != 2 or mask.dtype != np.uint8:
        raise ValueError('mask must be 2-d array of unit8 type')
    if mask.shape != img.shape[:2]:
        raise ValueError('mask shape must match image shape')

    return cv2.bitwise_and(img, img, None, mask)

def get_fgx_mask(
    img: np.ndarray,
    mask_rect: Optional[Iterable] = None,
    mask_array: Optional[np.ndarray] = None,
    model: Optional[Tuple] = None,
    num_iter: Optional[int] = 5,
) -> Tuple[np.ndarray, Tuple]:

    """ Extracts a foreground mask from image

    Args:
        img:        An OpenCV image
        mask_rect:  A rectangular mask which defines an rectangular area to process
        mask_array: A custom mask initialized with cv2.GC_xxx constants
        model:      Previously created extraction model
        num_iter:   Number of iterations

    Returns:
        Tuple containing generated mask and extraction model
    """

    if model is None:
        fgdModel, bgdModel = None, None
    else:
        fgdModel, bgdModel = model

        if bgdModel is None:
            bgdModel = np.zeros((1, 65), np.float64)
        elif bgdModel.shape != (1, 65):
            raise ValueError('bgdModel shape must be (1,65')

        if fgdModel is None:
            fgdModel = np.zeros((1, 65), np.float64)
        elif fgdModel.shape != (1, 65):
            raise ValueError('fgdModel shape must be (1,65')

    if mask_rect is not None:
        """ Rectangular mask """
        if len(mask_rect) != 4:
            raise ValueError('mask_rect must must have length 4')
        mask = np.full(img.shape[:2], cv2.GC_PR_BGD, np.uint8)
        mask, bgdModel, fgdModel = cv2.grabCut(img, mask, mask_rect, bgdModel, fgdModel, num_iter, cv2.GC_INIT_WITH_RECT)
        mask = np.where((mask == cv2.GC_PR_BGD) | (mask == cv2.GC_BGD), 0, 1).astype(np.uint8)

    elif mask_array is not None:
        """ Custom mask """
        if len(mask_array.shape) > 2 or mask_array.dtype != np.uint8:
            raise ValueError('mask_array must be 2-d numpy array of uint8')
        if mask_array.shape != img.shape[:2]:
            raise ValueError('mask shape must match image shape')

        mask, bgdModel, fgdModel = cv2.grabCut(img, mask_array, None, bgdModel, fgdModel, num_iter, cv2.GC_INIT_WITH_MASK)
        mask = np.where((mask == cv2.GC_PR_BGD) | (mask == cv2.GC_BGD), 0, 1).astype(np.uint8)
    else:
        raise ValueError('Either mask_rect or mask_array have to be provided')

    return mask, (bgdModel, fgdModel)
