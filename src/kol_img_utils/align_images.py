# Image processing functions package
# Image alignment routines
# (c) kol, 2019-2022
#
# Based on Satya Mallick article at
# https://www.learnopencv.com/image-alignment-ecc-in-opencv-c-python/

import cv2
import numpy as np
from typing import Optional, Tuple
from .colors import COLOR_BLACK

_MODES = {
    'translation': (cv2.MOTION_TRANSLATION, 2, 3),
    'affine': (cv2.MOTION_AFFINE, 2, 3),
    'homography': (cv2.MOTION_HOMOGRAPHY, 3, 3),
    'euclidian': (cv2.MOTION_EUCLIDEAN, 2, 3),
}

def align_images(
    im1: np.ndarray, 
    im2: np.ndarray, 
    mode: Optional[str] = 'affine',
    pad_color: Optional[Tuple] = COLOR_BLACK,
    num_iter: Optional[int] = 1000,
    term_eps: Optional[float] = 1e-10) -> np.ndarray:

    """Algins two images.  
    The function tries to align one image to another so they would converge
    as much as possible using specified transformation mode.  

    Briefly, `euclidian` could be used to align rotated images, 
    `affine` helps with shifts and skewness, and `homography` works in perspective distortions.
    There's also `translation` mode available, but I couldn't find any case it works.

    Please not that in case of unsuccessfull alignment (e.g.wrong mode), 
    the underlying OpenCV function could return the image without any modification.
    To catch that, extra checks should be performed (for example, with SSIM score 
    returned by `get_image_diff()`).

    See OpenCV's help for `findTransformECC` for more details.

    Args:
        im1:        An OpenCV image to align to
        im2:        An OpenCV image which should be aligned
        mode:       Transformation mode, one of `affine`, `homography`, `euclidian`
        pad_color:  Padding color
        num_iter:   Number of iterations during ECC matrix construction
        term_eps:   Threshold of coefficient correlation increment

    Returns:
        Aligned image produced from `im2` or None if alignment was unsuccessfull

    See Also:
        get_image_diff
    """

    # Define warp matrix
    if mode not in _MODES:
        raise ValueError(f'Unknown mode: {mode}')

    mode_params = _MODES[mode]
    warp_mode = mode_params[0]
    warp_matrix = np.eye(mode_params[1], mode_params[2], dtype=np.float32)

    # Convert images to grayscale (required by ECC algo)
    im1_gray = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)

    # Run the ECC algorithm and generate warp_matrix
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, num_iter, term_eps)
    (_, warp_matrix) = cv2.findTransformECC(im1_gray, im2_gray, warp_matrix,
                                             motionType=warp_mode,
                                             criteria=criteria)

    try:
        sz = im1.shape
        if warp_mode == cv2.MOTION_HOMOGRAPHY:
            # Use warpPerspective for Homography transformation
            im2_aligned = cv2.warpPerspective(im2, warp_matrix, (sz[1],sz[0]),
                                               flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                                               borderMode=cv2.BORDER_CONSTANT,
                                               borderValue=pad_color)

        else :
            # Use warpAffine for other transformations
            im2_aligned = cv2.warpAffine(im2, warp_matrix, (sz[1],sz[0]),
                                         flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                                         borderMode=cv2.BORDER_CONSTANT,
                                         borderValue=pad_color)
        return im2_aligned

    except:
        # Alignment unsuccessfull
        return None
