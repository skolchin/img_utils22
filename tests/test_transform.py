# Image processing functions package
# Unit tests for transform module
# (c) kol, 2022

import pytest
import numpy as np
from common import show_image
from img_utils22 import apply_patch

def test_apply_patch(white_image, small_black_image, debug_show):
    py, px = int(white_image.shape[0] / 2), int(white_image.shape[1] / 2)
    ph, pw = small_black_image.shape[:2]
    img = apply_patch(white_image, px, py, small_black_image)
    assert tuple(img.shape) == tuple(white_image.shape)
    assert tuple(img[py-1,px-1, :]) == (255,255,255)
    assert tuple(img[py,px, :]) == (0,0,0)
    assert tuple(img[py+ph-1,px+pw-1, :]) == (0,0,0)
    assert tuple(img[py+ph,px+pw, :]) == (255,255,255)
    show_image(img, debug_show, 'test_apply_patch:1 - no mask')

    mask = np.ones_like(small_black_image)
    mask[0:int(ph/2), :, :] = 0
    img = apply_patch(white_image, px, py, small_black_image, mask=mask)
    assert tuple(img.shape) == tuple(white_image.shape)
    assert tuple(img[py-1,px-1, :]) == (255,255,255)
    assert tuple(img[py,px, :]) == (255,255,255)
    assert tuple(img[py+ph-1,px+pw-1, :]) == (0,0,0)
    assert tuple(img[py+ph,px+pw, :]) == (255,255,255)
    show_image(img, debug_show, 'test_apply_patch:2 - with mask')

    img = apply_patch(white_image, px, py, small_black_image, alpha=0.5)
    assert tuple(img.shape) == tuple(white_image.shape)
    assert tuple(img[py-1,px-1, :]) == (255,255,255)
    assert tuple(img[py,px, :]) == (128,128,128)
    assert tuple(img[py+ph-1,px+pw-1, :]) == (128,128,128)
    assert tuple(img[py+ph,px+pw, :]) == (255,255,255)
    show_image(img, debug_show, 'test_apply_patch:3 - with alpha')

    with pytest.raises(ValueError):
        apply_patch(small_black_image, 0, 0, white_image)

    img = apply_patch(small_black_image, 0, 0, white_image, clip=True)
    assert tuple(img.shape) == tuple(small_black_image.shape)
    assert tuple(img[0,0, :]) == (255,255,255)
    show_image(img, debug_show, 'test_apply_patch:4 - with clipping')
