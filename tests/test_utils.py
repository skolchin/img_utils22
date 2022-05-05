# Image processing functions package
# Unit tests for utils module
# (c) kol, 2022

import pytest
import numpy as np
from common import show_image

from kol_img_utils import (
    img1_to_img3,
    get_image_area,
    apply_patch
)

def test_img1_to_img3(white_image_bw, debug_show):
    img = img1_to_img3(white_image_bw)
    assert tuple(img.shape) == (100, 100, 3)
    show_image(img, debug_show, 'test_img1_to_img3')

def test_get_image_area(white_image, debug_show):
    img = get_image_area(white_image, (10, 10, 20, 20))
    assert tuple(img.shape) == (10, 10, 3)
    show_image(img, debug_show, 'test_get_image_area')

    with pytest.raises(ValueError):
        get_image_area(white_image, 1)
        get_image_area(white_image, (10, 10))
        get_image_area(white_image, (-10, 10, 20, 20))
        get_image_area(white_image, (10, 10, 0, 20))

def test_apply_patch(white_image, small_black_image, debug_show):
    img = apply_patch(white_image, 10, 10, small_black_image)
    assert tuple(img.shape) == tuple(white_image.shape)
    assert tuple(img[9,9, :]) == (255,255,255)
    assert tuple(img[10,10, :]) == (0,0,0)
    assert tuple(img[19,19, :]) == (0,0,0)
    assert tuple(img[20,20, :]) == (255,255,255)
    show_image(img, debug_show, '1:test_apply_patch: no mask')

    mask = np.ones_like(small_black_image)
    mask[0, :, :] = 0
    img = apply_patch(white_image, 10, 10, small_black_image, mask=mask)
    assert tuple(img.shape) == tuple(white_image.shape)
    assert tuple(img[9,9, :]) == (255,255,255)
    assert tuple(img[10,10, :]) == (255,255,255)
    assert tuple(img[19,19, :]) == (0,0,0)
    assert tuple(img[20,20, :]) == (255,255,255)
    show_image(img, debug_show, '2:test_apply_patch: with mask')

    img = apply_patch(white_image, 10, 10, small_black_image, alpha=0.5)
    assert tuple(img.shape) == tuple(white_image.shape)
    assert tuple(img[9,9, :]) == (255,255,255)
    assert tuple(img[10,10, :]) == (128,128,128)
    assert tuple(img[19,19, :]) == (128,128,128)
    assert tuple(img[20,20, :]) == (255,255,255)
    show_image(img, debug_show, '3:test_apply_patch: with alpha')

    with pytest.raises(ValueError):
        apply_patch(small_black_image, 0, 0, white_image)

    img = apply_patch(small_black_image, 0, 0, white_image, clip=True)
    assert tuple(img.shape) == tuple(small_black_image.shape)
    assert tuple(img[0,0, :]) == (255,255,255)
    show_image(img, debug_show, '4:test_apply_patch: with clipping')
