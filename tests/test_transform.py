# Image processing functions package
# Unit tests for transform module
# (c) kol, 2022

import pytest
import numpy as np
from common import show_image

from img_utils22 import patch, resize, rescale, rotate, COLOR_WHITE

def test_apply_patch(white_square_image, small_black_square_image, debug_show):
    py, px = int(white_square_image.shape[0] / 2), int(white_square_image.shape[1] / 2)
    ph, pw = small_black_square_image.shape[:2]
    img = patch(white_square_image, px, py, small_black_square_image)
    assert tuple(img.shape) == tuple(white_square_image.shape)
    assert tuple(img[py-1,px-1, :]) == (255,255,255)
    assert tuple(img[py,px, :]) == (0,0,0)
    assert tuple(img[py+ph-1,px+pw-1, :]) == (0,0,0)
    assert tuple(img[py+ph,px+pw, :]) == (255,255,255)
    show_image(img, debug_show, 'test_apply_patch:1 - no mask')

    mask = np.ones_like(small_black_square_image)
    mask[0:int(ph/2), :, :] = 0
    img = patch(white_square_image, px, py, small_black_square_image, patch_mask=mask)
    assert tuple(img.shape) == tuple(white_square_image.shape)
    assert tuple(img[py-1,px-1, :]) == (255,255,255)
    assert tuple(img[py,px, :]) == (255,255,255)
    assert tuple(img[py+ph-1,px+pw-1, :]) == (0,0,0)
    assert tuple(img[py+ph,px+pw, :]) == (255,255,255)
    show_image(img, debug_show, 'test_apply_patch:2 - with mask')

    img = patch(white_square_image, px, py, small_black_square_image, alpha=0.5)
    assert tuple(img.shape) == tuple(white_square_image.shape)
    assert tuple(img[py-1,px-1, :]) == (255,255,255)
    assert tuple(img[py,px, :]) == (128,128,128)
    assert tuple(img[py+ph-1,px+pw-1, :]) == (128,128,128)
    assert tuple(img[py+ph,px+pw, :]) == (255,255,255)
    show_image(img, debug_show, 'test_apply_patch:3 - with alpha')

    with pytest.raises(ValueError):
        patch(small_black_square_image, 0, 0, white_square_image)

    img = patch(small_black_square_image, 0, 0, white_square_image, clip=True)
    assert tuple(img.shape) == tuple(small_black_square_image.shape)
    assert tuple(img[0,0, :]) == (255,255,255)
    show_image(img, debug_show, 'test_apply_patch:4 - with clipping')


def test_resize(black_rect_image, debug_show):
    show_image(black_rect_image, debug_show, 'test_resize:1 - original')

    new_size = max(black_rect_image.shape[:2])*2
    img, scale = resize(black_rect_image, new_size, return_extra=True)
    assert max(img.shape) == new_size
    assert scale == (2.0, 2.0)
    show_image(img, debug_show, 'test_resize:2 - upsize to max')

    new_size = int(min(black_rect_image.shape[:2])/2)
    img, scale = resize(black_rect_image, new_size, return_extra=True)
    assert scale == (0.25, 0.25)
    assert max(img.shape) == new_size
    show_image(img, debug_show, 'test_resize:3 - downsize to min')

    new_size = (int(black_rect_image.shape[0]/2), int(black_rect_image.shape[1]/2))
    img, scale = resize(black_rect_image, new_size, return_extra=True)
    assert scale == (0.5, 0.5)
    assert tuple(img.shape[:2]) == new_size
    show_image(img, debug_show, 'test_resize:4 - resize to exact size')

def test_rescale(black_rect_image, debug_show):
    show_image(black_rect_image, debug_show, 'test_rescale:1 - original')

    new_size = max(black_rect_image.shape[:2])*2
    img, scale, offs = rescale(black_rect_image, new_size, pad_color=COLOR_WHITE, return_extra=True)
    assert tuple(img.shape[:2]) == (black_rect_image.shape[:2][0]*2, black_rect_image.shape[:2][1]*2)
    assert scale == (2.0, 2.0)
    assert offs == (0, 0)
    show_image(img, debug_show, 'test_upscale:2 - upscale to max')

    new_size = max(black_rect_image.shape[:2])*2
    img, scale, offs = rescale(black_rect_image, new_size, pad_color=COLOR_WHITE, center=True, return_extra=True)
    assert tuple(img.shape[:2]) == (black_rect_image.shape[0]*2, black_rect_image.shape[1]*2)
    assert scale == (2.0, 2.0)
    assert offs == (int(black_rect_image.shape[1]/2), int(black_rect_image.shape[0]/2))
    show_image(img, debug_show, 'test_upscale:3 - upscale to max centered')

def test_rotate(blue_square_image, debug_show):
    show_image(blue_square_image, debug_show, 'test_rotate:1 - original')

    img = rotate(blue_square_image, 45, COLOR_WHITE, avoid_clipping=False)
    assert img.shape == blue_square_image.shape
    show_image(img, debug_show, 'test_rotate:1 - rotated as-is')

    img = rotate(blue_square_image, 45, COLOR_WHITE, avoid_clipping=True)
    assert img.shape != blue_square_image.shape
    show_image(img, debug_show, 'test_rotate:1 - rotated no-clip')
