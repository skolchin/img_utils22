# Image processing functions package
# Unit tests for utils module
# (c) kol, 2022

import pytest
from common import show_image
from img_utils22 import img1_to_img3, get_image_area

def test_img1_to_img3(white_square_image_bw, debug_show):
    img = img1_to_img3(white_square_image_bw)
    assert len(img.shape) == 3
    assert img.shape[2] == 3
    show_image(img, debug_show, 'test_img1_to_img3:1')

def test_get_image_area(white_square_image, debug_show):
    img = get_image_area(white_square_image, (10, 10, 20, 20))
    assert len(img.shape) == 3
    show_image(img, debug_show, 'test_get_image_area:1')

    with pytest.raises(ValueError):
        get_image_area(white_square_image, 1)
        get_image_area(white_square_image, (10, 10))
        get_image_area(white_square_image, (-10, 10, 20, 20))
        get_image_area(white_square_image, (10, 10, 0, 20))

