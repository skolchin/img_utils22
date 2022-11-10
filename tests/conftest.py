# Image processing functions package
# Fixtures for unit tests
# (c) kol, 2022

import pytest
import cv2
import numpy as np
from img_utils22 import rotate
from common import get_web_image

# from https://stackoverflow.com/questions/40880259/how-to-pass-arguments-in-pytest-by-command-line
def pytest_addoption(parser):
    parser.addoption('--show_images', action='store_true', default=False)

@pytest.fixture(scope="session")
def debug_show(request):
    return request.config.option.show_images

@pytest.fixture
def normal_square_size():
    return (500, 500)

@pytest.fixture
def small_square_size():
    return (50, 50)

@pytest.fixture
def normal_rect_size():
    return (250, 500)

@pytest.fixture
def white_square_image(normal_square_size):
    return np.full((normal_square_size[0], normal_square_size[1], 3), 255, dtype=np.uint8)

@pytest.fixture
def white_square_image_bw(normal_square_size):
    return np.full(normal_square_size, 255, dtype=np.uint8)

@pytest.fixture
def white_rect_image(normal_rect_size):
    return np.full((normal_rect_size[0], normal_rect_size[1], 3), 255, dtype=np.uint8)

@pytest.fixture
def black_square_image(normal_square_size):
    return np.full((normal_square_size[0], normal_square_size[1], 3), 0, dtype=np.uint8)

@pytest.fixture
def small_black_square_image(small_square_size):
    return np.full((small_square_size[0], small_square_size[1], 3), 0, dtype=np.uint8)

@pytest.fixture
def black_rect_image(normal_rect_size):
    return np.full((normal_rect_size[0], normal_rect_size[1], 3), 0, dtype=np.uint8)

@pytest.fixture
def blue_square_image(white_square_image):
    cy, cx = int(white_square_image.shape[0]/2), int(white_square_image.shape[1]/2)
    return cv2.rectangle(white_square_image.copy(), (int(cx/2), int(cy/2)), 
                         (cx+int(cx/2), cy+int(cy/2)), (255,0,0), -1)

@pytest.fixture
def blue_rotated_square_image(blue_square_image):
    return rotate(blue_square_image, 10)

@pytest.fixture
def blue_skewed_square_image(white_square_image):
    cy, cx = int(white_square_image.shape[0]/2), int(white_square_image.shape[1]/2)
    w, h, dw = int(cx/2), int(cy/2), int(cx/4)
    points = np.array([[[cx-w+dw, cy-h], [cx+w+dw, cy-h], 
                       [cx+w, cy+h], [cx-w, cy+h]]], np.int32)
    return cv2.fillPoly(white_square_image.copy(), points, (255,0,0), 8)

@pytest.fixture
def blue_shifted_square_image(white_square_image):
    cy, cx = int(white_square_image.shape[0]/2), int(white_square_image.shape[1]/2)
    return cv2.rectangle(white_square_image.copy(), (int(cx/2)+int(cx/4), int(cy/2)+int(cy/4)), 
                         (cx+int(cx/2), cy+int(cy/2)), (255,0,0), -1)

@pytest.fixture
def test_color_image():
    return get_web_image('https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png')

@pytest.fixture
def test_fgbg_images():
    img_fg = get_web_image('https://i.stack.imgur.com/oyrKo.jpg')
    img_bg = get_web_image('https://i.stack.imgur.com/rMoqy.jpg')
    return img_fg, img_bg
