# Image processing functions package
# Fixtures for unit tests
# (c) kol, 2022

import pytest
import cv2
import numpy as np
from kol_img_utils import rotate

# from https://stackoverflow.com/questions/40880259/how-to-pass-arguments-in-pytest-by-command-line
def pytest_addoption(parser):
    parser.addoption('--show_images', action='store_true', default=False)

@pytest.fixture(scope="session")
def debug_show(request):
    return request.config.option.show_images

@pytest.fixture
def normal_size():
    return (500, 500)

@pytest.fixture
def small_size():
    return (50, 50)

@pytest.fixture
def white_image(normal_size):
    return np.full((normal_size[0], normal_size[1], 3), 255, dtype=np.uint8)

@pytest.fixture
def white_image_bw(normal_size):
    return np.full(normal_size, 255, dtype=np.uint8)

@pytest.fixture
def black_image(normal_size):
    return np.full((normal_size[0], normal_size[1], 3), 0, dtype=np.uint8)

@pytest.fixture
def small_black_image(small_size):
    return np.full((small_size[0], small_size[1], 3), 0, dtype=np.uint8)

@pytest.fixture
def blue_square_image(white_image):
    cy, cx = int(white_image.shape[0]/2), int(white_image.shape[1]/2)
    return cv2.rectangle(white_image.copy(), (int(cx/2), int(cy/2)), 
                         (cx+int(cx/2), cy+int(cy/2)), (255,0,0), -1)

@pytest.fixture
def blue_rotated_square_image(blue_square_image):
    return rotate(blue_square_image, 10)

@pytest.fixture
def blue_skewed_square_image(white_image):
    cy, cx = int(white_image.shape[0]/2), int(white_image.shape[1]/2)
    w, h, dw = int(cx/2), int(cy/2), int(cx/4)
    points = np.array([[[cx-w+dw, cy-h], [cx+w+dw, cy-h], 
                       [cx+w, cy+h], [cx-w, cy+h]]], np.int32)
    return cv2.fillPoly(white_image.copy(), points, (255,0,0), 8)

@pytest.fixture
def blue_shifted_square_image(white_image):
    cy, cx = int(white_image.shape[0]/2), int(white_image.shape[1]/2)
    return cv2.rectangle(white_image.copy(), (int(cx/2)+int(cx/4), int(cy/2)+int(cy/4)), 
                         (cx+int(cx/2), cy+int(cy/2)), (255,0,0), -1)
