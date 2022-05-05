# Image processing functions package
# Fixtures for unit tests
# (c) kol, 2022

import pytest
import numpy as np

# from https://stackoverflow.com/questions/40880259/how-to-pass-arguments-in-pytest-by-command-line
def pytest_addoption(parser):
    parser.addoption('--show_images', action='store_true', default=False)

@pytest.fixture(scope="session")
def debug_show(request):
    return request.config.option.show_images

@pytest.fixture
def normal_size():
    return (100, 100)

@pytest.fixture
def small_size():
    return (10, 10)

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

