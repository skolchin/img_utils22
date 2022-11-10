# Image processing functions package
# Unit tests for transform module
# (c) kol, 2022

from common import show_image

from img_utils22 import filters

def test_filter_gray(test_color_image, debug_show):
    show_image(test_color_image, debug_show, title='gray:1 - source')

    img = filters.gray(test_color_image)
    assert len(img.shape) == 2
    assert img.shape == test_color_image.shape[:2]
    
    show_image(img, debug_show, title='gray:2 - converted')

def test_filter_pyramid(test_color_image, debug_show):
    show_image(test_color_image, debug_show, title='pyramid:1 - source')
    img = filters.pyramid_filter(test_color_image)
    show_image(img, debug_show, title='pyramid:2 - converted')

def test_filter_clahe(test_color_image, debug_show):
    show_image(test_color_image, debug_show, title='clahe:1 - source')
    img = filters.equalize_luminosity(test_color_image)
    show_image(img, debug_show, title='clahe:2 - converted')

def test_filter_brightness(test_color_image, debug_show):
    show_image(test_color_image, debug_show, title='brt:1 - source')
    img = filters.increase_brightness(test_color_image, 50)
    show_image(img, debug_show, title='brt:2 - converted')
