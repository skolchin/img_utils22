# Image processing functions package
# Unit tests for filters module
# (c) kol, 2022

import pytest
from common import show_image

from img_utils22 import filters, Pipe

def test_filter_edges(test_color_image, test_gray_image, debug_show):
    show_image(test_gray_image, debug_show, title='edges:1 - source')

    img = filters.Edges()(test_gray_image)
    show_image(img, debug_show, title='edges:2 - converted')

    with pytest.raises(AssertionError):
        img = filters.Edges()(test_color_image)

def test_filter_gray(test_color_image, test_gray_image, debug_show):
    show_image(test_color_image, debug_show, title='gray:1 - source')

    img = filters.Gray()(test_color_image)
    assert len(img.shape) == 2
    assert img.shape == test_color_image.shape[:2]
    
    show_image(img, debug_show, title='gray:2 - converted')

    with pytest.raises(AssertionError):
        img = filters.Gray()(test_gray_image)

def test_filter_pyramid(test_color_image, debug_show):
    show_image(test_color_image, debug_show, title='pyramid:1 - source')
    img = filters.PyramidFilter()(test_color_image)
    assert img is not None
    show_image(img, debug_show, title='pyramid:2 - converted')

def test_filter_channel(test_color_image, debug_show):
    show_image(test_color_image, debug_show, title='channel:1 - source')
    for n, ch in enumerate(filters.Channel.CHANNELS):
        img = filters.Channel(ch)(test_color_image)
        if n <= 2:
            assert img is not None
        show_image(img, debug_show, title=f'channel:{n+2} - channel {ch}')

def test_filter_threshold(test_color_image, test_gray_image, debug_show):
    show_image(test_color_image, debug_show, title='thresh:1 - source')

    img = filters.Threshold('binary')(test_color_image)
    assert img is not None
    show_image(img, debug_show, title=f'thresh:2 - binary')

    img = filters.Threshold('binary+otsu')(test_gray_image)
    assert img is not None
    show_image(img, debug_show, title=f'thresh:3 - binary+otsu')

def test_pipe(test_color_image, debug_show):
    pipe = Pipe() | \
        filters.PyramidFilter() | \
        filters.Gray() | \
        filters.Edges() | \
        filters.Ensure3()

    img = pipe(test_color_image)
    show_image(test_color_image, debug_show, title='pipe:1 - original')
    show_image(img, debug_show, title='pipe:2 - processed')

    assert filters.PyramidFilter in pipe
    assert filters.Gray in pipe
    assert 'Gray' in pipe
    assert 'Bla' not in pipe

# def test_filter_clahe(test_color_image, debug_show):
#     show_image(test_color_image, debug_show, title='clahe:1 - source')
#     img = filters.equalize_luminosity(test_color_image)
#     show_image(img, debug_show, title='clahe:2 - converted')

# def test_filter_brightness(test_color_image, debug_show):
#     show_image(test_color_image, debug_show, title='brt:1 - source')
#     img = filters.increase_brightness(test_color_image, 50)
#     show_image(img, debug_show, title='brt:2 - converted')
