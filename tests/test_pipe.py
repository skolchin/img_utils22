# Image processing functions package
# Unit tests for pipe module
# (c) kol, 2022

from common import show_image
from img_utils22 import filters, Pipe

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

def test_pipe_mixin(test_color_image, debug_show):
    pipe = filters.Blur() | filters.Gray() | filters.Edges()
    assert isinstance(pipe, Pipe)

    img = pipe(test_color_image)
    show_image(test_color_image, debug_show, title='pipe:1 - original')
    show_image(img, debug_show, title='pipe:2 - processed')
    assert filters.Gray in pipe
    assert 'Gray' in pipe
