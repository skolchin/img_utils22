# Image processing functions package
# Unit tests for align_images module
# (c) kol, 2022

from common import show_image
from kol_img_utils import align_images

def test_align_images(blue_square_image, blue_rotated_square_image, blue_skewed_square_image, debug_show):

    show_image(blue_square_image, debug_show, 'test_align_images:1 - original')
    show_image(blue_rotated_square_image, debug_show, 'test_align_images:2 - rotated')
    show_image(blue_skewed_square_image, debug_show, 'test_align_images:3 - skewed')

    # Rotation is an euclidian transformation
    img = align_images(blue_square_image, blue_rotated_square_image, 'euclidian')
    assert img is not None
    show_image(img, debug_show, 'test_align_images:4 - rotated aligned')

    # Skew is affine transformation
    img = align_images(blue_square_image, blue_skewed_square_image, 'homography')
    assert img is not None
    show_image(img, debug_show, 'test_align_images:5 - skewed aligned')
