# Image processing functions package
# Unit tests for align_images module
# (c) kol, 2022

from common import show_image, put_centered_text
from img_utils22 import align_images, get_image_diff, COLOR_WHITE

def test_align_images(blue_square_image, blue_rotated_square_image, blue_skewed_square_image, 
                        blue_shifted_square_image, debug_show):

    put_centered_text(blue_square_image, '1', COLOR_WHITE)
    put_centered_text(blue_rotated_square_image, '2', COLOR_WHITE)
    put_centered_text(blue_skewed_square_image, '3', COLOR_WHITE)
    put_centered_text(blue_shifted_square_image, '4', COLOR_WHITE)

    show_image(blue_square_image, debug_show, 'test_align_images:1 - original')
    show_image(blue_rotated_square_image, debug_show, 'test_align_images:2 - rotated')

    # Rotation is an euclidian transformation
    img = align_images(blue_square_image, blue_rotated_square_image, 'euclidian', pad_color=COLOR_WHITE)
    assert img is not None
    score, _ = get_image_diff(blue_square_image, img)
    print(f'\nSimilarity score after rotated image euclidian alignment: {score}')
    assert score > 0.98
    show_image(img, debug_show, 'test_align_images:3 - rotated aligned')

    # Skew is affine or homography transformation
    show_image(blue_skewed_square_image, debug_show, 'test_align_images:4 - skewed')

    img = align_images(blue_square_image, blue_skewed_square_image, 'affine', pad_color=COLOR_WHITE)
    assert img is not None
    show_image(img, debug_show, 'test_align_images:5 - skewed affine aligned')
    score, _ = get_image_diff(blue_square_image, img)
    print(f'Similarity score after skewed image affine alignment: {score}')
    assert score > 0.98

    img = align_images(blue_square_image, blue_skewed_square_image, 'homography', pad_color=COLOR_WHITE)
    assert img is not None
    show_image(img, debug_show, 'test_align_images:5 - skewed homography aligned')
    score, _ = get_image_diff(blue_square_image, img)
    print(f'Similarity score after skewed image homography alignment: {score}')
    assert score > 0.98
