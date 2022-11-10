import cv2
from img_utils22 import get_bgsub_mask, apply_image_mask
from common import show_image

def test_bgsub(test_fgbg_images, debug_show):
    img_fg, img_bg = test_fgbg_images
    mask = get_bgsub_mask(img_fg, img_bg, 21)
    new_img = apply_image_mask(img_fg, mask)

    show_image(img_fg, debug_show, title='bgsub:1 - original')
    show_image(mask, debug_show, title='bgsub:2 - mask')
    show_image(new_img, debug_show, title='bgsub:3 - result')
