import numpy as np
import cv2

from common import show_image
from img_utils22 import get_fgx_mask, apply_image_mask

def test_fgx(test_fgbg_images, debug_show):
    img_fg, img_bg = test_fgbg_images

    rect = (490, 180, 690, 380)
    mask, model = get_fgx_mask(img_fg, rect)
    new_img_rect = apply_image_mask(img_fg, mask)

    mask = np.full(img_fg.shape[:2], cv2.GC_PR_BGD, np.uint8)
    cv2.circle(mask, (610, 280), 80, cv2.GC_FGD, -1)

    mask, model = get_fgx_mask(img_fg, mask_array=mask)
    new_img_mask = apply_image_mask(img_fg, mask)

    show_image(img_fg, debug_show, 'fgx:1 - original')
    show_image(new_img_mask, debug_show, 'fgx:2 - masked (circle)')
    show_image(new_img_rect, debug_show, 'fgx:3 - masked (rect)')

