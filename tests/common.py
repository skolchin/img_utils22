# Image processing functions package
# Unit tests common functions
# (c) kol, 2022

import cv2

def show_image(img, debug_show, title='debug'):
    if debug_show:
        cv2.imshow(title, img)
        cv2.waitKey(0)

