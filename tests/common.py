# Image processing functions package
# Unit tests common functions
# (c) kol, 2022

import cv2

def show_image(img, debug_show, title='debug'):
    if debug_show:
        cv2.imshow(title, img)
        cv2.waitKey(0)

def put_centered_text(img, text, color=(0,0,0)):
    scale = max(img.shape[:2]) / 100.0
    xy = (int(img.shape[1]/2 - 4*scale*len(text)), int(img.shape[0]/2 + 4*scale))
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_PLAIN, scale, color, int(scale))
