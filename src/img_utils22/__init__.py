from .colors import (
    COLOR_BLACK,
    COLOR_WHITE,
    COLOR_BLUE,
    COLOR_GREEN,
    COLOR_RED,
    COLOR_GRAY,
    random_colors,
    gradient_colors,
    rgb_to_rgba,
    rgba_to_rgb,
)
from .transform import (
    center_image,
    extend_image,
    rescale,
    resize,
    rotate,
    patch
)
from .misc import (
    imshow,
    img1_to_img3,
    get_image_area,
    get_center_area,
)
from .align_images import align_images
from .mask import (
    get_image_diff,
    get_bgsub_mask,
    get_fgx_mask,
    apply_image_mask
)
from .filters import (
    LoadFile,
    ShowImage,
    Resize,
    Area,
    PyramidFilter,
    Gray,
    Edges,
    Ensure3,
    Channel,
    Threshold,
    Dilate,
    Erode,
    Blur,
    EqualizeLuminosity,
    IncreaseBrightness,
    AdjustGamma,
    ExtractForeground,
    ExtractFgObjectsArea,
)
from .pipe import Pipe

import cv2
import atexit
atexit.register(cv2.destroyAllWindows)
