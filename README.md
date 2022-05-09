# Image processing functions package

Version 1.0.0  
(c) kol 2019-2022  

This package contains various image processing functions I wrote on multiple projects.
I developed it mostly for personal use, but I welcome free usage by anyone.

Some parts of the code were obtained from public sources, references kept in files.

## Intro

All the functions are built on top of wonderful [OpenCV](https://docs.opencv.org/) package and therefore
use the same data structures and conventions.

Basically, an OpenCV's image is a [Numpy](https://numpy.org/) array (`ndarray`),
usually of `np.uint8` type, with 3 dimensions representing image's *height*, *width* and *color depth*
respectivelly.

Colors are encoded as 3-element tuple with unsigned integer values ranged from `0` (black) to `255` (white)
which occupy the last dimension of an array. Each tuple element specifies value of some basic
color. Other colors are encoded as combination of basic color values.  

Order of colors used by the OpenCV is *blue, green, red (BGR)*
which is different from other packages like [matplotlib](https://matplotlib.org/) or
[Pillow](https://pillow.readthedocs.io).

For example, 100x100 black color image will be represented as numpy array with shape `[100,100,3]`
with all its values equal to `0`, while the same size red image will have the same number of
dimensions, but its last value would be `(0,0,255)`.

## Functions

The list of functions included so far:

| Function              | Description |
|-----------------------|-------------|
| random_colors         | Iterator which returns a sequence of random colors |
| gradient_colors       | Gradient color generator |
| rgba_to_rgb           | Convert RGBA color to RGB color |
| rgb_to_rgba           | Convert RGB color to RGBA color |
| center_image          | Make a bigger image and center initial image on it |
| extend_image          | Extends or shrinks image dimensions to new size |
| rescale               | Proportionally extends an image either by specified scale or to specified size |
| resize                | Proportionally resizes an image either to specified scale or to specified size |
| rotate                | Rotate given image to specified angle |
| patch                 | Applies a patch image with optional masking and alpha-channel blending |
| img1_to_img3          | Converts 1-channel (BW) image to 3-channel (color) image |
| get_image_area        | Get part of an image defined by rectangular area |
| align_images          | Align two images |
| get_image_diff        | Get the difference of two images |
| pyramid_filter        | Pyramid filtering |
| gray                  | Converts to gray |
| extract_channel       | Extracts one channel |
| threshold             | Calculates thresholding mask |
| dilate                | Dilate image |
| erode                 | Erode image |
| blur                  | Blur image |
| equalize_luminosity   | Equalize luminocity |
| increase_brightness   | Increase image brightness by given value |

## Installation

Since I didn't publish it to PyPi yet, the package must be built from scratch, like this:

    >>> python -m pip install build
    >>> python -m build .
    >>> pip install dist/img_utils22-<version>-py3-none-any.whl

## Unit testing

Unit tests are written for [pytest](https://docs.pytest.org/). To perform all tests simple run
`pytest` in `tests\` directory, for example:

    >>> pytest -s -v

In order to show images produced during the tests, specify `--show_images` key:

    >>> pytest -s -v --show_images -k align

In this mode, while an image is showing, press `Q` to continue testing.

Not everything is currently covered by tests but I've been working on it. Please also note that I didn't test this package under *nix.
