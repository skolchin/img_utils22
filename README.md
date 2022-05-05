# Image processing functions package
Version 1.0.0  
(c) kol 2019-2022  

This package contains various image processing functions I wrote on multiple projects.
I developed it mostly for personal use, but I welcome free usage by anyone.

All the functions are built on top of wonderful OpenCV package and therefore use
the same data structures and conventions. 

Basically, an OpenCV's image is an [numpy](https://numpy.org/) array (`ndarray`), 
usually of `np.uint8` type, with 3 dimensions representing image's *height*, *width* and *color depth* 
respectivelly.

Colors are encoded as 3-element tuple with unsigned integer values ranged from `0` (black) to `255` (white)
which occupy the last dimension of an array. Each tuple element specifies value of some basic
color. Other colors are encoded as combination of basic color values. 

Order of colors used by the OpenCV is *blue, green, red (BGR)*
which is different from other packages like [matplotlib](https://matplotlib.org/) or 
[Pillow](https://pillow.readthedocs.io).

For example, 100x100 black color image will be represented as numpy array with shape `[100,100,3]`
and all its values equal to `0`, while the same size red image will have the same number of
dimensions, but its last value would be `(0,0,255)`.

## List of functions (version 1.0.0)

| Function | Description |
|----------|-------------|
| random_colors         | Iterator which returns a sequence of random colors |
| gradient_colors       | Gradient color generator |
| rgba_to_rgb           | Convert RGBA color to RGB color |
| rgb_to_rgba           | Convert RGB color to RGBA color |
| increase_brightness   | Increase image brightness by given value |
| center_image          | Make a bigger image and center initial image on it |
| resize                | Resizes an image so neither of its sides will be bigger specified size |
| rotate                | Rotate given image to specified angle |
| img1_to_img3          | Converts 1-channel (BW) image to 3-channel (color) image |
| get_image_area        | Get part of an image defined by rectangular area |
| apply_patch           | Applies a patch at given coordinates with optional masking and alpha-channel blending |


## Installation

Since I didn't publish it to PyPi, the package must be built from scratch, like this:

    >>> python -m pip install build
    >>> python -m build .
    >>> pip install dist/kol_img_utils-<version>-py3-none-any.whl

## Unit testing

Unit tests are written for [pytest](https://docs.pytest.org/). To perform all tests simple run
`pytest` in `tests\` directory, for example:

    >>> pytest -s -v

Not everything is covered by tests, though, but I've been working on it.

Please also note that I didn't test this package under *nix.
