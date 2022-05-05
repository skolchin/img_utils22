# Image processing functions package
Version 1.0.0  
(c) kol 2019-2022  

This package contains various image processing functions I wrote on multiple projects.
I developed it mostly for personal use, but I welcome free usage by anyone.

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
