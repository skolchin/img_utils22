# Image processing functions package
# (c) kol, 2022

""" Piping for image filters """

import numpy as np
import inspect
from functools import reduce, wraps
from typing import Callable

class Pipe:
    """ Pipeline of image processing functions.
    
    Use `Pipe()` to start pipeline construction, then add callables via `|` operator.
    Resulting pipe could be executed with a `call()` with a source image.

    While the pipe supports passing of custom keyword parameters to individual calls,
    it's better to use callable objects where all parameters are set at initialization.
    See `.filters` module for details.

    Examples:

        pipe = Pipe() | PyramidFilter() | Gray() | Edges()
        img = pipe(test_color_image)

    """
    def __init__(self):
        self.__funcs = []

    def __or__(self, func: Callable) -> 'Pipe':
        """ Add a function to pipeline """
        if not callable(func):
            raise TypeError(f'{type(func)!r} objects cannot be added to pipe')
        self.__funcs.append(func)
        return self

    def __call__(self, img: np.ndarray, **kwargs) -> np.ndarray:
        """ Call the pipeline """
        return reduce(lambda im, func: func(im, **kwargs), self.__funcs, img)

    def __len__(self):
        return len(self.__funcs)

    def __contains__(self, fclass):
        return self.find(fclass) is not None

    def __iter__(self):
        return iter(self.__funcs)

    def __getitem__(self, fclass):
        for func in self.__funcs:
            if isinstance(fclass, str):
                name = func.__class__.__name__ if hasattr(func, '__class__') else func.__name__
                if name == fclass:
                    return func
            elif isinstance(func, fclass):
                return func
        raise KeyError(fclass)

    def find(self, fclass):
        try:
            return self[fclass]
        except KeyError:
            return None

class PipedMixin:
    """ Mixin class to include into piped classes.

    Allows to make pipe from arbitrary function or class with no `Pipe()` call.

    Examples:

        pipe = Pipe() | Gray() | Edges() # traditional way
        assert isinstance(pipe, Pipe)

        pipe = Gray() | Edges() # this would work too
        assert isinstance(pipe, Pipe)

    """
    def __or__(self, func: Callable) -> 'Pipe':
        return Pipe() | self | func

def pipe(func: Callable):
    """ Pipe decorator
    
    Args:
        func: Any callable with at least one positional parameter (which is expected to be image)
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        sig = inspect.signature(func)
        params = list(sig.parameters.values())
        if not params or params[0].kind not in (inspect.Parameter.POSITIONAL_ONLY,
                                                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                                inspect.Parameter.VAR_POSITIONAL):
            raise TypeError(f'Pipe function {func} must have 1st positional parameter')

        # Limit kwargs to ones expected by the function
        kwargs = {k:v for k, v in kwargs.items() or {} if k in sig.parameters}

        # Bind parameters and call the function
        ba = sig.bind(*args, **kwargs)
        ba.apply_defaults()
        return func(*ba.args, **ba.kwargs)

    return wrapper
