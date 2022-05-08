# Image processing functions package
# Simple Python pipeline for filters
# (c) kol, 2022

import inspect
from functools import reduce, wraps
from typing import Callable, Any

class Pipe:
    """ Pipeline of functions passing results of one call to another.
    
    Use `Pipe()` to start pipeline construction and add any functions via `|` operator.
    Resulting pipe could be executed by calling it while providing all parameters
    required by individual pipe functions.

    Pipe functions must have at least 1 positional parameter, which will be set to results 
    of previous function's call during the pipe execution. The functions might have any number of other
    parameters, which will be passed by keyword.
    
    Results of function calls will be passed to the next function in pipe, result of the last function's 
    call will be returned to the caller (`reduce` pattern).

    Examples:
        >>> @pipe
        >>> def a(param_1, param_a):
        >>>     print(f'Call to a() with param_1={param_1}, param_a={param_a}')
        >>>     return param_1 + '_' + param_a
        >>> @pipe
        >>> def b(param_1, param_b):
        >>>     print(f'Call to b() with param_1={param_1}, param_b={param_b}')
        >>>     return param_1 + '_' + param_b
        >>> p = Pipe() | a | b
        >>> print(p('start', param_a='a', param_b='b')) # output: start_a_b
    """
    def __init__(self):
        self.__funcs = []

    def __or__(self, func: Callable) -> 'Pipe':
        """ Add a function to pipeline """
        if not callable(func):
            raise TypeError(f'{type(func)!r} objects cannot be added to pipe')
        self.__funcs.append(func)
        return self

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """ Call the pipeline """

        # First function argument must be positional
        if args is None or not len(args):
            args_list = [None]
        else:
            args_list = list(args)

        def call_fun(res, fun):
            args_list[0] = res
            return fun(*args_list, **kwargs)

        return reduce(call_fun, self.__funcs, args_list[0])

def pipe(func: Callable):
    """ Pipe decorator
    
    Args:
        func: Any callable with at least one positional parameter

    See Also:
        Pipe

    Examples:
        >>> @pipe
        >>> def a(param_1):
        >>>     print(f'Call to a() with param_1={param_1}')
        >>>     return param_1 + '_after_a'
        >>> p = Pipe() | a
        >>> p('start')  # output: start_after_a
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Get the function signature
        sig = inspect.signature(func)

        # Check its 1st argument is positional
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
