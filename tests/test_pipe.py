# Image processing functions package
# Unit tests for pipe module
# (c) kol, 2022

import pytest
from img_utils22 import pipe, Pipe

def test_pipe():
    @pipe
    def a(param_1, param_a):
        return param_1 + '_' + param_a

    @pipe
    def b(param_1, param_b):
        return param_1 + '_' + param_b

    @pipe
    def wrong_func():
        pass

    p = Pipe() | a | b
    assert p('start', param_a='a', param_b='b') == 'start_a_b'

    with pytest.raises(TypeError):
        p('start', param_a='a')

        p = Pipe() | wrong_func
        p()

        p = Pipe() | 'c'
        p()
