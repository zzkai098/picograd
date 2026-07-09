"""Basic gradient checks for the engine.

The first two tests pass with the minimal core (+ and *). The rest are marked
xfail/skip until you implement the corresponding ops and backward().
"""

import pytest

from zgrad import Value


def test_add_forward():
    a = Value(2.0)
    b = Value(3.0)
    assert (a + b).data == 5.0


def test_mul_forward():
    a = Value(2.0)
    b = Value(-4.0)
    assert (a * b).data == -8.0


@pytest.mark.xfail(reason="implement backward() first")
def test_simple_backward():
    a = Value(2.0)
    b = Value(-3.0)
    c = Value(10.0)
    d = a * b + c
    d.backward()
    assert a.grad == -3.0  # d(d)/d(a) = b
    assert b.grad == 2.0   # d(d)/d(b) = a
    assert c.grad == 1.0   # d(d)/d(c) = 1


@pytest.mark.xfail(reason="implement tanh() + backward() first")
def test_tanh_backward():
    x = Value(0.0)
    y = x.tanh()
    y.backward()
    # d(tanh)/dx at 0 == 1
    assert abs(x.grad - 1.0) < 1e-6
