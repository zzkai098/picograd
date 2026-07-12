"""Gradient checks for the autograd engine.

Run from the project root:
    python -m pytest tests/test_engine.py -v
"""

import math
from picograd import Value


# ------------------------------------------------------------------
# Forward pass
# ------------------------------------------------------------------
def test_add_forward():
    a = Value(2.0)
    b = Value(3.0)
    assert (a + b).data == 5.0


def test_mul_forward():
    a = Value(2.0)
    b = Value(-4.0)
    assert (a * b).data == -8.0


def test_ops_forward():
    a = Value(2.0)
    b = Value(-3.0)
    assert (a - b).data == 5.0
    assert (a ** 2).data == 4.0
    assert (a / b).data == 2.0 / -3.0
    assert (2 * a).data == 4.0          # __rmul__
    assert (2 + a).data == 4.0          # __radd__
    assert (2 - a).data == 0.0          # __rsub__
    assert (2 / a).data == 1.0          # __rtruediv__
    assert abs(a.exp().data - math.exp(2.0)) < 1e-9
    assert Value(-5.0).relu().data == 0.0
    assert Value(5.0).relu().data == 5.0
    assert abs(Value(0.5).tanh().data - math.tanh(0.5)) < 1e-9


# ------------------------------------------------------------------
# Backward pass
# ------------------------------------------------------------------
def test_simple_backward():
    a = Value(2.0)
    b = Value(-3.0)
    c = Value(10.0)
    d = a * b + c
    d.backward()
    assert a.grad == -3.0  # d(d)/d(a) = b
    assert b.grad == 2.0   # d(d)/d(b) = a
    assert c.grad == 1.0   # d(d)/d(c) = 1


def test_tanh_backward():
    x = Value(0.0)
    y = x.tanh()
    y.backward()
    # d(tanh)/dx at 0 == 1
    assert abs(x.grad - 1.0) < 1e-6


def test_grad_accumulates():
    # A node reused in the graph must accumulate (+=), not overwrite (=).
    x = Value(3.0)
    (x + x).backward()          # dy/dx = 2
    assert x.grad == 2.0

    x = Value(4.0)
    (x * x).backward()          # dy/dx = 2x = 8
    assert x.grad == 8.0


# ------------------------------------------------------------------
# Numerical gradient check: analytic backward vs. central difference.
# Catches an unattached _backward or a wrong derivative formula.
# ------------------------------------------------------------------
def _grad_check(build_fn, inputs, h=1e-6, tol=1e-4):
    vals = [Value(x) for x in inputs]
    build_fn(vals).backward()
    analytic = [v.grad for v in vals]

    for i in range(len(inputs)):
        xp = list(inputs); xp[i] += h
        xm = list(inputs); xm[i] -= h
        fp = build_fn([Value(x) for x in xp]).data
        fm = build_fn([Value(x) for x in xm]).data
        numeric = (fp - fm) / (2 * h)
        assert abs(analytic[i] - numeric) < tol, (
            f"input {i}: analytic={analytic[i]:+.6f} numeric={numeric:+.6f}"
        )


def test_gradcheck_mixed():
    # f = tanh(x0*x1 + x2**2) * exp(x0) - x1/x2 + relu(x0)
    _grad_check(
        lambda v: (v[0] * v[1] + v[2] ** 2).tanh() * v[0].exp() - v[1] / v[2] + v[0].relu(),
        [1.5, -2.0, 3.0],
    )
