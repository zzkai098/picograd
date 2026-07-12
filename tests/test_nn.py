"""Tests for the neural-net building blocks (picograd/nn.py).

Run from the project root:
    python -m pytest tests/test_nn.py -v
"""

import random

from picograd.engine import Value
from picograd.nn import Neuron, Layer, MLP


# ------------------------------------------------------------------
# Structure: parameter counts and forward-pass shapes
# ------------------------------------------------------------------
def test_neuron_params():
    n = Neuron(3)
    assert len(n.parameters()) == 4          # 3 weights + 1 bias


def test_layer_params_and_shape():
    layer = Layer(3, 5)
    assert len(layer.parameters()) == 5 * (3 + 1)   # 5 neurons x (3 w + 1 b)

    out = layer([1.0, 2.0, 3.0])
    assert isinstance(out, list) and len(out) == 5

    # A single-neuron layer returns a bare Value, not a list.
    single = Layer(3, 1)
    assert isinstance(single([1.0, 2.0, 3.0]), Value)


def test_mlp_params_and_forward():
    model = MLP(3, [4, 4, 1])
    # 4*(3+1) + 4*(4+1) + 1*(4+1) = 16 + 20 + 5 = 41
    assert len(model.parameters()) == 4 * (3 + 1) + 4 * (4 + 1) + 1 * (4 + 1)

    out = model([1.0, 2.0, 3.0])
    assert isinstance(out, Value)


# ------------------------------------------------------------------
# Activation selection
# ------------------------------------------------------------------
def test_activation_relu_clamps_negative():
    n = Neuron(1, activation='relu')
    n.w[0].data = -5.0
    n.b.data = 0.0
    assert n([1.0]).data == 0.0              # relu(-5) == 0


def test_activation_none_is_linear():
    n = Neuron(1, activation=None)
    n.w[0].data = 2.0
    n.b.data = 1.0
    assert n([3.0]).data == 7.0              # 2*3 + 1, no activation


def test_invalid_activation_raises():
    try:
        Neuron(2, activation='sigmoid')
    except AssertionError:
        return
    assert False, "expected AssertionError for unknown activation"


# ------------------------------------------------------------------
# Training: the MLP should drive a toy loss toward zero.
# ------------------------------------------------------------------
def test_mlp_trains():
    random.seed(0)
    xs = [[2.0, 3.0], [-1.0, -1.0], [0.5, -2.0], [1.0, 1.0]]
    ys = [1.0, -1.0, -1.0, 1.0]

    model = MLP(2, [8, 8, 1], activation='tanh')

    def mse():
        ypred = [model(x) for x in xs]
        return sum(((y - p) ** 2 for y, p in zip(ys, ypred)), Value(0.0)) / len(ys)

    start = mse().data
    for _ in range(50):
        loss = mse()
        model.zero_grad()
        loss.backward()
        for p in model.parameters():
            p.data -= 0.2 * p.grad

    end = mse().data
    assert end < start * 0.1, f"loss did not drop enough: {start:.4f} -> {end:.4f}"
