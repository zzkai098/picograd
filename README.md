# picograd

A tiny reverse-mode automatic differentiation engine in pure Python.

`picograd` builds a dynamic computation graph over scalar values and backpropagates
gradients through it with the chain rule — the same core mechanism behind PyTorch's
`autograd` and every other deep learning framework. Zero dependencies, a few hundred
lines, fully gradient-checked. Inspired by Andrej Karpathy's
[micrograd](https://github.com/karpathy/micrograd).

## Install

```bash
pip install picograd
```

Or install from source in editable mode:

```bash
git clone https://github.com/zzkai098/picograd.git
cd picograd
pip install -e .
```

## Quickstart

```python
from picograd import Value

a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)

d = a * b + c        # forward pass builds the graph
d.backward()         # reverse pass fills in every gradient

print(a.grad)        # d(d)/d(a) = b = -3.0
print(b.grad)        # d(d)/d(b) = a =  2.0
```

Gradients flow through arbitrarily deep expressions built from any supported op:

```python
from picograd import Value

x = Value(1.5)
y = Value(-2.0)

z = (x * y + x**2).tanh() * x.exp() - y / x
z.backward()

print(x.grad, y.grad)   # exact analytic gradients
```

## Features

- Scalar `Value` autograd node with a dynamically-built computation graph
- Full operator set: `+`, `-`, `*`, `/`, `**`, unary negation, and reflected
  ops (`2 * x`, `1 + x`, ...)
- Nonlinearities: `tanh`, `relu`, `exp`
- `backward()` via topological sort with correct gradient **accumulation** for
  nodes reused across the graph
- `nn` module with composable building blocks: `Neuron`, `Layer`, `MLP`
- No dependencies — just the standard library

## API

```python
from picograd import Value
from picograd.nn import Neuron, Layer, MLP

model = MLP(nin=3, nouts=[4, 4, 1])   # a 3 -> 4 -> 4 -> 1 network
params = model.parameters()            # all weights + biases as Values
model.zero_grad()                      # reset gradients before each backward
```

## Development

```bash
pip install -e .
python -m pytest -v        # run the test suite
```

The test suite includes a numerical gradient check (analytic `backward()` vs.
central-difference) across a mixed expression, so any incorrect derivative or
unattached backward closure is caught immediately.

## Project layout

```
picograd/
├── picograd/
│   ├── __init__.py
│   ├── engine.py     # core autograd: the Value class
│   └── nn.py         # neural net building blocks (Neuron / Layer / MLP)
├── tests/
│   └── test_engine.py
└── README.md
```

## License

MIT © 2026 Zhankai Zhang
