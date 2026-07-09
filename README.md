# zgrad

A tiny reverse-mode automatic differentiation engine, reimplemented from scratch
to understand how PyTorch's `autograd` works under the hood.

`zgrad` builds a dynamic computation graph over scalar values, then backpropagates
gradients through it via the chain rule — the same core idea that powers every
deep learning framework. Inspired by Andrej Karpathy's
[micrograd](https://github.com/karpathy/micrograd).

## Why

Reimplementing autograd is the fastest way to internalize that backpropagation is
not magic — it's just the chain rule applied automatically over a graph of
operations. This repo is my from-scratch build as part of studying deep learning
foundations (Neural Networks: Zero to Hero).

## Example

```python
from zgrad import Value

a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)
d = a * b + c        # forward pass builds the graph
d.backward()         # reverse pass fills in gradients

print(a.grad)        # d(d)/d(a) = b = -3.0
print(b.grad)        # d(d)/d(b) = a =  2.0
```

## Features

- [x] Scalar `Value` with `+`, `*`
- [ ] `pow`, `-`, `/`, `exp`
- [ ] Activations: `tanh`, `relu`
- [ ] Topological-sort `backward()`
- [ ] `nn` module: `Neuron`, `Layer`, `MLP`
- [ ] Train a small MLP on a toy dataset

## Layout

```
zgrad/
├── zgrad/
│   ├── __init__.py
│   ├── engine.py     # core autograd: the Value class
│   └── nn.py         # neural net building blocks (Neuron / Layer / MLP)
├── tests/
│   └── test_engine.py
├── examples/
│   └── demo.py
└── README.md
```

## Install / Run

Run from the repo root:

```bash
PYTHONPATH=. python3 examples/demo.py
PYTHONPATH=. python3 -m pytest    # run tests
```

## License

MIT © 2026 Zhankai Zhang
