"""Core autograd engine: a scalar Value that tracks its computation graph.

This is the heart of picograd. A `Value` wraps a single number and remembers the
operation that produced it, so gradients can flow backward through the graph via
the chain rule.

Start with the minimal working core below (+ and *), then implement the TODOs to
match micrograd's full feature set.
"""


class Value:
    """A scalar value node in the autograd computation graph."""

    def __init__(self, data, _children=(), _op=""):
        self.data = data
        self.grad = 0.0
        # internal: how to propagate the gradient into this node's inputs
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op  # the op that produced this node, for debugging/graphviz

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            # d(out)/d(self) = 1, d(out)/d(other) = 1  -> accumulate
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            # d(out)/d(self) = other, d(out)/d(other) = self
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    # TODO: implement __pow__ (self ** k) with correct local gradient
    # TODO: implement exp() and tanh() / relu() activations
    # TODO: implement __neg__, __sub__, __truediv__, and the __r*__ reflected ops

    def backward(self):
        """Backpropagate gradients from this node through the whole graph."""
        # TODO:
        #   1. Build a topological ordering of all nodes behind `self`.
        #   2. Set self.grad = 1.0 (seed the output gradient).
        #   3. Call node._backward() for each node in reverse topo order.
        raise NotImplementedError("implement topological-sort backward()")

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
