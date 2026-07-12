"""Core autograd engine: a scalar Value that tracks its computation graph.

This is the heart of picograd. A `Value` wraps a single number and remembers the
operation that produced it, so gradients can flow backward through the graph via
the chain rule.

Start with the minimal working core below (+ and *), then implement the TODOs to
match micrograd's full feature set.
"""
import math 

class Value:
    """A scalar value node in the autograd computation graph."""
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._prev = set(_children)
        self._backward = lambda: None
        self._op = _op
        self.label = label 
        
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        
        return out 
        
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward    
        
        return out
    
    def __pow__(self, other):
        assert isinstance(other, (float, int)), "Only supporting int/float powers for now"
        out = Value(self.data**other, (self, ), f"**{other}")
        
        def _backward():
            self.grad += other * self.data**(other - 1) * out.grad
        out._backward = _backward
        
        return out 
    
    def exp(self):
        out = Value(math.exp(self.data), (self, ), 'exp')
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out 
        
    def tanh(self):
        n = self.data
        t = (math.exp(2*n) - 1) / (math.exp(2*n) + 1)
        out = Value(t, (self, ), 'tanh')
        
        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward    
                
        return out 
    
    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self, ), 'ReLU')
        
        def _backward():
            self.grad += (out.data>0) * out.grad
        out._backward = _backward
        
        return out 
    
    def backward(self):
        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for children in v._prev:
                    build_topo(children)
                topo.append(v)
        
        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
    
    def __neg__(self):
        return self * -1.0
    
    def __sub__(self, other):
        return self + (-other)
        
    def __radd__(self, other): # other + self
        return self + other 
    
    def __rmul__(self, other):
        return self * other 

    def __rsub__(self, other): # other - self
        return other + (-self)
    
    def __truediv__(self, other): # self / other
        return self * other**-1
    
    def __rtruediv__(self, other): # other / self
        return other * self**-1      
            
    def __repr__(self):
        return f"Value(data={self.data})"