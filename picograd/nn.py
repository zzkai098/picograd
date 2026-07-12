"""Neural network building blocks on top of the autograd engine.

  Mirrors micrograd's nn module: a Neuron is a weighted sum + activation,
  a Layer is a list of Neurons, and an MLP is a stack of Layers.
"""
import random
from .engine import Value


class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0

class Neuron(Module):
    def __init__(self, nin, activation='relu'):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0.0)
        assert activation in ('relu', 'tanh', None), "activation must be 'relu', 'tanh', or None"
        self.activation = activation
        
    def __call__(self, x):
        #w * x + b
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        
        if self.activation == 'relu':
            return act.relu()
        elif self.activation == 'tanh':
            return act.tanh()
        else:
            return act
        
    def parameters(self):
        return self.w + [self.b]
    
    def __repr__(self):
        act = self.activation or 'linear'
        return f"Neuron(in={len(self.w)}, act={act})"

class Layer(Module):
    def __init__(self, nin, nout, **kwargs):
        self.nin = nin
        self.nout = nout
        self.activation = kwargs.get('activation', 'relu')
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]
    
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]        
        return outs[0] if len(outs) == 1 else outs 
    
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]
    
    def __repr__(self):
        return f"Layer(in={self.nin}, out={self.nout}, act={self.activation})"

class MLP(Module):
    def __init__(self, nin, nouts:list, **kwargs):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], **kwargs) for i in range(len(nouts))]
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x 
    
    def parameters(self):
        out = [p for layer in self.layers for p in layer.parameters()]
        return out 
    
    def __repr__(self):
        rows = "\n".join(f"  ({i}): {layer}" for i, layer in enumerate(self.layers))
        return f"MLP(\n{rows}\n)"
