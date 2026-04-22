# neural_network.py - feed-forward neural net built from scratch with numpy
# weights get evolved instead of trained via backprop
# optional GPU acceleration via CuPy (CUDA)

import numpy as np
from typing import List, Callable

# GPU support - try cupy for CUDA acceleration
from core import _gpu_bootstrap  # registers pip-installed nvidia DLL dirs  # noqa: F401
_GPU_AVAILABLE = False
_cp = None
try:
    import cupy as _cp
    _GPU_AVAILABLE = True
except Exception:
    pass


def gpu_available() -> bool:
    return _GPU_AVAILABLE


def get_xp(use_gpu: bool = False):
    """Get the array module (cupy for GPU, numpy for CPU)."""
    if use_gpu and _GPU_AVAILABLE:
        return _cp
    return np


# activation functions

def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)

def leaky_relu(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0, x, 0.01 * x)

def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))

ACTIVATIONS = {
    "tanh": tanh,
    "relu": relu,
    "leaky_relu": leaky_relu,
    "sigmoid": sigmoid,
}


class NeuralNetwork:
    """
    Simple fully-connected feed-forward net.
    layer_sizes is something like [40, 48, 32, 5].
    Weights get set externally by the Genome class.
    """

    def __init__(
        self,
        layer_sizes: List[int],
        activation: str = "tanh",
        output_activation: str = "tanh",
    ):
        self.layer_sizes = list(layer_sizes)
        self.num_layers = len(layer_sizes)
        self.activation_fn: Callable = ACTIVATIONS[activation]
        self.output_activation_fn: Callable = ACTIVATIONS[output_activation]

        # init weight matrices and bias vectors (genome fills these in later)
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        for i in range(self.num_layers - 1):
            w = np.zeros((layer_sizes[i], layer_sizes[i + 1]))
            b = np.zeros(layer_sizes[i + 1])
            self.weights.append(w)
            self.biases.append(b)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Run input through all layers and return the output."""
        a = x.astype(np.float64)
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = a @ w + b
            if i < len(self.weights) - 1:
                a = self.activation_fn(z)
            else:
                # last layer uses output activation (tanh for bounded outputs)
                a = self.output_activation_fn(z)
        return a

    @staticmethod
    def batched_forward(inputs, weight_stacks, bias_stacks, xp=np):
        """Batched forward pass for N networks with the same architecture.

        Args:
            inputs: (N, input_size) array
            weight_stacks: list of (N, fan_in, fan_out) arrays, one per layer
            bias_stacks: list of (N, fan_out) arrays, one per layer
            xp: array module (numpy or cupy)

        Returns:
            (N, output_size) array of outputs
        """
        a = inputs
        for W, B in zip(weight_stacks, bias_stacks):
            z = xp.einsum('ni,nio->no', a, W) + B
            a = xp.tanh(z)
        return a

    def get_flat_weights(self) -> np.ndarray:
        """Flatten all weights + biases into a single 1D array."""
        parts = []
        for w, b in zip(self.weights, self.biases):
            parts.append(w.ravel())
            parts.append(b.ravel())
        return np.concatenate(parts)

    def set_flat_weights(self, flat: np.ndarray):
        """Load weights + biases from a flat array (inverse of get_flat_weights)."""
        offset = 0
        for i in range(len(self.weights)):
            w_shape = self.weights[i].shape
            w_size = self.weights[i].size
            self.weights[i] = flat[offset : offset + w_size].reshape(w_shape)
            offset += w_size

            b_size = self.biases[i].size
            self.biases[i] = flat[offset : offset + b_size].reshape(self.biases[i].shape)
            offset += b_size

    def total_params(self) -> int:
        return sum(w.size + b.size for w, b in zip(self.weights, self.biases))

    def copy(self) -> "NeuralNetwork":
        nn = NeuralNetwork(
            self.layer_sizes,
            activation=self._activation_name(),
            output_activation=self._output_activation_name(),
        )
        nn.set_flat_weights(self.get_flat_weights().copy())
        return nn

    def _activation_name(self) -> str:
        for name, fn in ACTIVATIONS.items():
            if fn is self.activation_fn:
                return name
        return "tanh"

    def _output_activation_name(self) -> str:
        for name, fn in ACTIVATIONS.items():
            if fn is self.output_activation_fn:
                return name
        return "tanh"

    def __repr__(self):
        return (
            f"NeuralNetwork(layers={self.layer_sizes}, "
            f"params={self.total_params()}, "
            f"act={self._activation_name()})"
        )
