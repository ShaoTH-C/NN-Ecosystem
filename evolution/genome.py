# genome.py - encodes a neural network as a flat weight vector
# handles mutation, topology changes, and building the actual NN from genes

import numpy as np
from typing import List, Optional
from core.neural_network import NeuralNetwork
import config as cfg


class Genome:
    """
    Stores the DNA of a creature's brain. The genes array is just all the
    weights and biases flattened into one vector. Mutation tweaks those values,
    and occasionally adds or removes a neuron from a hidden layer.
    """

    def __init__(
        self,
        layer_sizes: Optional[List[int]] = None,
        genes: Optional[np.ndarray] = None,
        generation: int = 0,
    ):
        self.layer_sizes = layer_sizes or cfg.get_nn_architecture()
        self.generation = generation
        self.fitness = 0.0
        self.age = 0

        if genes is not None:
            self.genes = genes.copy()
        else:
            self.genes = self._xavier_init()

    def _xavier_init(self) -> np.ndarray:
        """Xavier init - gives reasonable starting weights so things aren't totally random."""
        parts = []
        for i in range(len(self.layer_sizes) - 1):
            fan_in = self.layer_sizes[i]
            fan_out = self.layer_sizes[i + 1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            w = np.random.randn(fan_in * fan_out) * std
            b = np.zeros(fan_out)
            parts.extend([w, b])
        return np.concatenate(parts)

    def build_network(self) -> NeuralNetwork:
        """Create the actual NN object from these genes."""
        nn = NeuralNetwork(
            self.layer_sizes,
            activation=cfg.NN_ACTIVATION,
            output_activation="tanh",
        )
        nn.set_flat_weights(self.genes.copy())
        return nn

    def mutate(self) -> "Genome":
        """
        Returns a mutated copy. Most of the time it just perturbs weights with
        gaussian noise. Occasionally it'll reset a weight entirely, and very
        rarely it'll add or remove a hidden neuron.
        """
        new_genes = self.genes.copy()
        new_layer_sizes = list(self.layer_sizes)

        # perturb a fraction of the weights
        mask = np.random.rand(len(new_genes)) < cfg.MUTATION_RATE
        perturbation = np.random.randn(len(new_genes)) * cfg.MUTATION_STRENGTH
        new_genes[mask] += perturbation[mask]

        # sometimes just slam a weight to a new random value
        if np.random.rand() < 0.05:
            idx = np.random.randint(len(new_genes))
            new_genes[idx] = np.random.randn() * cfg.WEIGHT_INIT_STD

        # rarely, mess with the topology
        if np.random.rand() < cfg.TOPOLOGY_MUTATION_RATE and len(new_layer_sizes) > 2:
            new_layer_sizes, new_genes = self._topology_mutate(
                new_layer_sizes, new_genes
            )

        child = Genome(
            layer_sizes=new_layer_sizes,
            genes=new_genes,
            generation=self.generation + 1,
        )
        return child

    def _topology_mutate(self, layer_sizes, genes):
        """Add or remove a neuron from a random hidden layer."""
        hidden_indices = list(range(1, len(layer_sizes) - 1))
        if not hidden_indices:
            return layer_sizes, genes

        layer_idx = np.random.choice(hidden_indices)

        if np.random.rand() < 0.6 and layer_sizes[layer_idx] < 32:
            # add a neuron
            new_sizes = list(layer_sizes)
            new_sizes[layer_idx] += 1
            new_genome = Genome(layer_sizes=new_sizes)
            new_genes = new_genome.genes
            min_len = min(len(genes), len(new_genes))
            new_genes[:min_len] = genes[:min_len]
            return new_sizes, new_genes
        elif layer_sizes[layer_idx] > 4:
            # remove a neuron (don't let layers get too small)
            new_sizes = list(layer_sizes)
            new_sizes[layer_idx] -= 1
            new_genome = Genome(layer_sizes=new_sizes)
            new_genes = new_genome.genes
            min_len = min(len(genes), len(new_genes))
            new_genes[:min_len] = genes[:min_len]
            return new_sizes, new_genes

        return layer_sizes, genes

    def copy(self) -> "Genome":
        return Genome(
            layer_sizes=list(self.layer_sizes),
            genes=self.genes.copy(),
            generation=self.generation,
        )

    def distance(self, other: "Genome") -> float:
        """How genetically different two genomes are. Used for speciation."""
        if self.layer_sizes != other.layer_sizes:
            return float("inf")
        return float(np.sqrt(np.mean((self.genes - other.genes) ** 2)))

    def __repr__(self):
        return (
            f"Genome(layers={self.layer_sizes}, "
            f"params={len(self.genes)}, "
            f"gen={self.generation}, "
            f"fitness={self.fitness:.1f})"
        )
