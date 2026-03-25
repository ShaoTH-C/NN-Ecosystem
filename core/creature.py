# creature.py - the autonomous agents that live in the ecosystem
# each one has sensors, a neural network brain, and an energy budget

import numpy as np
from typing import List, Tuple, Optional
from enum import Enum

from core.neural_network import NeuralNetwork
from evolution.genome import Genome
import config as cfg


class Species(Enum):
    HERBIVORE = 0
    CARNIVORE = 1


class Creature:
    """
    An agent in the world. It senses nearby objects with raycasts,
    feeds that into its NN brain, and decides whether to move, eat,
    or attack each tick. Energy runs out -> it dies. Enough energy -> it reproduces.
    """

    _id_counter = 0

    def __init__(
        self,
        x: float,
        y: float,
        species: Species,
        genome: Optional[Genome] = None,
    ):
        Creature._id_counter += 1
        self.id = Creature._id_counter

        # position and movement
        self.x = x
        self.y = y
        self.angle = np.random.uniform(0, 2 * np.pi)
        self.speed = 0.0

        # what kind of creature this is
        self.species = species
        self.color = (
            cfg.HERBIVORE_COLOR if species == Species.HERBIVORE
            else cfg.CARNIVORE_COLOR
        )
        # carnivores are faster but turn wider, herbivores are agile but slower
        if species == Species.HERBIVORE:
            self.max_speed = cfg.HERBIVORE_MAX_SPEED
            self.turn_rate = cfg.HERBIVORE_TURN_RATE
        else:
            self.max_speed = cfg.CARNIVORE_MAX_SPEED
            self.turn_rate = cfg.CARNIVORE_TURN_RATE

        # brain
        self.genome = genome if genome is not None else Genome()
        self.brain: NeuralNetwork = self.genome.build_network()

        # energy / life
        self.energy = cfg.INITIAL_ENERGY
        self.alive = True
        self.age = 0
        self.reproduce_cooldown = 0
        self.children_count = 0

        # lifetime stats
        self.food_eaten = 0
        self.kills = 0
        self.total_energy_gained = 0.0
        self.distance_traveled = 0.0

        # sensor data gets written every tick
        self.sensor_readings: np.ndarray = np.zeros(cfg.NUM_SENSOR_RAYS * 3)
        self.sensor_endpoints: List[Tuple[float, float]] = []
        self.sensor_hits: List[Optional[Tuple[float, float]]] = []

        # what the brain decided this tick
        self.wants_to_eat = False
        self.wants_to_attack = False

    # --- sensor system ---

    def cast_sensors(
        self,
        food_positions: np.ndarray,
        herbivore_positions: np.ndarray,
        carnivore_positions: np.ndarray,
    ):
        """
        Cast rays outward and see what's nearby. Each ray returns 3 values:
        distance to nearest food, herbivore, and carnivore (normalized 0-1, 1 = nothing).
        The whole thing is vectorized so it's reasonably fast even with lots of targets.
        """
        num_rays = cfg.NUM_SENSOR_RAYS
        sensor_range = cfg.SENSOR_RANGE
        readings = np.ones(num_rays * 3)

        pos = np.array([self.x, self.y])

        # compute all ray directions at once
        angles = self.angle + (2.0 * np.pi * np.arange(num_rays) / num_rays)
        ray_dirs = np.column_stack((np.cos(angles), np.sin(angles)))

        # save endpoints for the renderer to draw
        endpoints = pos + ray_dirs * sensor_range
        self.sensor_endpoints = list(map(tuple, endpoints))
        self.sensor_hits = [None] * num_rays

        detection_width = cfg.CREATURE_RADIUS + 5

        # check each target group (food, herbs, carns) against all rays at once
        for channel, targets in enumerate(
            [food_positions, herbivore_positions, carnivore_positions]
        ):
            if len(targets) == 0:
                continue

            diff = targets - pos

            # project each target onto each ray direction
            proj_all = (diff @ ray_dirs.T).T  # shape: (num_rays, num_targets)

            # perpendicular distance squared
            dist_sq = np.sum(diff ** 2, axis=1)
            perp_sq = dist_sq[np.newaxis, :] - proj_all ** 2
            perp_sq = np.maximum(perp_sq, 0.0)

            # only count targets that are in front, in range, and close to the ray
            valid = (
                (proj_all > 0)
                & (proj_all < sensor_range)
                & (perp_sq < detection_width ** 2)
            )

            # find the closest valid target per ray
            masked_proj = np.where(valid, proj_all, np.inf)
            min_dists = np.min(masked_proj, axis=1)

            found = min_dists < sensor_range
            readings[np.where(found)[0] * 3 + channel] = min_dists[found] / sensor_range

            # save hit positions for visualization
            for i in np.where(found)[0]:
                d = min_dists[i]
                hit_pos = (pos[0] + ray_dirs[i, 0] * d, pos[1] + ray_dirs[i, 1] * d)
                if self.sensor_hits[i] is None:
                    self.sensor_hits[i] = hit_pos
                else:
                    # keep whichever hit is closer
                    old_d = np.sqrt(
                        (self.sensor_hits[i][0] - pos[0]) ** 2
                        + (self.sensor_hits[i][1] - pos[1]) ** 2
                    )
                    if d < old_d:
                        self.sensor_hits[i] = hit_pos

        self.sensor_readings = readings

    # --- decision making ---

    def think(self):
        """
        Feed sensor data into the NN and interpret the outputs.
        Outputs: [turn, speed, eat, attack] all in roughly [-1, 1].
        """
        inputs = np.zeros(cfg.NN_INPUT_SIZE)
        inputs[: len(self.sensor_readings)] = self.sensor_readings
        inputs[-3] = self.energy / cfg.MAX_ENERGY
        inputs[-2] = self.speed / self.max_speed
        inputs[-1] = 1.0  # bias

        outputs = self.brain.forward(inputs)

        # steer
        turn = outputs[0] * self.turn_rate
        self.angle += turn
        self.angle %= 2 * np.pi

        # throttle (map [-1,1] to [0, max_speed])
        self.speed = (outputs[1] + 1.0) / 2.0 * self.max_speed

        self.wants_to_eat = outputs[2] > 0.0
        self.wants_to_attack = outputs[3] > 0.0

    # --- movement ---

    def move(self):
        dx = np.cos(self.angle) * self.speed
        dy = np.sin(self.angle) * self.speed
        self.x += dx
        self.y += dy

        # wrap around (toroidal world)
        self.x %= cfg.WORLD_WIDTH
        self.y %= cfg.WORLD_HEIGHT

        self.distance_traveled += self.speed

    # --- energy ---

    def consume_energy(self):
        if self.speed > 0.5:
            cost = cfg.MOVE_ENERGY_COST * (self.speed / self.max_speed)
        else:
            cost = cfg.IDLE_ENERGY_COST

        # carnivores burn a bit more just existing (higher metabolism)
        if self.species == Species.CARNIVORE:
            cost *= 1.2

        self.energy -= cost

        if self.energy <= 0:
            self.alive = False
            self.energy = 0

    def gain_energy(self, amount: float):
        self.energy = min(self.energy + amount, cfg.MAX_ENERGY)
        self.total_energy_gained += amount

    # --- reproduction ---

    def can_reproduce(self) -> bool:
        return (
            self.alive
            and self.energy >= cfg.REPRODUCE_ENERGY_THRESHOLD
            and self.reproduce_cooldown <= 0
        )

    def reproduce(self, partner: Optional["Creature"] = None) -> "Creature":
        """
        Make a baby. If there's a compatible partner nearby, do crossover first.
        Otherwise just clone + mutate.
        """
        if partner is not None and partner.genome.layer_sizes == self.genome.layer_sizes:
            from evolution.selection import crossover
            child_genome = crossover(self.genome, partner.genome)
            child_genome = child_genome.mutate()
        else:
            child_genome = self.genome.mutate()

        # spawn the kid nearby
        offset_angle = np.random.uniform(0, 2 * np.pi)
        offset_dist = cfg.CREATURE_RADIUS * 3
        child_x = (self.x + np.cos(offset_angle) * offset_dist) % cfg.WORLD_WIDTH
        child_y = (self.y + np.sin(offset_angle) * offset_dist) % cfg.WORLD_HEIGHT

        child = Creature(
            x=child_x,
            y=child_y,
            species=self.species,
            genome=child_genome,
        )

        self.energy -= cfg.REPRODUCE_ENERGY_COST
        self.reproduce_cooldown = cfg.REPRODUCE_COOLDOWN
        self.children_count += 1

        child.energy = cfg.REPRODUCE_ENERGY_COST * 0.7

        return child

    # --- per-tick bookkeeping ---

    def update(self):
        self.age += 1
        if self.reproduce_cooldown > 0:
            self.reproduce_cooldown -= 1

        # fitness is a combo of survival, energy gathered, and offspring produced
        self.genome.fitness = (
            self.age * 0.1
            + self.total_energy_gained
            + self.children_count * 20.0
        )
        self.genome.age = self.age

    # --- utils ---

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y])

    def distance_to(self, other) -> float:
        """Toroidal distance (wraps around edges)."""
        dx = abs(self.x - other.x)
        dy = abs(self.y - other.y)
        dx = min(dx, cfg.WORLD_WIDTH - dx)
        dy = min(dy, cfg.WORLD_HEIGHT - dy)
        return np.sqrt(dx * dx + dy * dy)

    def __repr__(self):
        return (
            f"Creature(id={self.id}, {self.species.name}, "
            f"energy={self.energy:.1f}, age={self.age}, "
            f"gen={self.genome.generation})"
        )
