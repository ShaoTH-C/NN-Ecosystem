# creature.py - the autonomous agents that live in the ecosystem
# each one has sensors, a neural network brain, an energy budget,
# a lifespan with aging, and maturity growth from child to adult

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
    An agent in the world. It senses nearby objects with raycasts + directional
    targeting, feeds that into its NN brain, and decides how to move, eat,
    attack, and breed. Has a finite lifespan with aging effects.
    """

    _id_counter = 0

    def __init__(
        self,
        x: float,
        y: float,
        species: Species,
        genome: Optional[Genome] = None,
        parent_id: Optional[int] = None,
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
        if species == Species.HERBIVORE:
            self.base_max_speed = cfg.HERBIVORE_MAX_SPEED
            self.base_turn_rate = cfg.HERBIVORE_TURN_RATE
        else:
            self.base_max_speed = cfg.CARNIVORE_MAX_SPEED
            self.base_turn_rate = cfg.CARNIVORE_TURN_RATE

        # effective stats (modified by maturity and aging)
        self.max_speed = self.base_max_speed
        self.turn_rate = self.base_turn_rate

        # brain
        self.genome = genome if genome is not None else Genome()
        self.brain: NeuralNetwork = self.genome.build_network()

        # energy / life
        self.energy = cfg.INITIAL_ENERGY
        self.alive = True
        self.age = 0
        self.lifespan = np.random.randint(cfg.MIN_LIFESPAN, cfg.MAX_LIFESPAN + 1)

        # maturity: 0 = newborn, 1 = fully grown
        self.maturity = 0.0
        self.parent_id: Optional[int] = parent_id
        self.children_count = 0

        # lifetime stats
        self.food_eaten = 0
        self.kills = 0
        self.total_energy_gained = 0.0
        self.distance_traveled = 0.0

        # circle detection — exponential moving average of signed turn values
        self.turn_accumulator = 0.0

        # sensor data gets written every tick
        self.sensor_readings: np.ndarray = np.zeros(cfg.NUM_SENSOR_RAYS * 3)
        self.sensor_endpoints: List[Tuple[float, float]] = []
        self.sensor_hits: List[Optional[Tuple[float, float]]] = []

        # nearest-target directional info (set during cast_sensors)
        self.nearest_food_rel: Tuple[float, float, float] = (0.0, 0.0, 1.0)
        self.nearest_same_rel: Tuple[float, float, float] = (0.0, 0.0, 1.0)
        self.nearest_threat_rel: Tuple[float, float, float] = (0.0, 0.0, 1.0)
        self.parent_direction: Tuple[float, float] = (0.0, 0.0)

        # what the brain decided this tick
        self.wants_to_eat = False
        self.wants_to_attack = False
        self.wants_to_breed = False

    # --- aging / maturity ---

    def _compute_age_factors(self):
        """Compute speed and turn-rate multipliers based on maturity and aging."""
        age_frac = self.age / max(self.lifespan, 1)

        # maturity ramps from 0 to 1 over MATURATION_TICKS
        self.maturity = min(1.0, self.age / cfg.MATURATION_TICKS)

        # young creatures are slower and turn slower
        mat_speed = cfg.CHILD_SPEED_SCALE + (1.0 - cfg.CHILD_SPEED_SCALE) * self.maturity
        mat_turn = 0.7 + 0.3 * self.maturity

        # old-age decline
        if age_frac > cfg.AGING_START_FRACTION:
            aging_progress = (age_frac - cfg.AGING_START_FRACTION) / (1.0 - cfg.AGING_START_FRACTION)
            aging_speed = 1.0 - aging_progress * (1.0 - cfg.OLD_AGE_SPEED_MULT)
        else:
            aging_speed = 1.0

        self.max_speed = self.base_max_speed * mat_speed * aging_speed
        self.turn_rate = self.base_turn_rate * mat_turn * max(aging_speed, 0.6)

    def _get_metabolism_multiplier(self) -> float:
        """How much faster this creature burns energy based on age."""
        age_frac = self.age / max(self.lifespan, 1)
        if age_frac > cfg.AGING_START_FRACTION:
            aging_progress = (age_frac - cfg.AGING_START_FRACTION) / (1.0 - cfg.AGING_START_FRACTION)
            return 1.0 + aging_progress * (cfg.OLD_AGE_METABOLISM_MULT - 1.0)
        return 1.0

    def get_energy_value(self) -> float:
        """How much energy a predator gets from killing this creature."""
        mat_mult = cfg.CHILD_ENERGY_VALUE_MULT + (1.0 - cfg.CHILD_ENERGY_VALUE_MULT) * self.maturity
        return (cfg.KILL_BASE_ENERGY + self.energy * cfg.KILL_ENERGY_FRACTION) * mat_mult

    # --- sensor system ---

    def cast_sensors(
        self,
        food_positions: np.ndarray,
        herbivore_positions: np.ndarray,
        carnivore_positions: np.ndarray,
        store_viz: bool = True,
    ):
        """
        Cast rays outward and compute nearest-target directional info.
        Each ray returns 3 values: distance to nearest food, herbivore,
        carnivore (normalized 0-1, 1 = nothing).
        Also computes direction + distance to nearest food, same-species,
        and threat for the directional NN inputs.
        """
        num_rays = cfg.NUM_SENSOR_RAYS
        sensor_range = cfg.SENSOR_RANGE
        readings = np.ones(num_rays * 3)

        pos = np.array([self.x, self.y])

        # compute all ray directions at once
        angles = self.angle + (2.0 * np.pi * np.arange(num_rays) / num_rays)
        ray_dirs = np.column_stack((np.cos(angles), np.sin(angles)))

        # only save endpoints/hits when the renderer actually needs them
        if store_viz:
            endpoints = pos + ray_dirs * sensor_range
            self.sensor_endpoints = list(map(tuple, endpoints))
            self.sensor_hits = [None] * num_rays

        detection_width = cfg.CREATURE_RADIUS + 5

        # check each target group against all rays at once
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

            # only count targets in front, in range, and close to the ray
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
            if store_viz:
                for i in np.where(found)[0]:
                    d = min_dists[i]
                    hit_pos = (pos[0] + ray_dirs[i, 0] * d, pos[1] + ray_dirs[i, 1] * d)
                    if self.sensor_hits[i] is None:
                        self.sensor_hits[i] = hit_pos
                    else:
                        old_d = np.sqrt(
                            (self.sensor_hits[i][0] - pos[0]) ** 2
                            + (self.sensor_hits[i][1] - pos[1]) ** 2
                        )
                        if d < old_d:
                            self.sensor_hits[i] = hit_pos

        self.sensor_readings = readings

        # directional nearest-target info
        self.nearest_food_rel = self._nearest_target_rel(food_positions, pos, sensor_range)

        if self.species == Species.HERBIVORE:
            self.nearest_same_rel = self._nearest_target_rel(herbivore_positions, pos, sensor_range)
            self.nearest_threat_rel = self._nearest_target_rel(carnivore_positions, pos, sensor_range)
        else:
            self.nearest_same_rel = self._nearest_target_rel(carnivore_positions, pos, sensor_range)
            # for carnivores, "threat" slot = nearest prey (herbivore) they should chase
            self.nearest_threat_rel = self._nearest_target_rel(herbivore_positions, pos, sensor_range)

    def _nearest_target_rel(self, positions: np.ndarray, my_pos: np.ndarray, max_range: float):
        """Find nearest target and return (sin_rel_angle, cos_rel_angle, normalized_dist)."""
        if len(positions) == 0:
            return (0.0, 0.0, 1.0)

        # toroidal distance vectors
        diff = positions - my_pos
        diff[:, 0] = np.where(
            np.abs(diff[:, 0]) > cfg.WORLD_WIDTH / 2,
            diff[:, 0] - np.sign(diff[:, 0]) * cfg.WORLD_WIDTH,
            diff[:, 0],
        )
        diff[:, 1] = np.where(
            np.abs(diff[:, 1]) > cfg.WORLD_HEIGHT / 2,
            diff[:, 1] - np.sign(diff[:, 1]) * cfg.WORLD_HEIGHT,
            diff[:, 1],
        )

        dists = np.sqrt(diff[:, 0] ** 2 + diff[:, 1] ** 2)

        # exclude self (distance ~0) and anything impossibly close
        valid_mask = dists > 0.5
        if not np.any(valid_mask):
            return (0.0, 0.0, 1.0)

        dists_valid = np.where(valid_mask, dists, np.inf)
        nearest_idx = int(np.argmin(dists_valid))
        nearest_dist = dists_valid[nearest_idx]

        if nearest_dist > max_range * 2:
            return (0.0, 0.0, 1.0)

        # relative angle (how far left/right the target is from our heading)
        dx, dy = diff[nearest_idx]
        abs_angle = np.arctan2(dy, dx)
        rel_angle = abs_angle - self.angle

        return (
            float(np.sin(rel_angle)),
            float(np.cos(rel_angle)),
            float(min(nearest_dist / max_range, 1.0)),
        )

    # --- decision making (split for batched processing) ---

    def build_nn_inputs(self) -> np.ndarray:
        """Assemble the NN input vector from sensor data and internal state."""
        inputs = np.zeros(cfg.NN_INPUT_SIZE)

        # ray sensors [0:24]
        inputs[: len(self.sensor_readings)] = self.sensor_readings

        # self info [24:28]
        inputs[24] = self.energy / cfg.MAX_ENERGY
        inputs[25] = self.speed / max(self.max_speed, 0.01)
        inputs[26] = self.age / max(self.lifespan, 1)
        inputs[27] = self.maturity

        # nearest food direction + distance [28:31]
        inputs[28] = self.nearest_food_rel[0]
        inputs[29] = self.nearest_food_rel[1]
        inputs[30] = self.nearest_food_rel[2]

        # nearest same-species [31:34]
        inputs[31] = self.nearest_same_rel[0]
        inputs[32] = self.nearest_same_rel[1]
        inputs[33] = self.nearest_same_rel[2]

        # nearest threat/prey [34:37]
        inputs[34] = self.nearest_threat_rel[0]
        inputs[35] = self.nearest_threat_rel[1]
        inputs[36] = self.nearest_threat_rel[2]

        # parent direction [37:39] (fades with maturity)
        inputs[37] = self.parent_direction[0]
        inputs[38] = self.parent_direction[1]

        # bias [39]
        inputs[39] = 1.0

        return inputs

    def apply_nn_outputs(self, outputs: np.ndarray):
        """Interpret the NN output vector into actions."""
        # steer
        turn = float(outputs[0]) * self.turn_rate
        self.angle += turn
        self.angle %= 2 * np.pi

        # track sustained turning for circle detection
        self.turn_accumulator = self.turn_accumulator * 0.95 + turn

        # throttle (map [-1,1] to [0, max_speed])
        raw_speed = (float(outputs[1]) + 1.0) / 2.0 * self.max_speed
        self.speed = max(0.0, raw_speed)

        self.wants_to_eat = float(outputs[2]) > 0.0
        self.wants_to_attack = float(outputs[3]) > 0.0
        self.wants_to_breed = float(outputs[4]) > 0.0

    def think(self):
        """Convenience: build inputs, run NN, apply outputs (used as fallback)."""
        inputs = self.build_nn_inputs()
        outputs = self.brain.forward(inputs)
        self.apply_nn_outputs(outputs)

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
        # base metabolism: starve even when idle (like real life)
        cost = cfg.BASE_METABOLISM

        # small extra cost for movement (proportional to speed)
        if self.speed > 0.5 and self.max_speed > 0:
            cost += cfg.MOVE_ENERGY_EXTRA * (self.speed / self.max_speed)

        # penalize sustained circling (but less than before)
        if abs(self.turn_accumulator) > 2.0:
            cost += cfg.BASE_METABOLISM * 0.3

        # carnivores burn more (higher metabolism)
        if self.species == Species.CARNIVORE:
            cost *= cfg.CARNIVORE_METABOLISM_MULT

        # aging increases metabolism
        cost *= self._get_metabolism_multiplier()

        # children have lower metabolism (smaller body)
        if self.maturity < 1.0:
            cost *= (0.5 + 0.5 * self.maturity)

        self.energy -= cost

        if self.energy <= 0:
            self.alive = False
            self.energy = 0

    def gain_energy(self, amount: float):
        self.energy = min(self.energy + amount, cfg.MAX_ENERGY)
        self.total_energy_gained += amount

    # --- reproduction ---

    def can_breed(self) -> bool:
        """Check if this creature is eligible to breed this tick."""
        if not self.alive:
            return False
        if self.energy < cfg.BREEDING_ENERGY_THRESHOLD:
            return False

        # age range check
        min_age = int(self.lifespan * cfg.BREEDING_AGE_MIN_FRAC)
        max_age = int(self.lifespan * cfg.BREEDING_AGE_MAX_FRAC)
        if self.age < min_age or self.age > max_age:
            return False

        # NN decision OR instinct when energy is very high
        if self.wants_to_breed:
            return True
        if self.energy > cfg.MAX_ENERGY * 0.8:
            return True  # instinct: breed when well-fed

        return False

    def is_compatible_mate(self, other: "Creature") -> bool:
        """Check if another creature is a valid mating partner."""
        if other is self or not other.alive:
            return False
        if other.species != self.species:
            return False
        if other.energy < cfg.BREEDING_ENERGY_THRESHOLD * 0.7:
            return False

        # age compatibility
        if abs(self.age - other.age) > cfg.MATE_AGE_TOLERANCE:
            return False

        # mate must also be in breeding age range
        min_age = int(other.lifespan * cfg.BREEDING_AGE_MIN_FRAC)
        max_age = int(other.lifespan * cfg.BREEDING_AGE_MAX_FRAC)
        if other.age < min_age or other.age > max_age:
            return False

        return True

    def reproduce_with(self, partner: "Creature") -> "Creature":
        """Breed with a partner using crossover + low-mutation knowledge transfer."""
        from evolution.selection import crossover

        # crossover parent genomes then apply teaching-level mutation
        if partner.genome.layer_sizes == self.genome.layer_sizes:
            child_genome = crossover(self.genome, partner.genome)
        else:
            better = self if self.genome.fitness >= partner.genome.fitness else partner
            child_genome = better.genome.copy()

        child_genome = child_genome.mutate_child()

        # spawn near midpoint of parents
        mid_x = (self.x + partner.x) / 2
        mid_y = (self.y + partner.y) / 2
        offset_angle = np.random.uniform(0, 2 * np.pi)
        offset_dist = cfg.CREATURE_RADIUS * 3
        child_x = (mid_x + np.cos(offset_angle) * offset_dist) % cfg.WORLD_WIDTH
        child_y = (mid_y + np.sin(offset_angle) * offset_dist) % cfg.WORLD_HEIGHT

        child = Creature(
            x=child_x,
            y=child_y,
            species=self.species,
            genome=child_genome,
            parent_id=self.id,
        )

        # energy costs split between parents
        self.energy -= cfg.BREEDING_ENERGY_COST * 0.6
        partner.energy -= cfg.BREEDING_ENERGY_COST * 0.4
        self.children_count += 1
        partner.children_count += 1

        # child starts with limited energy
        child.energy = cfg.BREEDING_ENERGY_COST * cfg.CHILD_INITIAL_ENERGY_FRAC

        return child

    # --- per-tick bookkeeping ---

    def update(self):
        self.age += 1
        self._compute_age_factors()

        # die of old age
        if self.age >= self.lifespan:
            self.alive = False
            return

        # fitness is a combo of survival, energy gathered, offspring, and kills
        self.genome.fitness = (
            self.age * 0.1
            + self.total_energy_gained
            + self.children_count * 30.0
            + self.kills * 20.0
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
            f"energy={self.energy:.1f}, age={self.age}/{self.lifespan}, "
            f"mat={self.maturity:.0%}, gen={self.genome.generation})"
        )
