# world.py - the main simulation loop
# handles spawning, sensing, batched NN thinking, movement,
# eating, attacking, mate-based reproduction, aging, and cleanup

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

from core.creature import Creature, Species
from core.food import Food
from core.neural_network import NeuralNetwork, get_xp
from evolution.genome import Genome
from evolution.selection import crossover
import config as cfg

# GPU support
try:
    import cupy as cp
    _CUPY_AVAILABLE = True
except ImportError:
    cp = None
    _CUPY_AVAILABLE = False


class World:
    """
    The 2D ecosystem. Each tick runs through:
    spawn food -> sense -> think (batched) -> move -> eat/attack -> breed -> cleanup
    """

    def __init__(self):
        self.tick = 0
        self.creatures: List[Creature] = []
        self.food_items: List[Food] = []

        # keep track of the best genomes we've seen (used for respawning)
        self.best_herbivore_genome: Optional[Genome] = None
        self.best_carnivore_genome: Optional[Genome] = None

        # running totals
        self.total_births = 0
        self.total_deaths = 0
        self.total_food_eaten = 0
        self.total_kills = 0
        self.max_generation_herb = 0
        self.max_generation_carn = 0

        # spatial grids for fast neighbor lookups
        self.grid_cell_size = cfg.SENSOR_RANGE
        self.grid: Dict[Tuple[int, int], List] = defaultdict(list)
        self.food_grid: Dict[Tuple[int, int], List] = defaultdict(list)

        # creature lookup by ID (for parent tracking)
        self.creature_lookup: Dict[int, Creature] = {}

        # set to False on non-render ticks to skip sensor visualization data
        self.store_sensor_viz = True

        self._spawn_initial_population()
        self._spawn_initial_food()

    def _spawn_initial_population(self):
        for _ in range(cfg.INITIAL_HERBIVORES):
            c = Creature(
                x=np.random.uniform(20, cfg.WORLD_WIDTH - 20),
                y=np.random.uniform(20, cfg.WORLD_HEIGHT - 20),
                species=Species.HERBIVORE,
            )
            self.creatures.append(c)

        for _ in range(cfg.INITIAL_CARNIVORES):
            c = Creature(
                x=np.random.uniform(20, cfg.WORLD_WIDTH - 20),
                y=np.random.uniform(20, cfg.WORLD_HEIGHT - 20),
                species=Species.CARNIVORE,
            )
            self.creatures.append(c)

    def _spawn_initial_food(self):
        for _ in range(cfg.INITIAL_FOOD):
            self.food_items.append(Food())

    # --- main loop ---

    def step(self):
        """One tick of the simulation."""
        self.tick += 1

        self._spawn_food()
        self._build_grid()
        self._build_creature_lookup()

        food_pos, herb_pos, carn_pos = self._gather_positions()

        # phase 1: sense (per-creature, vectorized rays + nearest-target)
        for creature in self.creatures:
            if not creature.alive:
                continue
            self._set_parent_direction(creature)
            creature.cast_sensors(
                food_pos, herb_pos, carn_pos,
                store_viz=self.store_sensor_viz,
            )

        # phase 2: think (batched NN forward pass for performance)
        self._batch_think()

        # phase 3: act
        for creature in self.creatures:
            if not creature.alive:
                continue
            creature.move()
            creature.consume_energy()
            if not creature.alive:
                continue
            creature.update()

        # phase 4: interactions
        self._handle_eating()
        self._handle_attacks()
        self._handle_reproduction()
        self._remove_dead()
        self._enforce_population_limits()
        self._track_best_genomes()

    # --- batched neural network forward pass ---

    def _batch_think(self):
        """Run all creatures' NNs in one batched operation per topology group.
        Uses GPU (cupy) if available and enabled, otherwise numpy."""
        living = [c for c in self.creatures if c.alive]
        if not living:
            return

        use_gpu = getattr(cfg, 'USE_GPU', False) and _CUPY_AVAILABLE
        xp = cp if use_gpu else np

        # group creatures by network topology for batching
        groups: Dict[tuple, List[Creature]] = defaultdict(list)
        for c in living:
            key = tuple(c.brain.layer_sizes)
            groups[key].append(c)

        for layer_sizes, creatures in groups.items():
            N = len(creatures)
            num_layers = len(layer_sizes) - 1
            input_size = layer_sizes[0]

            # build batched input matrix
            inputs = np.zeros((N, input_size))
            for i, c in enumerate(creatures):
                inputs[i] = c.build_nn_inputs()

            # stack weight matrices for each layer
            layer_W = []
            layer_B = []
            for l in range(num_layers):
                W = np.stack([c.brain.weights[l] for c in creatures])
                B = np.stack([c.brain.biases[l] for c in creatures])
                layer_W.append(W)
                layer_B.append(B)

            # transfer to GPU if available
            if use_gpu:
                inputs = xp.asarray(inputs)
                layer_W = [xp.asarray(w) for w in layer_W]
                layer_B = [xp.asarray(b) for b in layer_B]

            # batched forward pass
            outputs = NeuralNetwork.batched_forward(inputs, layer_W, layer_B, xp)

            # transfer back to CPU
            if use_gpu:
                outputs = xp.asnumpy(outputs)

            # distribute outputs to creatures
            for i, c in enumerate(creatures):
                c.apply_nn_outputs(outputs[i])

    # --- parent tracking ---

    def _build_creature_lookup(self):
        self.creature_lookup.clear()
        for c in self.creatures:
            if c.alive:
                self.creature_lookup[c.id] = c

    def _set_parent_direction(self, creature: Creature):
        """For immature creatures, compute direction toward parent."""
        if creature.maturity >= 1.0 or creature.parent_id is None:
            creature.parent_direction = (0.0, 0.0)
            return

        parent = self.creature_lookup.get(creature.parent_id)
        if parent is None or not parent.alive:
            creature.parent_direction = (0.0, 0.0)
            creature.parent_id = None
            return

        # toroidal direction to parent
        dx = parent.x - creature.x
        dy = parent.y - creature.y
        if abs(dx) > cfg.WORLD_WIDTH / 2:
            dx -= np.sign(dx) * cfg.WORLD_WIDTH
        if abs(dy) > cfg.WORLD_HEIGHT / 2:
            dy -= np.sign(dy) * cfg.WORLD_HEIGHT

        abs_angle = np.arctan2(dy, dx)
        rel_angle = abs_angle - creature.angle

        # strength fades as creature matures
        strength = 1.0 - creature.maturity
        creature.parent_direction = (
            float(np.sin(rel_angle) * strength),
            float(np.cos(rel_angle) * strength),
        )

    # --- food spawning ---

    def _spawn_food(self):
        alive_food = sum(1 for f in self.food_items if f.alive)
        if alive_food < cfg.MAX_FOOD:
            for _ in range(cfg.FOOD_SPAWN_RATE):
                if alive_food < cfg.MAX_FOOD:
                    self.food_items.append(Food())
                    alive_food += 1

    # --- spatial grid (for fast neighbor lookups) ---

    def _build_grid(self):
        self.grid.clear()
        self.food_grid.clear()
        for creature in self.creatures:
            if creature.alive:
                cell = self._get_cell(creature.x, creature.y)
                self.grid[cell].append(creature)
        for food in self.food_items:
            if food.alive:
                cell = self._get_cell(food.x, food.y)
                self.food_grid[cell].append(food)

    def _get_cell(self, x: float, y: float) -> Tuple[int, int]:
        return (int(x // self.grid_cell_size), int(y // self.grid_cell_size))

    def _get_neighbors(self, creature: Creature) -> List[Creature]:
        """All creatures in the same or adjacent grid cells."""
        cx, cy = self._get_cell(creature.x, creature.y)
        neighbors = []
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                cell = (cx + dx, cy + dy)
                neighbors.extend(self.grid.get(cell, []))
        return neighbors

    def _gather_positions(self):
        """Collect positions of food/herbs/carns into arrays for the sensor system."""
        food_list = []
        herb_list = []
        carn_list = []

        for f in self.food_items:
            if f.alive:
                food_list.append((f.x, f.y))

        for c in self.creatures:
            if not c.alive:
                continue
            if c.species == Species.HERBIVORE:
                herb_list.append((c.x, c.y))
            else:
                carn_list.append((c.x, c.y))

        food_pos = np.array(food_list, dtype=np.float64) if food_list else np.empty((0, 2))
        herb_pos = np.array(herb_list, dtype=np.float64) if herb_list else np.empty((0, 2))
        carn_pos = np.array(carn_list, dtype=np.float64) if carn_list else np.empty((0, 2))

        return food_pos, herb_pos, carn_pos

    # --- eating (herbivores eat food) ---

    def _get_nearby_food(self, creature: Creature) -> List[Food]:
        cx, cy = self._get_cell(creature.x, creature.y)
        nearby = []
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                nearby.extend(self.food_grid.get((cx + dx, cy + dy), []))
        return nearby

    def _handle_eating(self):
        for creature in self.creatures:
            if not creature.alive or creature.species != Species.HERBIVORE:
                continue

            for food in self._get_nearby_food(creature):
                if not food.alive:
                    continue
                dist = creature.distance_to(food)

                # instinct: auto-eat if very close, OR NN wants to eat and in range
                if dist < cfg.INSTINCT_EAT_RANGE or (creature.wants_to_eat and dist < cfg.EAT_RANGE):
                    energy = food.consume()
                    creature.gain_energy(energy)
                    creature.food_eaten += 1
                    self.total_food_eaten += 1
                    break  # one food per tick

    # --- attacking (carnivores hunt herbivores) ---

    def _handle_attacks(self):
        for creature in self.creatures:
            if not creature.alive or creature.species != Species.CARNIVORE:
                continue

            # find nearest herbivore in range
            best_target = None
            best_dist = cfg.ATTACK_RANGE

            for other in self._get_neighbors(creature):
                if (
                    other is creature
                    or not other.alive
                    or other.species != Species.HERBIVORE
                ):
                    continue
                dist = creature.distance_to(other)
                if dist < best_dist:
                    best_dist = dist
                    best_target = other

            if best_target is None:
                continue

            # instinct: auto-attack when very close, OR NN wants to attack
            do_attack = False
            if best_dist < cfg.INSTINCT_ATTACK_RANGE:
                do_attack = True  # reflex
            elif creature.wants_to_attack and best_dist < cfg.ATTACK_RANGE:
                do_attack = True

            if not do_attack:
                continue

            creature.energy -= cfg.ATTACK_ENERGY_COST

            # damage scales with speed (charging attack)
            speed_factor = 0.5 + 0.5 * (creature.speed / creature.max_speed if creature.max_speed > 0 else 0)
            damage = cfg.ATTACK_DAMAGE * speed_factor
            best_target.energy -= damage

            if best_target.energy <= 0:
                best_target.alive = False
                energy_gain = best_target.get_energy_value()
                creature.gain_energy(energy_gain)
                creature.kills += 1
                self.total_kills += 1

    # --- reproduction (mate-based) ---

    def _handle_reproduction(self):
        new_creatures = []

        herb_count = sum(
            1 for c in self.creatures
            if c.alive and c.species == Species.HERBIVORE
        )
        carn_count = sum(
            1 for c in self.creatures
            if c.alive and c.species == Species.CARNIVORE
        )

        # gather candidates and shuffle to avoid order bias
        breeding_candidates = [c for c in self.creatures if c.can_breed()]
        if breeding_candidates:
            np.random.shuffle(breeding_candidates)

        already_bred = set()

        for creature in breeding_candidates:
            if creature.id in already_bred:
                continue
            if not creature.can_breed():
                continue

            # population caps
            if (creature.species == Species.HERBIVORE
                    and herb_count >= cfg.MAX_POPULATION_HERBIVORE):
                continue
            if (creature.species == Species.CARNIVORE
                    and carn_count >= cfg.MAX_POPULATION_CARNIVORE):
                continue

            # find best compatible mate nearby
            partner = None
            best_mate_dist = cfg.MATE_SEARCH_RANGE

            for other in self._get_neighbors(creature):
                if other.id in already_bred:
                    continue
                if not creature.is_compatible_mate(other):
                    continue
                dist = creature.distance_to(other)
                if dist < best_mate_dist:
                    best_mate_dist = dist
                    partner = other

            if partner is None:
                continue

            child = creature.reproduce_with(partner)
            new_creatures.append(child)
            self.total_births += 1
            already_bred.add(creature.id)
            already_bred.add(partner.id)

            if creature.species == Species.HERBIVORE:
                herb_count += 1
            else:
                carn_count += 1

        self.creatures.extend(new_creatures)

    # --- cleanup ---

    def _remove_dead(self):
        alive_creatures = []
        for c in self.creatures:
            if c.alive:
                alive_creatures.append(c)
            else:
                self.total_deaths += 1
        self.creatures = alive_creatures
        self.food_items = [f for f in self.food_items if f.alive]

    def _enforce_population_limits(self):
        """If a species drops too low, respawn some using the best genome we've seen."""
        herb_count = sum(
            1 for c in self.creatures if c.species == Species.HERBIVORE
        )
        carn_count = sum(
            1 for c in self.creatures if c.species == Species.CARNIVORE
        )

        while herb_count < cfg.MIN_POPULATION:
            genome = self._get_seed_genome(Species.HERBIVORE)
            c = Creature(
                x=np.random.uniform(20, cfg.WORLD_WIDTH - 20),
                y=np.random.uniform(20, cfg.WORLD_HEIGHT - 20),
                species=Species.HERBIVORE,
                genome=genome.mutate(),
            )
            self.creatures.append(c)
            herb_count += 1

        while carn_count < max(2, cfg.MIN_POPULATION // 4):
            genome = self._get_seed_genome(Species.CARNIVORE)
            c = Creature(
                x=np.random.uniform(20, cfg.WORLD_WIDTH - 20),
                y=np.random.uniform(20, cfg.WORLD_HEIGHT - 20),
                species=Species.CARNIVORE,
                genome=genome.mutate(),
            )
            self.creatures.append(c)
            carn_count += 1

    def _get_seed_genome(self, species: Species) -> Genome:
        if species == Species.HERBIVORE and self.best_herbivore_genome is not None:
            return self.best_herbivore_genome
        if species == Species.CARNIVORE and self.best_carnivore_genome is not None:
            return self.best_carnivore_genome
        return Genome()

    def _track_best_genomes(self):
        for creature in self.creatures:
            if creature.species == Species.HERBIVORE:
                self.max_generation_herb = max(
                    self.max_generation_herb, creature.genome.generation
                )
                if (
                    self.best_herbivore_genome is None
                    or creature.genome.fitness > self.best_herbivore_genome.fitness
                ):
                    self.best_herbivore_genome = creature.genome.copy()
            else:
                self.max_generation_carn = max(
                    self.max_generation_carn, creature.genome.generation
                )
                if (
                    self.best_carnivore_genome is None
                    or creature.genome.fitness > self.best_carnivore_genome.fitness
                ):
                    self.best_carnivore_genome = creature.genome.copy()

    # --- public getters ---

    @property
    def herbivores(self) -> List[Creature]:
        return [c for c in self.creatures if c.species == Species.HERBIVORE]

    @property
    def carnivores(self) -> List[Creature]:
        return [c for c in self.creatures if c.species == Species.CARNIVORE]

    @property
    def alive_food_count(self) -> int:
        return sum(1 for f in self.food_items if f.alive)

    def get_stats(self) -> dict:
        herbs = self.herbivores
        carns = self.carnivores

        avg_energy_herb = (
            np.mean([c.energy for c in herbs]) if herbs else 0
        )
        avg_energy_carn = (
            np.mean([c.energy for c in carns]) if carns else 0
        )
        avg_age_herb = np.mean([c.age for c in herbs]) if herbs else 0
        avg_age_carn = np.mean([c.age for c in carns]) if carns else 0

        return {
            "tick": self.tick,
            "herbivores": len(herbs),
            "carnivores": len(carns),
            "food": self.alive_food_count,
            "avg_energy_herb": float(avg_energy_herb),
            "avg_energy_carn": float(avg_energy_carn),
            "avg_age_herb": float(avg_age_herb),
            "avg_age_carn": float(avg_age_carn),
            "total_births": self.total_births,
            "total_deaths": self.total_deaths,
            "total_food_eaten": self.total_food_eaten,
            "total_kills": self.total_kills,
            "max_gen_herb": self.max_generation_herb,
            "max_gen_carn": self.max_generation_carn,
            "best_fitness_herb": (
                self.best_herbivore_genome.fitness
                if self.best_herbivore_genome else 0
            ),
            "best_fitness_carn": (
                self.best_carnivore_genome.fitness
                if self.best_carnivore_genome else 0
            ),
        }
