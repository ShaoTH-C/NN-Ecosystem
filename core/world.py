# world.py - the main simulation loop
# handles spawning, sensing, movement, eating, attacking, reproduction, cleanup

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

from core.creature import Creature, Species
from core.food import Food
from evolution.genome import Genome
from evolution.selection import select_parent, crossover
import config as cfg


class World:
    """
    The 2D ecosystem. Each tick it runs through the full lifecycle:
    spawn food -> sense -> think -> move -> eat/attack -> reproduce -> cleanup
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

        # spatial grid so we don't have to check every creature against every other
        self.grid_cell_size = cfg.SENSOR_RANGE
        self.grid: Dict[Tuple[int, int], List] = defaultdict(list)

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

        food_pos, herb_pos, carn_pos = self._gather_positions()

        for creature in self.creatures:
            if not creature.alive:
                continue
            creature.cast_sensors(food_pos, herb_pos, carn_pos)
            creature.think()
            creature.move()
            creature.consume_energy()
            creature.update()

        self._handle_eating()
        self._handle_attacks()
        self._handle_reproduction()
        self._remove_dead()
        self._enforce_population_limits()
        self._track_best_genomes()

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
        for creature in self.creatures:
            if creature.alive:
                cell = self._get_cell(creature.x, creature.y)
                self.grid[cell].append(creature)

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
        food_pos = np.array(
            [[f.x, f.y] for f in self.food_items if f.alive]
        ) if any(f.alive for f in self.food_items) else np.empty((0, 2))

        herb_pos = np.array(
            [[c.x, c.y] for c in self.creatures
             if c.alive and c.species == Species.HERBIVORE]
        ) if any(c.alive and c.species == Species.HERBIVORE for c in self.creatures) else np.empty((0, 2))

        carn_pos = np.array(
            [[c.x, c.y] for c in self.creatures
             if c.alive and c.species == Species.CARNIVORE]
        ) if any(c.alive and c.species == Species.CARNIVORE for c in self.creatures) else np.empty((0, 2))

        return food_pos, herb_pos, carn_pos

    # --- eating (herbivores eat food) ---

    def _handle_eating(self):
        for creature in self.creatures:
            if (
                not creature.alive
                or creature.species != Species.HERBIVORE
                or not creature.wants_to_eat
            ):
                continue

            for food in self.food_items:
                if not food.alive:
                    continue
                dist = creature.distance_to(food)
                if dist < cfg.EAT_RANGE:
                    energy = food.consume()
                    creature.gain_energy(energy)
                    creature.food_eaten += 1
                    self.total_food_eaten += 1
                    break  # one food per tick

    # --- attacking (carnivores hunt herbivores) ---

    def _handle_attacks(self):
        for creature in self.creatures:
            if (
                not creature.alive
                or creature.species != Species.CARNIVORE
                or not creature.wants_to_attack
            ):
                continue

            creature.energy -= cfg.ATTACK_ENERGY_COST

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

            if best_target is not None:
                best_target.energy -= cfg.ATTACK_DAMAGE
                if best_target.energy <= 0:
                    best_target.alive = False
                    # successful kill = big energy reward
                    creature.gain_energy(cfg.FOOD_ENERGY * 2.5)
                    creature.kills += 1
                    self.total_kills += 1

    # --- reproduction ---

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

        for creature in self.creatures:
            if not creature.can_reproduce():
                continue

            # don't exceed population caps
            if (creature.species == Species.HERBIVORE
                    and herb_count >= cfg.MAX_POPULATION_HERBIVORE):
                continue
            if (creature.species == Species.CARNIVORE
                    and carn_count >= cfg.MAX_POPULATION_CARNIVORE):
                continue

            # maybe find a mate for crossover
            partner = None
            if np.random.rand() < cfg.CROSSOVER_RATE:
                for other in self._get_neighbors(creature):
                    if (
                        other is not creature
                        and other.alive
                        and other.species == creature.species
                        and creature.distance_to(other) < cfg.SENSOR_RANGE * 0.5
                    ):
                        partner = other
                        break

            child = creature.reproduce(partner)
            new_creatures.append(child)
            self.total_births += 1

            if creature.species == Species.HERBIVORE:
                herb_count += 1
            else:
                carn_count += 1

        self.creatures.extend(new_creatures)

    # --- cleanup ---

    def _remove_dead(self):
        dead_count = sum(1 for c in self.creatures if not c.alive)
        self.total_deaths += dead_count
        self.creatures = [c for c in self.creatures if c.alive]
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

        while carn_count < cfg.MIN_POPULATION // 2:
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
