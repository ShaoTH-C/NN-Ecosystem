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

# GPU support — cupy is optional; falls back silently to numpy if unavailable.
# When USE_GPU is on, the hot per-tick math (sensor casting, pairwise social
# distances, batched NN forward) runs on the GPU via the same code paths,
# selected through a single `xp` array module that points at cupy or numpy.
from core import _gpu_bootstrap  # registers pip-installed nvidia DLL dirs  # noqa: F401
try:
    import cupy as cp
    _CUPY_AVAILABLE = True
except Exception:
    cp = None
    _CUPY_AVAILABLE = False


def _to_cpu(arr, xp):
    """xp.asnumpy when xp is cupy; passthrough when xp is numpy."""
    if xp is np:
        return arr
    return cp.asnumpy(arr)


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

        # --- divine intervention state (god mode) ---
        # active environmental effects that decay over time
        # each effect: {kind, x, y, radius, ticks_remaining, strength}
        self.active_effects: List[dict] = []
        # transient particles for visualization (effects, blessings, etc.)
        self.particles: List[dict] = []
        # food spawn rate multiplier from "rain" effect
        self.food_spawn_multiplier = 1.0
        # divine action log for the HUD
        self.divine_log: List[str] = []
        self.divine_action_count = 0

        # GPU pacing: log once on first GPU/CPU decision so the user sees
        # what mode the sim is actually running in (and doesn't think it's
        # hung if cupy is doing first-time JIT compilation).
        self._gpu_announced = False

        self._spawn_initial_population()
        self._spawn_initial_food()

    def _pick_xp(self, n_living: int):
        """Return cupy if GPU is configured, available, and N is above the
        break-even population. Otherwise numpy. Logs the mode on first call."""
        use_gpu = (
            getattr(cfg, "USE_GPU", False)
            and _CUPY_AVAILABLE
            and n_living >= getattr(cfg, "GPU_MIN_POP", 150)
        )
        if not self._gpu_announced:
            self._gpu_announced = True
            if not getattr(cfg, "USE_GPU", False):
                print("  [GPU] disabled in config — running on CPU (numpy)")
            elif not _CUPY_AVAILABLE:
                print("  [GPU] cupy not available — running on CPU (numpy)")
            elif use_gpu:
                print(f"  [GPU] cupy active (N={n_living} ≥ {cfg.GPU_MIN_POP})")
            else:
                print(f"  [GPU] standby — will switch to cupy at N ≥ {cfg.GPU_MIN_POP}")
        return cp if use_gpu else np

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

        # phase 0.5: compute social awareness (nearby mates/density for NN inputs)
        self._compute_social_info()

        # phase 1: sense — batched across all creatures in one numpy op.
        # ~5x faster than the per-creature loop at population sizes > 100,
        # since the inner numpy work amortizes Python call overhead.
        for creature in self.creatures:
            if not creature.alive:
                continue
            self._set_parent_direction(creature)
        self._batch_sense(food_pos, herb_pos, carn_pos)

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

        # phase 5: social learning
        for creature in self.creatures:
            if creature.alive:
                creature.update_performance()
        self._handle_teaching()

        # phase 6: process active divine effects (disasters, blessings, plagues)
        self._tick_divine_effects()
        self._tick_particles()

        self._remove_dead()
        self._enforce_population_limits()
        self._track_best_genomes()

    # --- batched sensor casting ---

    def _batch_sense(self, food_pos: np.ndarray, herb_pos: np.ndarray, carn_pos: np.ndarray):
        """Cast sensor rays + find nearest targets for ALL alive creatures at once.

        Replaces N separate calls to `Creature.cast_sensors` with one batched
        operation. Mathematically identical to the per-creature path; the win
        is from amortizing Python overhead and (when N is high enough) running
        on the GPU via cupy.

        Shapes (with N creatures, T targets per channel, R rays):
          pos:        (N, 2)      world coords of each creature
          ray_dirs:   (N, R, 2)   per-creature ray vectors
          diff:       (N, T, 2)   target - creature position
          proj:       (N, R, T)   projection of each diff onto each ray
          readings:   (N, R, 3)   normalized [0,1] distance per ray per channel
        """
        living = [c for c in self.creatures if c.alive]
        if not living:
            return

        N = len(living)
        R = cfg.NUM_SENSOR_RAYS
        sensor_range = cfg.SENSOR_RANGE
        detection_width = cfg.CREATURE_RADIUS + 5
        det_w_sq = detection_width * detection_width

        # gather positions, angles, species into arrays (CPU side; this loop is
        # O(N) python and unavoidable since creature state lives on the host)
        pos = np.empty((N, 2), dtype=np.float64)
        angles = np.empty(N, dtype=np.float64)
        species_arr = np.empty(N, dtype=np.int8)
        for i, c in enumerate(living):
            pos[i, 0] = c.x
            pos[i, 1] = c.y
            angles[i] = c.angle
            species_arr[i] = 0 if c.species == Species.HERBIVORE else 1

        xp = self._pick_xp(N)

        # one transfer to xp's memory space (no-op for numpy, H2D for cupy).
        # everything below this line uses xp, so the heavy work runs on GPU
        # when xp is cupy.
        pos_x = xp.asarray(pos)
        angles_x = xp.asarray(angles)
        food_x = xp.asarray(food_pos)
        herb_x = xp.asarray(herb_pos)
        carn_x = xp.asarray(carn_pos)

        # per-creature ray directions: (N, R, 2)
        ray_offsets = (2.0 * xp.pi * xp.arange(R) / R)
        ray_angles = angles_x[:, None] + ray_offsets[None, :]  # (N, R)
        cos_a = xp.cos(ray_angles)
        sin_a = xp.sin(ray_angles)
        ray_dirs = xp.stack([cos_a, sin_a], axis=-1)  # (N, R, 2)

        readings = xp.ones((N, R, 3), dtype=xp.float64)
        # per (creature, ray) — track the closest hit across ALL channels
        # so we can reconstruct hit positions for the sensor visualization
        best_hit_dist = xp.full((N, R), sensor_range, dtype=xp.float64)
        best_hit_found = xp.zeros((N, R), dtype=bool)

        for ch, targets in enumerate((food_x, herb_x, carn_x)):
            T = targets.shape[0]
            if T == 0:
                continue
            # diff[n, t, :] = targets[t] - pos[n]
            diff = targets[None, :, :] - pos_x[:, None, :]              # (N, T, 2)
            # projection of each diff onto each ray
            proj = xp.einsum('ntx,nrx->nrt', diff, ray_dirs)            # (N, R, T)
            dist_sq = xp.sum(diff * diff, axis=2)                       # (N, T)
            perp_sq = dist_sq[:, None, :] - proj * proj                 # (N, R, T)
            xp.maximum(perp_sq, 0.0, out=perp_sq)
            valid = (proj > 0) & (proj < sensor_range) & (perp_sq < det_w_sq)
            masked = xp.where(valid, proj, xp.inf)
            min_dists = masked.min(axis=2)                              # (N, R)
            found = min_dists < sensor_range
            # write per-channel reading
            readings[:, :, ch] = xp.where(found, min_dists / sensor_range, readings[:, :, ch])
            # track the closest hit overall (any channel) per (creature, ray)
            closer = found & (min_dists < best_hit_dist)
            best_hit_dist = xp.where(closer, min_dists, best_hit_dist)
            best_hit_found = best_hit_found | found

        # nearest-target directional info (for NN inputs) — keep on xp
        nearest_food = self._batch_nearest(pos_x, angles_x, food_x, sensor_range, xp)
        nearest_herb = self._batch_nearest(pos_x, angles_x, herb_x, sensor_range, xp)
        nearest_carn = self._batch_nearest(pos_x, angles_x, carn_x, sensor_range, xp)

        store_viz = self.store_sensor_viz
        if store_viz:
            # endpoints[n, r] = pos[n] + ray_dirs[n, r] * sensor_range
            endpoints = pos_x[:, None, :] + ray_dirs * sensor_range  # (N, R, 2)
            # per-ray hit positions where any channel hit
            hit_xy = pos_x[:, None, :] + ray_dirs * best_hit_dist[:, :, None]  # (N, R, 2)

        # one D2H transfer batch (no-op for numpy) — pull everything we need
        # for the writeback loop in one go to avoid per-element transfers.
        readings_flat = _to_cpu(readings.reshape(N, R * 3), xp)
        nearest_food_np = _to_cpu(nearest_food, xp)
        nearest_herb_np = _to_cpu(nearest_herb, xp)
        nearest_carn_np = _to_cpu(nearest_carn, xp)
        if store_viz:
            endpoints_np = _to_cpu(endpoints, xp)
            hit_xy_np = _to_cpu(hit_xy, xp)
            best_hit_found_np = _to_cpu(best_hit_found, xp)
        herb_mask = (species_arr == 0)

        for i, c in enumerate(living):
            c.sensor_readings = readings_flat[i]
            c.nearest_food_rel = (
                float(nearest_food_np[i, 0]),
                float(nearest_food_np[i, 1]),
                float(nearest_food_np[i, 2]),
            )
            if herb_mask[i]:
                c.nearest_same_rel = (
                    float(nearest_herb_np[i, 0]),
                    float(nearest_herb_np[i, 1]),
                    float(nearest_herb_np[i, 2]),
                )
                c.nearest_threat_rel = (
                    float(nearest_carn_np[i, 0]),
                    float(nearest_carn_np[i, 1]),
                    float(nearest_carn_np[i, 2]),
                )
            else:
                c.nearest_same_rel = (
                    float(nearest_carn_np[i, 0]),
                    float(nearest_carn_np[i, 1]),
                    float(nearest_carn_np[i, 2]),
                )
                c.nearest_threat_rel = (
                    float(nearest_herb_np[i, 0]),
                    float(nearest_herb_np[i, 1]),
                    float(nearest_herb_np[i, 2]),
                )

            if store_viz:
                c.sensor_endpoints = [tuple(endpoints_np[i, r]) for r in range(R)]
                c.sensor_hits = [
                    (float(hit_xy_np[i, r, 0]), float(hit_xy_np[i, r, 1]))
                    if best_hit_found_np[i, r] else None
                    for r in range(R)
                ]

    @staticmethod
    def _batch_nearest(pos, angles, targets, max_range: float, xp):
        """For each creature, find nearest target with toroidal distance.
        Returns (N, 3) on the same xp module: [sin(rel_angle), cos(rel_angle),
        normalized_dist]. Default row [0, 0, 1] for "nothing in range".
        Inputs are expected to already live in xp's memory space."""
        N = pos.shape[0]
        out = xp.zeros((N, 3), dtype=xp.float64)
        out[:, 2] = 1.0

        T = targets.shape[0]
        if T == 0:
            return out

        diff = targets[None, :, :] - pos[:, None, :]  # (N, T, 2)
        W = cfg.WORLD_WIDTH
        H = cfg.WORLD_HEIGHT
        dx = diff[..., 0]
        dy = diff[..., 1]
        wrap_x = xp.abs(dx) > W * 0.5
        wrap_y = xp.abs(dy) > H * 0.5
        dx = xp.where(wrap_x, dx - xp.sign(dx) * W, dx)
        dy = xp.where(wrap_y, dy - xp.sign(dy) * H, dy)

        dists = xp.sqrt(dx * dx + dy * dy)            # (N, T)
        valid = dists > 0.5
        dists_v = xp.where(valid, dists, xp.inf)
        nearest_idx = xp.argmin(dists_v, axis=1)       # (N,)
        rows = xp.arange(N)
        nearest_dist = dists_v[rows, nearest_idx]      # (N,)

        in_range = nearest_dist <= max_range * 2.0
        # short-circuit only matters for numpy; on GPU it's faster to just run
        # the math than to copy a bool back to host to check
        if xp is np and not np.any(in_range):
            return out

        sel_dx = dx[rows, nearest_idx]
        sel_dy = dy[rows, nearest_idx]
        abs_angle = xp.arctan2(sel_dy, sel_dx)
        rel_angle = abs_angle - angles
        sin_r = xp.sin(rel_angle)
        cos_r = xp.cos(rel_angle)
        norm_d = xp.minimum(nearest_dist / max_range, 1.0)

        out[:, 0] = xp.where(in_range, sin_r, 0.0)
        out[:, 1] = xp.where(in_range, cos_r, 0.0)
        out[:, 2] = xp.where(in_range, norm_d, 1.0)
        return out

    # --- batched neural network forward pass ---

    def _batch_think(self):
        """Run all creatures' NNs in one batched operation per topology group.
        Uses GPU via _pick_xp when alive count crosses GPU_MIN_POP."""
        living = [c for c in self.creatures if c.alive]
        if not living:
            return

        xp = self._pick_xp(len(living))

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

            # stack weight matrices for each layer (CPU side; weights are
            # already numpy in the genome, so this is just np.stack)
            layer_W = [
                np.stack([c.brain.weights[l] for c in creatures])
                for l in range(num_layers)
            ]
            layer_B = [
                np.stack([c.brain.biases[l] for c in creatures])
                for l in range(num_layers)
            ]

            # one transfer to xp memory (no-op for numpy)
            inputs_x = xp.asarray(inputs)
            layer_W_x = [xp.asarray(w) for w in layer_W]
            layer_B_x = [xp.asarray(b) for b in layer_B]

            outputs = NeuralNetwork.batched_forward(inputs_x, layer_W_x, layer_B_x, xp)

            # one D2H batch, then distribute
            outputs_np = _to_cpu(outputs, xp)
            for i, c in enumerate(creatures):
                c.apply_nn_outputs(outputs_np[i])

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

    # --- social awareness ---

    def _compute_social_info(self):
        """Set nearby_mates_count for each creature using a single batched
        pairwise-compatibility computation. Feeds the NN's mate-seeking signal.

        Vectorized via a (N, N) distance matrix and boolean compatibility masks.
        Runs on GPU via cupy when N is high enough — pairwise ops scale O(N²)
        so this is where the GPU win is biggest at large populations.
        """
        living = [c for c in self.creatures if c.alive]
        if not living:
            return
        N = len(living)

        pos = np.empty((N, 2), dtype=np.float64)
        energy = np.empty(N, dtype=np.float64)
        age = np.empty(N, dtype=np.int32)
        lifespan = np.empty(N, dtype=np.int32)
        species_arr = np.empty(N, dtype=np.int8)
        for i, c in enumerate(living):
            pos[i, 0] = c.x
            pos[i, 1] = c.y
            energy[i] = c.energy
            age[i] = c.age
            lifespan[i] = c.lifespan
            species_arr[i] = 0 if c.species == Species.HERBIVORE else 1

        xp = self._pick_xp(N)
        pos_x = xp.asarray(pos)
        energy_x = xp.asarray(energy)
        age_x = xp.asarray(age)
        lifespan_x = xp.asarray(lifespan)
        species_x = xp.asarray(species_arr)

        # toroidal pairwise distance²
        dx = xp.abs(pos_x[:, None, 0] - pos_x[None, :, 0])
        dy = xp.abs(pos_x[:, None, 1] - pos_x[None, :, 1])
        xp.minimum(dx, cfg.WORLD_WIDTH - dx, out=dx)
        xp.minimum(dy, cfg.WORLD_HEIGHT - dy, out=dy)
        dist_sq = dx * dx + dy * dy

        # mate-compatibility checks (mirror Creature.is_compatible_mate)
        same_species = species_x[:, None] == species_x[None, :]
        energy_ok = energy_x[None, :] >= cfg.BREEDING_ENERGY_THRESHOLD * 0.7
        age_ok = xp.abs(age_x[:, None] - age_x[None, :]) <= cfg.MATE_AGE_TOLERANCE
        min_age = (lifespan_x * cfg.BREEDING_AGE_MIN_FRAC).astype(xp.int32)
        max_age = (lifespan_x * cfg.BREEDING_AGE_MAX_FRAC).astype(xp.int32)
        other_in_breeding_age = (
            (age_x[None, :] >= min_age[None, :]) & (age_x[None, :] <= max_age[None, :])
        )
        not_self = ~xp.eye(N, dtype=bool)
        in_range = dist_sq < (cfg.MATE_SEARCH_RANGE * cfg.MATE_SEARCH_RANGE)

        compat = same_species & energy_ok & age_ok & other_in_breeding_age & not_self & in_range
        mates_count = compat.sum(axis=1)

        # one D2H to feed back to creatures
        mates_count_np = _to_cpu(mates_count, xp)
        for i, c in enumerate(living):
            c.nearby_mates_count = int(mates_count_np[i])

    # --- food spawning ---

    def _spawn_food(self):
        alive_food = sum(1 for f in self.food_items if f.alive)
        # food spawn rate scales with active "rain" blessings
        spawn_rate = max(1, int(cfg.FOOD_SPAWN_RATE * self.food_spawn_multiplier))
        if alive_food < cfg.MAX_FOOD:
            for _ in range(spawn_rate):
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

            # satiated creatures don't eat — no benefit at full energy,
            # and in real life well-fed animals stop foraging
            if creature.is_satiated:
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

            # satiated carnivores only attack in self-defense range (instinct),
            # they don't actively hunt when full — they should be seeking mates
            do_attack = False
            if best_dist < cfg.INSTINCT_ATTACK_RANGE:
                do_attack = True  # reflex — always attack if prey is right there
            elif creature.is_satiated:
                do_attack = False  # full: don't bother hunting, focus on breeding
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
            # satiated creatures are more motivated to find mates (wider search)
            search_range = cfg.MATE_SEARCH_RANGE
            if creature.is_satiated:
                search_range *= 1.3  # well-fed animals actively seek mates

            partner = None
            best_mate_dist = search_range

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

    # --- social learning / teaching ---

    def _handle_teaching(self):
        """Best-performing creatures teach nearby group members.
        This simulates real-world social learning: animals in a group
        observe successful members and adjust their behavior.

        How it works:
          - Every TEACHING_INTERVAL ticks, each creature looks for the
            best performer of the same species within TEACHING_RADIUS
          - If the teacher is significantly better (above TEACHING_MIN_PERF_GAP),
            the learner blends its NN weights toward the teacher's
          - Blend rate scales with the performance gap (bigger gap = learn more)
        """
        if self.tick % cfg.TEACHING_INTERVAL != 0:
            return

        for creature in self.creatures:
            if not creature.alive:
                continue

            best_teacher = None
            best_perf = creature.performance_score

            for other in self._get_neighbors(creature):
                if other is creature or not other.alive:
                    continue
                if other.species != creature.species:
                    continue
                dist = creature.distance_to(other)
                if dist > cfg.TEACHING_RADIUS:
                    continue
                if other.performance_score > best_perf + cfg.TEACHING_MIN_PERF_GAP:
                    best_perf = other.performance_score
                    best_teacher = other

            if best_teacher is not None:
                # blend rate scales with how much better the teacher is
                gap = best_teacher.performance_score - creature.performance_score
                blend = min(cfg.TEACHING_BLEND_RATE * (gap / 0.3), cfg.TEACHING_MAX_BLEND)
                blend = max(blend, 0.02)  # minimum learning rate
                creature.learn_from(best_teacher, blend)

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

    # --- divine intervention (god mode tools) ---

    def _log_divine(self, msg: str):
        self.divine_log.append(f"[t{self.tick}] {msg}")
        if len(self.divine_log) > 8:
            self.divine_log.pop(0)
        self.divine_action_count += 1

    def divine_drop_food(self, x: float, y: float, count: int = 6, spread: float = 60.0):
        """Bless the land with food — drop a small grove of plants at (x, y)."""
        for _ in range(count):
            ox = x + np.random.uniform(-spread, spread)
            oy = y + np.random.uniform(-spread, spread)
            ox = max(10, min(cfg.WORLD_WIDTH - 10, ox))
            oy = max(10, min(cfg.WORLD_HEIGHT - 10, oy))
            self.food_items.append(Food(x=ox, y=oy))
            self._add_particle(ox, oy, kind="sparkle", color=(140, 230, 90),
                               life=40, vy=-0.6)
        self._log_divine(f"Dropped {count} plants")

    def divine_spawn_creature(self, x: float, y: float, species: Species):
        """Manifest a creature into the world. Uses best known genome if available."""
        if species == Species.HERBIVORE and self.best_herbivore_genome is not None:
            genome = self.best_herbivore_genome.mutate()
        elif species == Species.CARNIVORE and self.best_carnivore_genome is not None:
            genome = self.best_carnivore_genome.mutate()
        else:
            genome = None
        c = Creature(x=x, y=y, species=species, genome=genome)
        self.creatures.append(c)
        for _ in range(15):
            self._add_particle(x, y, kind="sparkle",
                               color=(255, 240, 180), life=30,
                               vx=np.random.uniform(-1.5, 1.5),
                               vy=np.random.uniform(-1.5, 1.5))
        name = species.name.title()
        self._log_divine(f"Manifested {name}")

    def divine_disaster(self, x: float, y: float, radius: float = 130.0):
        """Strike the land with disaster — a meteor/storm killing creatures in radius."""
        killed = 0
        for c in self.creatures:
            if not c.alive:
                continue
            d = np.sqrt((c.x - x) ** 2 + (c.y - y) ** 2)
            if d < radius:
                # closer = more lethal
                damage = (1.0 - d / radius) * 250.0
                c.energy -= damage
                if c.energy <= 0:
                    c.alive = False
                    killed += 1
        # destroy nearby food too
        food_destroyed = 0
        for f in self.food_items:
            if not f.alive:
                continue
            d = np.sqrt((f.x - x) ** 2 + (f.y - y) ** 2)
            if d < radius * 0.6:
                f.alive = False
                food_destroyed += 1
        # fire particles
        for _ in range(40):
            ang = np.random.uniform(0, 2 * np.pi)
            r = np.random.uniform(0, radius)
            px = x + np.cos(ang) * r
            py = y + np.sin(ang) * r
            self._add_particle(px, py, kind="fire",
                               color=(255, 110, 40), life=45,
                               vx=np.random.uniform(-0.8, 0.8),
                               vy=np.random.uniform(-1.5, -0.3))
        self.active_effects.append({
            "kind": "disaster", "x": x, "y": y,
            "radius": radius, "ticks_remaining": 30, "strength": 1.0,
        })
        self._log_divine(f"Disaster: {killed} died, {food_destroyed} plants burned")

    def divine_blessing(self, x: float, y: float, radius: float = 140.0):
        """Bless creatures in radius — restore their energy, like sunlight from above."""
        blessed = 0
        for c in self.creatures:
            if not c.alive:
                continue
            d = np.sqrt((c.x - x) ** 2 + (c.y - y) ** 2)
            if d < radius:
                c.gain_energy(120.0)
                blessed += 1
        for _ in range(35):
            ang = np.random.uniform(0, 2 * np.pi)
            r = np.random.uniform(0, radius)
            px = x + np.cos(ang) * r
            py = y + np.sin(ang) * r
            self._add_particle(px, py, kind="halo",
                               color=(255, 240, 150), life=60,
                               vy=-0.4)
        self.active_effects.append({
            "kind": "blessing", "x": x, "y": y,
            "radius": radius, "ticks_remaining": 40, "strength": 1.0,
        })
        self._log_divine(f"Blessing: {blessed} creatures healed")

    def divine_plague(self, x: float, y: float, radius: float = 160.0, duration: int = 400):
        """Cast a plague — a slow-acting energy drain on creatures in the area."""
        self.active_effects.append({
            "kind": "plague", "x": x, "y": y,
            "radius": radius, "ticks_remaining": duration, "strength": 0.45,
        })
        self._log_divine(f"Plague cast (r={int(radius)})")

    def divine_rain(self, duration: int = 600, multiplier: float = 4.0):
        """World-wide rain that boosts food spawn rate temporarily."""
        self.active_effects.append({
            "kind": "rain", "x": 0, "y": 0,
            "radius": 0, "ticks_remaining": duration, "strength": multiplier,
        })
        self._log_divine(f"Rain begins (x{multiplier:.0f})")

    def _add_particle(self, x: float, y: float, kind: str, color: tuple,
                      life: int, vx: float = 0.0, vy: float = 0.0):
        self.particles.append({
            "x": x, "y": y, "vx": vx, "vy": vy,
            "kind": kind, "color": color, "life": life, "max_life": life,
        })

    def _tick_particles(self):
        survivors = []
        for p in self.particles:
            p["life"] -= 1
            if p["life"] <= 0:
                continue
            p["x"] += p["vx"]
            p["y"] += p["vy"]
            # gentle gravity for fire, float-up for sparkles handled by initial vy
            if p["kind"] == "fire":
                p["vy"] += 0.02
            survivors.append(p)
        self.particles = survivors

    def _tick_divine_effects(self):
        # reset rain multiplier each tick — re-applied below if rain is active
        self.food_spawn_multiplier = 1.0
        survivors = []
        for eff in self.active_effects:
            eff["ticks_remaining"] -= 1

            if eff["kind"] == "plague":
                # drain energy from creatures inside the radius
                for c in self.creatures:
                    if not c.alive:
                        continue
                    d = np.sqrt((c.x - eff["x"]) ** 2 + (c.y - eff["y"]) ** 2)
                    if d < eff["radius"]:
                        c.energy -= eff["strength"]
                        if c.energy <= 0:
                            c.alive = False
                # occasional sickly green particle
                if self.tick % 3 == 0:
                    ang = np.random.uniform(0, 2 * np.pi)
                    r = np.random.uniform(0, eff["radius"])
                    self._add_particle(
                        eff["x"] + np.cos(ang) * r,
                        eff["y"] + np.sin(ang) * r,
                        kind="plague", color=(140, 200, 80),
                        life=40, vy=-0.2,
                    )

            elif eff["kind"] == "rain":
                self.food_spawn_multiplier = max(self.food_spawn_multiplier, eff["strength"])

            if eff["ticks_remaining"] > 0:
                survivors.append(eff)
        self.active_effects = survivors

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
