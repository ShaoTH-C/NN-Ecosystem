# Ecosystem — Technical Architecture

This document describes the **inner systems** of the simulation: the neural
network, the evolution loop, sensing, decision-making, social learning,
and the world tick. Rendering, audio, and the god-game UI are deliberately
out of scope — they are presentation layers built on top of what's described
here.

---

## 1. System overview

```
                    ┌─────────────────────────────────────┐
                    │             World (state)           │
                    │  creatures[]   food[]   particles   │
                    │  spatial grids   active_effects     │
                    └───────────────┬─────────────────────┘
                                    │  step() — one tick
                                    ▼
   ┌──────────────────────────────────────────────────────────────┐
   │                    PER-TICK PIPELINE                         │
   │                                                              │
   │   ① spawn food            (cfg.FOOD_SPAWN_RATE × rain mult)  │
   │   ② build spatial grid    (food_grid, creature grid)         │
   │   ③ social info           (N×N mate-compat — GPU-friendly)   │
   │   ④ sense (batched)       8 rays × 3 chan + nearest-target   │
   │   ⑤ think (batched)       N×[44]→[52]→[36]→[5] forward pass  │
   │   ⑥ act                   move • consume_energy • update     │
   │   ⑦ eat / attack          herbivores eat food, carns hunt    │
   │   ⑧ reproduce             mate-pairing + crossover           │
   │   ⑨ social learning       teaching every TEACHING_INTERVAL   │
   │   ⑩ divine effects        plague tick, rain, etc.            │
   │   ⑪ remove dead           + auto-respawn floor               │
   │   ⑫ track best genomes    (used to seed respawns)            │
   └──────────────────────────────────────────────────────────────┘
```

The whole simulation is **deterministic given a seed** (`--seed N`).
Pacing is decoupled from rendering by a fixed-timestep accumulator in
[main.py:96-117](../main.py#L96-L117).

Each system is detailed below.

---

## 2. The brain — neural network

**Architecture:** `[44, 52, 36, 5]` fully connected, `tanh` activation
throughout (including output). Built from scratch in NumPy in
[core/neural_network.py](../core/neural_network.py).

```
   inputs (44)        hidden 1 (52)     hidden 2 (36)      output (5)
   ──────────         ─────────────     ─────────────      ──────────
   sensors[24]    ─►                ─►                ─►   turn
   self stats[4]  ─►   tanh(W₁x+b₁) ─►  tanh(W₂h+b₂) ─►   speed
   nearest[9]     ─►                ─►                ─►   eat
   parent dir[2]  ─►                ─►                ─►   attack
   hunger[4]      ─►                ─►                ─►   breed
   bias[1]        ─►
```

**Total parameters** (from [main.py:78](../main.py#L78), printed at startup):

```
W₁: 44 × 52 = 2288        b₁: 52
W₂: 52 × 36 = 1872        b₂: 36
W₃: 36 × 5  =  180        b₃: 5
                  ─────────────
                  4433 parameters per brain
```

Every creature carries its own copy. With ~110 herbivores + ~22 carnivores
= ~132 brains, that's ~585k floats updated per tick.

### Batched forward pass

Naïvely calling `brain.forward(x)` per creature is ~5× slower than running
all of them at once via `einsum`. The world batches by topology group (so
that mid-evolution topology mutations don't break the batch):

```python
# world.py:_batch_think (simplified)
for topology, creatures in groups.items():
    inputs   = stack([c.build_nn_inputs() for c in creatures])  # (N, 44)
    layer_W  = [stack([c.brain.weights[l] for c in creatures])  # (N, fin, fout)
                for l in range(num_layers)]
    layer_B  = [...]                                            # (N, fout)
    outputs  = NeuralNetwork.batched_forward(inputs, layer_W, layer_B, xp)

# the einsum inside batched_forward:
z = xp.einsum('ni,nio->no', a, W) + B        # one matmul per layer for all N nets
a = xp.tanh(z)
```

`xp` is either `numpy` or `cupy` — see §10.

---

## 3. Sensing the world

Each creature casts **8 rays** evenly spaced around its facing direction,
and each ray reports the **normalized distance to the nearest hit on three
channels**: food, herbivores, carnivores.

```
                   (forward)
                       │
                       ▼
              ┌────────●────────┐
              │       /│\       │
              │      / │ \      │   ← 8 rays @ 45°
              │     /  │  \     │     each ray reports
            ──┼────●───●───●────┼──   3 floats:
              │     \  │  /     │       [food_dist,
              │      \ │ /      │        herb_dist,
              │       \│/       │        carn_dist]
              └────────●────────┘     normalized 0..1
                       ▲              (1 = nothing in range)
```

Total ray output: `8 × 3 = 24 floats` → indices `[0:24]` of the NN input.

### Ray hit-testing (vectorized)

For each ray the world projects every candidate target onto the ray
direction, rejects projections that are behind the creature or beyond
`SENSOR_RANGE`, and rejects targets whose perpendicular distance exceeds
`CREATURE_RADIUS + 5`. The closest valid hit per ray wins.

In [world.py:_batch_sense](../core/world.py#L189), this is one massive
einsum across all creatures, all rays, all targets, all channels:

```
   diff      = targets[None,:,:] - pos[:,None,:]      # (N, T, 2)
   proj      = einsum('ntx,nrx->nrt', diff, ray_dirs) # (N, R, T)
   perp_sq   = dist_sq[:,None,:] - proj²              # (N, R, T)
   valid     = (proj > 0) & (proj < range) & (perp_sq < det_w²)
   readings  = where(valid, proj/range, 1.0).min(axis=2)   # (N, R)
```

This is the single biggest hot-spot that benefits from GPU dispatch.

### Directional nearest-target info

In addition to the rays, the world also computes the **direction + distance
to the nearest food, same-species creature, and threat (or prey, for
carnivores)** using toroidal coordinates. Each is encoded as
`(sin(rel_angle), cos(rel_angle), normalized_distance)` so the NN sees a
smooth signal regardless of which direction the target is in.

These feed indices `[28:37]` of the input vector.

---

## 4. The 44-dimensional decision pipeline

[creature.py:build_nn_inputs](../core/creature.py#L375) assembles the input
vector. Layout:

| Index   | Source                  | Meaning                                     |
|---------|-------------------------|---------------------------------------------|
| `0:24`  | `sensor_readings`       | 8 rays × 3 channels (food / herb / carn)    |
| `24`    | `energy / MAX_ENERGY`   | normalized fullness                         |
| `25`    | `speed / max_speed`     | how fast am I going right now               |
| `26`    | `age / lifespan`        | how old am I (0=newborn, 1=dying)           |
| `27`    | `maturity`              | 0 = baby, 1 = adult                         |
| `28:31` | `nearest_food_rel`      | `(sin θ, cos θ, dist)` to nearest food      |
| `31:34` | `nearest_same_rel`      | nearest same-species (for mating / flocking)|
| `34:37` | `nearest_threat_rel`    | herb: nearest predator. carn: nearest prey  |
| `37:39` | `parent_direction`      | direction to parent (fades with maturity)   |
| `39`    | `hunger_urgency`        | nonlinear `(1 − e/E)^1.5`                   |
| `40`    | `is_satiated`           | step 0/1: above SATIATION_THRESHOLD?        |
| `41`    | `_breeding_ready`       | step 0/1: energy + age allow breeding?      |
| `42`    | `nearby_mates_count/3`  | how many compatible mates are nearby        |
| `43`    | `1.0`                   | bias unit                                   |

The 5-dim output is interpreted by
[creature.py:apply_nn_outputs](../core/creature.py#L418):

| Output | Range  | Mapped to                                      |
|--------|--------|------------------------------------------------|
| `0`    | `-1..1`| `turn = output × turn_rate` (radians/tick)     |
| `1`    | `-1..1`| `speed = (output+1)/2 × max_speed`             |
| `2`    | `>0`   | `wants_to_eat`                                 |
| `3`    | `>0`   | `wants_to_attack`                              |
| `4`    | `>0`   | `wants_to_breed`                               |

Boolean intents are **gated** by environment and instincts later (you can
"want" to attack, but if no prey is in range, nothing happens; if you're
satiated, the world skips `_handle_attacks` for you).

---

## 5. Behavioral instincts (the safety net)

Pure NN output produces lots of bad behaviors early in evolution:
standing still forever, spinning in tight circles, never eating.
[creature.py:_apply_instincts](../core/creature.py#L443) layers three hard
overrides on top of the NN's decision so newborns survive long enough to
matter:

```
  NN turn/speed ─┐
                 ▼
   ┌─────────────────────────────────────────────────────┐
   │  INSTINCT 1 — break idle                            │
   │  if speed < MIN_WANDER_SPEED_FRAC × max_speed:      │
   │      _idle_ticks += 1                               │
   │      if _idle_ticks > IDLE_BREAK_TICKS:             │
   │          random angle jolt + force speed = 0.4×max  │
   ├─────────────────────────────────────────────────────┤
   │  INSTINCT 2 — break circles                         │
   │  if |turn_accumulator| > CIRCLE_BREAK_THRESHOLD:    │
   │      _straight_lockout = CIRCLE_LOCKOUT_TICKS       │
   │      angle += rand(-0.6, 0.6)                       │
   │      turn_accumulator = 0                           │
   ├─────────────────────────────────────────────────────┤
   │  INSTINCT 3 — minimum wander speed                  │
   │  speed = max(speed, MIN_WANDER_SPEED_FRAC × max)    │
   └─────────────────────────────────────────────────────┘
                 │
                 ▼
            final action
```

While `_straight_lockout > 0`, the NN's turn output is zeroed —
[creature.py:422-425](../core/creature.py#L422-L425) — so the creature
**must** drive straight long enough to actually leave the loop. Without
this, the NN just resumes the same biased turn the next tick.

Anti-circling has a second layer covered in §12.

---

## 6. Energy & lifecycle

Every creature has `energy ∈ [0, MAX_ENERGY]`, dies when it hits 0 or when
`age >= lifespan`. The per-tick metabolism formula
([creature.py:consume_energy](../core/creature.py#L498)):

```
   cost = BASE_METABOLISM
        + (speed > 0.5 ? MOVE_ENERGY_EXTRA × speed/max : 0)
        + (|turn_accumulator| > 2 ? BASE × 0.3 : 0)              ← circle penalty
        × (CARNIVORE_METABOLISM_MULT  if carnivore)
        × aging_multiplier(age/lifespan)                          ← old-age tax
        × (0.5 + 0.5 × maturity         if child)                 ← cheaper to be small
```

Three lifecycle phases:

```
   age = 0                          MATURATION_TICKS              lifespan
   │                                       │                          │
   ├───────── child ───────────────────────┼──── adult ───┬─ old age ─┤
   │ slower, cheaper, parent               │ full stats   │ slower,
   │ direction signal                      │              │ pricier
   │                                       │              │
   maturity: 0 ──────────────────────────► 1.0     speed declines
   speed_mult: CHILD_SPEED_SCALE → 1.0             metabolism rises
                                                   (AGING_START_FRACTION
                                                    onward)
```

**Satiation** (`energy ≥ MAX_ENERGY × SATIATION_THRESHOLD`) flips behavior:
herbivores stop eating ([world.py:613](../core/world.py#L613)), carnivores
stop hunting ([world.py:660](../core/world.py#L660)), both prioritize
mating with a wider search radius ([world.py:720-721](../core/world.py#L720-L721)).

**Death by predation:** carnivores deal damage scaled by their charging
speed (`0.5 + 0.5×speed/max`) and gain energy
`KILL_BASE_ENERGY + prey.energy × KILL_ENERGY_FRACTION`, scaled down for
small prey via `CHILD_ENERGY_VALUE_MULT`.

---

## 7. Evolution — genome, mutation, crossover

The NN has no backprop. Brains improve by **selection**: surviving long
enough to reproduce passes your weights forward, dying ends them.

### Genome representation

[evolution/genome.py](../evolution/genome.py): a `Genome` is just a flat
NumPy array of all weights and biases concatenated in layer order, plus
the `layer_sizes` topology list. `.build_network()` reshapes it into
`NeuralNetwork.weights[]` and `.biases[]`.

### Xavier init **with a turn-bias prior**

A naïve Xavier init produces 44-input layers whose pre-activation sums
saturate `tanh` near `±1` from random noise alone — which means newborns
spin endlessly. So the output layer is hand-biased
([genome.py:_xavier_init](../evolution/genome.py#L32)):

```
   output index    | what it controls   |  init bias  |  init weight scale
   ────────────────┼────────────────────┼─────────────┼───────────────────
   0  turn         |   steering         |    0.0      |    × 0.15  ← tiny!
   1  speed        |   throttle         |    +0.5     |    × 1.0   (move forward)
   2  eat          |   herb eat intent  |    +0.3     |    × 1.0   (lean toward eating)
   3  attack       |   carn hunt intent |    −0.2     |    × 1.0   (don't attack by default)
   4  breed        |   breed intent     |    0.0      |    × 1.0   (neutral)
```

The 0.15× scaling on turn weights is the single most important hyper-prior
in the project: without it, every untrained newborn enters a death-spiral
and the system can't bootstrap learning. With it, newborns drive nearly
straight, then mutation/evolution discovers turning when it's actually
useful.

### Mutation operators

```
                         Genome.mutate()
                              │
        ┌─────────────────────┼──────────────────────┐
        ▼                     ▼                      ▼
   weight perturb       hard reassign          topology mutate
   (most edits)         (rare, 5%)             (very rare,
   mask = rand <        idx = rand;             TOPOLOGY_MUTATION_RATE)
   MUTATION_RATE;       genes[idx] = randn;      add / remove neuron
   genes[mask] +=                                  in a hidden layer
   randn × MUT_STR

```

`Genome.mutate_child()` is a **gentler** variant for parent→child
inheritance: lower rate (`CHILD_MUTATION_RATE`) and lower strength
(`CHILD_MUTATION_STRENGTH`). The metaphor: mutation = generational drift,
mutate_child = a parent teaching slightly imperfect lessons.

### Crossover

[evolution/selection.py:crossover](../evolution/selection.py#L21) is
**uniform crossover** — each gene independently comes from parent A or B
with 50/50 probability:

```
   parent A:  [w₀, w₁, w₂, w₃, w₄, ..., wₙ]
   parent B:  [v₀, v₁, v₂, v₃, v₄, ..., vₙ]
                  │   │   │   │   │
              mask:0  1   1   0   1   ...
                  ▼   ▼   ▼   ▼   ▼
   child:     [w₀, v₁, v₂, w₃, v₄, ..., vₙ]
```

If parents have different topologies (a topology mutation happened in one
of their lineages), crossover degrades to "clone the fitter parent."

### Fitness

Computed every tick in [creature.py:update](../core/creature.py#L624):

```
   fitness =   children_count × 150        ← reproductive success dominates
             + age × 0.08                  ← survival is cheap
             + food_eaten × 1.5            ← bootstrap: encourages eating early
             + kills × 25                  ← carnivore-specific
             + total_energy_gained × 0.3   ← efficiency hint
```

Reproductive success is intentionally king. Natural selection literally
doesn't care how long you live — only whether your genome propagates.

The world tracks `best_herbivore_genome` and `best_carnivore_genome` (the
all-time fitness leaders) to seed respawns when a population crashes
(see §11).

---

## 8. Social learning — the teaching system

Pure mutation-based evolution is slow. The world adds **horizontal weight
transfer** between living creatures: every `TEACHING_INTERVAL` ticks, each
creature looks at its high-performing neighbors and partially blends its
NN weights toward theirs. This is biologically analogous to imitation
learning in animal groups.

### Performance score

A rolling EMA score per creature
([creature.py:update_performance](../core/creature.py#L149)):

```
  score = 0.35 × energy/MAX                              ← well-fed
        + 0.25 × min(speed/max, 1) × (1 − 0.7×circle_pen) ← moving usefully
        + 0.15 × min(age/300, 1)                         ← experienced
        + 0.15 × min(children/3, 1)                      ← reproductive
        + 0.10 × min(food_eaten/k, 1)                    ← feeding success

  performance_score ← 0.92×prev + 0.08×score              ← EMA smoothing
```

### Teacher selection & blending

[world.py:_handle_teaching](../core/world.py#L754):

```
   for each living creature:
       teacher ← argmax of (neighbor.perf - my.perf)
                 over neighbors of same species within TEACHING_RADIUS
                 such that gap > TEACHING_MIN_PERF_GAP

       if teacher exists:
           gap   = teacher.perf − my.perf
           blend = clip( TEACHING_BLEND_RATE × gap/0.3,
                         0.02, TEACHING_MAX_BLEND )
           my.weights ← (1 − blend) × my.weights + blend × teacher.weights
           my.genome.genes ← my.weights        ← so children inherit it
```

Two important properties:

1. **Bigger gap → faster learning.** If the teacher is dramatically better,
   you blend more aggressively. If they're only marginally better, you
   stay mostly yourself.
2. **Genome sync.** After learning, the genome is overwritten with the
   blended weights. This means **learned behavior gets inherited** — a
   form of cultural transmission turning the system into a quasi-Lamarckian
   evolution rather than pure Darwinian.

---

## 9. World tick order — why this order matters

```
   STEP 1:  spawn food     ←─ before sensing, so newly-spawned plants
                              are visible this tick
   STEP 2:  build grids    ←─ rebuild every tick because everything moved
   STEP 3:  social info    ←─ before think, feeds NN input #42
   STEP 4:  set parent dir ←─ before sense, feeds NN inputs #37-38
   STEP 5:  batch sense    ←─ all rays for all creatures in one numpy op
   STEP 6:  batch think    ←─ all forward passes in one einsum per layer
   STEP 7:  move           ←─ apply turn + throttle decided by think
   STEP 8:  consume energy ←─ may kill creature → skip remaining phases
   STEP 9:  update         ←─ age, fitness, slow-circle-detection
   STEP 10: eat            ←─ herbivores grab food
   STEP 11: attack         ←─ carnivores damage prey (may kill)
   STEP 12: reproduce      ←─ pairing + crossover + spawn child
   STEP 13: update perf    ←─ for teaching
   STEP 14: teaching       ←─ horizontal weight blending
   STEP 15: divine effects ←─ plague drain, rain food multiplier, etc.
   STEP 16: particles tick ←─ visual effects only
   STEP 17: remove dead    ←─ compact creature list
   STEP 18: enforce floor  ←─ respawn if pop < MIN_POPULATION
   STEP 19: track best     ←─ remember the all-time leader genomes
```

Note that **dead creatures don't act**. After `consume_energy` may flip
`alive=False`, every later phase guards on `if not c.alive: continue`.

---

## 10. GPU acceleration — when, why, how

Every batched op in `world.py` uses an `xp` module that points at either
`numpy` or `cupy` (CUDA). The decision is per-tick and **population-gated**:

```python
# world.py:_pick_xp
def _pick_xp(self, n_living):
    use_gpu = (
        cfg.USE_GPU
        and _CUPY_AVAILABLE
        and n_living >= cfg.GPU_MIN_POP        # default: 150
    )
    return cp if use_gpu else np
```

**Why a threshold?** GPU dispatch has a fixed latency cost
(host-to-device transfer, kernel launch). At low N, that overhead dwarfs
the actual math. The break-even is around N=150 in this codebase — below
that, NumPy on CPU wins.

**Where the GPU win is biggest:**

```
   Operation                           Cost scaling     Best on
   ─────────────────────────────────  ───────────────   ───────
   _compute_social_info  (N×N pairs)  O(N²)  ✓✓✓✓     GPU
   _batch_sense          (N×R×T)      O(N·R·T) ✓✓✓    GPU (large T)
   _batch_think          (N×Σ Wₗ)     O(N·params) ✓✓  GPU
   per-creature update                O(N) python      CPU (Python-bound)
```

The pairwise compatibility matrix is by far the heaviest — at N=200 it's
40,000 cell-wise compatibility checks per tick, all five conditions
multiplied together. GPU reduces it to a couple of milliseconds.

**Bootstrap:** [core/_gpu_bootstrap.py](../core/_gpu_bootstrap.py) registers
the pip-installed CUDA DLL directories (both `os.add_dll_directory` and
`PATH` prepend) so cupy can find them on Windows. Imported with a `noqa`
because it has a side effect on import.

---

## 11. Population dynamics — caps and the floor

Three pressures keep the world balanced:

```
                            POPULATION CONTROL

                  ┌──────────────────────────────┐
                  │  MAX_POPULATION_HERBIVORE    │   hard cap on
                  │  MAX_POPULATION_CARNIVORE    │   reproduction
                  └─────────────┬────────────────┘
                                │
                                ▼
   reproduction loop checks count first — over cap → child denied
                                │
                                ▼
                  ┌──────────────────────────────┐
                  │  MAX_FOOD                    │   spawn loop stops
                  └─────────────┬────────────────┘
                                │
                                ▼
                  ┌──────────────────────────────┐
                  │  MIN_POPULATION              │   if pop drops below,
                  │  (also max(2, MIN_POP//4)    │   auto-respawn from
                  │   for carnivores)            │   best known genome
                  └──────────────────────────────┘
```

The **respawn floor** ([world.py:_enforce_population_limits](../core/world.py#L807))
is what makes the simulation feel "alive forever." When a species crashes,
fresh creatures spawn using the all-time best-fitness genome, mutated
once for variety. This means knowledge isn't lost when a population
collapses — the next generation literally inherits the prior best brain.

---

## 12. Anti-circling — two-layer defense

Circling is the single most common failure mode of evolved navigators:
the NN finds it can hold a constant turn and "wait" near food. There are
**two independent detectors**, because each catches a different failure
pattern:

### Layer A — turn accumulator (fast)

```
   every tick:
       turn_accumulator ← 0.95 × turn_accumulator + turn

   if |turn_accumulator| > CIRCLE_BREAK_THRESHOLD:
       trigger lockout, jolt angle, zero accumulator
```

This is an EMA of signed turn values. A creature that consistently turns
right will spike `turn_accumulator` positive and trigger the break.

**Catches:** sustained one-sided turning (tight circles).
**Misses:** the sneakier failure where the NN *alternates* turn signs
but still traces a loop.

### Layer B — displacement snapshot (slow)

```
   every CIRCLE_SNAPSHOT_INTERVAL ticks (80):

       path_walked = distance_traveled - _snap_path_len
       net         = toroidal distance from (_snap_x, _snap_y) to (x, y)

       if age >= CIRCLE_HARD_BREAK_MIN_AGE
          and path_walked >= CIRCLE_MIN_PATH_TO_CHECK   (50)
          and net < CIRCLE_MAX_NET_DISPLACEMENT:        (45)

           hard break:
              angle += random ±(π/2 .. π)
              _straight_lockout = CIRCLE_HARD_LOCKOUT_TICKS  (70)
              speed = max(speed, 0.6 × max_speed)

       always re-snapshot pose & path-length
```

This catches the case where the creature has *walked far* but *moved
nowhere*. The 70-tick straight-lockout is long enough to escape any loop
the world is small enough to contain.

```
   creature trajectory                       interpretation
   ─────────────────────────────────────     ──────────────
   straight line                             ✓ ok
   gentle curve                              ✓ ok (path ≈ net displacement)
   tight one-sided spiral                    ✗ Layer A catches it (turn EMA)
   zigzag loop returning to start            ✗ Layer B catches it (path >> net)
```

---

## Appendix A — Key config knobs

The most important tunables in [config.py](../config.py) for tech-side
behavior:

| Knob                          | Effect                                    |
|-------------------------------|-------------------------------------------|
| `NN_HIDDEN_LAYERS`            | brain capacity (currently `[52, 36]`)     |
| `MUTATION_RATE` / `_STRENGTH` | exploration intensity per generation      |
| `CHILD_MUTATION_RATE` / `_STR`| how much a child differs from its parents |
| `TEACHING_BLEND_RATE`         | how fast horizontal learning happens      |
| `TEACHING_MIN_PERF_GAP`       | who counts as "worth learning from"       |
| `BREEDING_ENERGY_THRESHOLD`   | how rich you must be to breed             |
| `BREEDING_AGE_MIN_FRAC` / `_MAX_FRAC` | reproductive age window           |
| `MAX_POPULATION_HERBIVORE/CARNIVORE` | hard caps                          |
| `MIN_POPULATION`              | respawn floor                             |
| `GPU_MIN_POP`                 | when to switch to cupy                    |
| `CIRCLE_BREAK_THRESHOLD`      | turn-accumulator trigger (Layer A)        |
| `CIRCLE_MAX_NET_DISPLACEMENT` | displacement-snapshot trigger (Layer B)   |

---

## Appendix B — Data flow summary (one creature, one tick)

```
   ┌─────────────────────────────────────────────────────────────┐
   │  STATE (Creature)                                           │
   │  x, y, angle, speed, energy, age, maturity,                 │
   │  turn_accumulator, _straight_lockout, _snap_*,              │
   │  performance_score, brain (NN), genome                      │
   └────────────────────┬────────────────────────────────────────┘
                        │
   World.step() ────────┤
                        ▼
              build_nn_inputs() ──► [44]
                        │
                        ▼
              brain forward (batched einsum) ──► [5]
                        │
                        ▼
              apply_nn_outputs():
                turn (lockout-gated)
                speed
                wants_to_eat / attack / breed
                ─► instinct overrides
                        │
                        ▼
              move() — toroidal wrap
                        │
                        ▼
              consume_energy() — may kill
                        │
                        ▼
              update() — age, fitness recompute,
                         slow-circle detector
                        │
                        ▼
              world handles eat / attack / breed / teach
                        │
                        ▼
              update_performance() — for next teach pass
```

Everything else (rendering, audio, intro, divine tools) is presentation
that observes this loop without affecting its semantics.
