# NeuroEvolution Ecosystem Simulator

A program I wrote for fun ;) (Later changed to a game to feature player modes for DSGN 0020, original NN now in debug mode)

2D world where creatures with neural network brains evolve through natural selection. The NNs are built from scratch with NumPy (no pytorch/tensorflow). Herbivores eat plants, carnivores hunt herbivores, and over time they actually get better at it.

## what it does

- creatures have 8 ray-cast sensors to see food and other creatures around them
- sensor data goes into a feed-forward neural network → outputs: turn, speed, eat, attack
- energy runs out = you die. eat enough = you reproduce (with a mutated copy of your brain)
- thats it. no hand-designed fitness function, survival is the only thing that matters
- carnivores are faster but turn wider, herbivores are slower but more agile

## setup

```
pip install -r requirements.txt
python main.py
```

headless mode (no window, just stats):
```
python main.py --headless --ticks 20000
```

## controls

- `SPACE` pause
- `S` show sensor rays
- `H` toggle HUD
- `F` turbo mode
- `1-6` speed presets (1x to 50x)
- click on a creature to inspect it
- `Q` quit (saves plots to `analytics_output/`)

## project structure

```
├── main.py              # entry point
├── config.py            # all the knobs
├── core/
│   ├── neural_network.py    # NN from scratch (numpy)
│   ├── creature.py          # agents with brains
│   ├── food.py              # plants
│   └── world.py             # simulation loop
├── evolution/
│   ├── genome.py            # weight encoding + mutation
│   └── selection.py         # tournament selection, crossover
├── visualization/
│   └── renderer.py          # pygame rendering
└── analytics/
    └── tracker.py           # stats tracking + matplotlib plots
```

## the neural net

```
27 inputs (8 rays × 3 channels + energy + speed + bias)
  → 20 neurons (tanh)
  → 16 neurons (tanh)
  → 4 outputs (turn, speed, eat, attack)
```

964 total parameters per creature, all evolved not trained.

## tech

python, numpy, pygame, matplotlib
