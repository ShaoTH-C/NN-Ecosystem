# AI-Generated Sprites

Drop PNG files into this folder. The game-mode renderer auto-loads any
file it finds; if a file is missing it falls back to a procedural sprite
so the project always runs.

| Filename            | Recommended size | Purpose                            |
|---------------------|------------------|------------------------------------|
| `background.png`    | 1400×900 (or any aspect — it's stretched to the play area) | Main world background (grass, forest, biome). Should look like a top-down nature scene. |
| `herbivore.png`     | 256×256 (transparent) | Cute herbivore creature, top-down view, facing UP. Will be rotated to match heading. |
| `carnivore.png`     | 256×256 (transparent) | Predator creature, top-down view, facing UP. Will be rotated to match heading. |
| `food.png`          | 128×128 (transparent) | Plant / berry / fruit sprite. |
| `icon_food.png`     | 64×64 (transparent) | (optional) Sidebar icon for "Bring Forth Food". |
| `icon_herbivore.png`| 64×64 (transparent) | (optional) Sidebar icon for spawn herbivore. |
| `icon_carnivore.png`| 64×64 (transparent) | (optional) Sidebar icon for spawn predator. |
| `icon_blessing.png` | 64×64 (transparent) | (optional) Sidebar icon for sun's blessing. |
| `icon_rain.png`     | 64×64 (transparent) | (optional) Sidebar icon for rain. |
| `icon_disaster.png` | 64×64 (transparent) | (optional) Sidebar icon for meteor / disaster. |
| `icon_plague.png`   | 64×64 (transparent) | (optional) Sidebar icon for plague. |

Important details:
- **Top-down view, facing up**, with a transparent background.
- Use a consistent art style across all sprites.
- Keep the sprite roughly centered inside the canvas.
