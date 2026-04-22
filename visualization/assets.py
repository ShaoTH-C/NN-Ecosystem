# assets.py - load AI-generated sprites for the game-mode renderer
# falls back to procedurally drawn surfaces if image files are missing,
# so the project always runs even before the user generates art.

import os
import pygame
import math
from typing import Optional, Tuple

import config as cfg


def _assets_path(*parts: str) -> str:
    base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        cfg.ASSETS_DIR)
    return os.path.join(base, *parts)


def _try_load(filename: str, size: Optional[Tuple[int, int]] = None) -> Optional[pygame.Surface]:
    """Try to load an image from assets/. Returns None if missing."""
    path = _assets_path(filename)
    if not os.path.exists(path):
        return None
    try:
        surf = pygame.image.load(path).convert_alpha()
        if size is not None:
            surf = pygame.transform.smoothscale(surf, size)
        return surf
    except Exception:
        return None


# --- procedural fallbacks (so the game runs without AI art) ---

def _make_herbivore_fallback(size: int) -> pygame.Surface:
    """Cute round green critter (placeholder for AI deer/rabbit sprite)."""
    surf = pygame.Surface((size, size), pygame.SRCALPHA)
    cx = cy = size // 2
    body_r = int(size * 0.42)
    # body
    pygame.draw.circle(surf, (76, 168, 96), (cx, cy + 2), body_r)
    pygame.draw.circle(surf, (96, 200, 116), (cx, cy + 2), body_r, 0)
    # belly highlight
    pygame.draw.circle(surf, (160, 230, 160), (cx, cy + body_r // 3), body_r // 2)
    # ears
    ear_r = body_r // 3
    pygame.draw.ellipse(surf, (76, 168, 96),
                        (cx - body_r, cy - body_r - 2, ear_r, ear_r * 2))
    pygame.draw.ellipse(surf, (76, 168, 96),
                        (cx + body_r - ear_r, cy - body_r - 2, ear_r, ear_r * 2))
    # eyes
    eye_r = max(2, size // 14)
    pygame.draw.circle(surf, (20, 20, 20), (cx - body_r // 3, cy - 2), eye_r)
    pygame.draw.circle(surf, (20, 20, 20), (cx + body_r // 3, cy - 2), eye_r)
    pygame.draw.circle(surf, (255, 255, 255), (cx - body_r // 3 + 1, cy - 3), max(1, eye_r // 2))
    pygame.draw.circle(surf, (255, 255, 255), (cx + body_r // 3 + 1, cy - 3), max(1, eye_r // 2))
    return surf


def _make_carnivore_fallback(size: int) -> pygame.Surface:
    """Sharp-eared red predator (placeholder for AI wolf/fox sprite)."""
    surf = pygame.Surface((size, size), pygame.SRCALPHA)
    cx = cy = size // 2
    body_r = int(size * 0.42)
    pygame.draw.circle(surf, (180, 60, 60), (cx, cy + 2), body_r)
    pygame.draw.circle(surf, (220, 80, 70), (cx, cy + 2), body_r, 0)
    # darker belly
    pygame.draw.circle(surf, (130, 40, 40), (cx, cy + body_r // 2), body_r // 2)
    # pointed ears
    ear_h = body_r
    pygame.draw.polygon(surf, (160, 50, 50), [
        (cx - body_r, cy - body_r // 4),
        (cx - body_r // 2, cy - body_r),
        (cx - body_r // 4, cy - body_r // 3),
    ])
    pygame.draw.polygon(surf, (160, 50, 50), [
        (cx + body_r, cy - body_r // 4),
        (cx + body_r // 2, cy - body_r),
        (cx + body_r // 4, cy - body_r // 3),
    ])
    # glowing yellow eyes
    eye_r = max(2, size // 12)
    pygame.draw.circle(surf, (255, 220, 80), (cx - body_r // 3, cy - 2), eye_r)
    pygame.draw.circle(surf, (255, 220, 80), (cx + body_r // 3, cy - 2), eye_r)
    pygame.draw.circle(surf, (20, 10, 0), (cx - body_r // 3, cy - 2), max(1, eye_r // 2))
    pygame.draw.circle(surf, (20, 10, 0), (cx + body_r // 3, cy - 2), max(1, eye_r // 2))
    return surf


def _make_food_fallback(size: int) -> pygame.Surface:
    """Glowing berry/plant placeholder."""
    surf = pygame.Surface((size, size), pygame.SRCALPHA)
    cx = cy = size // 2
    # outer glow
    for r in range(size // 2, 0, -2):
        a = int(60 * (r / (size / 2)))
        pygame.draw.circle(surf, (140, 230, 90, a), (cx, cy), r)
    # berry body
    pygame.draw.circle(surf, (220, 60, 90), (cx, cy + 1), size // 4)
    pygame.draw.circle(surf, (255, 120, 140), (cx - 1, cy), max(1, size // 10))
    # leaf
    pygame.draw.polygon(surf, (110, 200, 90), [
        (cx, cy - size // 4),
        (cx + size // 6, cy - size // 3),
        (cx + size // 12, cy - size // 5),
    ])
    return surf


def _make_background_fallback(w: int, h: int) -> pygame.Surface:
    """Soft vertical gradient grass background."""
    surf = pygame.Surface((w, h))
    top = cfg.GAME_BG_TOP
    bot = cfg.GAME_BG_BOTTOM
    for y in range(h):
        t = y / max(1, h - 1)
        r = int(top[0] + (bot[0] - top[0]) * t)
        g = int(top[1] + (bot[1] - top[1]) * t)
        b = int(top[2] + (bot[2] - top[2]) * t)
        pygame.draw.line(surf, (r, g, b), (0, y), (w, y))
    # scatter subtle "grass tuft" dots so the field has texture
    import random
    rng = random.Random(42)
    for _ in range(int(w * h / 600)):
        x = rng.randint(0, w - 1)
        y = rng.randint(0, h - 1)
        shade = rng.randint(-15, 15)
        col = (
            max(0, min(255, top[0] + shade - 10)),
            max(0, min(255, top[1] + shade)),
            max(0, min(255, top[2] + shade - 10)),
        )
        pygame.draw.circle(surf, col, (x, y), rng.choice([1, 1, 2]))
    return surf


# --- public asset registry ---

class Assets:
    """Loads game-mode sprites once. Falls back to procedural art if missing."""

    def __init__(self, sprite_size: int = 36):
        self.sprite_size = sprite_size
        self.herbivore: pygame.Surface = (
            _try_load("herbivore.png", (sprite_size, sprite_size))
            or _make_herbivore_fallback(sprite_size)
        )
        self.carnivore: pygame.Surface = (
            _try_load("carnivore.png", (sprite_size, sprite_size))
            or _make_carnivore_fallback(sprite_size)
        )
        # food sprite size — bumped so berries read clearly against the grass.
        # scales with sprite_size so custom sprite_size still works, floor of 28.
        food_size = max(28, int(sprite_size * 0.75))
        self.food: pygame.Surface = (
            _try_load("food.png", (food_size, food_size))
            or _make_food_fallback(food_size)
        )
        # background is loaded at the renderer's window size — handled below
        self._bg_cache: Optional[pygame.Surface] = None
        self._bg_cache_size: Tuple[int, int] = (0, 0)

        # tool icons (sidebar) — small square images
        icon_size = 38
        self.icon_food = _try_load("icon_food.png", (icon_size, icon_size))
        self.icon_disaster = _try_load("icon_disaster.png", (icon_size, icon_size))
        self.icon_blessing = _try_load("icon_blessing.png", (icon_size, icon_size))
        self.icon_plague = _try_load("icon_plague.png", (icon_size, icon_size))
        self.icon_rain = _try_load("icon_rain.png", (icon_size, icon_size))
        self.icon_herb = _try_load("icon_herbivore.png", (icon_size, icon_size))
        self.icon_carn = _try_load("icon_carnivore.png", (icon_size, icon_size))

    def background(self, w: int, h: int) -> pygame.Surface:
        if self._bg_cache is not None and self._bg_cache_size == (w, h):
            return self._bg_cache
        loaded = _try_load("background.png", (w, h))
        self._bg_cache = loaded if loaded is not None else _make_background_fallback(w, h)
        self._bg_cache_size = (w, h)
        return self._bg_cache

    def get_creature_sprite(self, species_name: str) -> pygame.Surface:
        if species_name == "HERBIVORE":
            return self.herbivore
        return self.carnivore


# --- pre-rendered sprite cache ---
# pygame.transform.smoothscale + pygame.transform.rotate run on every creature
# every frame is the renderer's biggest cost. Pre-computing N rotation angles ×
# a few maturity sizes lets _draw_creature do a single blit instead.

CACHE_ANGLES = 36          # 10° per bucket — visually indistinguishable from continuous
CACHE_MATURITY_BUCKETS = 4  # newborn / juvenile / adolescent / adult


class SpriteCache:
    """Pre-rotates each creature sprite at fixed angle steps and a few size
    buckets, then serves them by lookup. Avoids per-frame transforms.
    Behavior: roughly 5-10x faster than pygame.transform.rotate per draw call.
    """

    def __init__(self, assets: "Assets"):
        self.assets = assets
        self._cache: dict = {}  # (species_name, angle_bucket, maturity_bucket) -> Surface
        self._build()

    def _build(self):
        for species in ("HERBIVORE", "CARNIVORE"):
            base = self.assets.get_creature_sprite(species)
            sw, sh = base.get_size()
            for m_bucket in range(CACHE_MATURITY_BUCKETS):
                # maturity 0..1 mapped to 0.55..1.0 (children smaller, adults full)
                m_frac = m_bucket / max(1, CACHE_MATURITY_BUCKETS - 1)
                scale = 0.55 + 0.45 * m_frac                
                tw =  max(8, int(sw * scale))
                th = max(8, int(sh * scale))
                scaled = pygame.transform.smoothscale(base, (tw, th))
                for a_bucket in range(CACHE_ANGLES):
                    deg = -360.0 * a_bucket / CACHE_ANGLES - 90
                    rotated = pygame.transform.rotate(scaled, deg)
                    self._cache[(species, a_bucket, m_bucket)] = rotated

    def get(self, species_name: str, angle_rad: float, maturity: float) -> pygame.Surface:
        # bucket the angle into [0, CACHE_ANGLES)
        a = (angle_rad / (2.0 * math.pi)) % 1.0
        a_bucket = int(a * CACHE_ANGLES) % CACHE_ANGLES
        m_bucket = max(0, min(CACHE_MATURITY_BUCKETS - 1,
                              int(maturity * CACHE_MATURITY_BUCKETS)))
        return self._cache[(species_name, a_bucket, m_bucket)]
