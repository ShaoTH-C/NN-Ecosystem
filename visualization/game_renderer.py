# game_renderer.py - the "god game" view of the ecosystem.
#
# this is the design-focused renderer: pretty background, sprite-based
# creatures, particle effects, a left-hand sidebar of divine tools, and a
# stylized HUD. press TAB to swap to the technical/debug renderer.

import math
import time
import pygame
import pygame.gfxdraw as gfx
import numpy as np
from typing import List, Tuple, Optional

from core.creature import Creature, Species
from core.world import World
from core.sound import get_sounds
from visualization.assets import Assets, SpriteCache, CACHE_MATURITY_BUCKETS
import config as cfg


# ---- god-tool definitions (sidebar buttons) ----
TOOLS = [
    {"id": "food",     "label": "Bring Forth Food",     "hotkey": "1",
     "subtitle": "Drop a grove of plants",       "color": (130, 220, 110)},
    {"id": "herb",     "label": "Manifest Herbivore",   "hotkey": "2",
     "subtitle": "Spawn a peaceful grazer",      "color": (90, 200, 130)},
    {"id": "carn",     "label": "Manifest Predator",    "hotkey": "3",
     "subtitle": "Spawn a hungry hunter",        "color": (220, 90, 90)},
    {"id": "blessing", "label": "Sun's Blessing",       "hotkey": "4",
     "subtitle": "Heal all in the light",        "color": (255, 220, 130)},
    {"id": "rain",     "label": "Summon Rain",          "hotkey": "5",
     "subtitle": "World-wide food bloom",        "color": (130, 180, 240)},
    {"id": "disaster", "label": "Strike a Meteor",      "hotkey": "6",
     "subtitle": "Burn an area to ash",          "color": (255, 130, 60)},
    {"id": "plague",   "label": "Cast a Plague",        "hotkey": "7",
     "subtitle": "Slow drain across the land",   "color": (160, 220, 90)},
]

SIDEBAR_WIDTH = 250


class GameRenderer:
    """Stylized 'god game' renderer with a left tool sidebar.

    Players act as a deity reshaping a neuro-evolutionary ecosystem:
    bring food, manifest creatures, bless or smite the land, then watch
    natural selection re-balance everything underneath.
    """

    def __init__(self, world: World):
        pygame.init()
        pygame.display.set_caption("Ecosystem: God Game")

        self.win_w = cfg.WINDOW_WIDTH
        self.win_h = cfg.WINDOW_HEIGHT
        self.screen = pygame.display.set_mode(
            (self.win_w, self.win_h), pygame.RESIZABLE
        )
        self.clock = pygame.time.Clock()
        self.world = world
        self._rebuild_fonts()

        self.assets = Assets(sprite_size=42)
        self.sprite_cache = SpriteCache(self.assets)
        # pre-built drop-shadow per maturity bucket (avoid per-creature alloc)
        self._shadow_cache = self._build_shadow_cache()

        # rendering toggles
        self.show_sensors = False
        self.show_hud = True
        self.show_energy_bars = True
        self.selected_creature: Optional[Creature] = None
        self.paused = False
        # 1 = real-time (one sim tick per ~33ms render frame). Lowered from 2
        # because at sim_speed=2 each render frame had to crunch 2 sim ticks,
        # which dominated frame time at populated counts. Press UP to speed up.
        self.sim_speed = 1
        self.turbo = False

        # currently selected god tool (id from TOOLS), or None for "inspect"
        self.active_tool: Optional[str] = "food"

        # tooltip / button hover state
        self.hovered_tool: Optional[str] = None

        # graph history
        self.herb_history: List[int] = []
        self.carn_history: List[int] = []
        self.food_history: List[int] = []
        self.graph_max_points = 240

        # subtle global animation phase
        self.anim_t = 0.0

        # perf counters: render FPS comes from the pygame clock; sim TPS is
        # measured by sampling world.tick over a sliding ~0.5s window so it
        # reflects actual wall-clock simulation rate (not target_TPS).
        self._sim_tps = 0.0
        self._tps_window_start = time.time()
        self._tps_window_ticks = world.tick

        # render caches — see _build_static_text() / _build_sidebar_surface()
        self._static_text: dict = {}
        self._sidebar_surface: Optional[pygame.Surface] = None
        self._sidebar_cache_key = None
        self._hud_panel: Optional[pygame.Surface] = None
        self._graph_panel: Optional[pygame.Surface] = None
        self._log_panel: Optional[pygame.Surface] = None
        self._build_static_text()

    # --- fonts & layout helpers ---

    def _rebuild_fonts(self):
        self.font_tiny = pygame.font.SysFont("segoeui", 11)
        self.font_small = pygame.font.SysFont("segoeui", 13)
        self.font_medium = pygame.font.SysFont("segoeui", 16)
        self.font_large = pygame.font.SysFont("segoeui", 20, bold=True)
        self.font_title = pygame.font.SysFont("georgia", 26, bold=True)
        self.font_button = pygame.font.SysFont("segoeui", 14, bold=True)

    def _build_static_text(self):
        """Pre-render every label that never changes its string. Saves dozens
        of font.render() calls per frame — fonts are surprisingly expensive."""
        T = self._static_text
        T["title_eco"] = self.font_title.render("ECOSYSTEM", True, cfg.GAME_TITLE_GOLD)
        T["title_sub"] = self.font_small.render("a god game", True, cfg.GAME_SIDEBAR_TEXT)
        T["section_tools"] = self.font_small.render("DIVINE TOOLS", True, (200, 180, 140))
        T["section_controls"] = self.font_small.render("CONTROLS", True, (200, 180, 140))
        T["legend_lines"] = [
            self.font_tiny.render(line, True, cfg.GAME_SIDEBAR_TEXT)
            for line in (
                "1-7  pick a tool",
                "Click  use tool",
                "Right-click  inspect",
                "Space  pause   F  turbo",
                "Tab  debug view   Q  quit",
            )
        ]
        # per-tool: pre-render (label, subtitle, hotkey badge text). Background
        # color still varies with hover/active state, so we composite per cache.
        T["tool"] = {}
        for tool in TOOLS:
            T["tool"][tool["id"]] = {
                "label": self.font_button.render(tool["label"], True, cfg.GAME_SIDEBAR_TEXT),
                "subtitle": self.font_tiny.render(tool["subtitle"], True, (180, 165, 140)),
                "hotkey": self.font_small.render(tool["hotkey"], True, cfg.GAME_SIDEBAR_TEXT),
                # smaller numeral that fits inside the corner badge on the icon
                "hotkey_small": self.font_tiny.render(tool["hotkey"], True, cfg.GAME_SIDEBAR_TEXT),
            }
        # HUD static labels
        T["hud_title"] = self.font_large.render("THE WORLD", True, cfg.GAME_TITLE_GOLD)
        for k in ("Herbivores", "Predators", "Plants", "Generation (H/P)",
                  "Total births", "Total deaths", "Divine acts"):
            T[f"hud_lbl_{k}"] = self.font_small.render(k, True, (200, 195, 180))
        T["graph_title"] = self.font_small.render(
            "POPULATION OVER TIME", True, (200, 180, 140)
        )
        T["log_title"] = self.font_small.render(
            "DIVINE INTERVENTIONS", True, (200, 180, 140)
        )
        T["log_empty"] = self.font_tiny.render(
            "(use a tool to shape the world)", True, (160, 150, 130)
        )

    def _invalidate_panel_caches(self):
        """Force panels to be rebuilt next frame (call on resize)."""
        self._sidebar_surface = None
        self._sidebar_cache_key = None
        self._hud_panel = None
        self._graph_panel = None
        self._log_panel = None

    def _get_alpha_panel(self, attr: str, w: int, h: int) -> pygame.Surface:
        """Return a cached SRCALPHA panel of the given size, rebuilding if
        size changed. Eliminates one Surface alloc per HUD draw per frame."""
        cached = getattr(self, attr)
        if cached is None or cached.get_size() != (w, h):
            surf = pygame.Surface((w, h), pygame.SRCALPHA)
            surf.fill(cfg.GAME_PANEL_BG)
            setattr(self, attr, surf)
            return surf
        return cached

    @property
    def world_view_x(self) -> int:
        return SIDEBAR_WIDTH

    @property
    def world_view_w(self) -> int:
        return max(100, self.win_w - SIDEBAR_WIDTH)

    def _world_to_screen(self, wx: float, wy: float) -> Tuple[int, int]:
        sx = self.world_view_x + int(wx * self.world_view_w / cfg.WORLD_WIDTH)
        sy = int(wy * self.win_h / cfg.WORLD_HEIGHT)
        return sx, sy

    def _screen_to_world(self, sx: int, sy: int) -> Tuple[float, float]:
        wx = (sx - self.world_view_x) * cfg.WORLD_WIDTH / max(1, self.world_view_w)
        wy = sy * cfg.WORLD_HEIGHT / max(1, self.win_h)
        return wx, wy

    def _scale(self, val: float) -> int:
        return max(1, int(val * min(self.world_view_w / cfg.WORLD_WIDTH,
                                     self.win_h / cfg.WORLD_HEIGHT)))

    # --- main loop ---

    def process_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            self._handle_event(event)
        return True

    def render(self) -> bool:
        if not self.process_events():
            return False

        self.anim_t += 0.05

        # update sim-TPS sample once the window is wide enough
        elapsed = time.time() - self._tps_window_start
        if elapsed >= 0.5:
            self._sim_tps = (self.world.tick - self._tps_window_ticks) / elapsed
            self._tps_window_start = time.time()
            self._tps_window_ticks = self.world.tick

        # 1. world background fills the play area only
        bg = self.assets.background(self.world_view_w, self.win_h)
        self.screen.blit(bg, (self.world_view_x, 0))

        # subtle grid in play area
        self._draw_world_grid()

        self._draw_food()
        self._draw_creatures()
        self._draw_particles()
        self._draw_active_effects()

        # cursor preview ring for the active tool
        self._draw_tool_cursor()

        # left sidebar (drawn last over the play area at x=0)
        self._draw_sidebar()

        # right HUD + selection panel
        if self.show_hud:
            self._draw_hud()
            self._draw_population_graph()
            self._draw_divine_log()

        if self.selected_creature and self.selected_creature.alive:
            self._draw_selected_info()

        if self.paused:
            self._draw_paused_overlay()

        pygame.display.flip()
        if self.turbo:
            self.clock.tick(0)
        else:
            self.clock.tick(cfg.FPS)
        return True

    # --- input ---

    def _handle_event(self, event: pygame.event.Event):
        if event.type == pygame.VIDEORESIZE:
            self.win_w = max(event.w, 800)
            self.win_h = max(event.h, 500)
            self.screen = pygame.display.set_mode(
                (self.win_w, self.win_h), pygame.RESIZABLE
            )
            self.assets._bg_cache = None  # force regen at new size
            self._invalidate_panel_caches()

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                self.paused = not self.paused
            elif event.key == pygame.K_h:
                self.show_hud = not self.show_hud
            elif event.key == pygame.K_s:
                self.show_sensors = not self.show_sensors
            elif event.key == pygame.K_e:
                self.show_energy_bars = not self.show_energy_bars
            elif event.key == pygame.K_f:
                self.turbo = not self.turbo
                if self.turbo:
                    self.sim_speed = 20
            elif event.key == pygame.K_UP:
                self.sim_speed = min(self.sim_speed + 1, 50)
            elif event.key == pygame.K_DOWN:
                self.sim_speed = max(self.sim_speed - 1, 1)
            elif event.key == pygame.K_ESCAPE:
                self.selected_creature = None
                self.active_tool = None
            # tool hotkeys 1-7
            elif event.unicode in {"1", "2", "3", "4", "5", "6", "7"}:
                idx = int(event.unicode) - 1
                if 0 <= idx < len(TOOLS):
                    self.active_tool = TOOLS[idx]["id"]
                    get_sounds().play_sfx("click")

        elif event.type == pygame.MOUSEMOTION:
            self.hovered_tool = self._tool_at(event.pos)

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                self._handle_left_click(event.pos)
            elif event.button == 3:
                # right-click always inspects, regardless of active tool
                if event.pos[0] >= self.world_view_x:
                    self._select_creature(event.pos)

    def _handle_left_click(self, pos: Tuple[int, int]):
        # sidebar button click
        clicked_tool = self._tool_at(pos)
        if clicked_tool is not None:
            self.active_tool = clicked_tool
            get_sounds().play_sfx("click")
            return
        # world click — apply active tool, or inspect if no tool
        if pos[0] < self.world_view_x:
            return
        wx, wy = self._screen_to_world(*pos)
        if self.active_tool is None:
            self._select_creature(pos)
            return
        self._apply_tool(self.active_tool, wx, wy)

    def _apply_tool(self, tool_id: str, wx: float, wy: float):
        if tool_id == "food":
            self.world.divine_drop_food(wx, wy)
        elif tool_id == "herb":
            self.world.divine_spawn_creature(wx, wy, Species.HERBIVORE)
        elif tool_id == "carn":
            self.world.divine_spawn_creature(wx, wy, Species.CARNIVORE)
        elif tool_id == "blessing":
            self.world.divine_blessing(wx, wy)
        elif tool_id == "rain":
            self.world.divine_rain()
        elif tool_id == "disaster":
            self.world.divine_disaster(wx, wy)
        elif tool_id == "plague":
            self.world.divine_plague(wx, wy)
        # play matching sfx (no-op if sound disabled / unknown id).
        get_sounds().play_sfx(tool_id)

    def _tool_at(self, pos: Tuple[int, int]) -> Optional[str]:
        x, y = pos
        if x > SIDEBAR_WIDTH:
            return None
        # buttons start beneath the title block
        btn_top = 130
        btn_h = 62
        btn_gap = 6
        for i, tool in enumerate(TOOLS):
            by = btn_top + i * (btn_h + btn_gap)
            if 12 <= x <= SIDEBAR_WIDTH - 12 and by <= y <= by + btn_h:
                return tool["id"]
        return None

    def _select_creature(self, pos: Tuple[int, int]):
        mx, my = pos
        best_creature = None
        best_dist = 35
        for creature in self.world.creatures:
            sx, sy = self._world_to_screen(creature.x, creature.y)
            d = math.hypot(sx - mx, sy - my)
            if d < best_dist:
                best_dist = d
                best_creature = creature
        self.selected_creature = best_creature

    # --- world rendering ---

    def _draw_world_grid(self):
        spacing = 100
        for x in range(self.world_view_x, self.win_w, spacing):
            pygame.draw.line(self.screen, cfg.GAME_GRID_COLOR, (x, 0), (x, self.win_h), 1)
        for y in range(0, self.win_h, spacing):
            pygame.draw.line(self.screen, cfg.GAME_GRID_COLOR,
                             (self.world_view_x, y), (self.win_w, y), 1)

    def _draw_food(self):
        sprite = self.assets.food
        sw, sh = sprite.get_size()
        for food in self.world.food_items:
            if not food.alive:
                continue
            sx, sy = self._world_to_screen(food.x, food.y)
            self.screen.blit(sprite, (sx - sw // 2, sy - sh // 2))

    def _draw_creatures(self):
        for creature in self.world.creatures:
            if not creature.alive:
                continue
            self._draw_creature(creature)

    def _build_shadow_cache(self) -> list:
        """Pre-render one drop-shadow per maturity bucket to avoid per-creature alloc."""
        shadows = []
        base_size = self.assets.sprite_size
        for m in range(CACHE_MATURITY_BUCKETS):
            scale = 0.55 + 0.45 * (m / max(1, CACHE_MATURITY_BUCKETS - 1))
            w = max(8, int(base_size * scale))
            h = max(3, int(base_size * scale / 3))
            surf = pygame.Surface((w, h), pygame.SRCALPHA)
            pygame.draw.ellipse(surf, (0, 0, 0, 60), surf.get_rect())
            shadows.append(surf)
        return shadows

    def _draw_creature(self, creature: Creature):
        cx, cy = self._world_to_screen(creature.x, creature.y)
        # cache lookup instead of smoothscale + rotate per frame
        rotated = self.sprite_cache.get(
            creature.species.name, creature.angle, creature.maturity,
        )
        rw, rh = rotated.get_size()

        # selection halo (only for the one selected creature, so per-frame cost is fine)
        if creature is self.selected_creature:
            halo_r = max(rw, rh) // 2 + 8
            halo_surf = pygame.Surface((halo_r * 2 + 8, halo_r * 2 + 8), pygame.SRCALPHA)
            for r in range(halo_r, halo_r - 6, -1):
                a = int(160 * (1 - (halo_r - r) / 6))
                pygame.draw.circle(halo_surf, (255, 230, 100, a),
                                   (halo_r + 4, halo_r + 4), r, 2)
            self.screen.blit(halo_surf, (cx - halo_r - 4, cy - halo_r - 4))

        # cached drop-shadow (CACHE_MATURITY_BUCKETS imported at module top)
        m_bucket = max(0, min(CACHE_MATURITY_BUCKETS - 1,
                              int(creature.maturity * CACHE_MATURITY_BUCKETS)))
        shadow = self._shadow_cache[m_bucket]
        sw, sh = shadow.get_size()
        self.screen.blit(shadow, (cx - sw // 2, cy + sh))

        self.screen.blit(rotated, (cx - rw // 2, cy - rh // 2))

        # for energy-bar sizing below, recover the maturity-based target size
        target_w = int(self.assets.sprite_size * (0.55 + 0.45 * creature.maturity))
        target_h = target_w

        # sensors when toggled or for selected creature
        if self.show_sensors or creature is self.selected_creature:
            self._draw_sensors(creature)

        # energy bar
        if self.show_energy_bars:
            bar_w = max(20, target_w)
            bar_h = 4
            bar_x = cx - bar_w // 2
            bar_y = cy - target_h // 2 - 8
            energy_frac = max(0.0, min(1.0, creature.energy / cfg.MAX_ENERGY))
            pygame.draw.rect(self.screen, (20, 20, 20, 200),
                             (bar_x - 1, bar_y - 1, bar_w + 2, bar_h + 2),
                             border_radius=2)
            pygame.draw.rect(self.screen, (60, 60, 60),
                             (bar_x, bar_y, bar_w, bar_h), border_radius=2)
            fill_w = int(bar_w * energy_frac)
            if fill_w > 0:
                fill_color = (
                    int(255 * (1 - energy_frac)),
                    int(255 * energy_frac),
                    50,
                )
                pygame.draw.rect(self.screen, fill_color,
                                 (bar_x, bar_y, fill_w, bar_h), border_radius=2)

        # newborn indicator: little crown for very young creatures
        if creature.maturity < 0.3:
            ind_y = cy - target_h // 2 - 18
            pygame.draw.circle(self.screen, (255, 240, 200), (cx, ind_y), 3)

    def _draw_sensors(self, creature: Creature):
        cx, cy = self._world_to_screen(creature.x, creature.y)
        for endpoint in creature.sensor_endpoints:
            ex, ey = self._world_to_screen(endpoint[0], endpoint[1])
            pygame.draw.line(self.screen, (255, 255, 255, 60), (cx, cy), (ex, ey), 1)

    def _draw_particles(self):
        # gfxdraw.filled_circle does alpha blending directly on the screen,
        # so we skip the per-particle SRCALPHA Surface allocation that was
        # this function's biggest cost.
        screen = self.screen
        for p in self.world.particles:
            life_frac = p["life"] / max(1, p["max_life"])
            sx, sy = self._world_to_screen(p["x"], p["y"])
            cr, cg, cb = p["color"]
            kind = p["kind"]
            if kind == "sparkle":
                r = max(1, int(3 * life_frac))
                gfx.filled_circle(screen, sx, sy, r, (cr, cg, cb, int(220 * life_frac)))
            elif kind == "fire":
                r = max(2, int(5 * life_frac))
                gfx.filled_circle(screen, sx, sy, r, (cr, cg, cb, int(200 * life_frac)))
                gfx.filled_circle(screen, sx, sy, max(1, r // 2),
                                  (255, 220, 100, int(180 * life_frac)))
            elif kind == "halo":
                r = max(2, int(4 * life_frac))
                gfx.filled_circle(screen, sx, sy, r, (cr, cg, cb, int(180 * life_frac)))
            elif kind == "plague":
                r = max(2, int(4 * life_frac))
                gfx.filled_circle(screen, sx, sy, r, (cr, cg, cb, int(140 * life_frac)))

    def _draw_active_effects(self):
        """Persistent area indicators for plagues, etc."""
        for eff in self.world.active_effects:
            if eff["kind"] == "plague":
                cx, cy = self._world_to_screen(eff["x"], eff["y"])
                r = self._scale(eff["radius"])
                pulse = 0.6 + 0.4 * math.sin(self.anim_t * 2)
                surf = pygame.Surface((r * 2 + 4, r * 2 + 4), pygame.SRCALPHA)
                pygame.draw.circle(surf, (140, 200, 80, int(35 * pulse)),
                                   (r + 2, r + 2), r)
                pygame.draw.circle(surf, (140, 200, 80, 90),
                                   (r + 2, r + 2), r, 2)
                self.screen.blit(surf, (cx - r - 2, cy - r - 2))
            elif eff["kind"] == "rain":
                # falling streaks across the world view
                for _ in range(6):
                    rx = np.random.randint(self.world_view_x, self.win_w)
                    ry = np.random.randint(0, self.win_h)
                    pygame.draw.line(
                        self.screen, (160, 200, 240, 180),
                        (rx, ry), (rx - 3, ry + 12), 1,
                    )
            elif eff["kind"] in ("disaster", "blessing"):
                cx, cy = self._world_to_screen(eff["x"], eff["y"])
                r = self._scale(eff["radius"])
                life = eff["ticks_remaining"] / 40
                color = (255, 120, 50) if eff["kind"] == "disaster" else (255, 240, 150)
                surf = pygame.Surface((r * 2 + 6, r * 2 + 6), pygame.SRCALPHA)
                pygame.draw.circle(surf, (*color, int(120 * life)),
                                   (r + 3, r + 3), r, 3)
                self.screen.blit(surf, (cx - r - 3, cy - r - 3))

    def _draw_tool_cursor(self):
        if self.active_tool is None:
            return
        mx, my = pygame.mouse.get_pos()
        if mx < self.world_view_x:
            return
        radius_map = {
            "food": cfg.DROP_FOOD_SPREAD,
            "blessing": cfg.BLESSING_RADIUS,
            "disaster": cfg.DISASTER_RADIUS,
            "plague": cfg.PLAGUE_RADIUS,
        }
        if self.active_tool not in radius_map:
            # spawn tools just show a dot
            pygame.draw.circle(self.screen, (255, 255, 255), (mx, my), 6, 2)
            return
        wr = radius_map[self.active_tool]
        sr = self._scale(wr)
        color = next((t["color"] for t in TOOLS if t["id"] == self.active_tool), (255, 255, 255))
        ring = pygame.Surface((sr * 2 + 6, sr * 2 + 6), pygame.SRCALPHA)
        pulse = 0.7 + 0.3 * math.sin(self.anim_t * 4)
        pygame.draw.circle(ring, (*color, int(180 * pulse)),
                           (sr + 3, sr + 3), sr, 2)
        pygame.draw.circle(ring, (*color, 40), (sr + 3, sr + 3), sr)
        self.screen.blit(ring, (mx - sr - 3, my - sr - 3))

    # --- sidebar ---

    def _draw_sidebar(self):
        """Cache the entire sidebar as one surface, keyed by (size, active,
        hovered). Sidebar previously cost ~7 SRCALPHA allocs + ~25 font
        renders per frame; now it's one blit unless state changes."""
        key = (self.win_h, self.active_tool, self.hovered_tool)
        if self._sidebar_surface is None or self._sidebar_cache_key != key:
            self._sidebar_surface = self._build_sidebar_surface()
            self._sidebar_cache_key = key
        self.screen.blit(self._sidebar_surface, (0, 0))

    def _build_sidebar_surface(self) -> pygame.Surface:
        T = self._static_text
        surf = pygame.Surface((SIDEBAR_WIDTH, self.win_h), pygame.SRCALPHA)
        surf.fill(cfg.GAME_SIDEBAR_BG)
        pygame.draw.line(surf, cfg.GAME_SIDEBAR_ACCENT,
                         (SIDEBAR_WIDTH - 1, 0), (SIDEBAR_WIDTH - 1, self.win_h), 2)

        # title block
        title = T["title_eco"]
        subtitle = T["title_sub"]
        surf.blit(title, ((SIDEBAR_WIDTH - title.get_width()) // 2, 16))
        surf.blit(subtitle, ((SIDEBAR_WIDTH - subtitle.get_width()) // 2, 50))
        pygame.draw.line(surf, cfg.GAME_SIDEBAR_ACCENT,
                         (28, 78), (SIDEBAR_WIDTH - 28, 78), 1)

        # section label + rule
        surf.blit(T["section_tools"], (20, 96))
        pygame.draw.line(surf, (120, 100, 70),
                         (20, 116), (SIDEBAR_WIDTH - 20, 116), 1)

        # tool buttons
        btn_top = 130
        btn_h = 62
        btn_gap = 6
        for i, tool in enumerate(TOOLS):
            self._render_tool_button(surf, tool, 12, btn_top + i * (btn_h + btn_gap),
                                     SIDEBAR_WIDTH - 24, btn_h)

        # bottom legend
        legend_y = self.win_h - 130
        pygame.draw.line(surf, (120, 100, 70),
                         (20, legend_y - 8), (SIDEBAR_WIDTH - 20, legend_y - 8), 1)
        surf.blit(T["section_controls"], (20, legend_y))
        for i, line_surf in enumerate(T["legend_lines"]):
            surf.blit(line_surf, (20, legend_y + 20 + i * 14))

        return surf

    def _render_tool_button(self, surf, tool: dict, x: int, y: int, w: int, h: int):
        """Composite one tool button onto the sidebar surface."""
        is_active = self.active_tool == tool["id"]
        is_hover = self.hovered_tool == tool["id"]
        if is_active:
            bg_color = (90, 75, 110, 250)
        elif is_hover:
            bg_color = (75, 62, 95, 240)
        else:
            bg_color = (60, 50, 75, 230)
        # background (small per-button alloc, but only on cache rebuilds)
        bg = pygame.Surface((w, h), pygame.SRCALPHA)
        bg.fill(bg_color)
        surf.blit(bg, (x, y))

        pygame.draw.rect(surf, tool["color"], (x, y, 4, h))
        border_col = cfg.GAME_SIDEBAR_ACCENT if is_active else (90, 80, 100)
        pygame.draw.rect(surf, border_col, (x, y, w, h), 1)

        # icon (or hotkey badge fallback when the asset is missing)
        cached = self._static_text["tool"][tool["id"]]
        icon = self._tool_icon(tool["id"])
        icon_cx = x + 26
        icon_cy = y + h // 2
        if icon is not None:
            iw, ih = icon.get_size()
            surf.blit(icon, (icon_cx - iw // 2, icon_cy - ih // 2))
            # tiny hotkey badge in the bottom-right of the icon
            badge_r = 8
            bx = icon_cx + iw // 2 - 2
            by = icon_cy + ih // 2 - 2
            pygame.draw.circle(surf, (35, 30, 45), (bx, by), badge_r)
            pygame.draw.circle(surf, tool["color"], (bx, by), badge_r, 1)
            ht = cached["hotkey_small"]
            surf.blit(ht, (bx - ht.get_width() // 2, by - ht.get_height() // 2))
        else:
            badge_r = 11
            pygame.draw.circle(surf, (35, 30, 45), (icon_cx, icon_cy), badge_r)
            pygame.draw.circle(surf, tool["color"], (icon_cx, icon_cy), badge_r, 2)
            ht = cached["hotkey"]
            surf.blit(ht, (icon_cx - ht.get_width() // 2,
                           icon_cy - ht.get_height() // 2))

        # label + subtitle (pre-rendered in _build_static_text)
        surf.blit(cached["label"], (x + 50, y + 10))
        surf.blit(cached["subtitle"], (x + 50, y + 32))

    def _tool_icon(self, tool_id: str) -> Optional[pygame.Surface]:
        """Look up the loaded icon Surface for a tool id."""
        return {
            "food": self.assets.icon_food,
            "herb": self.assets.icon_herb,
            "carn": self.assets.icon_carn,
            "blessing": self.assets.icon_blessing,
            "rain": self.assets.icon_rain,
            "disaster": self.assets.icon_disaster,
            "plague": self.assets.icon_plague,
        }.get(tool_id)

    # --- HUD ---

    def _draw_hud(self):
        stats = self.world.get_stats()

        self.herb_history.append(stats["herbivores"])
        self.carn_history.append(stats["carnivores"])
        self.food_history.append(stats["food"])
        if len(self.herb_history) > self.graph_max_points:
            self.herb_history.pop(0)
            self.carn_history.pop(0)
            self.food_history.pop(0)

        hud_w = 270
        hud_h = 230
        hud_x = self.win_w - hud_w - 12
        hud_y = 12

        panel = self._get_alpha_panel("_hud_panel", hud_w, hud_h)
        self.screen.blit(panel, (hud_x, hud_y))
        pygame.draw.rect(self.screen, cfg.GAME_PANEL_BORDER,
                         (hud_x, hud_y, hud_w, hud_h), 1)

        # title bar (pre-rendered)
        self.screen.blit(self._static_text["hud_title"], (hud_x + 14, hud_y + 8))
        tick_text = self.font_tiny.render(
            f"day {stats['tick'] // 100:,}  ·  tick {stats['tick']:,}",
            True, (180, 170, 150),
        )
        self.screen.blit(tick_text, (hud_x + 14, hud_y + 32))

        # perf line: render fps + sim tps. green when sim keeps up, amber when
        # it's falling behind the wall-clock target, red when it's <half target.
        render_fps = self.clock.get_fps()
        target_tps = cfg.BASE_SIM_TPS * self.sim_speed
        if self._sim_tps >= target_tps * 0.85:
            tps_color = (140, 220, 140)
        elif self._sim_tps >= target_tps * 0.5:
            tps_color = (240, 200, 120)
        else:
            tps_color = (240, 130, 130)
        perf = self.font_tiny.render(
            f"{render_fps:4.1f} fps render  ·  {self._sim_tps:5.1f} / {target_tps:.0f} sim tps",
            True, tps_color,
        )
        self.screen.blit(perf, (hud_x + 14, hud_y + 44))

        y = hud_y + 56
        line_h = 19
        T = self._static_text

        def line(label_key, val, color):
            self.screen.blit(T[f"hud_lbl_{label_key}"], (hud_x + 16, y))
            val_surf = self.font_small.render(str(val), True, color)
            self.screen.blit(val_surf, (hud_x + hud_w - 16 - val_surf.get_width(), y))

        line("Herbivores", stats["herbivores"], cfg.HERBIVORE_COLOR); y += line_h
        line("Predators", stats["carnivores"], cfg.CARNIVORE_COLOR); y += line_h
        line("Plants", stats["food"], cfg.COLOR_FOOD); y += line_h + 4
        line("Generation (H/P)",
             f"{stats['max_gen_herb']}/{stats['max_gen_carn']}",
             (220, 220, 230)); y += line_h
        line("Total births", f"{stats['total_births']:,}", (180, 220, 180)); y += line_h
        line("Total deaths", f"{stats['total_deaths']:,}", (220, 180, 180)); y += line_h
        line("Divine acts", f"{self.world.divine_action_count}", cfg.GAME_TITLE_GOLD); y += line_h

    def _draw_population_graph(self):
        graph_w = 270
        graph_h = 110
        graph_x = self.win_w - graph_w - 12
        graph_y = 252

        panel = self._get_alpha_panel("_graph_panel", graph_w, graph_h)
        self.screen.blit(panel, (graph_x, graph_y))
        pygame.draw.rect(self.screen, cfg.GAME_PANEL_BORDER,
                         (graph_x, graph_y, graph_w, graph_h), 1)

        self.screen.blit(self._static_text["graph_title"], (graph_x + 14, graph_y + 6))

        if len(self.herb_history) < 2:
            return

        all_vals = self.herb_history + self.carn_history + self.food_history
        max_val = max(max(all_vals), 1)
        pad_top = 26
        pad_bot = 8
        usable_h = graph_h - pad_top - pad_bot

        def plot(history, color):
            n = len(history)
            step = max(1, (graph_w - 28) / max(n - 1, 1))
            pts = []
            for i, v in enumerate(history):
                px = graph_x + 14 + int(i * step)
                py = graph_y + pad_top + usable_h - int((v / max_val) * usable_h)
                pts.append((px, py))
            if len(pts) > 1:
                pygame.draw.lines(self.screen, color, False, pts, 2)

        plot(self.food_history, cfg.COLOR_GRAPH_FOOD)
        plot(self.herb_history, cfg.COLOR_GRAPH_HERB)
        plot(self.carn_history, cfg.COLOR_GRAPH_CARN)

    def _draw_divine_log(self):
        log_w = 270
        log_h = 150
        log_x = self.win_w - log_w - 12
        log_y = 374

        panel = self._get_alpha_panel("_log_panel", log_w, log_h)
        self.screen.blit(panel, (log_x, log_y))
        pygame.draw.rect(self.screen, cfg.GAME_PANEL_BORDER,
                         (log_x, log_y, log_w, log_h), 1)

        self.screen.blit(self._static_text["log_title"], (log_x + 14, log_y + 6))

        log = self.world.divine_log[-7:]
        if not log:
            self.screen.blit(self._static_text["log_empty"], (log_x + 14, log_y + 30))
            return
        y = log_y + 28
        for entry in log:
            text = self.font_tiny.render(entry, True, cfg.GAME_SIDEBAR_TEXT)
            self.screen.blit(text, (log_x + 14, y))
            y += 16

    def _draw_selected_info(self):
        c = self.selected_creature
        if c is None or not c.alive:
            return
        panel_w = 250
        panel_h = 180
        panel_x = self.world_view_x + 12
        panel_y = self.win_h - panel_h - 12

        panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        panel.fill(cfg.GAME_PANEL_BG)
        self.screen.blit(panel, (panel_x, panel_y))
        pygame.draw.rect(self.screen, cfg.GAME_PANEL_BORDER,
                         (panel_x, panel_y, panel_w, panel_h), 1)

        species_name = c.species.name.title()
        title = self.font_medium.render(
            f"{species_name} · gen {c.genome.generation}", True, c.color
        )
        self.screen.blit(title, (panel_x + 14, panel_y + 8))

        y = panel_y + 32
        lines = [
            f"Energy   {c.energy:5.1f} / {cfg.MAX_ENERGY:.0f}",
            f"Age      {c.age} / {c.lifespan}",
            f"Maturity {c.maturity:.0%}",
            f"Speed    {c.speed:.2f} / {c.max_speed:.2f}",
            f"Children {c.children_count}",
            f"Food     {c.food_eaten}    Kills {c.kills}",
            f"Brain    {c.genome.layer_sizes}",
        ]
        for line in lines:
            text = self.font_small.render(line, True, cfg.GAME_SIDEBAR_TEXT)
            self.screen.blit(text, (panel_x + 14, y))
            y += 18

    def _draw_paused_overlay(self):
        overlay = pygame.Surface((self.world_view_w, 50), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 140))
        self.screen.blit(overlay, (self.world_view_x, self.win_h // 2 - 25))
        text = self.font_title.render("PAUSED", True, cfg.GAME_TITLE_GOLD)
        self.screen.blit(text, (self.world_view_x + self.world_view_w // 2 - text.get_width() // 2,
                                self.win_h // 2 - text.get_height() // 2))

    def close(self):
        pygame.quit()
