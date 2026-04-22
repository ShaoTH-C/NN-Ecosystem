# intro.py — title screen for "Ecosystem: a god game".
#
# A standalone scene: builds its own pygame surface, draws an animated title +
# BEGIN button, and blocks until the user clicks BEGIN, presses Enter/Space,
# or closes the window. Returns "start" or "quit" so main.py can decide what
# to do next.

import math
import random
import time
from typing import List, Tuple

import pygame

import config as cfg
from core.sound import get_sounds


_TAGLINES = [
    "shape life.  teach it.  unmake it.",
    "every creature thinks for itself.",
    "you are the weather, the harvest, the plague.",
]


class _Drifter:
    """A slow-floating particle for the title background."""

    __slots__ = ("x", "y", "vx", "vy", "r", "alpha", "color")

    def __init__(self, w: int, h: int):
        self.x = random.uniform(0, w)
        self.y = random.uniform(0, h)
        self.vx = random.uniform(-0.15, 0.15)
        self.vy = random.uniform(-0.25, -0.05)
        self.r = random.uniform(1.8, 4.2)
        self.alpha = random.randint(60, 160)
        # tint: gold or pale green
        self.color = random.choice([
            (240, 210, 130),
            (180, 230, 170),
            (210, 240, 200),
        ])

    def step(self, w: int, h: int):
        self.x += self.vx
        self.y += self.vy
        if self.y < -10 or self.x < -10 or self.x > w + 10:
            self.x = random.uniform(0, w)
            self.y = h + random.uniform(0, 30)


def _blit_alpha_circle(surf: pygame.Surface, color, pos, radius, alpha):
    s = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
    pygame.draw.circle(s, (*color, alpha), (radius, radius), radius)
    surf.blit(s, (pos[0] - radius, pos[1] - radius))


def _gradient_bg(w: int, h: int) -> pygame.Surface:
    """Vertical gradient: deep teal-purple at top → moss green at bottom."""
    surf = pygame.Surface((w, h))
    top = (28, 36, 52)
    bot = (50, 78, 60)
    for y in range(h):
        t = y / max(1, h - 1)
        r = int(top[0] + (bot[0] - top[0]) * t)
        g = int(top[1] + (bot[1] - top[1]) * t)
        b = int(top[2] + (bot[2] - top[2]) * t)
        pygame.draw.line(surf, (r, g, b), (0, y), (w, y))
    return surf


def _draw_button(surf, rect, label, font, hover, pressed):
    base = (60, 48, 30)
    border = (240, 200, 120)
    text_col = (250, 235, 195)
    if pressed:
        bg = (35, 28, 18)
    elif hover:
        bg = (90, 70, 40)
    else:
        bg = base
    # subtle shadow
    shadow = pygame.Rect(rect.x + 3, rect.y + 4, rect.w, rect.h)
    pygame.draw.rect(surf, (0, 0, 0, 90), shadow, border_radius=14)
    pygame.draw.rect(surf, bg, rect, border_radius=14)
    pygame.draw.rect(surf, border, rect, width=2, border_radius=14)
    label_surf = font.render(label, True, text_col)
    lr = label_surf.get_rect(center=rect.center)
    surf.blit(label_surf, lr)


def run_intro(window_size: Tuple[int, int] = None) -> str:
    """Show the title screen. Returns 'start' or 'quit'."""
    if window_size is None:
        window_size = (cfg.WINDOW_WIDTH, cfg.WINDOW_HEIGHT)
    w, h = window_size

    pygame.init()
    screen = pygame.display.set_mode((w, h))
    pygame.display.set_caption("Ecosystem — a god game")
    clock = pygame.time.Clock()

    # warm up the sound manager + start menu music if available.
    sounds = get_sounds()
    sounds.play_music(cfg.SOUND_MUSIC_INTRO, loop=True, fade_ms=1200)

    # fonts: try a serif-y system font for the title, fallback to default.
    title_font = pygame.font.SysFont("georgia,timesnewroman,serif", 96, bold=True)
    sub_font = pygame.font.SysFont("georgia,timesnewroman,serif", 30, italic=True)
    tag_font = pygame.font.SysFont("georgia,timesnewroman,serif", 22)
    btn_font = pygame.font.SysFont("georgia,timesnewroman,serif", 32, bold=True)
    hint_font = pygame.font.SysFont("consolas,couriernew,monospace", 14)
    foot_font = pygame.font.SysFont("georgia,timesnewroman,serif", 14, italic=True)

    bg = _gradient_bg(w, h)
    drifters: List[_Drifter] = [_Drifter(w, h) for _ in range(70)]
    tagline = random.choice(_TAGLINES)

    btn_w, btn_h = 240, 64
    btn_rect = pygame.Rect((w - btn_w) // 2, int(h * 0.62), btn_w, btn_h)

    started = False
    pressed = False
    t0 = time.time()
    fade_in_dur = 0.7

    while True:
        mouse = pygame.mouse.get_pos()
        hover = btn_rect.collidepoint(mouse)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.mixer.music.stop()
                return "quit"
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_RETURN, pygame.K_SPACE):
                    sounds.play_sfx("click")
                    started = True
                elif event.key in (pygame.K_q, pygame.K_ESCAPE):
                    pygame.mixer.music.stop()
                    return "quit"
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if hover:
                    pressed = True
            if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                if pressed and hover:
                    sounds.play_sfx("click")
                    started = True
                pressed = False

        # update particles
        for d in drifters:
            d.step(w, h)

        # --- draw ---
        screen.blit(bg, (0, 0))
        # particles (additive-ish glow)
        for d in drifters:
            _blit_alpha_circle(screen, d.color, (int(d.x), int(d.y)), int(d.r), d.alpha)

        # vignette
        vignette = pygame.Surface((w, h), pygame.SRCALPHA)
        for i in range(8):
            a = 14 + i * 6
            pygame.draw.rect(vignette, (0, 0, 0, a),
                             (i * 12, i * 12, w - i * 24, h - i * 24), border_radius=20)
        screen.blit(vignette, (0, 0))

        # title with gentle pulse + drop shadow
        elapsed = time.time() - t0
        pulse = 1.0 + 0.015 * math.sin(elapsed * 2.0)
        title_surf = title_font.render("ECOSYSTEM", True, (245, 215, 130))
        tw, th = title_surf.get_size()
        pulsed = pygame.transform.smoothscale(
            title_surf, (int(tw * pulse), int(th * pulse))
        )
        ts_rect = pulsed.get_rect(center=(w // 2, int(h * 0.28)))
        # shadow pass
        shadow = title_font.render("ECOSYSTEM", True, (0, 0, 0))
        s_rect = shadow.get_rect(center=(w // 2 + 4, int(h * 0.28) + 5))
        shadow.set_alpha(140)
        screen.blit(shadow, s_rect)
        screen.blit(pulsed, ts_rect)

        # subtitle
        sub_surf = sub_font.render("a god game", True, (220, 200, 170))
        screen.blit(sub_surf, sub_surf.get_rect(center=(w // 2, int(h * 0.36))))

        # tagline
        tag_surf = tag_font.render(tagline, True, (200, 220, 200))
        screen.blit(tag_surf, tag_surf.get_rect(center=(w // 2, int(h * 0.46))))

        # button
        _draw_button(screen, btn_rect, "BEGIN", btn_font, hover, pressed)

        # controls hint
        hints = [
            "1-7  pick a divine tool      click  use it",
            "TAB  swap to debug view      SPACE  pause",
            "F  turbo speed               Q  quit",
        ]
        hy = int(h * 0.78)
        for line in hints:
            hs = hint_font.render(line, True, (200, 200, 210))
            screen.blit(hs, hs.get_rect(center=(w // 2, hy)))
            hy += 22

        # footer attribution
        foot = foot_font.render(
            "0020-402  ·  design final  ·  built on a neural-evolution sim",
            True, (150, 160, 160),
        )
        screen.blit(foot, foot.get_rect(center=(w // 2, h - 24)))

        # fade-in overlay over the whole intro for the first ~0.7s
        if elapsed < fade_in_dur:
            a = int(255 * (1.0 - elapsed / fade_in_dur))
            fade = pygame.Surface((w, h))
            fade.fill((0, 0, 0))
            fade.set_alpha(a)
            screen.blit(fade, (0, 0))

        pygame.display.flip()
        clock.tick(60)

        if started:
            # quick fade-out before handing control to the game
            for i in range(20):
                fade = pygame.Surface((w, h))
                fade.fill((0, 0, 0))
                fade.set_alpha(int(255 * (i + 1) / 20))
                screen.blit(fade, (0, 0))
                pygame.display.flip()
                clock.tick(60)
            return "start"
