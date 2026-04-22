# sound.py — music + sfx for the game.
#
# Mirrors the assets pattern: try to load an audio file from
# <ASSETS_DIR>/<SOUNDS_SUBDIR>/ for each sound; if it's missing, generate a
# short synth fallback so the game still has audio feedback before the user
# adds AI-generated tracks.
#
# Pygame mixer is initialized lazily (first time SoundManager is built) so
# this module is safe to import in headless contexts.

import os
import math
import numpy as np
from typing import Dict, Optional

import pygame

import config as cfg


_MIXER_READY = False


def _ensure_mixer() -> bool:
    """Initialize pygame.mixer once. Returns True on success."""
    global _MIXER_READY
    if _MIXER_READY:
        return True
    try:
        # 44.1kHz, 16-bit signed, stereo, smaller buffer for snappier sfx
        pygame.mixer.pre_init(44100, -16, 2, 512)
        pygame.mixer.init()
        _MIXER_READY = True
        return True
    except pygame.error:
        return False


def _sounds_dir() -> str:
    base = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        cfg.ASSETS_DIR,
        cfg.SOUNDS_SUBDIR,
    )
    return base


_AUDIO_EXTS = (".ogg", ".mp3", ".wav", ".flac")


def _resolve_audio_path(filename: str) -> Optional[str]:
    """Find <sounds_dir>/<filename>, trying common audio extensions if the
    exact name is missing. Lets the user drop in .mp3, .ogg, .wav, etc.
    without editing config."""
    base = _sounds_dir()
    direct = os.path.join(base, filename)
    if os.path.isfile(direct):
        return direct
    stem, _ = os.path.splitext(filename)
    for ext in _AUDIO_EXTS:
        alt = os.path.join(base, stem + ext)
        if os.path.isfile(alt):
            return alt
    return None


def _try_load(filename: str) -> Optional[pygame.mixer.Sound]:
    path = _resolve_audio_path(filename)
    if path is None:
        return None
    try:
        return pygame.mixer.Sound(path)
    except pygame.error:
        return None


# --- procedural fallbacks ----------------------------------------------------
#
# All synths return a pygame.mixer.Sound built from a stereo int16 numpy array.
# Kept short (≤0.6s) so they don't pile up if the player spams a tool.

SAMPLE_RATE = 44100


def _to_stereo_sound(mono: np.ndarray, peak: float = 0.7) -> pygame.mixer.Sound:
    """Normalize a float array to int16 stereo and wrap as a Sound."""
    if mono.size == 0:
        mono = np.zeros(64, dtype=np.float32)
    # gentle normalization — keep transients but cap peaks
    m = np.max(np.abs(mono)) or 1.0
    mono = (mono / m) * peak
    samples = (mono * 32767).astype(np.int16)
    stereo = np.column_stack((samples, samples))
    # pygame requires C-contiguous arrays
    return pygame.sndarray.make_sound(np.ascontiguousarray(stereo))


def _envelope(n: int, attack: float = 0.01, decay: float = 0.3) -> np.ndarray:
    """Quick AR envelope (no sustain, no release — perfect for stings)."""
    a = max(1, int(attack * SAMPLE_RATE))
    d = max(1, int(decay * SAMPLE_RATE))
    env = np.zeros(n, dtype=np.float32)
    env[:a] = np.linspace(0, 1, a, dtype=np.float32)
    if a < n:
        tail = n - a
        env[a:] = np.exp(-np.linspace(0, 4, tail, dtype=np.float32))
    return env


def _tone(freq: float, dur: float, wave: str = "sine") -> np.ndarray:
    n = int(dur * SAMPLE_RATE)
    t = np.linspace(0, dur, n, endpoint=False, dtype=np.float32)
    phase = 2 * math.pi * freq * t
    if wave == "saw":
        return 2 * (t * freq - np.floor(0.5 + t * freq)).astype(np.float32)
    if wave == "square":
        return np.sign(np.sin(phase)).astype(np.float32)
    if wave == "noise":
        return np.random.uniform(-1, 1, n).astype(np.float32)
    return np.sin(phase).astype(np.float32)


def _synth_click() -> pygame.mixer.Sound:
    n = int(0.07 * SAMPLE_RATE)
    sig = _tone(900, 0.07) * _envelope(n, 0.002, 0.06)
    return _to_stereo_sound(sig, peak=0.5)


def _synth_food() -> pygame.mixer.Sound:
    # bright pluck — two-tone chirp like a wind chime
    sig = _tone(740, 0.18) + 0.6 * _tone(1100, 0.18)
    sig *= _envelope(sig.size, 0.005, 0.18)
    return _to_stereo_sound(sig, peak=0.6)


def _synth_herb() -> pygame.mixer.Sound:
    # warm rising chord (C-E-G ascending arpeggio)
    parts = []
    for f, start in [(523, 0.0), (659, 0.05), (784, 0.10)]:
        n = int(0.3 * SAMPLE_RATE)
        s = _tone(f, 0.3) * _envelope(n, 0.01, 0.25)
        # pad with leading silence
        pad = np.zeros(int(start * SAMPLE_RATE), dtype=np.float32)
        parts.append(np.concatenate([pad, s]))
    length = max(p.size for p in parts)
    out = np.zeros(length, dtype=np.float32)
    for p in parts:
        out[: p.size] += p
    return _to_stereo_sound(out, peak=0.55)


def _synth_carn() -> pygame.mixer.Sound:
    # low growl: detuned saws + slight wobble
    n = int(0.4 * SAMPLE_RATE)
    base = _tone(110, 0.4, "saw") + _tone(112, 0.4, "saw")
    wobble = 1.0 + 0.15 * np.sin(2 * math.pi * 7 * np.linspace(0, 0.4, n, dtype=np.float32))
    sig = base[:n] * wobble * _envelope(n, 0.01, 0.35)
    return _to_stereo_sound(sig, peak=0.65)


def _synth_blessing() -> pygame.mixer.Sound:
    # bell-like: a fundamental + inharmonic overtones, long-ish decay
    n = int(0.55 * SAMPLE_RATE)
    f = 660
    sig = (
        _tone(f, 0.55)
        + 0.5 * _tone(f * 2.01, 0.55)
        + 0.3 * _tone(f * 3.02, 0.55)
        + 0.2 * _tone(f * 4.05, 0.55)
    )
    sig = sig[:n] * _envelope(n, 0.005, 0.5)
    return _to_stereo_sound(sig, peak=0.55)


def _synth_rain() -> pygame.mixer.Sound:
    # soft rushing noise with a low-pass-ish character (cumulative average)
    n = int(0.6 * SAMPLE_RATE)
    noise = _tone(0, 0.6, "noise")
    # cheap one-pole lowpass
    lp = np.zeros_like(noise)
    a = 0.08
    prev = 0.0
    for i in range(n):
        prev = prev + a * (noise[i] - prev)
        lp[i] = prev
    sig = lp * _envelope(n, 0.05, 0.5)
    return _to_stereo_sound(sig, peak=0.5)


def _synth_disaster() -> pygame.mixer.Sound:
    # impact: tight click + low boom
    n = int(0.5 * SAMPLE_RATE)
    click = _tone(0, 0.02, "noise") * _envelope(int(0.02 * SAMPLE_RATE), 0.001, 0.015)
    boom = _tone(55, 0.5) + 0.7 * _tone(82, 0.5)
    boom = boom[:n] * _envelope(n, 0.002, 0.45)
    out = np.zeros(n, dtype=np.float32)
    out[: click.size] += click
    out += boom
    return _to_stereo_sound(out, peak=0.8)


def _synth_plague() -> pygame.mixer.Sound:
    # sickly descending wobble: detuned chord that bends down
    n = int(0.55 * SAMPLE_RATE)
    t = np.linspace(0, 0.55, n, endpoint=False, dtype=np.float32)
    pitch = 1.0 - 0.4 * t / 0.55  # bend down 40%
    phase = np.cumsum(2 * math.pi * 220 * pitch / SAMPLE_RATE)
    sig = (
        np.sin(phase)
        + 0.7 * np.sin(phase * 1.005)  # detune
        + 0.4 * np.sin(phase * 1.5)    # eerie fifth
    )
    sig *= _envelope(n, 0.02, 0.5)
    return _to_stereo_sound(sig, peak=0.55)


_FALLBACK_BUILDERS = {
    "click": _synth_click,
    "food": _synth_food,
    "herb": _synth_herb,
    "carn": _synth_carn,
    "blessing": _synth_blessing,
    "rain": _synth_rain,
    "disaster": _synth_disaster,
    "plague": _synth_plague,
}

_SFX_FILES = {
    "click": cfg.SOUND_SFX_CLICK,
    "food": cfg.SOUND_SFX_FOOD,
    "herb": cfg.SOUND_SFX_HERB,
    "carn": cfg.SOUND_SFX_CARN,
    "blessing": cfg.SOUND_SFX_BLESSING,
    "rain": cfg.SOUND_SFX_RAIN,
    "disaster": cfg.SOUND_SFX_DISASTER,
    "plague": cfg.SOUND_SFX_PLAGUE,
}


# --- public API -------------------------------------------------------------

class SoundManager:
    """Loads sfx + manages music. Falls back to procedural synth for missing
    sfx. Music tracks have no fallback — silence is better than synthetic
    looping noise."""

    def __init__(self):
        self.enabled = bool(getattr(cfg, "SOUND_ENABLED", True))
        self.sfx: Dict[str, pygame.mixer.Sound] = {}
        self._current_music: Optional[str] = None
        if not self.enabled:
            return
        if not _ensure_mixer():
            self.enabled = False
            print("  [sound] mixer init failed — running silent")
            return

        # build sfx (load file or synth fallback)
        used_fallback = []
        for key, fname in _SFX_FILES.items():
            snd = _try_load(fname)
            if snd is None:
                snd = _FALLBACK_BUILDERS[key]()
                used_fallback.append(key)
            snd.set_volume(cfg.SFX_VOLUME)
            self.sfx[key] = snd

        if used_fallback:
            print(f"  [sound] sfx fallback (synth): {', '.join(used_fallback)}")
            print(f"          drop .ogg files in {_sounds_dir()} to replace")

    # --- sfx ---

    def play_sfx(self, key: str):
        if not self.enabled:
            return
        snd = self.sfx.get(key)
        if snd is not None:
            snd.play()

    # --- music ---

    def play_music(self, filename: str, loop: bool = True, fade_ms: int = 600):
        """Play a music file from the sounds dir. Tries .ogg/.mp3/.wav/.flac
        for the same stem if the exact filename is missing. No-op if none
        of them exist."""
        if not self.enabled:
            return
        path = _resolve_audio_path(filename)
        if path is None:
            if self._current_music != filename:  # log once per request
                print(f"  [sound] music file missing: {filename} (silent)")
                self._current_music = filename
            return
        try:
            pygame.mixer.music.load(path)
            pygame.mixer.music.set_volume(cfg.MUSIC_VOLUME)
            pygame.mixer.music.play(-1 if loop else 0, fade_ms=fade_ms)
            self._current_music = filename
        except pygame.error as e:
            print(f"  [sound] music load failed ({filename}): {e}")

    def stop_music(self, fade_ms: int = 400):
        if not self.enabled:
            return
        try:
            pygame.mixer.music.fadeout(fade_ms)
        except pygame.error:
            pass
        self._current_music = None

    def set_music_volume(self, vol: float):
        if not self.enabled:
            return
        pygame.mixer.music.set_volume(max(0.0, min(1.0, vol)))


# --- module-level singleton (lets game_renderer reach sounds without
# threading a SoundManager through every constructor) -----------------------

_INSTANCE: Optional[SoundManager] = None


def get_sounds() -> SoundManager:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = SoundManager()
    return _INSTANCE
