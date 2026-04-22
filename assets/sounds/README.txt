Drop AI-generated audio files in this folder using these exact filenames.
The game falls back to synthesized stings for any missing sfx, and to silence
for any missing music, so you can drop them in one at a time.

Format: any of .ogg / .mp3 / .wav / .flac — the loader auto-detects the
extension, so you can drop in whatever your generator gave you. Filenames
below are written as .ogg but .mp3 etc. work just as well (e.g.
music_main.mp3 is found automatically).
Sample rate: 44.1 kHz, stereo recommended.

MUSIC (looping)
  music_intro.ogg          title-screen music
  music_main.ogg           in-game ambient music

SOUND EFFECTS (one-shot, < 1 second)
  sfx_click.ogg            UI click (tool select / button press)
  sfx_food.ogg             "Bring Forth Food" tool
  sfx_spawn_herb.ogg       "Manifest Herbivore" tool
  sfx_spawn_carn.ogg       "Manifest Predator" tool
  sfx_blessing.ogg         "Sun's Blessing" tool
  sfx_rain.ogg             "Summon Rain" tool
  sfx_disaster.ogg         "Strike Meteor" tool
  sfx_plague.ogg           "Cast Plague" tool

Volumes are tuned in config.py:  MUSIC_VOLUME, SFX_VOLUME.
