#!/bin/bash

# Suppress ALSA and JACK audio error messages
export ALSA_PCM_CARD=default
export ALSA_PCM_DEVICE=0
export ALSA_MIXER_CARD=default
export SDL_AUDIODRIVER=pulse
export PULSE_RUNTIME_PATH=/dev/null

# Run TARS with stderr filtered to hide audio errors
python3 main.py "$@" 2>&1 | grep -v -E "(ALSA lib|Cannot connect to server|jack server|JackShmReadWritePtr)" 