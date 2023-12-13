#!/bin/bash

# This scripts tars up the contents for transport to the experiments server.
# It omits the cache dirs and the game assets.

tar --exclude='boop/game_assets' --exclude='.pytest_cache' \
  --exclude='boop/__pycache__' -cvzf boop.tgz *