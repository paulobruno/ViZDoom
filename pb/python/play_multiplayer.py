#!/usr/bin/env python
#####################################################################
# This script presents how to use Doom's native demo mechanism to
# replay episodes with perfect accuracy.
#
# Based on record_episodes.py
#####################################################################

from __future__ import print_function

import os
from random import choice
from vizdoom import *


episodes = 3

folder_path = "model_death_basic_multi_Player5/records/"

game = DoomGame()

# Use other config file if you wish.
game.load_config("../../scenarios/death_basic.cfg")
game.add_game_args("-host 1 -deathmatch")

# New render settings for replay
game.set_screen_resolution(ScreenResolution.RES_640X480)
game.set_render_hud(True)
game.set_window_visible(True)

# Replay can be played in any mode.
game.set_mode(Mode.SPECTATOR)

game.init()

print("\nREPLAY OF EPISODE")
print("************************\n")


for i in range(episodes):

    # Replays episodes stored in given file. Sending game command will interrupt playback.

    # only one POV per replay
    # player 1 POV
    game.replay_episode(folder_path + "ep_" + str(i+1) + "_rec.lmp", 1)

    # player 2 POV
#    game.replay_episode(folder_path + "ep_" + str(i+1) + "_rec.lmp", 2)

    while not game.is_episode_finished():
        # Use advance_action instead of make_action.
        game.advance_action()

    print("Game finished!")
    print("Player1 frags:", game.get_game_variable(GameVariable.PLAYER1_FRAGCOUNT))
    print("Player2 frags:", game.get_game_variable(GameVariable.PLAYER2_FRAGCOUNT))

game.close()
