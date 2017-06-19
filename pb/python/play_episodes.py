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


episodes = 5

folder_path = "model_pb_health_poison_rewards_floor/records/"


game = DoomGame()

# Use other config file if you wish.
game.load_config("../../scenarios/health_poison_rewards_floor.cfg")

# New render settings for replay
game.set_screen_resolution(ScreenResolution.RES_800X600)
game.set_render_hud(True)
game.set_window_visible(True)

# Replay can be played in any mode.
game.set_mode(Mode.SPECTATOR)

game.init()

print("\nREPLAY OF EPISODE")
print("************************\n")


for i in range(episodes):

    # Replays episodes stored in given file. Sending game command will interrupt playback.
    game.replay_episode(folder_path + "ep_" + str(i+1) + "_rec.lmp")

    while not game.is_episode_finished():
        s = game.get_state()

        # Use advance_action instead of make_action.
        game.advance_action()

        r = game.get_last_reward()
        # game.get_last_action is not supported and don't work for replay at the moment.

        print("State #" + str(s.number))
        #print("Game variables:", s.game_variables[0])
        print("Reward:", r)
        print("=====================")

    print("Episode finished.")
    print("total reward:", game.get_total_reward())
    print("************************")

game.close()
