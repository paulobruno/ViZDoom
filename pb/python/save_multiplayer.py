#!/usr/bin/env python
#####################################################################
# This script presents how to use save an image of every frame of a
# stored episode.
#
#####################################################################

from __future__ import print_function

import os
from vizdoom import *
import numpy as np
import skimage.color, skimage.transform
import matplotlib.pyplot as plt
import errno
from tqdm import trange


game_map = 'death_basic'

if (game_map == 'death_basic'):
    config_file_path = '../../scenarios/death_basic.cfg'
    load_path = 'death_basic/'
    game_resolution = (48, 64)
    bw_size = (9.6, 7.2)
    color_size = (9.6, 7.2)
    comparison_size = (19.2, 7.2)


def preprocess(image):
    img = skimage.transform.resize(image, game_resolution)
    img = skimage.color.rgb2gray(img) # convert to gray
    img = img.astype(np.float32)
    return img

def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


episodes = 1

game = DoomGame()
game.load_config(config_file_path)
game.add_game_args("-host 1 -deathmatch")
game.set_screen_resolution(ScreenResolution.RES_640X480)
game.set_render_hud(True)
game.set_window_visible(False)
game.set_screen_format(ScreenFormat.BGR24)

# Replay must be played in PLAYER mode to save every frame.
game.set_mode(Mode.PLAYER)

game.init()

print("\nREPLAY OF EPISODE")
print("************************\n")

make_sure_path_exists(load_path + '/black_white')
make_sure_path_exists(load_path + '/color')
make_sure_path_exists(load_path + '/comparison')
                
for i in trange(episodes):

    # Replays episodes stored in given file. Sending game command will interrupt playback.
    game.replay_episode(load_path + "/ep_" + str(i+1) + "_rec.lmp")

    current_frame = 0

    while not game.is_episode_finished():

        frame = game.get_state().screen_buffer

        '''
        fig_color = plt.figure(frameon=False, figsize=color_size)
        color_ax = plt.Axes(fig_color, [0., 0., 1., 1.])
        color_ax.set_axis_off()
        fig_color.add_axes(color_ax)
        color_ax.imshow(frame, interpolation="none")
        plt.savefig(load_path + '/color/' + str(i) + '_' + str(current_frame) + '.png', facecolor='black')
        plt.close(fig_color)
        '''
        fig_comparison = plt.figure(frameon=False, figsize=comparison_size)  
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        comparison_ax = fig_comparison.add_subplot(121)
        comparison_ax.set_axis_off()
        comparison_ax2 = fig_comparison.add_subplot(122)
        comparison_ax2.set_axis_off()
        comparison_ax.imshow(frame, interpolation="none")
        state = preprocess(frame)
        state = np.repeat(state[:, :, np.newaxis], 3, axis=2)
        comparison_ax2.imshow(state, interpolation="none")
        plt.savefig(load_path + '/comparison/' + str(i) + '_' + str(current_frame) + '.png', facecolor='black')
        plt.close(fig_comparison)
        '''
        fig_bw = plt.figure(frameon=False, figsize=bw_size)
        bw_ax = plt.Axes(fig_bw, [0., 0., 1., 1.])
        bw_ax.set_axis_off()
        fig_bw.add_axes(bw_ax)
        state = preprocess(frame)
        state = np.repeat(state[:, :, np.newaxis], 3, axis=2)
        bw_ax.imshow(state, interpolation="none")
        plt.savefig(load_path + '/black_white/' + str(i) + '_' + str(current_frame) + '.png', facecolor='black')
        plt.close(fig_bw)
        '''
        current_frame += 1

        # Use advance_action instead of make_action.
        game.advance_action()

game.close()
