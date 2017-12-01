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
    load_path_p1 = 'death_basic/player1'
    load_path_p2 = 'death_basic/player2'
    game_resolution = (48, 64)
    bw_size = (9.6, 7.2)
    color_size = (9.6, 7.2)
    comparison_size = (19.2, 7.2)
    both_size = (19.2, 7.2)


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


episodes = 5

game1 = DoomGame()
game1.load_config(config_file_path)
game1.add_game_args("-host 1 -deathmatch")
game1.set_screen_resolution(ScreenResolution.RES_640X480)
game1.set_render_hud(True)
game1.set_window_visible(False)
game1.set_screen_format(ScreenFormat.BGR24)
# Replay must be played in PLAYER mode to save every frame.
game1.set_mode(Mode.PLAYER)
game1.init()

game2 = DoomGame()
game2.load_config(config_file_path)
game2.add_game_args("-host 1 -deathmatch")
game2.set_screen_resolution(ScreenResolution.RES_640X480)
game2.set_render_hud(True)
game2.set_window_visible(False)
game2.set_screen_format(ScreenFormat.BGR24)
# Replay must be played in PLAYER mode to save every frame.
game2.set_mode(Mode.PLAYER)
game2.init()


print("\nREPLAY OF EPISODE")
print("************************\n")

make_sure_path_exists(load_path_p1 + '/black_white')
make_sure_path_exists(load_path_p1 + '/color')
make_sure_path_exists(load_path_p1 + '/comparison')
make_sure_path_exists(load_path_p1 + '/both')

make_sure_path_exists(load_path_p2 + '/black_white')
make_sure_path_exists(load_path_p2 + '/color')
make_sure_path_exists(load_path_p2 + '/comparison')
                
for i in trange(episodes):

    # Replays episodes stored in given file. Sending game command will interrupt playback.
    game1.replay_episode(load_path_p1 + "/ep_" + str(i+1) + "_rec.lmp")
    game2.replay_episode(load_path_p2 + "/ep_" + str(i+1) + "_rec.lmp")

    current_frame = 0

    while not game1.is_episode_finished():

        frame1 = game1.get_state().screen_buffer
        frame2 = game2.get_state().screen_buffer

        
        fig_both = plt.figure(frameon=False, figsize=both_size)
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        both_ax = fig_both.add_subplot(121)
        both_ax.set_axis_off()
        both_ax2 = fig_both.add_subplot(122)
        both_ax2.set_axis_off()
        both_ax.imshow(frame1, interpolation="none")
        both_ax2.imshow(frame2, interpolation="none")
        plt.savefig(load_path_p1 + '/both/' + str(i) + '_' + str(current_frame) + '.png', facecolor='black')
        plt.close(fig_both)
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
        game1.advance_action()
        game2.advance_action()    

game1.close()
game2.close()
