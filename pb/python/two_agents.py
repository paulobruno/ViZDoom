#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

from random import choice
from vizdoom import *
from time import time, sleep
from tqdm import trange
from ReplayMemory import *
from DeepNetwork import *

import itertools as it
import numpy as np
import skimage.color, skimage.transform
import tensorflow as tf
import os
import errno

# For multiplayer game use process (ZDoom's multiplayer sync mechanism prevents threads to work as expected).
from multiprocessing import Process

# For singleplayer games threads can also be used.
# from threading import Thread


# game parameters
game_map = 'md_floor'
game_resolution = (3, 4)
img_channels = 1
frame_repeat = 1

learn_model = True
load_model = False
view_window = False


if (learn_model):
    save_model = True
    save_log = True
    skip_learning = False
else:
    save_model = False
    save_log = False
    skip_learning = True
    
log_savefile = 'log.txt'
model_savefile = 'model.ckpt'

if (game_map == 'basic'):
    config_file_path = '../../scenarios/basic.cfg'
    save_path = 'model_basic_'
elif (game_map == 'health'):
    config_file_path = '../../scenarios/health_gathering.cfg'
    save_path = 'model_health_'
elif (game_map == 'health_poison'):
    config_file_path = '../../scenarios/health_poison.cfg'
    save_path = 'model_health_poison_'
elif (game_map == 'health_poison_rewards'):
    config_file_path = '../../scenarios/health_poison_rewards.cfg'
    save_path = 'model_hp_rewards_'
elif (game_map == 'health_poison_rewards_floor'):
    config_file_path = '../../scenarios/health_poison_rewards_floor.cfg'
    save_path = 'model_hpr_floor_'
elif (game_map == 'multiplayer'):
    config_file_path = '../../scenarios/multi_duel_floor.cfg'
    save_path = 'model_multi_duel_'
elif (game_map == 'death_basic'):
    config_file_path = '../../scenarios/death_basic.cfg'
    save_path = 'model_death_basic_5'
elif (game_map == 'md_floor'):
    config_file_path = '../../scenarios/md_floor.cfg'
    save_path = 'model_mdfloor_2_'
else:
    print('ERROR: wrong game map.')


# training regime
num_epochs = 2
train_episodes_per_epoch = 10
learning_steps_per_epoch = 10000
test_episodes_per_epoch = 30
episodes_to_watch = 3

# NN learning settings
batch_size = 2

# NN architecture
conv_width = 4
conv_height = 4
features_layer1 = 1
features_layer2 = 2
fc_num_outputs = 4

# Q-learning settings
learning_rate = 0.0001
discount_factor = 0.99
replay_memory_size = 1
dropout_keep_prob = 0.7

reward_multiplier = 60;

# 50 random seeds previously generated to use during test
test_map = [48, 839, 966, 520, 134, 713, 939, 591, 666, 286, 552, 843, 940, 290, 826, 321, 476, 278, 831, 685, 473, 113, 795, 32, 90, 631, 587, 350, 117, 577, 394, 34, 815, 925, 148, 584, 890, 209, 466, 980, 246, 406, 240, 214, 288, 400, 787, 236, 465, 836]


def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

def preprocess(image):
    img = skimage.transform.resize(image, game_resolution)
    #img = skimage.color.rgb2gray(img) # convert to gray
    img = img.astype(np.float32)
    return img

def exploration_rate(epoch):
    start_eps = 1.0
    end_eps = 0.1
    const_eps_epochs = 0.1 * num_epochs
    eps_decay_epochs = 0.6 * num_epochs

    if load_model:
        return end_eps
    else:
        if epoch < const_eps_epochs:
            return start_eps
        elif epoch < eps_decay_epochs:
            return start_eps - (epoch - const_eps_epochs) / \
                               (eps_decay_epochs - const_eps_epochs) * (start_eps - end_eps)
        else:
            return end_eps




if __name__ == '__main__':
    player1 = player
    player2 = Player

    p1 = Process(target=player1.run)
    p1.start()
    player2.run()

    print("Done")
