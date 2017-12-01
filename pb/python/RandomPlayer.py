#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

from random import choice
from vizdoom import *
from time import time, sleep

import os
import errno
import ConfigParser


# read configuration from file
config = ConfigParser.RawConfigParser()
config.read('settings.cfg')

game_map = config.get('game', 'map')

learn_model = config.getboolean('model', 'learn_model')
view_window = config.getboolean('model', 'view_window')

num_epochs = config.getint('regime', 'num_epochs')
train_episodes_per_epoch = config.getint('regime', 'train_episodes_per_epoch')
test_episodes_per_epoch = config.getint('regime', 'test_episodes_per_epoch')
episodes_to_watch = config.getint('regime', 'episodes_to_watch')


if (learn_model):
    save_model = True
    save_log = True
    skip_learning = False
else:
    save_model = False
    save_log = False
    skip_learning = True

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
    save_path = 'model_death_basic_multi_'
elif (game_map == 'md_floor'):
    config_file_path = '../../scenarios/md_floor.cfg'
    save_path = 'model_mdfloor_2_'
else:
    print('ERROR: wrong game map.')

# 50 random seeds previously generated to use during test
test_map = [48, 839, 966, 520, 134, 713, 939, 591, 666, 286, 552, 843, 940, 290, 826, 321, 476, 278, 831, 685, 473, 113, 795, 32, 90, 631, 587, 350, 117, 577, 394, 34, 815, 925, 148, 584, 890, 209, 466, 980, 246, 406, 240, 214, 288, 400, 787, 236, 465, 836]

def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


class RandomPlayer():

    def __init__(self, name, colorset, is_host, can_shoot):
        self.name = name
        self.colorset = colorset
        self.is_host = is_host
        self.can_shoot = can_shoot
    
    def init_doom(self):
        print('Initializing doom...')
        self.game = DoomGame()
        self.game.load_config(config_file_path)
        self.game.set_window_visible(view_window)
        self.game.set_mode(Mode.PLAYER)
        self.game.set_screen_format(ScreenFormat.GRAY8)
        self.game.set_screen_resolution(ScreenResolution.RES_640X480)
        if (self.is_host):
            self.game.add_game_args('-host 2 -deathmatch +timelimit 1 +sv_spawnfarthest 1')
        else:
            self.game.add_game_args('-join 127.0.0.1')
        self.game.add_game_args('+name ' + self.name + ' +colorset ' + self.colorset)
        self.game.init()
        print('Doom initizalized.')


    def run(self):    

        self.init_doom()

        if (self.can_shoot):
            actions = [[True, False, False], [False, True, False], [False, False, True]]
        else:
            actions = [[True, False, False], [False, True, False]]

        if not skip_learning:
            for _ in range(num_epochs):
                for i in range(train_episodes_per_epoch):
                    while not self.game.is_episode_finished():
                        if self.game.is_player_dead():
                            self.game.respawn_player()

                        # player 2 artificially staticos
                        #self.game.make_action(actions[0])
                        #self.game.make_action(actions[1])
                        self.game.make_action(choice(actions))

                    self.game.new_episode()
                                    
                for i in range(test_episodes_per_epoch):
                    self.game.set_seed(test_map[i])
                    
                    while not self.game.is_episode_finished():
                        if self.game.is_player_dead():
                            self.game.respawn_player()

                        # player 2 artificially static
                        #self.game.make_action(actions[0])
                        #self.game.make_action(actions[1])
                        self.game.make_action(choice(actions))

                    self.game.new_episode()

        self.game.close()

        self.game.set_window_visible(True)
        self.game.set_mode(Mode.ASYNC_PLAYER)

        save_path_player = save_path + self.name + '/'
        make_sure_path_exists(save_path_player + "records")
        video_index = 1

        for i in range(episodes_to_watch):
            self.game.clear_game_args()
            if (self.is_host):
                self.game.add_game_args('-host 2 -deathmatch +timelimit 1 +sv_spawnfarthest 1')
            else:
                self.game.add_game_args('-join 127.0.0.1')
            self.game.add_game_args('+name ' + self.name + ' +colorset ' + self.colorset)
            self.game.add_game_args("-record " + save_path_player + "records/ep_" + str(video_index) + "_rec.lmp")
            video_index += 1


            self.game.init()
                
            while not self.game.is_episode_finished():
                if self.game.is_player_dead():
                    self.game.respawn_player()

                # player 2 artificially static
    #            self.game.make_action(actions[0])
    #            self.game.make_action(actions[1])
                self.game.make_action(choice(actions))

            sleep(1.0)

            self.game.close()
