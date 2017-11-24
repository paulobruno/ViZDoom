#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

from random import choice
from vizdoom import *
from time import time, sleep
from tqdm import trange
from shutil import copyfile
from ReplayMemory import *
from DeepNetwork import *

import itertools as it
import numpy as np
import skimage.color, skimage.transform
import tensorflow as tf
import os
import errno
import ConfigParser

# For multiplayer game use process (ZDoom's multiplayer sync mechanism prevents threads to work as expected).
from multiprocessing import Process

# For singleplayer games threads can also be used.
# from threading import Thread


# read configuration from file
config = ConfigParser.RawConfigParser()
config.read('settings.cfg')

game_map = config.get('game', 'map')
game_w = config.getint('game', 'resolution_width')
game_h = config.getint('game', 'resolution_height')
game_resolution = (game_h, game_w)
img_channels = config.getint('game', 'img_channels')

learn_model = config.getboolean('model', 'learn_model')
load_model = config.getboolean('model', 'load_model')
view_window = config.getboolean('model', 'view_window')
log_savefile = config.get('model', 'log_savefile')
model_savefile = config.get('model', 'model_savefile')
path_num = config.get('model', 'save_path_num')

num_epochs = config.getint('regime', 'num_epochs')
train_episodes_per_epoch = config.getint('regime', 'train_episodes_per_epoch')
learning_steps_per_epoch = config.getint('regime', 'learning_steps_per_epoch')
test_episodes_per_epoch = config.getint('regime', 'test_episodes_per_epoch')
episodes_to_watch = config.getint('regime', 'episodes_to_watch')

conv_width = config.getint('network', 'conv_width')
conv_height = config.getint('network', 'conv_height')
num_feat_layers = config.getint('network', 'num_feat_layers')
features_layer = []
for i in range(num_feat_layers):
    features_layer.append(config.getint('network', 'features_layer_' + str(i+1)))
learning_rate = config.getfloat('network', 'learning_rate')
dropout_keep_prob = config.getfloat('network', 'dropout_keep_prob')

frame_repeat = config.getint('learning', 'frame_repeat')
batch_size = config.getint('learning', 'batch_size')
discount_factor = config.getfloat('learning', 'discount_factor')
replay_memory_size = config.getint('learning', 'replay_memory_size')


reward_multiplier = 60


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


class DqnPlayer():

#    def __init__(self, name, colorset, is_host, sess, learn, get_q_values, get_best_action):
    def __init__(self, name, colorset, is_host):
        self.name = name
        self.colorset = colorset
        self.is_host = is_host
        #self.sess = sess
        #self.learn = learn
        #self.get_q_values = get_q_values
        #self.get_best_action = get_best_action
        
#    def load_config():
#        with open('config_read_file.txt') as f:
            
    
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
#            print('d: -host 2 -deathmatch +timelimit 1 +sv_spawnfarthest 1')
        else:
            self.game.add_game_args('-join 127.0.0.1')
#            print('d: -join 127.0.0.1')
        self.game.add_game_args('+name ' + self.name + ' +colorset ' + self.colorset)
        self.game.init()
        print('Doom initizalized.')
                
            
    def init_config(self):
        self.save_path_player = save_path + self.name + '/'

        if save_log:
            make_sure_path_exists(self.save_path_player)
            if load_model:
                self.log_file = open(self.save_path_player + log_savefile, 'a')
            else:
                copyfile('settings.cfg', self.save_path_player + log_savefile)
                self.log_file = open(self.save_path_player + log_savefile, 'a')
                print('\nTotal_elapsed_time Training_episodes Training_min Training_mean Training_max Testing_min Testing_mean Testing_max', file=self.log_file)
                '''
                self.log_file = open(self.save_path_player + log_savefile, 'w')
                print('Map: ' + game_map, file=self.log_file)
                print('Resolution: ' + str(game_resolution), file=self.log_file)
                print('Image channels: ' + str(img_channels), file=self.log_file)
                print('Frame repeat: ' + str(frame_repeat), file=self.log_file)
                print('Learning rate: ' + str(learning_rate), file=self.log_file)
                print('Discount: ' + str(discount_factor), file=self.log_file)
                print('Replay memory size: ' + str(replay_memory_size), file=self.log_file)
                print('Dropout probability: ' + str(dropout_keep_prob), file=self.log_file)
                print('Batch size: ' + str(batch_size), file=self.log_file)
                print('Convolution kernel size: (' + str(conv_width) + ',' + str(conv_height) + ')', file=self.log_file)
                feat_layers = ''
                for i in range(num_feat_layers):
                    feat_layers += str(features_layer[i]) + ' '
                print('Layers size: ' + feat_layers, file=self.log_file),
                print('Fully connected size: ' + str(fc_num_outputs), file=self.log_file)
                print('Epochs: ' + str(num_epochs), file=self.log_file)
                print('Training episodes: ' + str(train_episodes_per_epoch), file=self.log_file)
                #print('Learning steps: ' + str(learning_steps_per_epoch), file=self.log_file)
                print('Testing episodes: ' + str(test_episodes_per_epoch), file=self.log_file)
                print('Total_elapsed_time Training_episodes Training_min Training_mean Training_max Testing_min Testing_mean Testing_max', file=self.log_file)
                '''
            

    def perform_learning_step(self, eps):
        
        #print('inicio...')
        s1 = preprocess(self.game.get_state().screen_buffer)

        #print('eps...')
        if random() <= eps:
            a = randint(0, len(self.actions) - 1)
        else:
            a = self.get_best_action(s1, True)
            
        #print('reward...')
        last_frags = self.game.get_game_variable(GameVariable.FRAGCOUNT)
        self.game.make_action(self.actions[a], frame_repeat)
        current_frags = self.game.get_game_variable(GameVariable.FRAGCOUNT)
        reward = reward_multiplier * (current_frags - last_frags) - 1.0
        
        #print('temrinal...')
        isterminal = self.game.is_episode_finished()
        
        #print('s2...')
        s2 = preprocess(self.game.get_state().screen_buffer) if not isterminal else None

        #print('memory...')
        self.memory.add_transition(s1, a, s2, isterminal, reward)

        #print('batch...')
        if self.memory.size > batch_size:
            s1, a, s2, isterminal, r = self.memory.get_sample(batch_size)

            q2 = np.max(self.get_q_values(s2, True), axis=1)
            target_q = self.get_q_values(s1, True)

            target_q[np.arange(target_q.shape[0]), a] = r + discount_factor * (1-isterminal) * q2

            self.learn(s1, target_q, True)


    def run(self):
        #game = initialize_vizdoom(config_file_path)
                        
        self.init_doom()
        self.init_config()
        
        num_actions = self.game.get_available_buttons_size()
        self.actions = np.zeros((num_actions, num_actions), dtype=np.int32)
        for i in range(num_actions):
            self.actions[i, i] = 1
        self.actions = self.actions.tolist()
                        
        self.memory = ReplayMemory(capacity=replay_memory_size, game_resolution=game_resolution, num_channels=img_channels)
                
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
#        config.gpu_options.per_process_gpu_memory_fraction = 0.4
        sess = tf.Session(config=config)
        
        self.learn, self.get_q_values, self.get_best_action = create_network(sess, num_actions)
        
        saver = tf.train.Saver()
                    
        if load_model:
            make_sure_path_exists(self.save_path_player+model_savefile)
            print('Loading model from: ', self.save_path_player+model_savefile)
            saver.restore(sess, self.save_path_player+model_savefile)
        else:
            init = tf.global_variables_initializer()
            sess.run(init)
        
        time_start = time()
        
        if not skip_learning:
            print(self.name + ': Starting the training!')

            for epoch in range(num_epochs):
                print('\nEpoch %d\n-------' % (epoch+1))
                train_episodes_finished = 0
                train_scores = []

                print(self.name + ': Training...')
                
                eps = exploration_rate(epoch)

                for _ in trange(train_episodes_per_epoch):

                    while not self.game.is_episode_finished():
                        if self.game.is_player_dead():
                            self.game.respawn_player()
                            
                        if not self.game.is_episode_finished():
                            self.perform_learning_step(eps)
                                                
                    score = reward_multiplier * self.game.get_game_variable(GameVariable.FRAGCOUNT) - 300.0
    #                score = game.get_total_reward()
                    train_scores.append(score)
                    train_episodes_finished += 1
                    
    #                print("Episode finished!")
    #                print("Player1 frags:", score)
                    
                    self.game.new_episode()

    #            while not game.is_episode_finished():
    #                if game.is_player_dead():
    #                    game.respawn_player()
    #                game.make_action(choice(actions))


                print('%d training episodes played.' % train_episodes_finished)
     
                train_scores = np.array(train_scores)
                print('Results: mean: %.1f±%.1f,' % (train_scores.mean(), train_scores.std()), \
                      'min: %.1f,' % train_scores.min(), 'max: %.1f,' % train_scores.max())


                print('\n' + self.name + ': Testing...')
                test_scores = []
                for test_episode in trange(test_episodes_per_epoch):
                    self.game.set_seed(test_map[test_episode])
                    
                    while not self.game.is_episode_finished():
                        #print(self.name + 'd: in while')
                        if self.game.is_player_dead():
                            self.game.respawn_player()
                            #print(self.name + 'd: respawn')
                        
                        if not self.game.is_episode_finished():
                            #print(self.name + 'd: if not finished')
                            state = preprocess(self.game.get_state().screen_buffer)
                            #print(self.name + 'd: preprocess')
                            best_action_index = self.get_best_action(state, False)
                            #print(self.name + 'd: best action')
                                                
                            self.game.make_action(self.actions[best_action_index], frame_repeat)

                    #print(self.name + 'd: after episode finished')
                    
                    r = reward_multiplier * self.game.get_game_variable(GameVariable.FRAGCOUNT) - 300.0
    #                r = game.get_total_reward() 
                    test_scores.append(r)
                    
                    #print('d: after socre append')
                    self.game.new_episode()
                    
                    #print('d: after new episode')

                test_scores = np.array(test_scores)
                print('Results: mean: %.1f±%.1f,' % (test_scores.mean(), test_scores.std()), \
                      'min: %.1f,' % test_scores.min(), 'max: %.1f,' % test_scores.max())

            # Starts a new episode. All players have to call new_episode() in multiplayer mode.

                if save_model:
                    make_sure_path_exists(self.save_path_player+model_savefile)
                    print('Saving the network weights to:', self.save_path_player+model_savefile)
                    #saver.save(self.sess, self.save_path_player+model_savefile)
                    saver.save(sess, self.save_path_player+model_savefile)

                total_elapsed_time = (time() - time_start) / 60.0
                print('Total elapsed time: %.2f minutes' % total_elapsed_time)


                # log to file
                if save_log:
                    print(total_elapsed_time, train_episodes_finished, 
                          train_scores.min(), train_scores.mean(), train_scores.max(), 
                          test_scores.min(), test_scores.mean(), test_scores.max(), file=self.log_file)
                    self.log_file.flush()


        if save_log:
            self.log_file.close()

        self.game.close()
        print('======================================')
        print(self.name + ': Training finished. ')
        
        
        print(self.name + ': It\'s time to watch!')
    #    raw_input('Press Enter to continue...') # in python3 use input() instead

        self.game.set_window_visible(True)
        self.game.set_mode(Mode.ASYNC_PLAYER)
        
        video_index = 1
        make_sure_path_exists(self.save_path_player + "records")

        for i in range(episodes_to_watch):
            self.game.clear_game_args()
            if (self.is_host):
                self.game.add_game_args('-host 2 -deathmatch +timelimit 1 +sv_spawnfarthest 1')
            else:
                self.game.add_game_args('-join 127.0.0.1')
            self.game.add_game_args('+name ' + self.name + ' +colorset ' + self.colorset)
            self.game.add_game_args("-record " + self.save_path_player + "records/ep_" + str(video_index) + "_rec.lmp")
            video_index += 1
            
            self.game.init()
            
            while not self.game.is_episode_finished():
                if self.game.is_player_dead():
                    self.game.respawn_player()
                
                if not self.game.is_episode_finished():
                    state = preprocess(self.game.get_state().screen_buffer)
                    best_action_index = self.get_best_action(state, False)
                                        
                    self.game.set_action(self.actions[best_action_index])
                    for _ in range(frame_repeat):
                        self.game.advance_action()

            sleep(1.0)
            score = self.game.get_game_variable(GameVariable.FRAGCOUNT)
            print(self.name + ': Total score: ', score)

            self.game.close()
