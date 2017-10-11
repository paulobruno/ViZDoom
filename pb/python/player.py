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
import ConfigParser

# For multiplayer game use process (ZDoom's multiplayer sync mechanism prevents threads to work as expected).
from multiprocessing import Process

# For singleplayer games threads can also be used.
# from threading import Thread

config = ConfigParser.RawConfigParser()
config.read('settings.cfg')



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
    save_path = 'model_death_basic_5'
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

def record_episodes():
    print('It\'s time to watch!')
#    raw_input('Press Enter to continue...') # in python3 use input() instead

    game.set_window_visible(True)
    game.set_mode(Mode.ASYNC_PLAYER)
    
    video_index = 1
    make_sure_path_exists(save_path_player1 + "records")

    for i in range(episodes_to_watch):
        game.clear_game_args()
        game.add_game_args("-host 2 -deathmatch +timelimit 1 +sv_spawnfarthest 1")
        game.add_game_args("+name Player1 +colorset 0")
        game.add_game_args("-record " + save_path_player1 + "records/ep_" + str(video_index) + "_rec.lmp")
        video_index += 1
        
        game.init()    
        tf
        while not game.is_episode_finished():
            if game.is_player_dead():
                game.respawn_player()
            
            if not game.is_episode_finished():
                state = preprocess(game.get_state().screen_buffer)
                best_action_index = get_best_action(state, False)
                                    
                game.set_action(actions[best_action_index])
                for _ in range(frame_repeat):
                    game.advance_action()

        sleep(1.0)
        score = game.get_game_variable(GameVariable.FRAGCOUNT)
        print('Total score: ', score)

        game.close()

def player(name, colorset, is_host, session, config_file_path, save_path, log_save_file, view_window):

    def load_config():
        with open('config_read_file.txt') as f:
            
    
    def init_doom():
        print('Initializing doom...')
        game = DoomGame()
        game.load_config(config_file_path)
        game.set_window_visible(view_window)
        game.set_mode(Mode.PLAYER)
        game.set_screen_format(ScreenFormat.GRAY8)
        game.set_screen_resolution(ScreenResolution.RES_640X480)
        if (is_host):
            game.add_game_args('-host 2 -deathmatch +timelimit 1 +sv_spawnfarthest 1')
            game.add_game_args('+name ' + name + ' +colorset ' + colorset)
        else:
            game.add_game_args('-join 127.0.0.1')
            game.add_game_args('+name ' + name + ' +colorset ' + colorset)
        game.init()
        print('Doom initizalized.')
            
            
    def init_config():
        save_path_player = save_path + name + '/'

        if save_log:
            make_sure_path_exists(save_path_player)
            if load_model:
                log_file = open(save_path_player + log_savefile, 'a')
            else:
                log_file = open(save_path_player1+log_savefile, 'w')
                print('Map: ' + game_map, file=log_file)
                print('Resolution: ' + str(game_resolution), file=log_file)
                print('Image channels: ' + str(img_channels), file=log_file)
                print('Frame repeat: ' + str(frame_repeat), file=log_file)
                print('Learning rate: ' + str(learning_rate), file=log_file)
                print('Discount: ' + str(discount_factor), file=log_file)
                print('Replay memory size: ' + str(replay_memory_size), file=log_file)
                print('Dropout probability: ' + str(dropout_keep_prob), file=log_file)
                print('Batch size: ' + str(batch_size), file=log_file)
                print('Convolution kernel size: (' + str(conv_width) + ',' + str(conv_height) + ')', file=log_file)
                print('Layers size: ' + str(features_layer1) + ' ' + str(features_layer2), file=log_file)
                print('Fully connected size: ' + str(fc_num_outputs), file=log_file)
                print('Epochs: ' + str(num_epochs), file=log_file)
                print('Learning steps: ' + str(learning_steps_per_epoch), file=log_file)
                print('Test episodes: ' + str(test_episodes_per_epoch), file=log_file)
                print('Total_elapsed_time Training_episodes Training_min Training_mean Training_max Testing_min Testing_mean Testing_max', file=log_file)
                log_file.flush()

        num_actions = game.get_available_buttons_size()
        actions = np.zeros((num_actions, num_actions), dtype=np.int32)
        for i in range(num_actions):
            actions[i, i] = 1
        actions = actions.tolist()
        
        memory = ReplayMemory(capacity=replay_memory_size, game_resolution=game_resolution, num_channels=img_channels)



    def perform_learning_step(eps):
        
        #print('inicio...')
        s1 = preprocess(game.get_state().screen_buffer)

        #print('eps...')
        if random() <= eps:
            a = randint(0, len(actions) - 1)
        else:
            a = get_best_action(s1, True)
            
        #print('reward...')
        last_frags = game.get_game_variable(GameVariable.FRAGCOUNT)
        game.make_action(actions[a], frame_repeat)
        current_frags = game.get_game_variable(GameVariable.FRAGCOUNT)
        reward = reward_multiplier * (current_frags - last_frags) - 1.0
        
        #print('temrinal...')
        isterminal = game.is_episode_finished()
        
        #print('s2...')
        s2 = preprocess(game.get_state().screen_buffer) if not isterminal else None

        #print('memory...')
        memory.add_transition(s1, a, s2, isterminal, reward)

        #print('batch...')
        if memory.size > batch_size:
            s1, a, s2, isterminal, r = memory.get_sample(batch_size)

            q2 = np.max(get_q_values(s2, True), axis=1)
            target_q = get_q_values(s1, True)

            target_q[np.arange(target_q.shape[0]), a] = r + discount_factor * (1-isterminal) * q2

            learn(s1, target_q, True)


    #game = initialize_vizdoom(config_file_path)
    
    
    
    sess = tf.Session()   
        
    learn, get_q_values, get_best_action, simple_q = create_network(sess, len(actions), game_resolution, img_channels, conv_width, conv_height, features_layer1, features_layer2, fc_num_outputs, learning_rate, dropout_keep_prob)
    
    saver = tf.train.Saver()
    
    
    if load_model:
        make_sure_path_exists(save_path_player1+model_savefile)
        print('Loading model from: ', save_path_player1+model_savefile)
        saver.restore(sess, save_path_player1+model_savefile)
    else:
        init = tf.global_variables_initializer()
        sess.run(init)

    time_start = time()
    
    if not skip_learning:
        print('Starting the training!')

        for epoch in range(num_epochs):
            print('\nEpoch %d\n-------' % (epoch+1))
            train_episodes_finished = 0
            train_scores = []

            print('Training...')
            
            eps = exploration_rate(epoch)

            for _ in trange(train_episodes_per_epoch):

                while not game.is_episode_finished():
                    if game.is_player_dead():
                        game.respawn_player()
                        
                    if not game.is_episode_finished():
                        perform_learning_step(eps)
                                            
                score = reward_multiplier * game.get_game_variable(GameVariable.FRAGCOUNT) - 300.0
#                score = game.get_total_reward()
                train_scores.append(score)
                train_episodes_finished += 1
                
#                print("Episode finished!")
#                print("Player1 frags:", score)
                
                game.new_episode()

#            while not game.is_episode_finished():
#                if game.is_player_dead():
#                    game.respawn_player()
#                game.make_action(choice(actions))


            print('%d training episodes played.' % train_episodes_finished)
 
            train_scores = np.array(train_scores)
            print('Results: mean: %.1f±%.1f,' % (train_scores.mean(), train_scores.std()), \
                  'min: %.1f,' % train_scores.min(), 'max: %.1f,' % train_scores.max())


            print('\nTesting...')
            test_scores = []
            for test_episode in trange(test_episodes_per_epoch):
                game.set_seed(test_map[test_episode])
                
                while not game.is_episode_finished():
                    if game.is_player_dead():
                        game.respawn_player()
                    
                    if not game.is_episode_finished():    
                        state = preprocess(game.get_state().screen_buffer)
                        best_action_index = get_best_action(state, False)
                                            
                        game.make_action(actions[best_action_index], frame_repeat)

                r = reward_multiplier * game.get_game_variable(GameVariable.FRAGCOUNT) - 300.0
#                r = game.get_total_reward() 
                test_scores.append(r)
                
                game.new_episode()

            test_scores = np.array(test_scores)
            print('Results: mean: %.1f±%.1f,' % (test_scores.mean(), test_scores.std()), \
                  'min: %.1f,' % test_scores.min(), 'max: %.1f,' % test_scores.max())

        # Starts a new episode. All players have to call new_episode() in multiplayer mode.

            if save_model:
                make_sure_path_exists(save_path_player1+model_savefile)
                print('Saving the network weights to:', save_path_player1+model_savefile)
                saver.save(sess, save_path_player1+model_savefile)

            total_elapsed_time = (time() - time_start) / 60.0
            print('Total elapsed time: %.2f minutes' % total_elapsed_time)


            # log to file
            if save_log:
                print(total_elapsed_time, train_episodes_finished, 
                      train_scores.min(), train_scores.mean(), train_scores.max(), 
                      test_scores.min(), test_scores.mean(), test_scores.max(), file=log_file)
                log_file.flush()


    if save_log:
        log_file.close()

    game.close()
    print('======================================')
    print('Training finished. ')

    record_episodes()
