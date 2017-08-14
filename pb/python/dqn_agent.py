#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

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

import matplotlib.pyplot as plt

# game parameters
game_map = 'health_poison_rewards_floor'
game_resolution = (48, 64)
img_channels = 1
frame_repeat = 8

learn_model = False
load_model = True


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
    save_path = 'model_pb_basic/'
elif (game_map == 'line'):
    config_file_path = '../../scenarios/defend_the_line.cfg'
    save_path = 'model_pb_line_temp/'
elif (game_map == 'corridor'):
    config_file_path = '../../scenarios/deadly_corridor.cfg'
    save_path = 'model_pb_corridor/'
elif (game_map == 'health'):
    config_file_path = '../../scenarios/health_gathering.cfg'
    save_path = 'model_pb_health/'
elif (game_map == 'health_poison'):
    config_file_path = '../../scenarios/health_poison.cfg'
    save_path = 'model_pb_health_poison/'
elif (game_map == 'health_poison_rewards'):
    config_file_path = '../../scenarios/health_poison_rewards.cfg'
    save_path = 'model_pb_health_poison_rewards_4/'
elif (game_map == 'health_poison_rewards_floor'):
    config_file_path = '../../scenarios/health_poison_rewards_floor.cfg'
    save_path = 'model_pb_health_poison_rewards_floor/'
else:
    print('ERROR: wrong game map.')


## Q-learning settings
#discount_factor = 0.99
#replay_memory_size = 10000

## NN learning settings
#batch_size = 64

## training regime
#num_epochs = 10
#learning_steps_per_epoch = 1000
#test_episodes_per_epoch = 10
#episodes_to_watch = 5

# training regime
num_epochs = 70
learning_steps_per_epoch = 15000
test_episodes_per_epoch = 10
episodes_to_watch = 5

# NN learning settings
batch_size = 64

# NN architecture
conv_width = 4
conv_height = 4
features_layer1 = 32
features_layer2 = 64
fc_num_outputs = 1024

# Q-learning settings
learning_rate = 0.0001
discount_factor = 0.99
replay_memory_size = 10000
dropout_keep_prob = 0.7



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

def perform_learning_step(epoch):
    
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

    s1 = preprocess(game.get_state().screen_buffer)

    eps = exploration_rate(epoch)
    if random() <= eps:
        a = randint(0, len(actions) - 1)
    else:
        a = get_best_action(s1, 0.7)
    reward = game.make_action(actions[a], frame_repeat)
    
    isterminal = game.is_episode_finished()
    s2 = preprocess(game.get_state().screen_buffer) if not isterminal else None

    memory.add_transition(s1, a, s2, isterminal, reward)

    if memory.size > batch_size:
        s1, a, s2, isterminal, r = memory.get_sample(batch_size)

        q2 = np.max(get_q_values(s2, 0.7), axis=1)
        target_q = get_q_values(s1, 0.7)

        target_q[np.arange(target_q.shape[0]), a] = r + discount_factor * (1-isterminal) * q2

        learn(s1, target_q, 0.7)
        

def initialize_vizdoom(config_file_path):
    print('Initializing doom...')
    game = DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(False)
    game.set_mode(Mode.PLAYER)
    game.set_screen_format(ScreenFormat.GRAY8)
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.init()
    print('Doom initizalized.')
    return game


if __name__ == '__main__':
    game = initialize_vizdoom(config_file_path)

#    if save_log:
#        make_sure_path_exists(save_path)
#        if load_model:
#            log_file = open(save_path+log_savefile, 'a')
#        else:
#            log_file = open(save_path+log_savefile, 'w')            


    if save_log:
        make_sure_path_exists(save_path)
        if load_model:
            log_file = open(save_path+log_savefile, 'a')
        else:
            log_file = open(save_path+log_savefile, 'w')
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

    sess = tf.Session()
#    learn, get_q_values, get_best_action = create_network(sess, len(actions), game_resolution, img_channels)
    learn, get_q_values, get_best_action, simple_q = create_network(sess, len(actions), game_resolution, img_channels, conv_width, conv_height, features_layer1, features_layer2, fc_num_outputs, learning_rate)
    
    saver = tf.train.Saver()

    if load_model:
        make_sure_path_exists(save_path+model_savefile)
        print('Loading model from: ', save_path+model_savefile)
        saver.restore(sess, save_path+model_savefile)
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
            dropout_keep_prob = 0.7
            game.new_episode()
            
            for learning_step in trange(learning_steps_per_epoch):
                perform_learning_step(epoch)
                if game.is_episode_finished():
                    score = game.get_total_reward()                    
                    train_scores.append(score)
                    game.new_episode()
                    train_episodes_finished += 1

            print('%d training episodes played.' % train_episodes_finished)
 
            train_scores = np.array(train_scores)

            print('Results: mean: %.1f±%.1f,' % (train_scores.mean(), train_scores.std()), \
                  'min: %.1f,' % train_scores.min(), 'max: %.1f,' % train_scores.max())

            print('\nTesting...')
            dropout_keep_prob = 1.0
            test_episode = []
            test_scores = []
            for test_episode in trange(test_episodes_per_epoch):        
                game.set_seed(666)
                game.new_episode()
                while not game.is_episode_finished():
                    state = preprocess(game.get_state().screen_buffer)
                    best_action_index = get_best_action(state, 1.0)
                    
                    game.make_action(actions[best_action_index], frame_repeat)             
                r = game.get_total_reward()
                test_scores.append(r)

            test_scores = np.array(test_scores)
            print('Results: mean: %.1f±%.1f,' % (test_scores.mean(), test_scores.std()), \
                  'min: %.1f,' % test_scores.min(), 'max: %.1f,' % test_scores.max())
            
            if save_model:
                make_sure_path_exists(save_path+model_savefile)
                print('Saving the network weights to:', save_path+model_savefile)
                saver.save(sess, save_path+model_savefile)


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
    print('Training finished. It\'s time to watch!')

    raw_input('Press Enter to continue...') # in python3 use input() instead

    game.set_window_visible(True)
    game.set_mode(Mode.ASYNC_PLAYER)
    game.init()
    
    dropout_keep_prob = 1.0
    
    video_index = 1
    make_sure_path_exists(save_path + "records")

    for _ in range(episodes_to_watch):
        game.set_seed(666)
        game.new_episode(save_path + "records/ep_" + str(video_index) + "_rec.lmp")
        video_index += 1
        while not game.is_episode_finished():
            state = preprocess(game.get_state().screen_buffer)
            best_action_index = get_best_action(state, 1.0)
            
            game.set_action(actions[best_action_index])
            for _ in range(frame_repeat):
                game.advance_action()

        sleep(1.0)
        score = game.get_total_reward()
        print('Total score: ', score)
