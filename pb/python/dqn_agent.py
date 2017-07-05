#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

from vizdoom import *
from random import sample, randint, random
from time import time, sleep
from tqdm import trange

import itertools as it
import numpy as np
import skimage.color, skimage.transform
import tensorflow as tf
import os
import errno


# game parameters
game_map = 'basic'
game_resolution = (30, 45)
img_channels = 1
frame_repeat = 12

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
else:
    print('ERROR: wrong game map.')


# Q-learning settings
learning_rate = 0.00025
discount_factor = 0.99
replay_memory_size = 10000

# NN architecture
conv_width = 5
conv_height = 5
features_layer1 = 32
features_layer2 = 64
fc_num_outputs = 1024

# NN learning settings
batch_size = 64

# training regime
num_epochs = 60
learning_steps_per_epoch = 5000
num_episodes = 100
test_episodes_per_epoch = 10
episodes_to_watch = 5


# ceil of a division, source: http://stackoverflow.com/questions/14822184/is-there-a-ceiling-equivalent-of-operator-in-python
def ceildiv(a, b):
    return -(-a // b)

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


def create_network(session, num_available_actions):
    """ creates the network with 
    conv_relu + max_pool + conv_relu + max_pool + fc + dropout + fc """

    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

#    def max_pool_2x2(x):
#        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


    s1_ = tf.placeholder(tf.float32, [None] + list(game_resolution) + [img_channels], name='State')

    target_q_ = tf.placeholder(tf.float32, [None, num_available_actions], name='TargetQ')

    # first convolutional layer
    W_conv1 = weight_variable([conv_height, conv_width, img_channels, features_layer1]) # [5, 5, 1, 32]
    #print('w_conv1: ', W_conv1)
    b_conv1 = bias_variable([features_layer1]) # [32]
    #print('b_conv1: ', b_conv1)

    h_conv1 = tf.nn.relu(conv2d(s1_, W_conv1) + b_conv1)
    #print('h_conv1: ', h_conv1)
    #h_pool1 = max_pool_2x2(h_conv1)
    #print('h_pool1: ', h_pool1)

    # second convolutional layer
    W_conv2 = weight_variable([conv_height, conv_width, features_layer1, features_layer2]) # [5, 5, 32, 64]
    #print('w_conv2: ', W_conv2)
    b_conv2 = bias_variable([features_layer2]) # [64]
    #print('b_conv2: ', b_conv2)

    h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)
    #print('h_conv2: ', h_conv2)
    #h_pool2 = max_pool_2x2(h_conv2)
    #print('h_pool2: ', h_pool2)

    # densely connected layer
    W_fc1 = weight_variable([game_resolution[0]*game_resolution[1]*features_layer2, fc_num_outputs]) # [8*12*(64), 1024]
    b_fc1 = bias_variable([fc_num_outputs]) # [1024]

    h_pool2_flat = tf.reshape(h_conv2, [-1, game_resolution[0]*game_resolution[1]*features_layer2]) # [-1, 8*12*(64)]
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # dropout
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # readout layer
    W_fc2 = weight_variable([fc_num_outputs, num_available_actions]) # [1024, 8]
    b_fc2 = bias_variable([num_available_actions]) # [8]

    q = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    best_a = tf.argmax(q, 1)

    loss = tf.losses.mean_squared_error(q, target_q_)

    optimizer = tf.train.RMSPropOptimizer(learning_rate)

    train_step = optimizer.minimize(loss)

    def function_learn(s1, target_q):
        feed_dict = {s1_: s1, target_q_: target_q, keep_prob: dropout_keep_prob}
        l, _ = session.run([loss, train_step], feed_dict=feed_dict)
        return l

    def function_get_q_values(state):
        return session.run(q, feed_dict={s1_: state, keep_prob: dropout_keep_prob})

    def function_get_best_action(state):
        return session.run(best_a, feed_dict={s1_: state, keep_prob: dropout_keep_prob})

    def function_simple_get_best_action(state):
        return function_get_best_action(state.reshape([1, game_resolution[0], game_resolution[1], 1]))[0]
    
    return function_learn, function_get_q_values, function_simple_get_best_action


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
        a = get_best_action(s1)
    reward = game.make_action(actions[a], frame_repeat)
    
    isterminal = game.is_episode_finished()
    s2 = preprocess(game.get_state().screen_buffer) if not isterminal else None

    memory.add_transition(s1, a, s2, isterminal, reward)

    if memory.size > batch_size:
        s1, a, s2, isterminal, r = memory.get_sample(batch_size)

        q2 = np.max(get_q_values(s2), axis=1)
        target_q = get_q_values(s1)

        target_q[np.arange(target_q.shape[0]), a] = r + discount_factor * (1-isterminal) * q2

        learn(s1, target_q)
        

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

    if save_log:
        make_sure_path_exists(save_path)
        if load_model:
            log_file = open(save_path+log_savefile, 'a')
        else:
            log_file = open(save_path+log_savefile, 'w')            

    num_actions = game.get_available_buttons_size()
    actions = np.zeros((num_actions, num_actions), dtype=np.int32)
    for i in range(num_actions):
        actions[i, i] = 1
    actions = actions.tolist()
    
    memory = ReplayMemory(capacity=replay_memory_size, game_resolution=game_resolution, img_channels=img_channels)

    sess = tf.Session()
    learn, get_q_values, get_best_action = create_network(sess, len(actions))
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
            dropout_keep_prob = 0.5
            game.new_episode()
                
            for learning_step in trange(learning_steps_per_epoch):
                perform_learning_step(epoch)
                if game.is_episode_finished():
                    score = game.get_total_reward()
                    train_scores.append(score)
                    game.new_episode()
                    train_episodes_finished += 1

#            for episode in trange(num_episodes):
#                while not (game.is_episode_finished()):
#                    perform_learning_step(epoch)
#                score = game.get_total_reward()                    
#                train_scores.append(score)
#                train_episodes_finished += 1
#                game.new_episode()

            print('%d training episodes played.' % train_episodes_finished)
 
            train_scores = np.array(train_scores)

            print('Results: mean: %.1f±%.1f,' % (train_scores.mean(), train_scores.std()), \
                  'min: %.1f,' % train_scores.min(), 'max: %.1f,' % train_scores.max())

            print('\nTesting...')
            dropout_keep_prob = 1.0
            test_episode = []
            test_scores = []
            for test_episode in trange(test_episodes_per_epoch):
                game.new_episode()
                while not game.is_episode_finished():
                    state = preprocess(game.get_state().screen_buffer)
                    best_action_index = get_best_action(state)
                    
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

    for _ in range(episodes_to_watch):
        game.new_episode()
        while not game.is_episode_finished():
            state = preprocess(game.get_state().screen_buffer)
            best_action_index = get_best_action(state)
            
            game.set_action(actions[best_action_index])
            for _ in range(frame_repeat):
                game.advance_action()

        sleep(1.0)
        score = game.get_total_reward()
        print('Total score: ', score)
