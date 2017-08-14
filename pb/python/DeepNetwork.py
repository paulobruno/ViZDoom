import tensorflow as tf


def create_network(session, num_available_actions, game_resolution, img_channels, conv_width, conv_height, features_layer1, features_layer2, fc_num_outputs, learning_rate):

    s1_ = tf.placeholder(tf.float32, [None] + list(game_resolution) + [img_channels], name='State')
    target_q_ = tf.placeholder(tf.float32, [None, num_available_actions], name='TargetQ')

    # Add 2 convolutional layers with ReLu activation
    conv1 = tf.contrib.layers.convolution2d(s1_, num_outputs=features_layer1, kernel_size=[conv_width, conv_height], stride=[2, 2],
                                            activation_fn=tf.nn.relu,
                                            weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                            biases_initializer=tf.constant_initializer(0.1))
    conv2 = tf.contrib.layers.convolution2d(conv1, num_outputs=features_layer2, kernel_size=[conv_width, conv_height], stride=[2, 2],
                                            activation_fn=tf.nn.relu,
                                            weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                            biases_initializer=tf.constant_initializer(0.1))
    conv2_flat = tf.contrib.layers.flatten(conv2)
    fc1 = tf.contrib.layers.fully_connected(conv2_flat, num_outputs=fc_num_outputs, activation_fn=tf.nn.relu,
                                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                                            biases_initializer=tf.constant_initializer(0.1))

    is_in_training = tf.placeholder(tf.bool)
    fc1_drop = tf.contrib.layers.dropout(fc1, keep_prob=0.7, is_training=is_in_training)

    q = tf.contrib.layers.fully_connected(fc1_drop, num_outputs=num_available_actions, activation_fn=None,
                                          weights_initializer=tf.contrib.layers.xavier_initializer(),
                                          biases_initializer=tf.constant_initializer(0.1))
    best_a = tf.argmax(q, 1)

    loss = tf.losses.mean_squared_error(q, target_q_)

    optimizer = tf.train.RMSPropOptimizer(learning_rate)
    # Update the parameters according to the computed gradient using RMSProp.
    train_step = optimizer.minimize(loss)
    

    def function_learn(s1, target_q, is_training):
        feed_dict = {s1_: s1, target_q_: target_q, is_in_training: is_training}
        l, _ = session.run([loss, train_step], feed_dict=feed_dict)
        return l

    def function_get_q_values(state, is_training):
        return session.run(q, feed_dict={s1_: state, is_in_training: is_training})

    def function_simple_get_q_values(state, is_training):
        return function_get_q_values(state.reshape([1, game_resolution[0], game_resolution[1], img_channels]), is_training)[0]

    def function_get_best_action(state, is_training):
        return session.run(best_a, feed_dict={s1_: state, is_in_training: is_training})

    def function_simple_get_best_action(state, is_training):
        return function_get_best_action(state.reshape([1, game_resolution[0], game_resolution[1], img_channels]), is_training)[0]
    
    return function_learn, function_get_q_values, function_simple_get_best_action, function_simple_get_q_values
