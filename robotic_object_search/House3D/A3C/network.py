#! /usr/bin/env python

import tensorflow as tf
import tensorflow.contrib.slim as slim




class Lowlevel_Network():
    def __init__(self,
                 window_size,
                 num_labels,
                 action_size,
                 history_steps,
                 scope='global'
                 ):
        with tf.variable_scope(scope):
            self.visions = tf.placeholder(shape=[None, history_steps * window_size * window_size, num_labels],
                                          dtype=tf.float32)
            self.depths = tf.placeholder(shape=[None, history_steps * window_size * window_size, 1], dtype=tf.float32)
            self.targets = tf.placeholder(shape=[None, num_labels], dtype=tf.float32)

            targets_expanded = tf.tile(tf.expand_dims(self.targets, 1),
                                       [1, history_steps * window_size * window_size, 1])
            masked_visions = tf.reduce_sum(self.visions * targets_expanded, axis=-1)
            masked_visions = slim.flatten(masked_visions)

            depths = slim.flatten(self.depths)

            hidden_visions = slim.fully_connected(inputs=masked_visions,
                                                  num_outputs=256,
                                                  activation_fn=tf.nn.relu,
                                                  weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                  biases_initializer=tf.zeros_initializer(),
                                                  scope='vision_hidden')

            hidden_depths = slim.fully_connected(inputs=depths,
                                                 num_outputs=256,
                                                 activation_fn=tf.nn.relu,
                                                 weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                 biases_initializer=tf.zeros_initializer(),
                                                 scope='depth_hidden')

            vision_depth_feature = tf.concat([hidden_visions, hidden_depths], 1)

            embed_feature = slim.fully_connected(inputs=vision_depth_feature,
                                                 num_outputs=256,
                                                 activation_fn=tf.nn.relu,
                                                 weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                 biases_initializer=tf.zeros_initializer(),
                                                 scope='embed')

            # policy estimation

            hidden_policy = slim.fully_connected(inputs=embed_feature,
                                                 num_outputs=20,
                                                 activation_fn=tf.nn.relu,
                                                 weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                 biases_initializer=tf.zeros_initializer(),
                                                 scope='policy_hidden')

            self.policy = slim.fully_connected(inputs=hidden_policy,
                                               num_outputs=action_size,
                                               activation_fn=tf.nn.softmax,
                                               weights_initializer=tf.contrib.layers.xavier_initializer(),
                                               biases_initializer=tf.zeros_initializer(),
                                               scope='policy')

            # value estimation

            hidden_value = slim.fully_connected(inputs=embed_feature,
                                                num_outputs=20,
                                                activation_fn=tf.nn.relu,
                                                weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                biases_initializer=tf.zeros_initializer(),
                                                scope='value_hidden')


            self.value = slim.fully_connected(inputs=hidden_value,
                                              num_outputs=1,
                                              activation_fn=None,
                                              weights_initializer=tf.contrib.layers.xavier_initializer(),
                                              biases_initializer=tf.zeros_initializer(),
                                              scope='value')

        # Lowlevel training
        self.chosen_actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.advantages = tf.placeholder(shape=[None], dtype=tf.float32)
        self.target_values = tf.placeholder(shape=[None], dtype=tf.float32)
        self.lowlevel_lr = tf.placeholder(dtype=tf.float32)
        self.er = tf.placeholder(dtype=tf.float32)

        actions_onehot = tf.one_hot(self.chosen_actions, action_size, dtype=tf.float32)
        log_policy = tf.log(tf.clip_by_value(self.policy, 0.000001, 0.999999))
        log_pi_for_action = tf.reduce_sum(tf.multiply(log_policy, actions_onehot), axis=1)

        self.value_loss = 0.5 * tf.reduce_mean(tf.square(self.target_values - self.value))

        self.policy_loss = -tf.reduce_mean(log_pi_for_action * self.advantages)

        self.entropy_loss = -tf.reduce_mean(tf.reduce_sum(self.policy * (-log_policy), axis=1))

        self.lowlevel_loss = self.value_loss + self.policy_loss + self.er * self.entropy_loss

        local_lowlevel_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        gradients = tf.gradients(self.lowlevel_loss, local_lowlevel_params)
        norm_gradients, _ = tf.clip_by_global_norm(gradients, 40.0)

        lowlevel_trainer = tf.train.RMSPropOptimizer(learning_rate=self.lowlevel_lr)
        global_lowlevel_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
        self.lowlevel_update = lowlevel_trainer.apply_gradients(zip(norm_gradients, global_lowlevel_params))





















