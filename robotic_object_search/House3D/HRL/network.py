#! /usr/bin/env python

import tensorflow as tf
import tensorflow.contrib.slim as slim


class Highlevel_Network():
    def __init__(self,
                 window_size,
                 num_labels,
                 # action_size,
                 history_steps,
                 scope
                 ):
        with tf.variable_scope('highlevel'):
            with tf.variable_scope(scope):
                self.visions = tf.placeholder(shape=[None, history_steps * window_size * window_size, num_labels],
                                              dtype=tf.float32)
                self.depths = tf.placeholder(shape=[None, history_steps * window_size * window_size, 1], dtype=tf.float32)
                self.targets = tf.placeholder(shape=[None, num_labels], dtype=tf.float32)


                related_visions = self.fc2d(inputs=self.visions,
                                            num_outputs=1,
                                            activation_fn=None,
                                            scope='vision_preprocess')
                related_visions = slim.flatten(related_visions)

                depths = slim.flatten(self.depths)

                hidden_visions = slim.fully_connected(inputs=related_visions,
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

                vision_depth_feature = tf.concat([hidden_visions, hidden_depths, self.targets], -1)

                embed_feature = slim.fully_connected(inputs=vision_depth_feature,
                                                     num_outputs=256,
                                                     activation_fn=tf.nn.relu,
                                                     weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                     biases_initializer=tf.zeros_initializer(),
                                                     scope='embed')

                q_values = slim.fully_connected(inputs=embed_feature,
                                                num_outputs=num_labels,
                                                activation_fn=None,
                                                weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                biases_initializer=tf.zeros_initializer(),
                                                scope='qvalue')

                self.q_values = q_values

                # highlevel training
                if scope != 'global':
                    self.chosen_objects = tf.placeholder(shape=[None], dtype=tf.int32)
                    self.target_q_values = tf.placeholder(shape=[None], dtype=tf.float32)
                    self.highlevel_lr = tf.placeholder(dtype=tf.float32)

                    objects_onehot = tf.one_hot(self.chosen_objects, num_labels, dtype=tf.float32)
                    q_values_for_chosen_objects = tf.reduce_sum(self.q_values*objects_onehot, axis=1)
                    td_error = tf.square(self.target_q_values - q_values_for_chosen_objects)
                    self.qvalue_loss = 0.5*tf.reduce_mean(td_error)

                    highlevel_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'highlevel/%s' % scope)
                    gradients = tf.gradients(self.qvalue_loss, highlevel_params)
                    norm_gradients, _ = tf.clip_by_global_norm(gradients, 40.0)

                    highlevel_trainer = tf.train.RMSPropOptimizer(learning_rate=self.highlevel_lr)
                    global_highlevel_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'highlevel/global')
                    self.highlevel_update = highlevel_trainer.apply_gradients(zip(norm_gradients, global_highlevel_params))

    def fc2d(self,
             inputs,
             num_outputs,
             activation_fn,
             scope, ):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE) as s:
            n0, n1, n2 = inputs.get_shape().as_list()
            weights = tf.get_variable(name='weights',
                                      shape=[n2, num_outputs],
                                      initializer=tf.contrib.layers.xavier_initializer(),
                                      trainable=True)
            wx = tf.einsum('ijk,kl->ijl', inputs, weights)
            biases = tf.get_variable(name='biases',
                                     shape=[num_outputs],
                                     initializer=tf.zeros_initializer(),
                                     trainable=True)
            wx_b = wx + biases
            result = wx_b if activation_fn is None else activation_fn(wx_b, name=s.name)
            return result

    def conv3d(self,
               scope_name,
               input,
               filter_size):
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
            conv_filter = tf.get_variable(name='weights',
                                          shape=filter_size,
                                          initializer=tf.contrib.layers.xavier_initializer(),
                                          trainable=True)
            conv = tf.nn.conv3d(input=input,
                                filter=conv_filter,
                                strides=[1, 1, 1, 1, 1],
                                padding='VALID')
            biases = tf.get_variable(name='biases',
                                     shape=[filter_size[-1]],
                                     initializer=tf.zeros_initializer(),
                                     trainable=True)
            bias = tf.nn.bias_add(conv, biases)

            result = tf.nn.relu(bias, name=scope.name)
            return result


class Lowlevel_Network():
    def __init__(self,
                 window_size,
                 num_labels,
                 action_size,
                 history_steps,
                 scope
                 ):
        with tf.variable_scope('lowlevel'):
            with tf.variable_scope(scope):
                self.visions = tf.placeholder(
                    shape=[None, history_steps * window_size * window_size, num_labels],
                    dtype=tf.float32)
                self.depths = tf.placeholder(shape=[None, history_steps * window_size * window_size, 1],
                                             dtype=tf.float32)
                self.subtargets = tf.placeholder(shape=[None, num_labels], dtype=tf.float32)

                subtargets_expanded = tf.tile(tf.expand_dims(self.subtargets, 1),
                                              [1, history_steps * window_size * window_size, 1])
                masked_visions = tf.reduce_sum(self.visions * subtargets_expanded, axis=-1)
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
                if scope != 'global':
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

                    local_lowlevel_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'lowlevel/%s'%scope)
                    gradients = tf.gradients(self.lowlevel_loss, local_lowlevel_params)
                    norm_gradients, _ = tf.clip_by_global_norm(gradients, 40.0)

                    lowlevel_trainer = tf.train.RMSPropOptimizer(learning_rate=self.lowlevel_lr)
                    global_lowlevel_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'lowlevel/global')
                    self.lowlevel_update = lowlevel_trainer.apply_gradients(zip(norm_gradients, global_lowlevel_params))


























