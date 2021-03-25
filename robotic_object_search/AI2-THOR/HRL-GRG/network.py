#! /usr/bin/env python

import tensorflow as tf
import tensorflow.contrib.slim as slim


class Highlevel_Network():
    def __init__(self,
                 window_size,
                 history_steps,
                 scope
                 ):
        with tf.variable_scope('highlevel'):
            with tf.variable_scope(scope):
                self.visions = tf.placeholder(shape=[None, history_steps * window_size * window_size, 1], dtype=tf.float32)
                self.depths = tf.placeholder(shape=[None, history_steps * window_size * window_size, 1], dtype=tf.float32)

                hidden_visions = tf.reshape(self.visions, [-1, window_size, window_size, history_steps])
                hidden_depths = tf.reshape(self.depths, [-1, window_size, window_size, history_steps])

                result = tf.concat((hidden_visions, hidden_depths), axis=-1)
                conv_layers = [(50, [3, 3]),
                               (100, [3, 3])]
                pool_layers = [[2, 2],
                               [2, 2]]
                for i in range(len(conv_layers)):
                    num_filters, kernel_size = conv_layers[i]
                    result = slim.conv2d(inputs=result,
                                         num_outputs=num_filters,
                                         kernel_size=kernel_size,
                                         stride=1,
                                         padding='SAME',
                                         scope='conv_%d' % i)
                    if pool_layers[i] is not None:
                        result = slim.max_pool2d(inputs=result,
                                                 kernel_size=pool_layers[i],
                                                 scope='pool_%d' % i)
                vision_depth_feature = slim.flatten(result)

                self.feature = vision_depth_feature

                embed_feature = slim.fully_connected(inputs=vision_depth_feature,
                                                     num_outputs=256,
                                                     activation_fn=tf.nn.relu,
                                                     weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                     biases_initializer=tf.zeros_initializer(),
                                                     scope='embed')
                q_values = slim.fully_connected(inputs=embed_feature,
                                                num_outputs=1,
                                                activation_fn=None,
                                                weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                biases_initializer=tf.zeros_initializer(),
                                                scope='qvalue')
                self.q_values = q_values

                # highlevel training
                if not scope.startswith('global'):
                    self.chosen_objects = tf.placeholder(shape=[None], dtype=tf.int32)
                    self.target_q_values = tf.placeholder(shape=[None], dtype=tf.float32)
                    self.highlevel_lr = tf.placeholder(dtype=tf.float32)

                    # objects_onehot = tf.one_hot(self.chosen_objects, num_labels, dtype=tf.float32)
                    q_values_for_chosen_objects = tf.reduce_sum(self.q_values, axis=1)
                    td_error = tf.square(self.target_q_values - q_values_for_chosen_objects)
                    self.qvalue_loss = 0.5*tf.reduce_mean(td_error)

                    highlevel_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'highlevel/%s' % scope)
                    gradients = tf.gradients(self.qvalue_loss, highlevel_params)
                    norm_gradients, _ = tf.clip_by_global_norm(gradients, 40.0)

                    highlevel_trainer = tf.train.RMSPropOptimizer(learning_rate=self.highlevel_lr)
                    global_highlevel_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'highlevel/global/main')
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


class Lowlevel_Network():
    def __init__(self,
                 window_size,
                 action_size,
                 history_steps,
                 scope
                 ):
        with tf.variable_scope('lowlevel'):
            with tf.variable_scope(scope):
                self.visions = tf.placeholder(shape=[None, history_steps * window_size * window_size, 1], dtype=tf.float32)
                self.depths = tf.placeholder(shape=[None, history_steps * window_size * window_size, 1], dtype=tf.float32)

                visions = slim.flatten(self.visions)
                depths = slim.flatten(self.depths)

                hidden_visions = slim.fully_connected(inputs=visions,
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
                if not scope.startswith('global'):
                    self.chosen_actions = tf.placeholder(shape=[None], dtype=tf.int32)
                    self.advantages = tf.placeholder(shape=[None], dtype=tf.float32)
                    self.target_values = tf.placeholder(shape=[None], dtype=tf.float32)
                    self.lowlevel_lr = tf.placeholder(dtype=tf.float32)

                    actions_onehot = tf.one_hot(self.chosen_actions, action_size, dtype=tf.float32)
                    log_policy = tf.log(tf.clip_by_value(self.policy, 0.000001, 0.999999))
                    log_pi_for_action = tf.reduce_sum(tf.multiply(log_policy, actions_onehot), axis=1)

                    self.value_loss = 0.5 * tf.reduce_mean(tf.square(self.target_values - self.value))

                    self.policy_loss = -tf.reduce_mean(log_pi_for_action * self.advantages)

                    self.entropy_loss = -tf.reduce_mean(tf.reduce_sum(self.policy * (-log_policy), axis=1))

                    self.lowlevel_loss = self.value_loss + self.policy_loss + 0.01 * self.entropy_loss

                    local_lowlevel_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'lowlevel/%s' % scope)
                    gradients = tf.gradients(self.lowlevel_loss, local_lowlevel_params)
                    norm_gradients, _ = tf.clip_by_global_norm(gradients, 40.0)

                    lowlevel_trainer = tf.train.RMSPropOptimizer(learning_rate=self.lowlevel_lr)
                    global_lowlevel_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'lowlevel/global')
                    self.lowlevel_update = lowlevel_trainer.apply_gradients(zip(norm_gradients, global_lowlevel_params))




if __name__ == '__main__':
    a = tf.placeholder(shape=[None], dtype=tf.int32)
    t = tf.placeholder(shape=[None], dtype=tf.float32)
    b = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    a1 = tf.one_hot(a, 5, dtype=tf.float32)
    b1 = tf.reduce_sum(b, axis=1)
    c = t - b1
    # c = a1#tf.reduce_sum(a1*b, axis=1)
    # c = c - t
    c = tf.reduce_mean(tf.square(c))
    with tf.Session() as sess:
        c1 = sess.run(c, feed_dict={a:[1,2,3,4],
                                    t:[1.,2.,3.,4.],
                                    b:[[1.], [2.], [3.], [4.]]}
                      )
        print c1

















