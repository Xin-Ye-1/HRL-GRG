#! /usr/bin/env python

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import sys
sys.path.append('..')
from utils.constant import *
flags = tf.app.flags
FLAGS = flags.FLAGS

class Scene_Prior_Network():
    def __init__(self,
                 vision_size,
                 word_size,
                 score_size,
                 action_size,
                 history_steps,
                 scope):
        with tf.variable_scope(scope):
            self.visions = tf.placeholder(shape=[None, history_steps, vision_size], dtype=tf.float32)
            self.targets = tf.placeholder(shape=[None, word_size], dtype=tf.float32)
            self.scores = tf.placeholder(shape=[None, score_size], dtype=tf.float32)

            visions = slim.flatten(self.visions)
            hidden_visions = slim.fully_connected(inputs=visions,
                                                  num_outputs=512,
                                                  activation_fn=tf.nn.relu,
                                                  weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                  biases_initializer=tf.zeros_initializer(),
                                                  scope='vision_hidden')
            hidden_targets = slim.fully_connected(inputs=self.targets,
                                                  num_outputs=512,
                                                  activation_fn=tf.nn.relu,
                                                  weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                  biases_initializer=tf.zeros_initializer(),
                                                  scope='target_hidden')
            # scene prior
            self.adjmat = self.get_adjmat()
            hidden_nodes_scores = slim.fully_connected(inputs=self.scores,
                                                       num_outputs=512,
                                                       activation_fn=None,
                                                       weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                       biases_initializer=tf.zeros_initializer(),
                                                       scope='score_hidden')
            nodes_word_embedding = self.get_nodes_word_embedding()
            hidden_nodes_words = slim.fully_connected(inputs=nodes_word_embedding,
                                                      num_outputs=512,
                                                      activation_fn=None,
                                                      weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                      biases_initializer=tf.zeros_initializer(),
                                                      scope='word_hidden')
            # batch_size = self.scores.get_shape().as_list()[0]
            num_nodes = nodes_word_embedding.get_shape().as_list()[0]
            hidden_nodes_scores = tf.tile(tf.expand_dims(hidden_nodes_scores, 1),
                                          [1, num_nodes, 1])
            hidden_nodes_words = tf.tile(tf.expand_dims(hidden_nodes_words, 0),
                                         [tf.shape(hidden_nodes_scores)[0], 1, 1])
            hidden_nodes = tf.concat([hidden_nodes_scores, hidden_nodes_words], -1)
            hidden_nodes = self.gcn_layer(inputs=hidden_nodes,
                                          num_outputs=1024,
                                          activation_fn=tf.nn.relu,
                                          scope='gcn_1')
            hidden_nodes = self.gcn_layer(inputs=hidden_nodes,
                                          num_outputs=1024,
                                          activation_fn=tf.nn.relu,
                                          scope='gcn_2')
            hidden_nodes = self.gcn_layer(inputs=hidden_nodes,
                                          num_outputs=1,
                                          activation_fn=tf.nn.relu,
                                          scope='gcn_3')
            hidden_nodes = slim.flatten(hidden_nodes)
            hidden_nodes = slim.fully_connected(inputs=hidden_nodes,
                                                num_outputs=512,
                                                activation_fn=None,
                                                weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                biases_initializer=tf.zeros_initializer(),
                                                scope='node_hidden')
            joint_embedding = tf.concat([hidden_visions, hidden_targets, hidden_nodes], -1)
            # joint_embedding = tf.concat([hidden_visions, hidden_targets], -1)
            hidden_joint = slim.fully_connected(inputs=joint_embedding,
                                                num_outputs=512,
                                                activation_fn=tf.nn.relu,
                                                weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                biases_initializer=tf.zeros_initializer(),
                                                scope='joint_hidden')

            self.policy = slim.fully_connected(inputs=hidden_joint,
                                               num_outputs=action_size,
                                               activation_fn=tf.nn.softmax,
                                               weights_initializer=tf.contrib.layers.xavier_initializer(),
                                               biases_initializer=tf.zeros_initializer(),
                                               scope='policy')

            self.value = slim.fully_connected(inputs=hidden_joint,
                                              num_outputs=1,
                                              activation_fn=None,
                                              weights_initializer=tf.contrib.layers.xavier_initializer(),
                                              biases_initializer=tf.zeros_initializer(),
                                              scope='value')

            # Training
            if not scope.startswith('global'):
                self.chosen_actions = tf.placeholder(shape=[None], dtype=tf.int32)
                self.advantages = tf.placeholder(shape=[None], dtype=tf.float32)
                self.target_values = tf.placeholder(shape=[None], dtype=tf.float32)
                self.lr = tf.placeholder(dtype=tf.float32)

                actions_onehot = tf.one_hot(self.chosen_actions, action_size, dtype=tf.float32)
                log_policy = tf.log(tf.clip_by_value(self.policy, 0.000001, 0.999999))
                log_pi_for_action = tf.reduce_sum(tf.multiply(log_policy, actions_onehot), axis=1)

                self.value_loss = 0.5 * tf.reduce_mean(tf.square(self.target_values - self.value))

                self.policy_loss = -tf.reduce_mean(log_pi_for_action * self.advantages)

                self.entropy_loss = -tf.reduce_mean(tf.reduce_sum(self.policy * (-log_policy), axis=1))

                self.loss = self.value_loss + self.policy_loss + 0.01 * self.entropy_loss

                local_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                gradients = tf.gradients(self.loss, local_params)
                norm_gradients, _ = tf.clip_by_global_norm(gradients, 40.0)

                trainer = tf.train.RMSPropOptimizer(learning_rate=self.lr)
                global_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.update = trainer.apply_gradients(zip(norm_gradients, global_params))


    def gcn_layer(self,
                  inputs,
                  num_outputs,
                  activation_fn,
                  scope
                 ):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE) as scope:
            n0, n1, n2 = inputs.get_shape().as_list()
            ah = tf.einsum('ij,bjk->bik', self.adjmat, inputs)
            weights = tf.get_variable(name='weights',
                                      shape=[n2, num_outputs],
                                      initializer=tf.contrib.layers.xavier_initializer(),
                                      trainable=True)
            ahw = tf.einsum('bij,jk->bik', ah, weights)
            result = ahw if activation_fn is None else activation_fn(ahw, name=scope.name)
            return result


    def get_adjmat(self):
        import scipy.io as sio
        import scipy.sparse as sp
        adjmat_path = FLAGS.adjmat_path
        adjmat = sio.loadmat(adjmat_path)['adjmat']
        # nomalization
        adjmat = sp.coo_matrix(adjmat)
        rowsum = np.array(adjmat.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        adjmat = adjmat.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
        return tf.constant(adjmat.tocsr().toarray(), dtype=tf.float32)


    def get_nodes_word_embedding(self):
        import h5py
        f = h5py.File(FLAGS.wemb_path, 'r')
        # objects = open('gcn/objects.txt').readlines()
        # objects = [o.strip() for o in objects]
        objects = ALL_OBJECTS_LIST
        num_objs = len(objects)
        word_size = len(f[objects[0]][:])
        nodes_word_embedding = np.zeros((num_objs, word_size))
        for i, obj in enumerate(objects):
            nodes_word_embedding[i,:] = f[obj][:]
        return tf.constant(nodes_word_embedding, dtype=tf.float32)






















