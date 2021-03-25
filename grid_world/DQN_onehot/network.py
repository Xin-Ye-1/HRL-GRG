import tensorflow as tf
import tensorflow.contrib.slim as slim

class DQN_Network():
    def __init__(self,
                 window_size,
                 channel_size,
                 num_goals,
                 num_actions,
                 history_steps,
                 scope='global'):
        with tf.variable_scope(scope):
            self.state = tf.placeholder(shape=[None, history_steps, window_size, window_size, channel_size], dtype=tf.float32)
            self.goal = tf.placeholder(shape=[None, num_goals], dtype=tf.float32)
            result = self.conv3d(scope_name='conv3d',
                                 input=self.state,
                                 filter_size=[history_steps, 1, 1, channel_size, 1])
            result = tf.reshape(result, [-1, window_size, window_size, 1])
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
                                     scope='conv_%d'%i)
                if pool_layers[i] is not None:
                    result = slim.max_pool2d(inputs=result,
                                             kernel_size=pool_layers[i],
                                             scope='pool_%d'%i)

            flatten = slim.flatten(result)
            flatten = tf.concat([flatten, self.goal], 1)
            hidden_embed = slim.fully_connected(inputs=flatten,
                                                num_outputs=100,
                                                activation_fn=None,
                                                weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                biases_initializer=tf.zeros_initializer(),
                                                scope='embed')
            qvalues = slim.fully_connected(inputs=hidden_embed,
                                           num_outputs=num_actions,
                                           activation_fn=None,
                                           weights_initializer=tf.contrib.layers.xavier_initializer(),
                                           biases_initializer=tf.zeros_initializer(),
                                           scope='qvalue')
            self.qvalues = qvalues


            # training
            if scope != 'global':
                self.chosen_actions = tf.placeholder(shape=[None], dtype=tf.int32)
                self.target_q_values = tf.placeholder(shape=[None], dtype=tf.float32)
                self.lr = tf.placeholder(dtype=tf.float32)

                actions_onehot = tf.one_hot(self.chosen_actions, num_actions, dtype=tf.float32)
                qvalues_for_chosen_actions = tf.reduce_sum(self.qvalues*actions_onehot, axis=1)
                td_error = tf.square(self.target_q_values - qvalues_for_chosen_actions)
                self.qvalue_loss = 0.5*tf.reduce_mean(td_error)

                params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                gradients = tf.gradients(self.qvalue_loss, params)
                norm_gradients, _ = tf.clip_by_global_norm(gradients, 40.0)

                trainer = tf.train.RMSPropOptimizer(learning_rate=self.lr)
                global_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.update = trainer.apply_gradients(zip(norm_gradients, global_params))



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








