#!/usr/bin/env python
import sys
sys.path.append('..')
from utils.environment import *

from worker import *
from network import *
import threading
from time import sleep
import os
from graph import *

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean('is_approaching_policy', False, 'If learning approaching policy.')
flags.DEFINE_integer('max_episodes', 100000, 'Maximum episodes.')
flags.DEFINE_multi_integer('max_episode_steps', [100, 200, 100, 100], 'Maximum steps for different scene types at each episode.')
flags.DEFINE_integer('max_lowlevel_episode_steps', 10,
                     'Maximum number of steps the robot can take during one episode in low-level policy.')
flags.DEFINE_integer('batch_size', 64,
                     'The size of replay memory used for training.')
flags.DEFINE_float('gamma', 0.99, 'Discount factor.')
flags.DEFINE_integer('window_size', 30, 'The size of vision window.')
flags.DEFINE_integer('history_steps', 4, 'The number of steps need to remember during training.')
flags.DEFINE_float('highlevel_lr', 0.0001, 'Highlevel learning rate.')
flags.DEFINE_float('lowlevel_lr', 0.0001, 'Lowlevel learning rate.')
flags.DEFINE_integer('replay_start_size', 0, 'The number of observations stored in the replay buffer before training.')
flags.DEFINE_integer('skip_frames', 1, 'The times for low-level action to repeat.')
flags.DEFINE_integer('highlevel_update_freq', 100, 'Highlevel network update frequency.')
flags.DEFINE_integer('lowlevel_update_freq', 10,  'Lowlevel network update frequency.')
flags.DEFINE_integer('target_update_freq', 100000, 'Target network update frequency.')
flags.DEFINE_multi_float('epsilon', [1, 10000, 0.1], ['Initial exploration rate', 'anneal steps', 'final exploration rate'] )
flags.DEFINE_boolean('load_model', True, 'If load previous trained model or not.')
flags.DEFINE_boolean('curriculum_training', True, 'If use curriculum training or not.')
flags.DEFINE_boolean('continuing_training', False, 'If continue training or not.')
flags.DEFINE_string('pretrained_model_path', '../A3C/result_pretrain/model', 'The path to load pretrained model from.')
flags.DEFINE_string('model_path', './result_pretrain/model', 'The path to store or load model from.')
flags.DEFINE_integer('num_threads', 1, 'The number of threads to train one scene one target.')
flags.DEFINE_multi_string('scene_types', ["kitchen", "living_room", "bedroom", "bathroom"], 'The scene types used for training.')
flags.DEFINE_integer('num_train_scenes', 20, 'The number of scenes used for training.')
flags.DEFINE_integer('num_validate_scenes', 5, 'The number of scenes used for validation.')
flags.DEFINE_integer('num_test_scenes', 5, 'The number of scenes used for testing.')
flags.DEFINE_integer('min_step_threshold', 0, 'The number of scenes used for testing.')
flags.DEFINE_string('evaluate_file', '', '')
flags.DEFINE_boolean('is_training', False, 'If is training or not.')


def set_up():
    if not os.path.exists(FLAGS.model_path):
        os.makedirs(FLAGS.model_path)

    tf.reset_default_graph()
    graph = tf.Graph()
    with graph.as_default():
        global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
        global_frames = tf.Variable(0, dtype=tf.int32, name='global_frames', trainable=False)

        highlevel_network_main = Highlevel_Network(window_size=FLAGS.window_size,
                                                   history_steps=FLAGS.history_steps,
                                                   scope='global/main')

        highlevel_network_target = Highlevel_Network(window_size=FLAGS.window_size,
                                                     history_steps=FLAGS.history_steps,
                                                     scope='global/target')

        lowlevel_network = Lowlevel_Network(window_size=FLAGS.window_size,
                                            action_size=NUM_ACTIONS,
                                            history_steps=FLAGS.history_steps,
                                            scope='global')

        relation_graph = Dirichlet_Multimodel_Graph(num_nodes=NUM_OBJECTS+1,
                                                    num_categories=FLAGS.max_lowlevel_episode_steps + 1,
                                                    gamma=FLAGS.gamma)

        graph_params = np.zeros((NUM_OBJECTS+1, NUM_OBJECTS+1, FLAGS.max_lowlevel_episode_steps + 1))
        graph_params[:, :, -1] = 1
        relation_graph.set_params(graph_params)

        envs = []
        for scene_no in range(1, FLAGS.num_train_scenes + FLAGS.num_validate_scenes + FLAGS.num_test_scenes +1):
            for scene_type in FLAGS.scene_types:
                env = Environment(scene_type=scene_type,
                                  scene_no=scene_no,
                                  window_size=FLAGS.window_size)
                envs.append(env)

        workers = []
        for i in range(FLAGS.num_threads):
            local_highlevel_network = Highlevel_Network(window_size=FLAGS.window_size,
                                                        history_steps=FLAGS.history_steps,
                                                        scope='local_%d'%i)
            local_lowlevel_network = Lowlevel_Network(window_size=FLAGS.window_size,
                                                      action_size=NUM_ACTIONS,
                                                      history_steps=FLAGS.history_steps,
                                                      scope='local_%d'%i)
            worker = Worker(name=i,
                            envs=envs,
                            highlevel_networks=(local_highlevel_network, highlevel_network_target),
                            lowlevel_networks=(local_lowlevel_network),
                            graph=relation_graph,
                            global_episodes=global_episodes,
                            global_frames=global_frames)
            workers.append(worker)
        return graph, workers


def train():
    graph, workers = set_up()
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.per_process_gpu_memory_fraction = 0.5
    gpu_config.gpu_options.allow_growth = True
    with tf.Session(graph=graph, config=gpu_config) as sess:
        coord = tf.train.Coordinator()
        all_threads = []
        for worker in workers:
            thread = threading.Thread(target=lambda: worker.work(sess))
            thread.start()
            sleep(0.1)
            all_threads.append(thread)
        thread = threading.Thread(target=lambda:workers[0].validate(sess))
        thread.start()
        sleep(0.1)
        all_threads.append(thread)
        coord.join(all_threads)



if __name__ == '__main__':
    train()
