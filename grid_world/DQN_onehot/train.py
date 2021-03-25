from network import *
import os
from worker import *
import threading
from time import sleep

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('max_episodes', 100000, 'Maximum episodes.')
flags.DEFINE_integer('max_episode_steps', 100, 'Maximum steps for each episode.')
flags.DEFINE_integer('max_lowlevel_episode_steps', 10,
                     'Maximum number of steps the robot can take during one episode in low-level policy.')
flags.DEFINE_integer('batch_size', 64,
                     'The size of replay memory used for training.')
flags.DEFINE_float('gamma', 0.99, 'Discount factor.')
flags.DEFINE_integer('window_size', 7, 'The size of vision window.')
flags.DEFINE_integer('num_channels', 2, 'The number of vision channels.')
flags.DEFINE_integer('num_actions', 4, 'The size of action space.')
flags.DEFINE_integer('num_envs', 100, 'The number of scenes used for validation.')
flags.DEFINE_integer('num_goals', 16, 'The number of targets for each scene that are used for training ')
flags.DEFINE_integer('num_threads', 1, 'The number of threads to train one scene one target.')
flags.DEFINE_string('env_dir', 'maps_16X16_v6', 'The path to save the results.')
flags.DEFINE_float('highlevel_lr', 0.0001, 'Highlevel learning rate.')
flags.DEFINE_float('lowlevel_lr', 0.0001, 'Lowlevel learning rate.')
flags.DEFINE_integer('history_steps', 1, 'The number of steps need to remember during training.')
flags.DEFINE_integer('skip_frames', 1, 'The times for low-level action to repeat.')
flags.DEFINE_integer('highlevel_update_freq', 10, 'Highlevel network update frequency.')
flags.DEFINE_integer('lowlevel_update_freq', 10,  'Lowlevel network update frequency.')
flags.DEFINE_integer('target_update_freq', 10000, 'Target network update frequency.')
flags.DEFINE_multi_float('epsilon', [1, 10000, 0.1], ['Initial exploration rate', 'anneal steps', 'final exploration rate'] )
flags.DEFINE_boolean('load_model', True, 'If load previous trained model or not.')
flags.DEFINE_boolean('curriculum_training', True, 'If use curriculum training or not.')
flags.DEFINE_boolean('continuing_training', False, 'If continue training or not.')
flags.DEFINE_string('pretrained_model_path', './result_for_pretrain/model', 'The path to load pretrained model from.')
flags.DEFINE_string('model_path', './result_pretrain/model', 'The path to store or load model from.')
flags.DEFINE_string('evaluate_file', '', '')
flags.DEFINE_boolean('evaluate_during_training', False, 'If evaluate during training.')


def set_up():
    if not os.path.exists(FLAGS.model_path):
        os.makedirs(FLAGS.model_path)

    tf.reset_default_graph()
    graph = tf.Graph()
    with graph.as_default():
        global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
        global_frames = tf.Variable(0, dtype=tf.int32, name='global_frames', trainable=False)

        DQN_network_main = DQN_Network(window_size=FLAGS.window_size,
                                       channel_size=FLAGS.num_channels,
                                       num_goals=FLAGS.num_goals,
                                       num_actions=FLAGS.num_actions,
                                       history_steps=FLAGS.history_steps,
                                       scope='global/main')

        DQN_network_target = DQN_Network(window_size=FLAGS.window_size,
                                         channel_size=FLAGS.num_channels,
                                         num_goals=FLAGS.num_goals,
                                         num_actions=FLAGS.num_actions,
                                         history_steps=FLAGS.history_steps,
                                         scope='global/target')

        workers = []
        for i in range(FLAGS.num_threads):
            local_DQN_network = DQN_Network(window_size=FLAGS.window_size,
                                            channel_size=FLAGS.num_channels,
                                            num_goals=FLAGS.num_goals,
                                            num_actions=FLAGS.num_actions,
                                            history_steps=FLAGS.history_steps,
                                            scope='local_%d'%i)

            worker = Worker(name=i,
                            DQN_network_main=local_DQN_network,
                            DQN_network_target=DQN_network_target,
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
        thread = threading.Thread(target=lambda: workers[0].evaluate(sess))
        thread.start()
        sleep(0.1)
        all_threads.append(thread)
        coord.join(all_threads)

if __name__ == '__main__':
    train()