#!/usr/bin/env python


from worker import *
from network import *
import threading
from time import sleep

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean('is_approaching_policy', False, 'If learning approaching policy.')
flags.DEFINE_integer('max_episodes', 100000, 'Maximum episodes.')
flags.DEFINE_integer('max_episode_steps', 1000, 'Maximum steps for each episode.')
flags.DEFINE_integer('max_lowlevel_episode_steps', 50,
                     'Maximum number of steps the robot can take during one episode in low-level policy.')
flags.DEFINE_integer('batch_size', 64,
                     'The size of replay memory used for training.')
flags.DEFINE_float('gamma', 0.99, 'Discount factor.')
flags.DEFINE_boolean('use_gt', False, 'If use ground truth detection.')
flags.DEFINE_integer('window_size', 10, 'The size of vision window.')
flags.DEFINE_integer('num_labels', 78, 'The size of option space.')
flags.DEFINE_integer('a_size', 6, 'The size of action space.')
flags.DEFINE_integer('history_steps', 4, 'The number of steps need to remember during training.')
flags.DEFINE_multi_float('er', [0.01, 1000, 0.01],
                         '[Initial exploration rate, anneal steps, final exploration rate]')
flags.DEFINE_float('highlevel_lr', 0.0001, 'Highlevel learning rate.')
flags.DEFINE_float('lowlevel_lr', 0.0001, 'Lowlevel learning rate.')
flags.DEFINE_string('vision_feature_pattern', '_deeplab_depth_logits_10', 'Which feature to use to represent vision.')
flags.DEFINE_string('depth_feature_pattern', '_deeplab_depth_depth1_10', 'Which feature to use to represent depth.')
flags.DEFINE_integer('replay_start_size', 0, 'The number of observations stored in the replay buffer before training.')
flags.DEFINE_integer('skip_frames', 1, 'The times for low-level action to repeat.')
flags.DEFINE_integer('highlevel_update_freq', 100, 'Highlevel network update frequency.')
flags.DEFINE_integer('lowlevel_update_freq', 10,  'Lowlevel network update frequency.')
flags.DEFINE_integer('target_update_freq', 100000, 'Target network update frequency.')
flags.DEFINE_multi_float('epsilon', [1, 10000, 0.1], ['Initial exploration rate', 'anneal steps', 'final exploration rate'] )
flags.DEFINE_boolean('load_model', True, 'If load previous trained model or not.')
flags.DEFINE_boolean('curriculum_training', True, 'If use curriculum training or not.')
flags.DEFINE_boolean('continuing_training', False, 'If continue training or not.')
flags.DEFINE_string('pretrained_model_path', '../A3C/result_se_for_pretrain/model', 'The path to load pretrained model from.')
flags.DEFINE_string('model_path', './result_se_pretrain/model', 'The path to store or load model from.')
flags.DEFINE_integer('num_scenes', 8, 'The number of scenes used for training.')
flags.DEFINE_integer('num_targets', 15, 'The number of targets for each scene that are used for training ')
flags.DEFINE_integer('num_threads', 1, 'The number of threads to train one scene one target.')
flags.DEFINE_boolean('use_default_scenes', True, 'If use default scenes for training.')
flags.DEFINE_boolean('use_default_targets', True, 'If use default targets for training.')
flags.DEFINE_multi_string('default_scenes',['5cf0e1e9493994e483e985c436b9d3bc',
                                            '0c9a666391cc08db7d6ca1a926183a76',
                                            '0c90efff2ab302c6f31add26cd698bea',
                                            '00d9be7210856e638fa3b1addf2237d6',
                                            '07d1d46444ca33d50fbcb5dc12d7c103',
                                            '026c1bca121239a15581f32eb27f2078',
                                            '0147a1cce83b6089e395038bb57673e3',
                                            '0880799c157b4dff08f90db221d7f884'
                                            ],
                          'Default scenes')
flags.DEFINE_multi_string('default_targets',
                          ['music', 'television', 'heater', 'stand', 'dressing_table', 'table',
                           'bed', 'mirror', 'ottoman', 'sofa', 'desk', 'picture_frame', 'tv_stand',
                           'toilet', 'bathtub'
                           ],
                          'Default targets.')
flags.DEFINE_string('evaluate_file', '', '')


def get_trainable_scenes_and_targets():
    if FLAGS.use_default_scenes:
        scenes = FLAGS.default_scenes
    else:
        scenes = json.load(open('%s/Environment/collected_houses.json' % cfg['codeDir'], 'r'))['houses']

    targets = []
    starting_points = []
    target_points = []
    for scene in scenes[:FLAGS.num_scenes]:
        scene_dir = '%s/Environment/houses/%s/' % (cfg['codeDir'], scene)

        all_target_points = get_target_points(scene, class2id.keys(), use_gt=True)

        if FLAGS.use_default_targets:
            all_targets = FLAGS.default_targets
        else:
            if FLAGS.use_gt:
                all_targets = json.load(open('%s/targets_info_all.json' % scene_dir, 'r')).keys()
            else:
                all_targets = json.load(open('%s/targets_info_all_pred.json' % scene_dir, 'r')).keys()


        all_starting_points = get_starting_points(scene, all_targets, use_gt=FLAGS.use_gt) \
            if FLAGS.is_approaching_policy else get_starting_points_according_to_distance(scene, all_targets)

        scene_targets = []
        #scene_target_points = []
        scene_starting_points = []
        num_targets = 0
        for i,t in enumerate(all_targets):
            t_points = all_target_points[t]
            s_points = [p for p in all_starting_points[i] if p not in t_points]
            if len(t_points) != 0 and len(s_points) != 0:
                scene_targets.append(t)
                #scene_target_points.append(t_points)
                scene_starting_points.append(s_points)
                num_targets += 1
                if num_targets == FLAGS.num_targets: break
        if FLAGS.is_approaching_policy and FLAGS.curriculum_training:
            scene_starting_points = sort_starting_points_according_to_distance(scene, scene_targets, scene_starting_points)

        targets.append(scene_targets)
        starting_points.append(scene_starting_points)
        target_points.append(all_target_points)
    return scenes, targets, starting_points, target_points

def select_starting_points(starting_points, targets, min_steps, threshold=20):
    selected_starting_points = []

    for sid in range(len(targets)):
        selected_scene_starting_points = []
        scene_min_steps = min_steps[sid]
        for tid in range(len(targets[sid])):
            target_starting_points = starting_points[sid][tid]
            target = targets[sid][tid]
            selected_target_starting_points = [sp for sp in target_starting_points
                                               if scene_min_steps[str(sp)][target] < threshold]
            selected_scene_starting_points.append(selected_target_starting_points)
        selected_starting_points.append(selected_scene_starting_points)
    return selected_starting_points

def set_up():
    if not os.path.exists(FLAGS.model_path):
        os.makedirs(FLAGS.model_path)

    tf.reset_default_graph()
    graph = tf.Graph()
    with graph.as_default():
        global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
        global_frames = tf.Variable(0, dtype=tf.int32, name='global_frames', trainable=False)

        highlevel_network_main = Highlevel_Network(window_size=FLAGS.window_size,
                                                   num_labels=FLAGS.num_labels,
                                                   action_size=FLAGS.a_size,
                                                   history_steps=FLAGS.history_steps,
                                                   scope='global/main')

        highlevel_network_target = Highlevel_Network(window_size=FLAGS.window_size,
                                                     num_labels=FLAGS.num_labels,
                                                     action_size=FLAGS.a_size,
                                                     history_steps=FLAGS.history_steps,
                                                     scope='global/target')

        lowlevel_network = Lowlevel_Network(window_size=FLAGS.window_size,
                                            num_labels=FLAGS.num_labels,
                                            action_size=FLAGS.a_size,
                                            history_steps=FLAGS.history_steps,
                                            scope='global')

        relation_graph = Dirichlet_Multimodel_Graph(num_nodes=FLAGS.num_labels,
                                                    num_categories=FLAGS.max_lowlevel_episode_steps + 1,
                                                    gamma=FLAGS.gamma)

        graph_params = np.zeros((FLAGS.num_labels, FLAGS.num_labels, FLAGS.max_lowlevel_episode_steps + 1))
        graph_params[:, :, -1] = 1
        relation_graph.set_params(graph_params)

        scenes, targets, starting_points, target_points = get_trainable_scenes_and_targets()
        # print scenes
        # print targets
        tools = []
        min_steps = []
        for scene in scenes:
            vision_feature_tool = Feature_Tool(scene_name=scene, feature_pattern=FLAGS.vision_feature_pattern)
            depth_feature_tool = Feature_Tool(scene_name=scene, feature_pattern=FLAGS.depth_feature_pattern)
            bbox_tool = Bbox_Tool(scene, use_gt=FLAGS.use_gt)
            tools.append([vision_feature_tool, depth_feature_tool, bbox_tool])
            min_steps.append(json.load(open('%s/Environment/houses/%s/minimal_steps_1.json' % (cfg['codeDir'], scene), 'r')))

        starting_points = select_starting_points(starting_points=starting_points,
                                                 targets=targets,
                                                 min_steps=min_steps)

        workers = []
        for i in range(FLAGS.num_threads):
            local_highlevel_network = Highlevel_Network(window_size=FLAGS.window_size,
                                                        num_labels=FLAGS.num_labels,
                                                        action_size=FLAGS.a_size,
                                                        history_steps=FLAGS.history_steps,
                                                        scope='local_%d'%i)
            local_lowlevel_network = Lowlevel_Network(window_size=FLAGS.window_size,
                                                      num_labels=FLAGS.num_labels,
                                                      action_size=FLAGS.a_size,
                                                      history_steps=FLAGS.history_steps,
                                                      scope='local_%d'%i)
            worker = Worker(name=i,
                            scenes=scenes,
                            targets=targets,
                            min_steps=min_steps,
                            starting_points=starting_points,
                            target_points=target_points,
                            tools=tools,
                            highlevel_network_main=local_highlevel_network,
                            highlevel_network_target=highlevel_network_target,
                            lowlevel_network=local_lowlevel_network,
                            graph=relation_graph,
                            global_episodes=global_episodes,
                            global_frames=global_frames)
            workers.append(worker)
        return graph, workers

def train():
    graph, workers = set_up()
    with tf.Session(graph=graph) as sess:
        coord = tf.train.Coordinator()
        all_threads = []
        for worker in workers:
            thread = threading.Thread(target=lambda: worker.work(sess))
            thread.start()
            sleep(0.1)
            all_threads.append(thread)
        coord.join(all_threads)



if __name__ == '__main__':
    train()
