#!/usr/bin/env python
import tensorflow as tf

import sys
sys.path.append('..')
from utils.helper import *

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('max_episodes', 100000, 'Maximum episodes.')
flags.DEFINE_integer('max_episode_steps', 1000, 'Maximum steps for each episode.')
flags.DEFINE_integer('a_size', 6, 'The size of action space.')
flags.DEFINE_float('gamma', 0.99, 'Discount factor.')
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
flags.DEFINE_string('save_path', '*.txt', 'The path to save the results.')
flags.DEFINE_string('evaluate_file', '', '')

cfg = json.load(open('../config.json', 'r'))
class2id = json.load(open(os.path.join(cfg['codeDir'], 'Environment', 'class2id.json'), 'r'))


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

        if FLAGS.use_default_targets:
            all_targets = FLAGS.default_targets
        else:
            if FLAGS.use_gt:
                all_targets = json.load(open('%s/targets_info_all.json' % scene_dir, 'r')).keys()
            else:
                all_targets = json.load(open('%s/targets_info_all_pred.json' % scene_dir, 'r')).keys()
        all_target_points = get_target_points(scene, all_targets, use_gt=True)

        all_starting_points = get_starting_points_according_to_distance(scene, all_targets)

        scene_targets = []
        scene_target_points = []
        scene_starting_points = []
        num_targets = 0
        for i,t in enumerate(all_targets):
            t_points = all_target_points[t]
            s_points = [p for p in all_starting_points[i] if p not in t_points]
            if len(t_points) != 0 and len(s_points) != 0:
                scene_targets.append(t)
                scene_target_points.append(t_points)
                scene_starting_points.append(s_points)
                num_targets += 1
                if num_targets == FLAGS.num_targets: break

        targets.append(scene_targets)
        starting_points.append(scene_starting_points)
        target_points.append(scene_target_points)
    return scenes[:FLAGS.num_scenes], targets, starting_points, target_points


def select_starting_points(starting_points, targets, min_steps, threshold=100):
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


def get_spl(success_records,
            min_steps,
            steps):
    spl = 0
    n = 0
    for i in range(len(success_records)):
        if min_steps[i] != 0:
            spl += float(success_records[i] * min_steps[i]) / max(min_steps[i], steps[i])
            n += 1
    spl = spl / n
    return spl


def random_walk(scene,
                target,
                starting_points,
                target_points,
                start_state=None):
    env = Semantic_Environment(scene)
    all_min_steps = json.load(open('%s/Environment/houses/%s/minimal_steps_1.json' % (cfg['codeDir'], scene), 'r'))

    if start_state == None:
        num_starting_points = len(starting_points)

        state = env.start(starting_points[np.random.choice(num_starting_points)])
        min_step = all_min_steps[str(state)][target]
        while min_step > 40 or min_step < 2:
            state = env.start(starting_points[np.random.choice(num_starting_points)])
            min_step = all_min_steps[str(state)][target]
    else:
        state = env.start(start_state)

    min_step = all_min_steps[str(state)][target]
    # print min_step
    done = False
    episode_steps = 0
    gamma = 1
    disc_cumu_rewards = 0

    for _ in range(FLAGS.max_episode_steps):
        action = np.random.choice(FLAGS.a_size)
        new_state = env.action_step(action)
        done = new_state in target_points
        reward = 1 if done else 0
        disc_cumu_rewards += gamma*reward
        gamma *= FLAGS.gamma
        episode_steps += 1
        if done:
            break
        # state = new_state
    left_step = json.load(open('%s/Environment/houses/%s/minimal_steps_1.json'%(cfg['codeDir'],scene), 'r'))[str(state)][target]

    return state, disc_cumu_rewards, episode_steps, min_step, done, left_step




def main():
    scenes, targets, starting_points, target_points = get_trainable_scenes_and_targets()
    min_steps = []
    for scene in scenes:
        min_steps.append(
            json.load(open('%s/Environment/houses/%s/minimal_steps_1.json' % (cfg['codeDir'], scene), 'r')))

    starting_points = select_starting_points(starting_points=starting_points,
                                             targets=targets,
                                             min_steps=min_steps)
    print scenes
    print targets

    episode_count = 0

    selected_scenes = []
    selected_targets = []
    selected_states = []

    rewards = []
    steps = []
    min_steps = []
    is_success = []
    left_steps = []

    while episode_count < FLAGS.max_episodes:
        sid = np.random.choice(len(scenes))
        scene = scenes[sid]
        tid = np.random.choice(len(targets[sid]))
        target = targets[sid][tid]


        state, disc_cumu_rewards, episode_steps, min_step, done, left_step = random_walk(scene=scene,
                                                                                         target=target,
                                                                                         starting_points=starting_points[sid][tid],
                                                                                         target_points=target_points[sid][tid])

        selected_scenes.append(scene)
        selected_targets.append(target)
        selected_states.append(state)

        rewards.append(disc_cumu_rewards)
        steps.append(episode_steps)
        min_steps.append(min_step)
        is_success.append(done)
        left_steps.append(left_step)

        episode_count += 1

    mean_rewards = np.mean(rewards)
    success_steps = np.array(steps)[np.array(is_success) == 1]
    mean_success_steps = np.mean(success_steps) if sum(is_success) != 0 else 0
    success_rate = np.mean(is_success)

    spl = get_spl(success_records=is_success, min_steps=min_steps, steps=steps)
    print "discounted cumulative rewards: " + str(mean_rewards)
    print "success steps: " + str(mean_success_steps)
    print "success rate: " + str(success_rate)
    print "spl: " + str(spl)
    print "left steps: " + str(np.mean(left_steps))
    print "average min steps: %4f" % np.mean(min_steps)
    print "average min rewards: %4f" % np.mean([(0.99**(s-1)) for s in min_steps])
    save_file(selected_scenes=selected_scenes,
              selected_targets=selected_targets,
              selected_states=selected_states,
              SR=success_rate,
              AS=mean_success_steps,
              SPL=spl,
              AR=mean_rewards,
              LS=np.mean(left_steps))


def test():
    np.random.seed(12345)
    rewards = []
    steps = []
    min_steps = []
    is_success = []
    left_steps = []

    scenes, targets, starting_points, target_points = get_trainable_scenes_and_targets()
    # print scenes
    # print targets

    with open(FLAGS.evaluate_file, 'r') as f:
        for line in f:
            nums = line.split()
            if len(nums) == 5:
                scene = nums[0]
                target = nums[1]
                start_state = (int(nums[2]), int(nums[3]), int(nums[4]))

                sid = scenes.index(scene)
                tid = targets[sid].index(target)

                state, disc_cumu_rewards, episode_steps, min_step, done, left_step = random_walk(scene=scene,
                                                                                                 target=target,
                                                                                                 starting_points=starting_points[sid][tid],
                                                                                                 target_points=target_points[sid][tid],
                                                                                                 start_state=start_state)

                rewards.append(disc_cumu_rewards)
                steps.append(episode_steps)
                min_steps.append(min_step)
                is_success.append(done)
                left_steps.append(left_step)


    success_steps = np.array(steps)[np.array(is_success) == 1]
    mean_success_steps = np.mean(success_steps) if sum(is_success) != 0 else 0

    success_min_steps = np.array(min_steps)[np.array(is_success) == 1]
    mean_success_min_steps = np.mean(success_min_steps) if sum(is_success) != 0 else 0

    print "SR:%4f" % np.mean(is_success)
    print "AS:%4f / %4f" % (mean_success_steps, mean_success_min_steps)
    print "SPL:%4f" % get_spl(success_records=is_success,
                              min_steps=min_steps,
                              steps=steps)
    print "AR:%4f" % np.mean(rewards)
    print "LS:%4f" % np.mean(left_steps)
    print "average min steps: %4f" % np.mean(min_steps)
    print "average min rewards: %4f" % np.mean([(0.99**(s-1)) for s in min_steps])




def save_file(selected_scenes,
              selected_targets,
              selected_states,
              SR,
              AS,
              SPL,
              AR,
              LS):
    with open(FLAGS.save_path, 'w') as f:
        f.write('SR:%4f, AS:%4f, SPL:%4f, AR:%4f\n'%(SR, AS, SPL, AR))
        for i, scene in enumerate(selected_scenes):
            f.write('%s %s %d %d %d\n'%(scene, selected_targets[i], selected_states[i][0], selected_states[i][1], selected_states[i][2]))


if __name__ == '__main__':
    if FLAGS.evaluate_file == '':
        main()
    else:
        test()

