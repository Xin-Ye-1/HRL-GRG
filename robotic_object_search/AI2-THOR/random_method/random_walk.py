#!/usr/bin/env python
import tensorflow as tf
import sys
sys.path.append('..')
from utils.constant import *
from utils.environment import *
np.random.seed(seed=None)

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean('is_approaching_policy', False, 'If learning approaching policy.')
flags.DEFINE_integer('max_episodes', 100000, 'Maximum episodes.')
flags.DEFINE_multi_integer('max_episode_steps', [100, 200, 100, 100], 'Maximum steps for different scene types at each episode.')
flags.DEFINE_float('gamma', 0.99, 'Discount factor.')
flags.DEFINE_integer('window_size', 30, 'The size of vision window.')
flags.DEFINE_multi_string('scene_types', ["kitchen", "living_room", "bedroom", "bathroom"], 'The scene types used for training.')
flags.DEFINE_integer('num_train_scenes', 20, 'The number of scenes used for training.')
flags.DEFINE_integer('num_validate_scenes', 5, 'The number of scenes used for validation.')
flags.DEFINE_integer('num_test_scenes', 5, 'The number of scenes used for testing.')
flags.DEFINE_integer('min_step_threshold', 0, 'The number of scenes used for testing.')
flags.DEFINE_string('save_path', '*.txt', 'The path to save the results.')
flags.DEFINE_string('evaluate_file', '', '')
flags.DEFINE_boolean('seen_scenes', True, 'If test on seen scenes.')
flags.DEFINE_boolean('seen_objects', True, 'If test on seen objects.')


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


def random_walk(scene_type,
                scene_no,
                target,
                start_pos=None):
    env = Environment(scene_type=scene_type,
                      scene_no=scene_no,
                      window_size=FLAGS.window_size)
    if start_pos == None:
        all_start_positions = env.get_visible_positions(target) \
            if FLAGS.is_approaching_policy else env.get_train_positions(target)
        all_start_positions = [p for p in all_start_positions
                               if env.get_minimal_steps(target, p) > FLAGS.min_step_threshold
                               and p not in env.get_visible_positions(target)]
        num_start_positions = len(all_start_positions)
        if num_start_positions == 0:
            return None
        state = env.start(all_start_positions[np.random.choice(num_start_positions)])
    else:
        state = env.start(start_pos)

    min_step = env.get_minimal_steps(target)
    done = False
    episode_steps = 0
    gamma = 1
    disc_cumu_rewards = 0

    max_episode_steps = FLAGS.max_episode_steps[FLAGS.scene_types.index(ALL_SCENES[env.scene_type])]
    for _ in range(max_episode_steps):
        action = np.random.choice(NUM_ACTIONS)
        new_state = env.action_step(action)
        done = env.is_done(target)
        reward = 1 if done else 0
        disc_cumu_rewards += gamma*reward
        gamma *= FLAGS.gamma
        episode_steps += 1
        if done:
            break
        # state = new_state
    left_step = env.get_minimal_steps(target)
    return state, disc_cumu_rewards, episode_steps, min_step, done, left_step


def main():
    selected_scene_type = []
    selected_scene_no = []
    selected_targets = []
    selected_states = []

    rewards = []
    steps = []
    min_steps = []
    is_success = []
    left_steps = []

    if FLAGS.seen_scenes:
        all_scene_no = range(1, FLAGS.num_train_scenes + 1)
    else:
        all_scene_no = range(FLAGS.num_train_scenes + 1,
                             FLAGS.num_train_scenes + FLAGS.num_validate_scenes + FLAGS.num_test_scenes + 1)

    for scene_type in FLAGS.scene_types:
        scene_type_idx = ALL_SCENES.index(scene_type)
        if FLAGS.seen_objects:
            all_objects = TRAIN_OBJECTS[scene_type_idx]
        else:
            all_objects = TEST_OBJECTS[scene_type_idx]
        episode_count = 0
        while episode_count < FLAGS.max_episodes:
            scene_no = np.random.choice(all_scene_no)
            target = np.random.choice(all_objects)
            result = random_walk(scene_type=scene_type,
                                 scene_no=scene_no,
                                 target=target
                                 )
            if result is None: continue
            state, disc_cumu_rewards, episode_steps, min_step, done, left_step = result
            selected_scene_type.append(scene_type)
            selected_scene_no.append(scene_no)
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
    save_file(selected_scene_type=selected_scene_type,
              selected_scene_no=selected_scene_no,
              selected_targets=selected_targets,
              selected_states=selected_states,
              SR=success_rate,
              AS=mean_success_steps,
              SPL=spl,
              AR=mean_rewards,
              LS=np.mean(left_steps))



def test():
    np.random.seed(0)

    rewards = []
    steps = []
    min_steps = []
    is_success = []
    left_steps = []

    with open(FLAGS.evaluate_file, 'r') as f:
        for line in f:
            nums = line.split()
            if len(nums) == 8:
                scene_type = nums[0]
                scene_no = int(nums[1])
                target = nums[2]
                start_pos = [float(nums[3]), float(nums[4]), float(nums[5]), int(nums[6]), int(nums[7])]

                result = random_walk(scene_type=scene_type,
                                     scene_no=scene_no,
                                     target=target,
                                     start_pos=start_pos)
                if result is None: continue
                state, disc_cumu_rewards, episode_steps, min_step, done, left_step = result
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


def save_file(selected_scene_type,
              selected_scene_no,
              selected_targets,
              selected_states,
              SR,
              AS,
              SPL,
              AR,
              LS):
    with open(FLAGS.save_path, 'w') as f:
        f.write('SR:%4f, AS:%4f, SPL:%4f, AR:%4f\n'%(SR, AS, SPL, AR))
        for i, scene_type in enumerate(selected_scene_type):
            f.write('%s %d %s %.02f %.02f %.02f %d %d\n'%
                    (scene_type, selected_scene_no[i], selected_targets[i],
                     selected_states[i][0], selected_states[i][1], selected_states[i][2],
                     selected_states[i][3], selected_states[i][4]))


if __name__ == '__main__':
    if FLAGS.evaluate_file == '':
        main()
    else:
        test()

