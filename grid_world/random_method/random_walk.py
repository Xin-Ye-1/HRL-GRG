import tensorflow as tf
import sys
sys.path.append('..')
from utils.environment import *

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('max_episodes', 100, 'Maximum episodes.')
flags.DEFINE_integer('max_episode_steps', 100, 'Maximum steps for each episode.')
flags.DEFINE_integer('num_actions', 4, 'The size of action space.')
flags.DEFINE_float('gamma', 0.99, 'Discount factor.')
flags.DEFINE_integer('num_envs', 30, 'The number of scenes used for validation.')
flags.DEFINE_integer('num_goals', 16, 'The number of targets for each scene that are used for training ')
flags.DEFINE_integer('num_threads', 1, 'The number of threads to train one scene one target.')
flags.DEFINE_string('env_dir', 'maps_16X16', 'The path to save the results.')
flags.DEFINE_string('save_path', '*.txt', 'The path to save the results.')
flags.DEFINE_string('evaluate_file', '', '')

cfg = json.load(open('../config.json', 'r'))

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
                start_pos=None):
    env = Environment(scene)
    if start_pos is None:
        start_pos, min_step = env.random_start(target, start_from=0)
    else:
        start_pos, min_step = env.start(start_pos, target)

    done = False
    episode_steps = 0
    gamma = 1
    disc_cumu_rewards = 0

    for _ in range(FLAGS.max_episode_steps):
        action = np.random.choice(FLAGS.num_actions)
        pos, reward, done = env.action_step(action)
        disc_cumu_rewards += gamma*reward
        gamma *= FLAGS.gamma
        episode_steps += 1
        if done:
            break
    return start_pos, disc_cumu_rewards, episode_steps, min_step, done




def evaluate():
    seeds = [0]
    episode_count = 0

    selected_scenes = []
    selected_targets = []
    selected_states = []

    SR = []
    AS = []
    MS = []
    SPL = []
    AR = []

    if FLAGS.evaluate_file != '':
        seeds = [1, 5, 13, 45, 99]
        with open(FLAGS.evaluate_file, 'r') as f:
            for line in f:
                if not line.startswith('SR:'):
                    nums = line.split()
                    scene = nums[0]
                    target = int(nums[1])
                    start_state = (int(nums[2]), int(nums[3]))
                    selected_scenes.append(scene)
                    selected_targets.append(target)
                    selected_states.append(start_state)

    mean_min_steps = mean_min_rewards = 0

    for seed in seeds:
        np.random.seed(seed)
        rewards = []
        steps = []
        min_steps = []
        is_success = []
        episode_count = 0

        while episode_count < FLAGS.max_episodes or episode_count < len(selected_scenes):
            if FLAGS.evaluate_file == '':
                scene = '%s/valid/map_%04d'%(FLAGS.env_dir, np.random.choice(FLAGS.num_envs))
                # target = np.random.choice(range(FLAGS.num_goals))
                # target = np.random.choice([2, 5, 10, 13])
                target = np.random.choice([0, 1, 3, 4, 6, 7, 8, 9, 11, 12, 14, 15])
                start_pos = None
            else:
                scene = selected_scenes[episode_count]
                target = selected_targets[episode_count]
                start_pos = selected_states[episode_count]

            state, disc_cumu_rewards, episode_steps, min_step, done = random_walk(scene=scene,
                                                                                  target=target,
                                                                                  start_pos=start_pos)
            if FLAGS.evaluate_file == '':
                selected_scenes.append(scene)
                selected_targets.append(target)
                selected_states.append(state)

            rewards.append(disc_cumu_rewards)
            steps.append(episode_steps)
            min_steps.append(min_step)
            is_success.append(done)

            episode_count += 1

        mean_rewards = np.mean(rewards)
        success_steps = np.array(steps)[np.array(is_success) == 1]
        mean_success_steps = np.mean(success_steps) if sum(is_success) != 0 else 0
        success_min_steps = np.array(min_steps)[np.array(is_success) == 1]
        mean_success_min_steps = np.mean(success_min_steps) if sum(is_success) != 0 else 0
        success_rate = np.mean(is_success)
        spl = get_spl(success_records=is_success, min_steps=min_steps, steps=steps)
        mean_min_steps = np.mean(min_steps)
        mean_min_rewards = np.mean([(0.99**(s-1)) for s in min_steps])

        SR.append(success_rate)
        AS.append(mean_success_steps)
        MS.append(mean_success_min_steps)
        SPL.append(spl)
        AR.append(mean_rewards)



    print 'mean'
    print 'SR:{:5} AS:{:5}/{:5} SPL:{:5} AR:{:5}'.format(
        round(np.mean(SR), 2),
        round(np.mean(AS), 2),
        round(np.mean(MS), 2),
        round(np.mean(SPL), 2),
        round(np.mean(AR), 2))
    print 'var'
    print 'SR:{:5} AS:{:5}/{:5} SPL:{:5} AR:{:5}'.format(
        round(np.var(SR), 2),
        round(np.var(AS), 2),
        round(np.var(MS), 2),
        round(np.var(SPL), 2),
        round(np.var(AR), 2))

    # print "discounted cumulative rewards: %s/%s"%(str(mean_rewards), str(mean_min_rewards))
    # print "success steps: %s/%s/%s"%(str(mean_success_steps), str(mean_success_min_steps), str(mean_min_steps))
    # print "success rate: " + str(success_rate)
    # print "spl: " + str(spl)
    if FLAGS.evaluate_file == '':
        save_file(selected_scenes=selected_scenes,
                  selected_targets=selected_targets,
                  selected_states=selected_states,
                  SR=(np.mean(SR), 1),
                  AS=(np.mean(AS), np.mean(MS), mean_min_steps),
                  SPL=(np.mean(SPL), 1),
                  AR=(np.mean(AR), mean_min_rewards))



def save_file(selected_scenes,
              selected_targets,
              selected_states,
              SR,
              AS,
              SPL,
              AR):
    with open(FLAGS.save_path, 'w') as f:
        f.write('SR:%4f/%4f, AS:%4f/%4f/%4f, SPL:%4f/%4f, AR:%4f/%4f\n'%(SR[0], SR[1], AS[0], AS[1], AS[2], SPL[0], SPL[1], AR[0], AR[1]))
        for i, scene in enumerate(selected_scenes):
            f.write('%s %s %d %d\n'%(scene, selected_targets[i], selected_states[i][0], selected_states[i][1]))


if __name__ =='__main__':
    evaluate()