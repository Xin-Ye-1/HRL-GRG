#!/usr/bin/env python
import sys

sys.path.append('..')
from utils.helper import *
from utils.offline_feature import *
import time
from replay_buffer import *
import json
import tensorflow.contrib.slim as slim

flags = tf.app.flags
FLAGS = flags.FLAGS

id2class = json.load(open(os.path.join(cfg['codeDir'], 'Environment', 'id2class.json'), 'r'))
class2id = json.load(open(os.path.join(cfg['codeDir'], 'Environment', 'class2id.json'), 'r'))

cfg = json.load(open('../config.json', 'r'))

np.random.seed(12345)


class Worker():
    def __init__(self,
                 name,
                 scenes,
                 targets,
                 min_steps,
                 starting_points,
                 target_points,
                 tools,
                 lowlevel_network,
                 global_episodes,
                 global_frames):
        self.name = name
        self.scenes = scenes
        self.targets = targets
        self.min_steps = min_steps
        self.starting_points = starting_points
        self.target_points = target_points
        self.tools = tools
        self.lowlevel_network = lowlevel_network

        self.global_episodes = global_episodes
        self.global_frames = global_frames

        self.episode_increment = self.global_episodes.assign_add(1)
        self.frame_increment = self.global_frames.assign_add(1)

        self.update_local_ops = update_target_graph('global', 'local_%d' % self.name)

        self.saver = tf.train.Saver(max_to_keep=1)

    def _initialize_network(self,
                            sess,
                            testing=False):
        with sess.as_default():
            if FLAGS.load_model:
                print 'Loading model ...'
                if testing:
                    ckpt = tf.train.get_checkpoint_state(FLAGS.model_path)
                    self.saver.restore(sess, ckpt.model_checkpoint_path)
                else:
                    sess.run(tf.global_variables_initializer())
                    ckpt = tf.train.get_checkpoint_state(FLAGS.pretrained_model_path)
                    variable_to_restore = slim.get_variables_to_restore()
                    # variable_to_restore = [val for val in variable_to_restore if 'termination' not in val.name]
                    # for var in variable_to_restore:
                    #     print var.name
                    temp_saver = tf.train.Saver(variable_to_restore)
                    temp_saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                sess.run(tf.global_variables_initializer())

    def _get_learning_rate(self,
                           lr):
        if self.episode_count < 500000:
            e = 0
        elif self.episode_count < 300000:
            e = 1
        elif self.episode_count < 500000:
            e = 2
        else:
            e = 3
        return lr / (10 ** e)

    def _train_lowlevel(self,
                        sess,
                        bootstrap_value):
        # replay_buffer:
        # [0:vision_feature, 1:depth_feature, 2:target_input, 3:value, 4:action, 5:reward]
        with sess.as_default():
            batch = self.replay_buffer.get_buffer()
            N = batch.shape[0]
            R = bootstrap_value
            discounted_rewards = np.zeros(N)
            for t in reversed(range(N)):
                R = batch[t, 5] + FLAGS.gamma * R
                discounted_rewards[t] = R
            advantages = np.array(discounted_rewards) - np.array(batch[:, 3])

            lowlevel_lr = self._get_learning_rate(FLAGS.lowlevel_lr)
            er_start, anneal_steps, er_end = FLAGS.er
            ratio = max((anneal_steps - self.episode_count) / float(anneal_steps), 0)
            er = (er_start - er_end) * ratio + er_end

            loss, policy_loss, value_loss, entropy_loss, _ = sess.run([self.lowlevel_network.lowlevel_loss,
                                                                       self.lowlevel_network.policy_loss,
                                                                       self.lowlevel_network.value_loss,
                                                                       self.lowlevel_network.entropy_loss,
                                                                       self.lowlevel_network.lowlevel_update],
                                                                      feed_dict={
                                                                          self.lowlevel_network.visions: np.stack(
                                                                              batch[:, 0]),
                                                                          self.lowlevel_network.depths: np.stack(
                                                                              batch[:, 1]),
                                                                          self.lowlevel_network.targets: np.stack(
                                                                              batch[:, 2]),
                                                                          self.lowlevel_network.chosen_actions: batch[:,
                                                                                                                4],
                                                                          self.lowlevel_network.advantages: advantages,
                                                                          self.lowlevel_network.target_values: discounted_rewards,
                                                                          self.lowlevel_network.lowlevel_lr: lowlevel_lr,
                                                                          self.lowlevel_network.er: er})
            return loss, policy_loss, value_loss, entropy_loss

    def _run_training_episode(self,
                              sess,
                              scene,
                              target,
                              min_steps,
                              starting_points,
                              target_points,
                              scene_tools,
                              testing=False,
                              start_state=None):
        env = Semantic_Environment(scene)
        target_input = np.zeros(FLAGS.num_labels)
        target_input[int(class2id[target])] = 1

        vision_feature_tool, depth_feature_tool, bbox_tool = scene_tools

        if start_state is not None:
            state = env.start(start_state)
        else:
            num_starting_points = len(starting_points)
            if testing:
                state = env.start(starting_points[np.random.choice(num_starting_points)])
                # state = env.start((6, 11, 3))
            else:
                scope = max(int(num_starting_points * min(float(self.episode_count + 10) / 10000, 1)), 1) \
                    if FLAGS.curriculum_training else num_starting_points
                state = env.start(starting_points[np.random.choice(scope)])

        min_step = min_steps[str(state)][target]


        done = False

        target_bbox = bbox_tool.get_gt_bbox(env.position, target)
        prev_area, _ = gt_bbox_reward(target_bbox)

        states_buffer = []

        disc_cumu_rewards = 0
        episode_steps = 0
        action_steps = 0
        gamma = 1

        actions_buffer = []

        losses = []
        policy_losses = []
        value_losses = []
        entropy_losses = []

        for _ in range(FLAGS.max_episode_steps):
            states_buffer.append(state)
            vision_feature = vision_feature_tool.get_last_history_features(states_buffer, steps=FLAGS.history_steps)
            depth_feature = depth_feature_tool.get_last_history_features(states_buffer, steps=FLAGS.history_steps)

            action_policy, value = sess.run([self.lowlevel_network.policy,
                                             self.lowlevel_network.value],
                                            feed_dict={self.lowlevel_network.visions: [vision_feature],
                                                       self.lowlevel_network.depths: [depth_feature],
                                                       self.lowlevel_network.targets: [target_input]})

            action = np.random.choice(FLAGS.a_size, p=action_policy[0])

            for _ in range(FLAGS.skip_frames):
                new_state = env.action_step(action)
                action_steps += 1
                done = new_state in target_points
                if done:
                    break

            extrinsic_reward = 1 if done else 0

            target_bbox = bbox_tool.get_gt_bbox(env.position, target)
            curr_area, _ = gt_bbox_reward(target_bbox)
            intrinsic_reward, _ = increasing_bbox_reward(scene, target, prev_area, curr_area, use_gt=FLAGS.use_gt)
            if curr_area > prev_area:
                prev_area = curr_area

            reward = extrinsic_reward
            # if not env.action_success():
            #     reward -= 0.01
            disc_cumu_rewards += gamma * reward
            gamma *= FLAGS.gamma

            actions_buffer.append(action)

            if not testing:
                if done or (episode_steps != 0 and episode_steps % FLAGS.lowlevel_update_freq == 0):
                    if done:
                        self.replay_buffer.add(np.reshape(np.array([vision_feature,
                                                                    depth_feature,
                                                                    target_input,
                                                                    value,
                                                                    action,
                                                                    reward]), [1, -1]))
                    bootstrap_value = 0 if done else value[0]
                    loss, policy_loss, value_loss, entropy_loss = self._train_lowlevel(sess=sess,
                                                                                       bootstrap_value=bootstrap_value)
                    self.replay_buffer.clear_buffer()
                    losses.append(loss)
                    policy_losses.append(policy_loss)
                    value_losses.append(value_loss)
                    entropy_losses.append(entropy_loss)
                self.replay_buffer.add(np.reshape(np.array([vision_feature,
                                                            depth_feature,
                                                            target_input,
                                                            value,
                                                            action,
                                                            reward]), [1, -1]))

            episode_steps += 1
            self.frame_count += 1
            if self.name == 0:
                sess.run(self.frame_increment)

            if done:
                states_buffer.append(new_state)
                break
            state = new_state

        if testing:
            left_step = min_steps[str(state)][target]

            return disc_cumu_rewards, action_steps, min_step, done, left_step, states_buffer, actions_buffer

        l = np.mean(losses) if len(losses) != 0 else 0
        pl = np.mean(policy_losses) if len(policy_losses) != 0 else 0
        vl = np.mean(value_losses) if len(value_losses) != 0 else 0
        el = np.mean(entropy_losses) if len(entropy_losses) != 0 else 0

        return disc_cumu_rewards, action_steps, min_step, done, l, pl, vl, el

    def _get_spl(self,
                 success_records,
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

    def _load_map_loc2idx(self,
                          scene):
        loc2idx = {}
        map_path = '%s/Environment/houses/%s/map.txt' % (cfg['codeDir'], str(scene))
        with open(map_path, 'r') as f:
            for line in f:
                nums = line.split()
                if len(nums) == 3:
                    idx = int(nums[0])
                    loc = (int(nums[1]), int(nums[2]))
                    loc2idx[loc] = idx
        return loc2idx

    def _local2global(self,
                      loc2idx,
                      lid):
        (x, y, orien) = lid
        idx = loc2idx[(x, y)]
        gid = 4 * idx + orien
        return gid

    def _save_trajectory(self,
                         scene,
                         target,
                         states_buffer,
                         actions_buffer):
        print 'len(states_buffer): ' + str(len(states_buffer))

        file_path = 'evaluate_%s.txt' % FLAGS.model_path.split('/')[-2]
        loc2idx = self._load_map_loc2idx(scene)

        n = len(states_buffer)
        with open(file_path, 'a') as f:
            f.write('%s\n' % scene)
            f.write('%s\n' % target)
            for i in range(n - 1):
                lid = states_buffer[i]
                gid = self._local2global(loc2idx, lid)
                # oid = options_buffer[i]
                # olabel = id2class[str(self.local2global[str(oid)])]
                f.write('%d %s %d %d\n' % (
                    i, str(lid), gid, actions_buffer[i]))
            lid = states_buffer[n - 1]
            gid = self._local2global(loc2idx, lid)
            f.write('%d %s %d \n' % (n - 1, str(lid), gid))
            f.write("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

    def test(self):
        with tf.Session() as sess:
            self._initialize_network(sess, testing=True)
            sess.run(self.update_local_ops)
            self.episode_count = 0
            self.frame_count = 0

            rewards = []
            steps = []
            min_steps = []
            is_success = []
            left_steps = []

            while self.episode_count < FLAGS.max_episodes:
                sid = np.random.choice(len(self.scenes))
                scene = self.scenes[sid]
                tid = np.random.choice(len(self.targets[sid]))
                target = self.targets[sid][tid]
                starting_points = self.starting_points[sid][tid]
                target_points = self.target_points[sid][tid]
                scene_tools = self.tools[sid]
                scene_min_steps = self.min_steps[sid]

                disc_cumu_rewards, episode_steps, min_step, done, left_step, \
                states_buffer, actions_buffer = self._run_training_episode(
                    sess=sess,
                    scene=scene,
                    target=target,
                    min_steps=scene_min_steps,
                    starting_points=starting_points,
                    target_points=target_points,
                    scene_tools=scene_tools,
                    testing=True)

                rewards.append(disc_cumu_rewards)
                steps.append(episode_steps)
                min_steps.append(min_step)
                is_success.append(done)
                left_steps.append(left_step)

                self._save_trajectory(scene=scene,
                                      target=target,
                                      states_buffer=states_buffer,
                                      actions_buffer=actions_buffer)

                self.episode_count += 1

            success_steps = np.array(steps)[np.array(is_success) == 1]
            mean_success_steps = np.mean(success_steps) if sum(is_success) != 0 else 0

            print "SR:%4f" % np.mean(is_success)
            print "AS:%4f" % mean_success_steps
            print "SPL:%4f" % self._get_spl(success_records=is_success,
                                            min_steps=min_steps,
                                            steps=steps)
            print "AR:%4f" % np.mean(rewards)
            print "LS:%4f" % np.mean(left_steps)

    def work(self,
             sess):
        print 'starting worker %s' % str(self.name)
        np.random.seed(self.name)
        with sess.as_default(), sess.graph.as_default():
            self._initialize_network(sess)
            self.episode_count = 0 #sess.run(self.global_episodes)
            self.frame_count = 0 #sess.run(self.global_frames)

            # [t, s, s', o, a, r, done]
            self.replay_buffer = ReplayBuffer()

            rewards = []
            steps = []
            min_steps = []
            is_success = []

            losses = []
            policy_losses = []
            value_losses = []
            entropy_losses = []

            if self.name == 0:
                self.summary_writer = tf.summary.FileWriter(
                    os.path.dirname(FLAGS.model_path) + '/' + str(self.name), graph=tf.get_default_graph())

            while self.episode_count < FLAGS.max_episodes:
                sess.run(self.update_local_ops)

                sid = np.random.choice(len(self.scenes))
                scene = self.scenes[sid]
                tid = np.random.choice(len(self.targets[sid]))
                target = self.targets[sid][tid]
                starting_points = self.starting_points[sid][tid]
                target_points = self.target_points[sid][tid]
                scene_tools = self.tools[sid]
                scene_min_steps = self.min_steps[sid]

                disc_cumu_rewards, episode_steps, min_step, done, \
                l, pl, vl, el = self._run_training_episode(sess=sess,
                                                           scene=scene,
                                                           target=target,
                                                           min_steps=scene_min_steps,
                                                           starting_points=starting_points,
                                                           target_points=target_points,
                                                           scene_tools=scene_tools,
                                                           testing=False)
                if self.name == 0:
                    print 'episode:{:6}, scene:{} target:{:20} reward:{:5} steps:{:5}/{:5} done:{}'.format(
                        self.episode_count, scene, target, round(disc_cumu_rewards, 2), episode_steps, min_step, done)
                    rewards.append(disc_cumu_rewards)
                    steps.append(episode_steps)
                    min_steps.append(min_step)
                    is_success.append(done)

                    losses.append(l)
                    policy_losses.append(pl)
                    value_losses.append(vl)
                    entropy_losses.append(el)

                    success_steps = np.array(steps)[np.array(is_success) == 1]
                    mean_success_steps = np.mean(success_steps[-100:]) if sum(is_success[-100:]) != 0 else 0

                    summary = tf.Summary()
                    summary.value.add(tag='Training/discounted cumulative rewards',
                                      simple_value=np.mean(rewards[-100:]))
                    summary.value.add(tag='Training/steps', simple_value=mean_success_steps)
                    summary.value.add(tag='Training/success rate', simple_value=np.mean(is_success[-100:]))
                    summary.value.add(tag='Training/spl', simple_value=self._get_spl(success_records=is_success[-100:],
                                                                                     min_steps=min_steps[-100:],
                                                                                     steps=steps[-100:]))
                    summary.value.add(tag='Loss/loss', simple_value=np.mean(losses[-100:]))
                    summary.value.add(tag='Loss/policy_loss', simple_value=np.mean(policy_losses[-100:]))
                    summary.value.add(tag='Loss/value_loss', simple_value=np.mean(value_losses[-100:]))
                    summary.value.add(tag='Loss/entropy_loss', simple_value=np.mean(entropy_losses[-100:]))

                    self.summary_writer.add_summary(summary, self.episode_count)
                    self.summary_writer.flush()

                    if self.episode_count % 1000 == 0 and self.episode_count != 0:
                        self.saver.save(sess, FLAGS.model_path + '/model' + str(self.episode_count) + '.cptk')

                    sess.run(self.episode_increment)

                self.episode_count += 1


    def evaluate(self,
                 read_file='../random_method/1s1t.txt'):
        with tf.Session() as sess:
            self._initialize_network(sess, testing=True)
            self.episode_count = 0
            self.frame_count = 0

            rewards = []
            steps = []
            min_steps = []
            is_success = []
            left_steps = []

            with open(read_file, 'r') as f:
                for line in f:
                    nums = line.split()
                    if len(nums) != 5:
                        continue
                    scene = nums[0]
                    target = nums[1]
                    start_state = (int(nums[2]), int(nums[3]), int(nums[4]))
                    # print (scene, target, start_state)

                    sid = self.scenes.index(scene)
                    tid = self.targets[sid].index(target)

                    starting_points = self.starting_points[sid][tid]
                    target_points = self.target_points[sid][tid]
                    scene_tools = self.tools[sid]
                    scene_min_steps = self.min_steps[sid]

                    disc_cumu_rewards, episode_steps, min_step, done, left_step, \
                    _, _, = self._run_training_episode(sess=sess,
                                                       scene=scene,
                                                       target=target,
                                                       min_steps=scene_min_steps,
                                                       starting_points=starting_points,
                                                       target_points=target_points,
                                                       scene_tools=scene_tools,
                                                       testing=True,
                                                       start_state=start_state)
                    # print "min_step: " + str(min_step)
                    # print "episode_step: " + str(episode_steps)
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
            print "SPL:%4f" % self._get_spl(success_records=is_success,
                                            min_steps=min_steps,
                                            steps=steps)
            print "AR:%4f" % np.mean(rewards)
            print "LS:%4f" % np.mean(left_steps)

    def _get_valid_options(self,
                           env_dir,
                           positions):
        mode = 'gt' if FLAGS.use_gt else 'pred'
        semantic_dynamic = json.load(open('%s/%s_dynamic.json' % (env_dir, mode), 'r'))
        valid_options = [FLAGS.num_labels-1]
        for pos in positions:
            transition = semantic_dynamic[str(pos)]
            options_str = transition.keys()
            valid_options += [int(o) for o in options_str]
        return list(set(valid_options))
        # return valid_options























