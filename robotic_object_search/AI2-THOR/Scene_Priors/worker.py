#!/usr/bin/env python
import sys
sys.path.append('..')
from replay_buffer import *
import tensorflow.contrib.slim as slim
from scipy.special import softmax
from utils.common import *
from utils.environment import *
import os
from time import sleep

flags = tf.app.flags
FLAGS = flags.FLAGS


np.random.seed(12345)
visibility = np.inf

class Worker():
    def __init__(self,
                 name,
                 envs,
                 obj_embeddings,
                 lowlevel_networks,
                 global_episodes,
                 global_frames):
        self.name = name
        self.envs = envs
        self.obj_embeddings = obj_embeddings
        self.lowlevel_network = lowlevel_networks

        self.global_episodes = global_episodes
        self.global_frames = global_frames

        self.episode_increment = self.global_episodes.assign_add(1)
        self.frame_increment = self.global_frames.assign_add(1)

        self.update_local_ops = update_multiple_target_graphs(from_scopes=['global'],
                                                              to_scopes=['local_%d' % self.name])

        self.saver = tf.train.Saver(max_to_keep=1)

        if self.name == 0 and FLAGS.is_training:
            self.summary_writer = tf.summary.FileWriter(
                os.path.dirname(FLAGS.model_path) + '/' + str(self.name), graph=tf.get_default_graph())

        self.episode_count = 0
        self.frame_count = 0
        self.lowlevel_replay_buffer = ReplayBuffer()

    def _initialize_network(self,
                            sess,
                            testing=False):
        with sess.as_default():
            if FLAGS.load_model:
                print 'Loading model ...'
                if testing or FLAGS.continuing_training:
                    ckpt = tf.train.get_checkpoint_state(FLAGS.model_path)
                    self.saver.restore(sess, ckpt.model_checkpoint_path)
                else:
                    sess.run(tf.global_variables_initializer())
                    ckpt = tf.train.get_checkpoint_state(FLAGS.pretrained_model_path)
                    variable_to_restore = slim.get_variables_to_restore(exclude=['global_episodes', 'global_frames'])
                    temp_saver = tf.train.Saver(variable_to_restore)
                    temp_saver.restore(sess, ckpt.model_checkpoint_path)

            else:
                sess.run(tf.global_variables_initializer())

    def _get_learning_rate(self):
        # if self.episode_count < 500000:
        #     e = 0
        # elif self.episode_count < 1000000:
        #     e = 1
        # elif self.episode_count < 5000000:
        #     e = 2
        # else:
        #     e = 3
        # return FLAGS.lowlevel_lr / (10 ** e)
        return (1-(0.95*self.episode_count)/FLAGS.max_episodes)*FLAGS.lowlevel_lr

    def _train_lowlevel(self,
                        sess,
                        bootstrap_value):
        # replay_buffer:
        # [0:vision, 1:target, 2:score 3:value 4:action, 5:reward]
        with sess.as_default():
            batch = self.lowlevel_replay_buffer.get_buffer()
            N = batch.shape[0]
            R = bootstrap_value
            discounted_rewards = np.zeros(N)
            for t in reversed(range(N)):
                R = batch[t, 5] + FLAGS.gamma * R
                discounted_rewards[t] = R
            advantages = np.array(discounted_rewards) - np.array(batch[:, 3])

            lowlevel_lr = self._get_learning_rate()

            entropy_loss, _ = sess.run([self.lowlevel_network.entropy_loss,
                                        self.lowlevel_network.update],
                                       feed_dict={self.lowlevel_network.visions: np.stack(batch[:, 0]),
                                                  self.lowlevel_network.targets: np.stack(batch[:, 1]),
                                                  self.lowlevel_network.scores: np.stack(batch[:, 2]),
                                                  self.lowlevel_network.chosen_actions: batch[:, 4],
                                                  self.lowlevel_network.advantages: advantages,
                                                  self.lowlevel_network.target_values: discounted_rewards,
                                                  self.lowlevel_network.lr: lowlevel_lr})
            return entropy_loss

    def _run_training_episode(self,
                              sess,
                              env,
                              target,
                              testing=False,
                              start_pos=None):
        if start_pos is not None:
            state = env.start(start_pos)
        else:
            all_start_positions = env.get_visible_positions(target) \
                if FLAGS.is_approaching_policy else env.get_train_positions(target)
            all_start_positions = [p for p in all_start_positions
                                   if env.get_minimal_steps(target, p) > FLAGS.min_step_threshold]
            num_start_positions = len(all_start_positions)
            if num_start_positions == 0:
                return None
            if testing:
                state = env.start(all_start_positions[np.random.choice(num_start_positions)])
            else:
                scope = max(int(num_start_positions * min(float(self.episode_count + 10) / 10000, 1)), 1) \
                    if FLAGS.curriculum_training and not testing else num_start_positions
                state = env.start(all_start_positions[np.random.choice(scope)])

        min_step = env.get_minimal_steps(target)

        done = False

        states_buffer = []

        disc_cumu_rewards = 0

        episode_steps = 0
        action_steps = 0

        gamma = 1

        subtargets_buffer = []
        actions_buffer = []

        lowlevel_entropy_losses = []

        vision, score = env.get_state_feature()
        vision = [vision for _ in range(FLAGS.history_steps)]

        max_episode_steps = FLAGS.max_lowlevel_episode_steps if FLAGS.is_approaching_policy else \
            FLAGS.max_episode_steps[FLAGS.scene_types.index(ALL_SCENES[env.scene_type])]
        for _ in range(max_episode_steps):
            states_buffer.append(env.position)

            action_policy, value = sess.run([self.lowlevel_network.policy,
                                             self.lowlevel_network.value],
                                            feed_dict={self.lowlevel_network.visions: [np.vstack(vision)],
                                                       self.lowlevel_network.targets: [self.obj_embeddings[target][:]],
                                                       self.lowlevel_network.scores:[score]})

            action = np.random.choice(NUM_ACTIONS, p=action_policy[0])

            # print action_policy[0]
            # print (episode_steps, env.scene_name, env.position, target, ACTIONS[action])

            for _ in range(FLAGS.skip_frames):
                new_state = env.action_step(action)
                action_steps += 1
                done = env.is_done(target) #and ACTIONS[action] == 'Done'
                if done:
                    break

            # extrinsic_reward = 1 if done else 0
            extrinsic_reward = 10 if done else -0.01

            disc_cumu_rewards += gamma * extrinsic_reward
            gamma *= FLAGS.gamma

            subtargets_buffer.append(target)
            actions_buffer.append(action)

            new_vision, new_score = env.get_state_feature()
            new_vision = vision[1:] + [new_vision]

            if not testing:
                # [0:vision, 1:target, 2:score 3:value 4:action, 5:reward]
                self.lowlevel_replay_buffer.add(np.reshape(np.array([np.vstack(vision),
                                                                     self.obj_embeddings[target][:],
                                                                     score,
                                                                     value[0],
                                                                     action,
                                                                     extrinsic_reward]), [1, -1]))

                if len(self.lowlevel_replay_buffer.buffer) > 0 and \
                        (done or (episode_steps!= 0 and episode_steps % FLAGS.lowlevel_update_freq == 0) or
                        episode_steps == max_episode_steps-1):
                    if done:
                        bootstrap_value = 0
                    else:
                        bootstrap_value = sess.run(self.lowlevel_network.value,
                                                   feed_dict={self.lowlevel_network.visions: [np.vstack(new_vision)],
                                                              self.lowlevel_network.targets: [self.obj_embeddings[target][:]],
                                                              self.lowlevel_network.scores: [new_score]})[0]
                    entropy_loss = self._train_lowlevel(sess=sess, bootstrap_value=bootstrap_value)
                    lowlevel_entropy_losses.append(entropy_loss)
                    self.lowlevel_replay_buffer.clear_buffer()
                    # sess.run(self.update_local_ops)

                self.frame_count += 1
                if self.name == 0:
                    sess.run(self.frame_increment)

            episode_steps += 1

            vision = new_vision
            score = new_score

            if done:
                states_buffer.append(env.position)
                break

        if testing:
            left_step = env.get_minimal_steps(target)

            return disc_cumu_rewards, episode_steps, min_step, done, left_step, states_buffer, subtargets_buffer, actions_buffer

        lel = np.mean(lowlevel_entropy_losses) if len(lowlevel_entropy_losses) != 0 else 0

        return disc_cumu_rewards, episode_steps, min_step, done, lel

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

    def work(self,
             sess):
        print('starting worker %s' % str(self.name))
        np.random.seed(self.name)
        with sess.as_default(), sess.graph.as_default():
            self._initialize_network(sess)
            self.episode_count = sess.run(self.global_episodes)
            self.frame_count = sess.run(self.global_frames)

            num_record = 100

            rewards = np.zeros(num_record)
            steps = np.zeros(num_record)
            min_steps = np.zeros(num_record)
            is_success = np.zeros(num_record)

            lentropy_losses = np.zeros(num_record)

            while self.episode_count <= FLAGS.max_episodes:
                sess.run(self.update_local_ops)

                env_idx = np.random.choice(range(len(FLAGS.scene_types) * FLAGS.num_train_scenes))
                env = self.envs[env_idx]
                if FLAGS.is_approaching_policy:
                    all_targets = env.get_scene_objects()
                else:
                    all_targets = [t for t in TRAIN_OBJECTS[env.scene_type] if t in env.get_scene_objects()]
                if len(all_targets) == 0: continue
                target = np.random.choice(all_targets)
                target_idx = ALL_OBJECTS_LIST.index(target)

                result = self._run_training_episode(sess=sess,
                                                    env=env,
                                                    target=target,
                                                    testing=False)
                if result is None: continue
                disc_cumu_rewards, action_steps, min_step, done, lel = result
                if self.name == 0:
                    print 'episode:{:6}, scene:{:15} target:{:20} reward:{:5} steps:{:5}/{:5} done:{}'.format(
                        self.episode_count, env.scene_name, target, round(disc_cumu_rewards, 2), action_steps, min_step, done)
                    rewards[self.episode_count%num_record] = disc_cumu_rewards
                    steps[self.episode_count%num_record] = action_steps
                    min_steps[self.episode_count%num_record] = min_step
                    is_success[self.episode_count%num_record] = done

                    lentropy_losses[self.episode_count % num_record] = lel

                    success_steps = steps[is_success == 1]
                    mean_success_steps = np.mean(success_steps) if sum(is_success) != 0 else 0

                    summary = tf.Summary()
                    summary.value.add(tag='Training/discounted cumulative rewards',
                                      simple_value=np.mean(rewards))
                    summary.value.add(tag='Training/steps', simple_value=mean_success_steps)
                    summary.value.add(tag='Training/success rate', simple_value=np.mean(is_success))
                    summary.value.add(tag='Training/spl', simple_value=self._get_spl(success_records=is_success,
                                                                                     min_steps=min_steps,
                                                                                     steps=steps))
                    summary.value.add(tag='Training/lowlevel_entropy_loss', simple_value=np.mean(lentropy_losses))

                    self.summary_writer.add_summary(summary, self.episode_count)
                    self.summary_writer.flush()

                    if self.episode_count % 1000 == 0 and self.episode_count != 0:
                        self.saver.save(sess, FLAGS.model_path + '/model' + str(self.episode_count) + '.cptk')

                    sess.run(self.episode_increment)

                self.episode_count += 1

    def validate(self, sess):
        np.random.seed(self.name)
        with sess.as_default(), sess.graph.as_default():
            validate_count = -1
            max_validate_count = FLAGS.max_episodes / 1000
            validate_env_idx = range(len(FLAGS.scene_types) * FLAGS.num_train_scenes,
                                     len(FLAGS.scene_types) * (FLAGS.num_train_scenes+FLAGS.num_validate_scenes))
            while validate_count < max_validate_count:
                sleep(4)
                if self.episode_count / 1000 > validate_count:
                    validate_count = self.episode_count / 1000
                    sess.run(self.update_local_ops)

                    rewards = []
                    steps = []
                    min_steps = []
                    is_success = []
                    for _ in range(100):
                        env_idx = np.random.choice(validate_env_idx)
                        env = self.envs[env_idx]
                        all_targets = [t for t in TRAIN_OBJECTS[env.scene_type] if t in env.get_scene_objects()]
                        if len(all_targets) == 0: continue
                        target = np.random.choice(all_targets)
                        target_idx = ALL_OBJECTS_LIST.index(target)

                        result = self._run_training_episode(sess=sess,
                                                            env=env,
                                                            target=target,
                                                            testing=True)
                        if result is None: continue
                        disc_cumu_rewards, action_steps, min_step, done, _, _, _, _ = result
                        rewards.append(disc_cumu_rewards)
                        steps.append(action_steps)
                        min_steps.append(min_step)
                        is_success.append(done)
                    success_steps = np.array(steps)[np.array(is_success) == 1]
                    mean_success_steps = np.mean(success_steps) if sum(is_success) != 0 else 0
                    mean_rewards = np.mean(rewards) if len(rewards) != 0 else 0
                    success_rate = np.mean(is_success) if len(is_success) != 0 else 0
                    spl = self._get_spl(success_records=is_success,
                                        min_steps=min_steps,
                                        steps=steps)
                    if self.name == 0:
                        summary = tf.Summary()
                        summary.value.add(tag='Validate/Seen objects/discounted cumulative rewards', simple_value=mean_rewards)
                        summary.value.add(tag='Validate/Seen objects/steps', simple_value=mean_success_steps)
                        summary.value.add(tag='Validate/Seen objects/success rate', simple_value=success_rate)
                        summary.value.add(tag='Validate/Seen objects/spl', simple_value=spl)
                        self.summary_writer.add_summary(summary, self.episode_count)
                        self.summary_writer.flush()

                    rewards = []
                    steps = []
                    min_steps = []
                    is_success = []
                    for _ in range(100):
                        env_idx = np.random.choice(validate_env_idx)
                        env = self.envs[env_idx]
                        all_targets = [t for t in TEST_OBJECTS[env.scene_type] if t in env.get_scene_objects()]
                        if len(all_targets) == 0: continue
                        target = np.random.choice(all_targets)
                        target_idx = ALL_OBJECTS_LIST.index(target)

                        result = self._run_training_episode(sess=sess,
                                                            env=env,
                                                            target=target,
                                                            testing=True)
                        if result is None: continue
                        disc_cumu_rewards, action_steps, min_step, done, _, _, _, _ = result
                        rewards.append(disc_cumu_rewards)
                        steps.append(action_steps)
                        min_steps.append(min_step)
                        is_success.append(done)
                    success_steps = np.array(steps)[np.array(is_success) == 1]
                    mean_success_steps = np.mean(success_steps) if sum(is_success) != 0 else 0
                    mean_rewards = np.mean(rewards) if len(rewards) != 0 else 0
                    success_rate = np.mean(is_success) if len(is_success) != 0 else 0
                    spl = self._get_spl(success_records=is_success,
                                        min_steps=min_steps,
                                        steps=steps)
                    if self.name == 0:
                        summary = tf.Summary()
                        summary.value.add(tag='Validate/Unseen objects/discounted cumulative rewards', simple_value=mean_rewards)
                        summary.value.add(tag='Validate/Unseen objects/steps', simple_value=mean_success_steps)
                        summary.value.add(tag='Validate/Unseen objects/success rate', simple_value=success_rate)
                        summary.value.add(tag='Validate/Unseen objects/spl', simple_value=spl)
                        self.summary_writer.add_summary(summary, self.episode_count)
                        self.summary_writer.flush()

    def evaluate(self,
                 read_file=''):
        np.random.seed(self.name)
        with tf.Session() as sess:
            self._initialize_network(sess, testing=True)

            rewards = []
            steps = []
            min_steps = []
            is_success = []
            left_steps = []

            with open(read_file, 'r') as f:
                for line in f:
                    nums = line.split()
                    if len(nums) != 8:
                        continue
                    scene_type = nums[0]
                    scene_no = int(nums[1])
                    target = nums[2]
                    start_pos = [float(nums[3]), float(nums[4]), float(nums[5]), int(nums[6]), int(nums[7])]

                    env_idx = (scene_no - 1) * len(FLAGS.scene_types) + FLAGS.scene_types.index(scene_type)
                    # print (scene_type, scene_no, env_idx)
                    env = self.envs[env_idx]
                    # print (env.scene_name)
                    target_idx = ALL_OBJECTS_LIST.index(target)

                    disc_cumu_rewards, episode_steps, min_step, done, left_step, \
                    _, _, _ = self._run_training_episode(sess=sess,
                                                         env=env,
                                                         target=target,
                                                         testing=True,
                                                         start_pos=start_pos)
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


    def _save_trajectory(self,
                         env,
                         target,
                         min_step,
                         states_buffer,
                         options_buffer,
                         actions_buffer):
        print 'len(states_buffer): ' + str(len(states_buffer))

        file_path = 'evaluate_%s.txt' % FLAGS.model_path.split('/')[-2]

        n = len(states_buffer)
        with open(file_path, 'a') as f:
            f.write('%d / %d\n' % (min_step, len(states_buffer)))
            f.write('%s\n' % env.scene_name)
            f.write('%s\n' % target)
            for i in range(n - 1):
                lid = states_buffer[i]
                gid = env.pos2idx[str(lid)]
                oid = options_buffer[i]
                olabel = ALL_OBJECTS_LIST[oid] if oid != -1 else 'random'
                f.write('%d %s %d %d %s %d\n' % (
                i, str(lid), gid, oid, olabel, actions_buffer[i]))
            lid = states_buffer[n - 1]
            gid = env.pos2idx[str(lid)]
            f.write('%d %s %d \n' % (n - 1, str(lid), gid))
            f.write("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

    def test(self):
        np.random.seed(self.name)
        with tf.Session() as sess:
            self._initialize_network(sess, testing=True)
            sess.run(self.update_local_ops)

            rewards = []
            steps = []
            min_steps = []
            is_success = []
            left_steps = []

            while self.episode_count < FLAGS.max_episodes:
                env_idx = np.random.choice(len(FLAGS.scene_types) *
                                           (FLAGS.num_train_scenes + FLAGS.num_validate_scenes + FLAGS.num_test_scenes))
                env = self.envs(env_idx)
                all_targets = env.get_scene_objects()
                if len(all_targets) == 0: continue
                target = np.random.choice(all_targets)
                target_idx = ALL_OBJECTS_LIST.index(target)

                result = self._run_training_episode(sess=sess,
                                                    env=env,
                                                    target=target_idx,
                                                    testing=True,
                                                    start_pos=None)#(10, 1, 2))
                if result is None: continue
                disc_cumu_rewards, episode_steps, min_step, done, left_step, \
                states_buffer, options_buffer, actions_buffer = result
                print (env.scene_name, target)
                print "min_step: " + str(min_step)
                print "episode_step: " + str(episode_steps)

                rewards.append(disc_cumu_rewards)
                steps.append(episode_steps)
                min_steps.append(min_step)
                is_success.append(done)
                left_steps.append(left_step)

                if done and float(min_step) / episode_steps > 0.4:
                    self._save_trajectory(env=env,
                                          target=target,
                                          min_step=min_step,
                                          states_buffer=states_buffer,
                                          options_buffer=options_buffer,
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























