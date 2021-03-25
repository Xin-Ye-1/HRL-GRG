import sys
sys.path.append('..')
import tensorflow.contrib.slim as slim
from utils.environment import *
from utils.helper import *
from replay_buffer import *
import tensorflow as tf
import os
import time
from time import sleep
flags = tf.app.flags
FLAGS = flags.FLAGS


cfg = json.load(open('../config.json', 'r'))

class Worker():
    def __init__(self,
                 name,
                 DQN_network_main,
                 DQN_network_target,
                 global_episodes,
                 global_frames):
        self.name = name
        self.DQN_network_main = DQN_network_main
        self.DQN_network_target = DQN_network_target
        self.global_episodes = global_episodes
        self.global_frames = global_frames

        self.episode_increment = self.global_episodes.assign_add(1)
        self.frame_increment = self.global_frames.assign_add(1)

        self.update_local_ops = update_multiple_target_graphs(from_scopes=['global'],
                                                              to_scopes=['local_%d' % self.name])
        self.update_target_ops = update_target_graph('global/main', 'global/target')

        self.saver = tf.train.Saver(max_to_keep=1)

        if self.name == 0 and FLAGS.evaluate_during_training:
            self.summary_writer = tf.summary.FileWriter(
                os.path.dirname(FLAGS.model_path) + '/' + str(self.name), graph=tf.get_default_graph())

        self.episode_count = 0
        self.frame_count = 0
        self.replay_buffer = ReplayBuffer()

    def _initialize_network(self,
                            sess,
                            testing=False):
        with sess.as_default():
            if FLAGS.load_model or testing:
                print 'Loading model ...'
                if testing or FLAGS.continuing_training:
                    ckpt = tf.train.get_checkpoint_state(FLAGS.model_path)
                    self.saver.restore(sess, ckpt.model_checkpoint_path)
                else:
                    ckpt = tf.train.get_checkpoint_state(FLAGS.pretrained_model_path)
                    self.saver.restore(sess, ckpt.model_checkpoint_path)

                    sess.run(tf.global_variables_initializer())
                    ckpt = tf.train.get_checkpoint_state(FLAGS.pretrained_model_path)
                    variable_to_restore = slim.get_variables_to_restore(exclude=['global_episodes', 'global_frames'])
                    temp_saver = tf.train.Saver(variable_to_restore)
                    temp_saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                sess.run(tf.global_variables_initializer())

    def _train(self,
               sess):
        # replay_buffer: [0:goal, 1:state, 2:action, 3:reward, 4:done, 5:new_state]
        with sess.as_default():
            batch = self.replay_buffer.sample(FLAGS.batch_size)
            next_q_prime = sess.run(self.DQN_network_target.qvalues,
                                    feed_dict={self.DQN_network_target.state: np.stack(batch[:, 5]),
                                               self.DQN_network_target.goal: np.stack(batch[:, 0])})
            next_q = sess.run(self.DQN_network_main.qvalues,
                              feed_dict={self.DQN_network_main.state: np.stack(batch[:, 5]),
                                         self.DQN_network_main.goal: np.stack(batch[:, 0])})

            target_qvalues = batch[:, 3] + (1 - batch[:, 4]) * FLAGS.gamma * \
                             (next_q_prime[range(FLAGS.batch_size), np.argmax(next_q, axis=-1)])
            qvalue_loss, _ = sess.run([self.DQN_network_main.qvalue_loss,
                                       self.DQN_network_main.update],
                                      feed_dict={self.DQN_network_main.state: np.stack(batch[:, 1]),
                                                 self.DQN_network_main.goal: np.stack(batch[:, 0]),
                                                 self.DQN_network_main.chosen_actions: batch[:, 2],
                                                 self.DQN_network_main.target_q_values: target_qvalues,
                                                 self.DQN_network_main.lr: FLAGS.lowlevel_lr})
            return qvalue_loss

    def _run_training_episode(self,
                              sess,
                              scene,
                              target,
                              testing=False,
                              start_pos=None):
        env = Environment(scene)
        target_input = np.zeros(FLAGS.num_goals)
        target_input[target] = 1
        if start_pos is not None:
            start_pos, min_step = env.start(start_pos, target)
        else:
            # sorted_start_positions = env.get_sorted_start_positions_for_approaching(target, around_size=FLAGS.window_size/2)
            sorted_start_positions = env.get_sorted_start_positions(target)
            num_start_positions = len(sorted_start_positions)
            scope = max(int(num_start_positions * min(float(self.episode_count + 10) / 10000, 1)), 1) \
                    if FLAGS.curriculum_training and not testing else num_start_positions
            start_pos, min_step = env.start(sorted_start_positions[np.random.choice(scope)], target)

        done = False
        states_buffer = []
        actions_buffer = []
        disc_cumu_rewards = 0
        gamma = 1
        episode_steps = 0
        action_steps = 0
        qvalue_losses = []

        state = env.get_around_state(FLAGS.window_size / 2, state_type='map_goal')
        # state = env.get_full_state()
        state = [state for _ in range(FLAGS.history_steps)]

        for _ in range(FLAGS.max_episode_steps):
            states_buffer.append(env.position)
            (ep_start, anneal_steps, ep_end) = FLAGS.epsilon
            if testing:
                epsilon = ep_end
            else:
                ratio = max((anneal_steps - self.episode_count) / float(anneal_steps),
                            0)
                epsilon = (ep_start - ep_end) * ratio + ep_end

            if np.random.rand() < epsilon:
                action = np.random.choice(FLAGS.num_actions)
            else:
                qvalues = sess.run(self.DQN_network_main.qvalues,
                                   feed_dict={self.DQN_network_main.state:[np.stack(state)],
                                              self.DQN_network_main.goal:[target_input]})
                action = np.argmax(qvalues)
                # print (scene, target, env.position)
                # print qvalues


            actions_buffer.append(action)

            new_pos = None
            reward = 0
            for _ in range(FLAGS.skip_frames):
                new_pos, reward, done = env.action_step(action)
                action_steps += 1
                if done:
                    break


            disc_cumu_rewards += gamma * reward
            gamma *= FLAGS.gamma

            new_state = state[1:] + [env.get_around_state(FLAGS.window_size/2, state_type='map_goal')]
            # new_state = env.get_full_state()
            if not testing:
                self.replay_buffer.add(np.reshape(np.array([target_input,
                                                            np.stack(state),
                                                            action,
                                                            reward,
                                                            done,
                                                            np.stack(new_state),
                                                            ]), [1, -1]))
                if len(self.replay_buffer.buffer) >= FLAGS.batch_size and \
                        (self.frame_count % FLAGS.lowlevel_update_freq == 0 or done):
                    qvalue_loss = self._train(sess)
                    qvalue_losses.append(qvalue_loss)
                if self.frame_count > 0 and self.frame_count % FLAGS.target_update_freq == 0:
                    sess.run(self.update_target_ops)

                self.frame_count += 1
                if self.name == 0:
                    sess.run(self.frame_increment)

            state = new_state
            episode_steps += 1


            if done:
                states_buffer.append(env.position)
                break

        if testing:
            return disc_cumu_rewards, action_steps, min_step, done, states_buffer, actions_buffer

        ql = np.mean(qvalue_losses) if len(qvalue_losses) != 0 else 0
        return disc_cumu_rewards, action_steps, min_step, done, ql



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

    def work(self, sess):
        print 'starting worker %s'%str(self.name)
        np.random.seed(self.name)
        with sess.as_default(), sess.graph.as_default():
            self._initialize_network(sess)
            self.episode_count = sess.run(self.global_episodes)
            self.frame_count = sess.run(self.global_frames)
            self.replay_buffer = ReplayBuffer()

            num_records = 100
            rewards = np.zeros(num_records)
            steps = np.zeros(num_records)
            min_steps = np.zeros(num_records)
            is_success = np.zeros(num_records)
            qvalue_losses = np.zeros(num_records)

            while self.episode_count <= FLAGS.max_episodes:
                sess.run(self.update_local_ops)

                scene = '%s/train/map_%04d' % (FLAGS.env_dir, np.random.choice(FLAGS.num_envs))
                # [2, 5, 10, 13]
                target = np.random.choice([0, 1, 3, 4, 6, 7, 8, 9, 11, 12, 14, 15])

                disc_cumu_rewards, action_steps, min_step, done, ql = self._run_training_episode(sess=sess,
                                                                                                 scene=scene,
                                                                                                 target=target)
                if self.name == 0:
                    print 'episode:{:6}, scene:{} target:{:5} reward:{:5} steps:{:5}/{:5} done:{}'.format(
                        self.episode_count, scene, target, round(disc_cumu_rewards, 2), action_steps, min_step, done)
                    rewards[self.episode_count % num_records] = disc_cumu_rewards
                    steps[self.episode_count % num_records] = action_steps
                    min_steps[self.episode_count % num_records] = min_step
                    is_success[self.episode_count % num_records] = done
                    qvalue_losses[self.episode_count % num_records] = ql
                    success_steps = steps[is_success == 1]
                    mean_success_steps = np.mean(success_steps) if sum(is_success) != 0 else 0

                    summary = tf.Summary()
                    summary.value.add(tag='Training/discounted cumulative rewards', simple_value=np.mean(rewards))
                    summary.value.add(tag='Training/steps', simple_value=mean_success_steps)
                    summary.value.add(tag='Training/success rate', simple_value=np.mean(is_success))
                    summary.value.add(tag='Training/spl', simple_value=self._get_spl(success_records=is_success,
                                                                                     min_steps=min_steps,
                                                                                     steps=steps))
                    summary.value.add(tag='Training/qvalue_loss', simple_value=np.mean(qvalue_losses))

                    self.summary_writer.add_summary(summary, self.episode_count)
                    self.summary_writer.flush()

                    if self.episode_count % 1000 == 0 and self.episode_count != 0:
                        self.saver.save(sess, FLAGS.model_path + '/model' + str(self.episode_count) + '.cptk')
                    sess.run(self.episode_increment)

                self.episode_count += 1

    def evaluate(self, sess):
        seeds = [1, 5, 13, 45, 99]
        # np.random.seed(self.name)
        with sess.as_default(), sess.graph.as_default():
            selected_scenes = []
            selected_targets = []
            selected_pos = []
            if FLAGS.evaluate_file != '':
                with open(FLAGS.evaluate_file, 'r') as f:
                    for line in f:
                        if not line.startswith('SR:'):
                            nums = line.split()
                            scene = nums[0]
                            target = int(nums[1])
                            start_state = (int(nums[2]), int(nums[3]))
                            selected_scenes.append(scene)
                            selected_targets.append(target)
                            selected_pos.append(start_state)

            evaluate_count = 0

            max_evaluate_count = FLAGS.max_episodes / 1000 if FLAGS.evaluate_during_training else 1

            while evaluate_count < max_evaluate_count:
                sleep(5)
                if self.episode_count / 1000 > evaluate_count or not FLAGS.evaluate_during_training:
                    if FLAGS.evaluate_during_training:
                        evaluate_count = self.episode_count / 1000
                    else:
                        evaluate_count += 1
                        self._initialize_network(sess, testing=True)
                    sess.run(self.update_local_ops)
                    SR = []
                    AS = []
                    MS = []
                    SPL = []
                    AR = []

                    for seed in seeds:
                        np.random.seed(seed)
                        rewards = []
                        steps = []
                        min_steps = []
                        is_success = []
                        for i in range(len(selected_scenes)):
                            scene = selected_scenes[i]
                            target = selected_targets[i]
                            start_pos = selected_pos[i]
                            disc_cumu_rewards, action_steps, min_step, done, _, _ = self._run_training_episode(sess=sess,
                                                                                                               scene=scene,
                                                                                                               target=target,
                                                                                                               testing=True,
                                                                                                               start_pos=start_pos)
                            rewards.append(disc_cumu_rewards)
                            steps.append(action_steps)
                            min_steps.append(min_step)
                            is_success.append(done)

                        success_steps = np.array(steps)[np.array(is_success) == 1]
                        mean_success_steps = np.mean(success_steps) if sum(is_success) != 0 else 0
                        success_min_steps = np.array(min_steps)[np.array(is_success) == 1]
                        mean_success_min_steps = np.mean(success_min_steps) if sum(is_success) != 0 else 0
                        mean_rewards = np.mean(rewards) if len(rewards) != 0 else 0
                        success_rate = np.mean(is_success) if len(is_success) != 0 else 0
                        spl = self._get_spl(success_records=is_success,
                                            min_steps=min_steps,
                                            steps=steps)

                        SR.append(success_rate)
                        AS.append(mean_success_steps)
                        MS.append(mean_success_min_steps)
                        SPL.append(spl)
                        AR.append(mean_rewards)

                    print 'mean'
                    print 'evaluate:{:6}, SR:{:5} AS:{:5}/{:5} SPL:{:5} AR:{:5}'.format(
                        evaluate_count,
                        round(np.mean(SR), 2),
                        round(np.mean(AS), 2),
                        round(np.mean(MS), 2),
                        round(np.mean(SPL), 2),
                        round(np.mean(AR), 2))
                    print 'var'
                    print 'evaluate:{:6}, SR:{:5} AS:{:5}/{:5} SPL:{:5} AR:{:5}'.format(
                        evaluate_count,
                        round(np.var(SR), 2),
                        round(np.var(AS), 2),
                        round(np.var(MS), 2),
                        round(np.var(SPL), 2),
                        round(np.var(AR), 2))

                    if self.name == 0 and FLAGS.evaluate_during_training:
                        summary = tf.Summary()
                        summary.value.add(tag='Evaluate/discounted cumulative rewards', simple_value=np.mean(AR))
                        summary.value.add(tag='Evaluate/steps', simple_value=np.mean(AS))
                        summary.value.add(tag='Evaluate/success rate', simple_value=np.mean(SR))
                        summary.value.add(tag='Evaluate/spl', simple_value=np.mean(SPL))
                        self.summary_writer.add_summary(summary, self.episode_count)
                        self.summary_writer.flush()

    def test(self):
        with tf.Session() as sess:
            self._initialize_network(sess, testing=True)
            sess.run(self.update_local_ops)
            rewards = []
            steps = []
            min_steps = []
            is_success = []

            while self.episode_count < FLAGS.max_episodes:
                # scene = '%s/train/map_%04d' % (FLAGS.env_dir, np.random.choice(FLAGS.num_envs))
                # target = np.random.choice(FLAGS.num_goals)

                scene = '%s/valid/map_%04d' % (FLAGS.env_dir, 16)
                target = 2
                start_postions = [4, 6]
                disc_cumu_rewards, action_steps, min_step, done, states_bufffer, actions_buffer \
                    = self._run_training_episode(sess=sess,
                                                 scene=scene,
                                                 target=target,
                                                 testing=True,
                                                 start_pos=start_postions)
                rewards.append(disc_cumu_rewards)
                steps.append(action_steps)
                min_steps.append(min_step)
                is_success.append(done)

                self._save_trajectory(scene=scene,
                                      target=target,
                                      states_buffer=states_bufffer,
                                      actions_buffer=actions_buffer)

                self.episode_count += 1

            success_steps = steps[is_success == 1]
            mean_success_steps = np.mean(success_steps) if sum(is_success) != 0 else 0
            mean_rewards = np.mean(rewards) if len(rewards) != 0 else 0
            success_rate = np.mean(is_success) if len(is_success) != 0 else 0
            spl = self._get_spl(success_records=is_success,
                                min_steps=min_steps,
                                steps=steps)

            print 'SR:{:5} AS:{:5} SPL:{:5} AR:{:5}'.format(
                round(success_rate, 2), round(mean_success_steps, 2), round(spl, 2),
                round(mean_rewards, 2))


    def _save_trajectory(self,
                         scene,
                         target,
                         states_buffer,
                         actions_buffer,
                         options_buffer=None):
        print 'len(states_buffer): ' + str(len(states_buffer))

        file_path = 'evaluate_%s.txt' % FLAGS.model_path.split('/')[-2]

        n = len(states_buffer)
        with open(file_path, 'a') as f:
            f.write('%s\n' % scene)
            f.write('%s\n' % target)
            for i in range(n - 1):
                lid = states_buffer[i]
                oid = options_buffer[i] if options_buffer is not None else 'None'
                f.write('%d %s %s %d\n' % (
                i, str(lid), str(oid), actions_buffer[i]))
            lid = states_buffer[n - 1]
            f.write('%d %s \n' % (n - 1, str(lid)))
            f.write("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")



if __name__ == '__main__':
    pass


