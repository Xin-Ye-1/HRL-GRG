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
                 highlevel_networks,
                 lowlevel_networks,
                 graph,
                 global_episodes,
                 global_frames):
        self.name = name
        self.envs = envs
        self.highlevel_network_main, self.highlevel_network_target = highlevel_networks
        self.lowlevel_network = lowlevel_networks
        self.graph = graph

        self.global_episodes = global_episodes
        self.global_frames = global_frames

        self.episode_increment = self.global_episodes.assign_add(1)
        self.frame_increment = self.global_frames.assign_add(1)

        self.update_local_ops = update_multiple_target_graphs(from_scopes=['highlevel/global/main',
                                                                           'lowlevel/global'],
                                                              to_scopes=['highlevel/local_%d' % self.name,
                                                                         'lowlevel/local_%d' % self.name])
        self.update_target_ops = update_multiple_target_graphs(from_scopes=['highlevel/global/main'],
                                                               to_scopes=['highlevel/global/target'],)

        self.saver = tf.train.Saver(max_to_keep=1)

        if self.name == 0 and FLAGS.is_training:
            self.summary_writer = tf.summary.FileWriter(
                os.path.dirname(FLAGS.model_path) + '/' + str(self.name), graph=tf.get_default_graph())

        self.episode_count = 0
        self.frame_count = 0
        self.lowlevel_replay_buffer = ReplayBuffer()
        self.highlevel_replay_buffer = ReplayBuffer()
        self.exp_counts = np.zeros((NUM_OBJECTS+1, NUM_OBJECTS+1, FLAGS.max_lowlevel_episode_steps + 1))

    def _initialize_network(self,
                            sess,
                            testing=False):
        with sess.as_default():
            if FLAGS.load_model:
                print 'Loading model ...'
                if testing or FLAGS.continuing_training:
                    ckpt = tf.train.get_checkpoint_state(FLAGS.model_path)
                    self.saver.restore(sess, ckpt.model_checkpoint_path)
                    with open(FLAGS.model_path + '/graphcheckpoint.json', 'r') as f:
                        graph_path = json.load(f)['graph_checkpoint_path']
                    graph_params = sio.loadmat(graph_path)['graph']
                    self.graph.set_params(graph_params)

                else:
                    # sess.run(tf.global_variables_initializer())
                    # ckpt = tf.train.get_checkpoint_state(FLAGS.pretrained_model_path)
                    # variable_to_restore = slim.get_variables_to_restore(exclude=['global_episodes', 'global_frames'])
                    # temp_saver = tf.train.Saver(variable_to_restore)
                    # temp_saver.restore(sess, ckpt.model_checkpoint_path)
                    # with open(FLAGS.pretrained_model_path + '/graphcheckpoint.json', 'r') as f:
                    #     graph_path = json.load(f)['graph_checkpoint_path']
                    # graph_params = sio.loadmat(graph_path)['graph']
                    # self.graph.set_params(graph_params)
                    sess.run(tf.global_variables_initializer())
                    ckpt = tf.train.get_checkpoint_state(FLAGS.pretrained_model_path)
                    variable_to_restore = slim.get_variables_to_restore(include=['lowlevel/global'])
                    # var_list = {}
                    # for var in variable_to_restore:
                    #     var_list[var.name.replace('lowlevel/', '').split(':')[0]] = var
                    # temp_saver = tf.train.Saver(var_list)
                    temp_saver = tf.train.Saver(variable_to_restore)
                    temp_saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                sess.run(tf.global_variables_initializer())


    def _get_learning_rate(self,
                           lr):
        if self.episode_count < 500000:
            e = 0
        elif self.episode_count < 3000000:
            e = 1
        elif self.episode_count < 5000000:
            e = 2
        else:
            e = 3
        return lr / (10 ** e)



    def _train_highlevel(self,
                         sess):
        # replay_buffer:
        # [0:vision, 1:depth, 2:subtarget, 3:reward, 4:done, 5:new_vision, 6:new_depth,
        #  7:planning_results, 8:new_visible_objects
        with sess.as_default():
            batch = self.highlevel_replay_buffer.sample(FLAGS.batch_size)

            new_visions = [(batch[i, 5] == t) * batch[i, 7][t] for i in range(FLAGS.batch_size) for t in batch[i, 8]]
            new_depths = [batch[i, 6] for i in range(FLAGS.batch_size) for _ in batch[i, 8]]

            next_q_prime = sess.run(self.highlevel_network_target.q_values,
                                    feed_dict={self.highlevel_network_target.visions:np.stack(new_visions),
                                               self.highlevel_network_target.depths:np.stack(new_depths)})
            next_q_prime = next_q_prime.flatten()
            next_q = sess.run(self.highlevel_network_main.q_values,
                              feed_dict={self.highlevel_network_main.visions:np.stack(new_visions),
                                         self.highlevel_network_main.depths:np.stack(new_depths)})
            next_q = next_q.flatten()

            valid_index = [0]
            s = 0
            for i in range(FLAGS.batch_size):
                s += len(batch[i, 8])
                valid_index.append(s)

            target_q_values = [next_q_prime[valid_index[i]:valid_index[i+1]][np.argmax(next_q[valid_index[i]:valid_index[i+1]])]
                               for i in range(FLAGS.batch_size)]
            target_q_values = batch[:, 3] + (1 - batch[:, 4]) * FLAGS.gamma*target_q_values


            highlevel_lr = self._get_learning_rate(FLAGS.highlevel_lr)

            qvalue_loss, _ = sess.run([self.highlevel_network_main.qvalue_loss,
                                       self.highlevel_network_main.highlevel_update],
                                      feed_dict={self.highlevel_network_main.visions:np.stack(batch[:, 0]),
                                                 self.highlevel_network_main.depths:np.stack(batch[:, 1]),
                                                 self.highlevel_network_main.chosen_objects:batch[:, 2],
                                                 self.highlevel_network_main.target_q_values:target_q_values,
                                                 self.highlevel_network_main.highlevel_lr:highlevel_lr})
            return qvalue_loss



    def _train_lowlevel(self,
                        sess,
                        bootstrap_value):
        # replay_buffer:
        # [0:vision, 1:depth, 2:value 3:action, 4:reward]
        with sess.as_default():
            batch = self.lowlevel_replay_buffer.get_buffer()
            N = batch.shape[0]
            R = bootstrap_value
            discounted_rewards = np.zeros(N)
            for t in reversed(range(N)):
                R = batch[t, 4] + FLAGS.gamma * R
                discounted_rewards[t] = R
            advantages = np.array(discounted_rewards) - np.array(batch[:, 2])

            lowlevel_lr = self._get_learning_rate(FLAGS.lowlevel_lr)

            entropy_loss, _ = sess.run([self.lowlevel_network.entropy_loss,
                                        self.lowlevel_network.lowlevel_update],
                                       feed_dict={self.lowlevel_network.visions: np.stack(batch[:, 0]),
                                                  self.lowlevel_network.depths: np.stack(batch[:, 1]),
                                                  self.lowlevel_network.chosen_actions: batch[:, 3],
                                                  self.lowlevel_network.advantages:advantages,
                                                  self.lowlevel_network.target_values: discounted_rewards,
                                                  self.lowlevel_network.lowlevel_lr: lowlevel_lr})
            return entropy_loss



    def _plan_on_graph(self,
                       valid_options,
                       planning_results):
        trajectories, rewards = planning_results
        all_trajectories = [trajectories[o] for o in valid_options]
        all_rewards = [rewards[o] for o in valid_options]
        distribution = softmax(all_rewards)
        return all_trajectories, all_rewards, distribution


    def _run_training_episode(self,
                              sess,
                              env,
                              target,
                              planning_results,
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
            if testing:
                state = env.start(all_start_positions[np.random.choice(num_start_positions)])
            else:
                scope = max(int(num_start_positions * min(float(self.episode_count + 10) / 10000, 1)), 1) \
                    if FLAGS.curriculum_training and not testing else num_start_positions
                state = env.start(all_start_positions[np.random.choice(scope)])

        min_step = env.get_minimal_steps(target)

        done = False
        sub_done = False

        subtarget = None

        states_buffer = []

        disc_cumu_rewards = 0
        step_extrinsic_cumu_rewards = 0
        disc_extrinsic_cumu_rewards = 0
        disc_intrinsic_cumu_rewards = 0.0
        lowlevel_disc_rewards = 0

        episode_steps = 0
        action_steps = 0
        highlevel_steps = 0
        lowlevel_steps = 0
        avg_lowlevel_steps = 0.0
        avg_lowlevel_sr = 0.0

        gamma = 1
        highlevel_gamma = 1
        lowlevel_gamma = 1

        subtargets_buffer = []
        actions_buffer = []


        highlevel_qvalue_losses = []
        lowlevel_entropy_losses = []

        termination = False
        seen_signal = np.zeros(NUM_OBJECTS+1)
        action = -1

        graph_plan = []

        (ep_start, anneal_steps, ep_end) = FLAGS.epsilon
        if testing:
            highlevel_epsilon = ep_end
        else:
            ratio = max((anneal_steps - max(self.episode_count - 0, 0)) / float(anneal_steps), 0)
            highlevel_epsilon = (ep_start - ep_end) * ratio + ep_end

        depth, semantic = env.get_state_feature(visibility=visibility)
        semantic = [semantic for _ in range(FLAGS.history_steps)]
        depth = [depth for _ in range(FLAGS.history_steps)]
        visible_objects = list(np.unique(semantic).astype('int'))

        if -1 not in visible_objects:
            visible_objects.append(-1)

        max_episode_steps = FLAGS.max_episode_steps[FLAGS.scene_types.index(ALL_SCENES[env.scene_type])]
        for _ in range(max_episode_steps):
            states_buffer.append(env.position)

            if not testing and lowlevel_steps != 0:
                for t in visible_objects:
                    if t != -1 and seen_signal[t] == 0:
                        seen_signal[t] = 1
                        self.exp_counts[subtarget, t, lowlevel_steps-1] += 1

            if highlevel_steps == 0 or sub_done or termination or lowlevel_steps == FLAGS.max_lowlevel_episode_steps:
                if not testing and highlevel_steps != 0 and not termination:
                    for i in range(NUM_OBJECTS):
                        if seen_signal[i] == 0:
                            self.exp_counts[subtarget, i, -1] += 1

                if highlevel_steps != 0:
                    avg_lowlevel_steps += lowlevel_steps
                    avg_lowlevel_sr += sub_done
                    disc_intrinsic_cumu_rewards += lowlevel_disc_rewards
                    disc_extrinsic_cumu_rewards += highlevel_gamma * step_extrinsic_cumu_rewards
                    highlevel_gamma *= FLAGS.gamma

                all_trajectories, all_rewards, distribution = self._plan_on_graph(visible_objects, planning_results)

                if np.random.rand() < highlevel_epsilon:
                    # if not testing:
                    #     print "------------graph planning-------------------"
                    #     # print all_rewards
                    #     print distribution
                    subtarget = np.random.choice(visible_objects, p=distribution)

                else:
                    # if not testing:
                    #     print "-------------Q learning----------------------"
                    semantics = [(np.vstack(semantic) == t)*planning_results[1][t] for t in visible_objects]

                    depths = [np.vstack(depth) for _ in visible_objects]
                    qvalues, features = sess.run([self.highlevel_network_main.q_values,
                                       self.highlevel_network_main.feature],
                                       feed_dict={self.highlevel_network_main.visions: np.stack(semantics),
                                                  self.highlevel_network_main.depths: np.stack(depths)})
                    qvalues = qvalues.flatten()

                    subtarget = visible_objects[int(np.argmax(qvalues))]
                    # subtarget = visible_objects[np.argmax(all_rewards)]

                graph_plan = planning_results[0][subtarget]

                highlevel_steps += 1

                sub_done = env.is_done(subtarget)
                done = env.is_done(target)
                seen_signal[:] = 0
                lowlevel_steps = 0
                lowlevel_disc_rewards = 0
                lowlevel_gamma = 1
                step_extrinsic_cumu_rewards = 0
                action = -1



            if not sub_done:
                refine_subtarget = target if subtarget == -1 else subtarget

                action_policy, value = sess.run([self.lowlevel_network.policy,
                                                 self.lowlevel_network.value],
                                                feed_dict={
                                                    self.lowlevel_network.visions: [(np.vstack(semantic) ==refine_subtarget)],
                                                    self.lowlevel_network.depths: [np.vstack(depth)]})

                action = np.random.choice(NUM_ACTIONS, p=action_policy[0])

                for _ in range(FLAGS.skip_frames):
                    new_state = env.action_step(action)
                    action_steps += 1
                    sub_done = env.is_done(subtarget)
                    done = env.is_done(target)
                    if sub_done or done:
                        break

            intrinsic_reward = 1 if sub_done else 0
            extrinsic_reward = 1 if done else 0

            disc_cumu_rewards += gamma * extrinsic_reward
            gamma *= FLAGS.gamma

            lowlevel_disc_rewards += lowlevel_gamma * intrinsic_reward
            lowlevel_gamma *= FLAGS.gamma

            step_extrinsic_cumu_rewards += extrinsic_reward
            subtargets_buffer.append(subtarget)
            actions_buffer.append(action)

            new_depth, new_semantic = env.get_state_feature(visibility=visibility)
            new_depth = depth[1:] + [new_depth]
            new_semantic = semantic[1:] + [new_semantic]
            new_visible_objects = list(np.unique(new_semantic).astype('int'))
            if -1 not in new_visible_objects:
                new_visible_objects.append(-1)

            termination = ((len(graph_plan) > 1) and (graph_plan[1] in new_visible_objects))
            for g in graph_plan[1:]:
                termination = termination or (g in new_visible_objects)

            if not testing:
                if action != -1 and subtarget != -1:
                    # [0:vision, 1:depth, 2:value 3:action, 4:reward]
                    self.lowlevel_replay_buffer.add(np.reshape(np.array([(np.vstack(semantic) == subtarget),
                                                                         np.vstack(depth),
                                                                         value[0],
                                                                         action,
                                                                         intrinsic_reward]), [1, -1]))

                    if len(self.lowlevel_replay_buffer.buffer) > 0 and \
                            (done or sub_done or termination or
                             (lowlevel_steps != 0 and lowlevel_steps % FLAGS.lowlevel_update_freq == 0) or
                             lowlevel_steps == FLAGS.max_lowlevel_episode_steps - 1):
                        if sub_done:
                            bootstrap_value = 0
                        else:
                            bootstrap_value = sess.run(self.lowlevel_network.value,
                                                       feed_dict={self.lowlevel_network.visions:
                                                                      [(np.vstack(new_semantic) == subtarget)],
                                                                  self.lowlevel_network.depths: [np.vstack(new_depth)]})[0]
                        entropy_loss = self._train_lowlevel(sess=sess, bootstrap_value=bootstrap_value)
                        lowlevel_entropy_losses.append(entropy_loss)
                        self.lowlevel_replay_buffer.clear_buffer()
                        # sess.run(self.update_local_ops)

                # [0:vision, 1:depth, 2:subtarget, 3:reward, 4:done, 5:new_vision, 6:new_depth,
                #  7:planning_results, 8:new_visible_objects
                semantics = (np.vstack(semantic) == subtarget) * planning_results[1][subtarget]
                self.highlevel_replay_buffer.add(np.reshape(np.array([semantics,
                                                                      np.vstack(depth),
                                                                      subtarget,
                                                                      extrinsic_reward,
                                                                      done,
                                                                      np.vstack(new_semantic),
                                                                      np.vstack(new_depth),
                                                                      planning_results[1],
                                                                      new_visible_objects]), [1, -1]))

                if len(self.highlevel_replay_buffer.buffer) >= FLAGS.batch_size and \
                        self.frame_count % FLAGS.highlevel_update_freq == 0:
                    qvalue_loss = self._train_highlevel(sess=sess)
                    highlevel_qvalue_losses.append(qvalue_loss)

                if self.frame_count > 0 and self.frame_count % FLAGS.target_update_freq == 0:
                    sess.run(self.update_target_ops)

                self.frame_count += 1
                if self.name == 0:
                    sess.run(self.frame_increment)

            episode_steps += 1
            lowlevel_steps += 1

            semantic = new_semantic
            depth = new_depth
            visible_objects = new_visible_objects

            if done:
                states_buffer.append(env.position)
                disc_extrinsic_cumu_rewards += highlevel_gamma * step_extrinsic_cumu_rewards

                break

        if testing:
            left_step = env.get_minimal_steps(target)

            return disc_cumu_rewards, episode_steps, min_step, done, left_step, states_buffer, subtargets_buffer, actions_buffer

        hql = np.mean(highlevel_qvalue_losses) if len(highlevel_qvalue_losses) != 0 else 0
        lel = np.mean(lowlevel_entropy_losses) if len(lowlevel_entropy_losses) != 0 else 0

        avg_lowlevel_steps /= highlevel_steps
        avg_lowlevel_sr /= highlevel_steps
        disc_intrinsic_cumu_rewards /= highlevel_steps

        return disc_cumu_rewards, episode_steps, min_step, done, hql, lel, \
               disc_extrinsic_cumu_rewards, highlevel_steps, disc_intrinsic_cumu_rewards, avg_lowlevel_steps, avg_lowlevel_sr

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
            highlevel_rewards = np.zeros(num_record)
            lowlevel_rewards = np.zeros(num_record)
            steps = np.zeros(num_record)
            all_highlevel_steps = np.zeros(num_record)
            all_lowlevel_steps = np.zeros(num_record)
            lowlevel_sr = np.zeros(num_record)
            min_steps = np.zeros(num_record)
            is_success = np.zeros(num_record)

            hqvalue_losses = np.zeros(num_record)
            lentropy_losses = np.zeros(num_record)

            while self.episode_count <= FLAGS.max_episodes:
                sess.run(self.update_local_ops)
                self.graph.update_graph(self.exp_counts)

                env_idx = np.random.choice(range(len(FLAGS.scene_types) * FLAGS.num_train_scenes))
                env = self.envs[env_idx]
                all_targets = [t for t in TRAIN_OBJECTS[env.scene_type] if t in env.get_scene_objects()]
                if len(all_targets) == 0: continue
                target = np.random.choice(all_targets)
                target_idx = ALL_OBJECTS_LIST.index(target)

                planning_results = self.graph.dijkstra_plan(range(NUM_OBJECTS+1), target_idx)

                disc_cumu_rewards, action_steps, min_step, done, hql, lel,\
                disc_extrinsic_cumu_rewards, highlevel_steps, disc_intrinsic_cumu_rewards, avg_lowlevel_steps, avg_lowlevel_sr \
                    = self._run_training_episode(sess=sess,
                                                 env=env,
                                                 target=target_idx,
                                                 planning_results=planning_results,
                                                 testing=False)
                if self.name == 0:
                    print 'episode:{:6}, scene:{:15} target:{:20} reward:{:5} steps:{:5}/{:5} done:{}'.format(
                        self.episode_count, env.scene_name, target, round(disc_cumu_rewards, 2), action_steps, min_step, done)
                    rewards[self.episode_count%num_record] = disc_cumu_rewards
                    highlevel_rewards[self.episode_count%num_record] = disc_extrinsic_cumu_rewards
                    lowlevel_rewards[self.episode_count%num_record] = disc_intrinsic_cumu_rewards
                    steps[self.episode_count%num_record] = action_steps
                    all_highlevel_steps[self.episode_count%num_record] = highlevel_steps
                    all_lowlevel_steps[self.episode_count%num_record] = avg_lowlevel_steps
                    lowlevel_sr[self.episode_count%num_record] = avg_lowlevel_sr
                    min_steps[self.episode_count%num_record] = min_step
                    is_success[self.episode_count%num_record] = done

                    hqvalue_losses[self.episode_count%num_record] = hql
                    lentropy_losses[self.episode_count % num_record] = lel

                    success_steps = steps[is_success == 1]
                    mean_success_steps = np.mean(success_steps) if sum(is_success) != 0 else 0

                    summary = tf.Summary()
                    summary.value.add(tag='Training/discounted cumulative rewards',
                                      simple_value=np.mean(rewards))
                    summary.value.add(tag='Training/highlevel rewards',
                                      simple_value=np.mean(highlevel_rewards))
                    summary.value.add(tag='Training/lowlevel rewards',
                                      simple_value=np.mean(lowlevel_rewards))
                    summary.value.add(tag='Training/steps', simple_value=mean_success_steps)
                    summary.value.add(tag='Training/highlevel steps', simple_value=np.mean(all_highlevel_steps))
                    summary.value.add(tag='Training/lowlevel steps', simple_value=np.mean(all_lowlevel_steps))
                    summary.value.add(tag='Training/success rate', simple_value=np.mean(is_success))
                    summary.value.add(tag='Training/lowlevel success rate', simple_value=np.mean(lowlevel_sr))
                    summary.value.add(tag='Training/spl', simple_value=self._get_spl(success_records=is_success,
                                                                                     min_steps=min_steps,
                                                                                     steps=steps))
                    summary.value.add(tag='Training/highlevel_qvalue_loss', simple_value=np.mean(hqvalue_losses))
                    summary.value.add(tag='Training/lowlevel_entropy_loss', simple_value=np.mean(lentropy_losses))

                    self.summary_writer.add_summary(summary, self.episode_count)
                    self.summary_writer.flush()

                    if self.episode_count % 1000 == 0 and self.episode_count != 0:
                        self.saver.save(sess, FLAGS.model_path + '/model' + str(self.episode_count) + '.cptk')

                        graph_path = FLAGS.model_path + '/graph' + str(self.episode_count) + '.mat'
                        self.graph.save(graph_path)
                        with open(FLAGS.model_path + '/graphcheckpoint.json', 'w') as f:
                            graphcheckpoint = {"graph_checkpoint_path": graph_path}
                            json.dump(graphcheckpoint, f)

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

                        planning_results = self.graph.dijkstra_plan(range(NUM_OBJECTS + 1), target_idx)

                        disc_cumu_rewards, action_steps, min_step, done, _, _, _, _ \
                            = self._run_training_episode(sess=sess,
                                                         env=env,
                                                         target=target_idx,
                                                         planning_results=planning_results,
                                                         testing=True)
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

                        planning_results = self.graph.dijkstra_plan(range(NUM_OBJECTS + 1), target_idx)

                        disc_cumu_rewards, action_steps, min_step, done, _, _, _, _ \
                            = self._run_training_episode(sess=sess,
                                                         env=env,
                                                         target=target_idx,
                                                         planning_results=planning_results,
                                                         testing=True)
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

                    env_idx = (scene_no-1) * len(FLAGS.scene_types) + FLAGS.scene_types.index(scene_type)
                    env = self.envs[env_idx]
                    target_idx = ALL_OBJECTS_LIST.index(target)

                    planning_results = self.graph.dijkstra_plan(range(NUM_OBJECTS+1), target_idx)

                    disc_cumu_rewards, episode_steps, min_step, done, left_step, \
                    states_buffer, options_buffer, actions_buffer = self._run_training_episode(sess=sess,
                                                                                               env=env,
                                                                                               target=target_idx,
                                                                                               testing=True,
                                                                                               planning_results=planning_results,
                                                                                               start_pos=start_pos)
                    # if self.name == 0:
                    #     print 'env_idx:{:5} scene:{:15} target:{:20} start_pos:{:30} reward:{:5} steps:{:5}/{:5} done:{}'.format(
                    #         env_idx, env.scene_name, target, start_pos, round(disc_cumu_rewards, 2), episode_steps,
                    #         min_step, done)


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
                env = self.envs[env_idx]
                all_targets = env.get_scene_objects()
                if len(all_targets) == 0: continue
                target = np.random.choice(all_targets)
                target_idx = ALL_OBJECTS_LIST.index(target)

                start_pos = None

                planning_results = self.graph.dijkstra_plan(range(NUM_OBJECTS+1), target_idx)

                disc_cumu_rewards, episode_steps, min_step, done, left_step, \
                states_buffer, options_buffer, actions_buffer = self._run_training_episode(
                    sess=sess,
                    env=env,
                    target=target_idx,
                    planning_results=planning_results,
                    testing=True,
                    start_pos=start_pos)#(10, 1, 2))

                print (env.scene_name, target)
                print "min_step: " + str(min_step)
                print "episode_step: " + str(episode_steps)

                rewards.append(disc_cumu_rewards)
                steps.append(episode_steps)
                min_steps.append(min_step)
                is_success.append(done)
                left_steps.append(left_step)

                if done and min_step > 6 and float(min_step) / episode_steps > 0.4:
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























