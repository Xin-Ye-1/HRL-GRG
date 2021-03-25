#!/usr/bin/env python
import sys

sys.path.append('..')
from utils.helper import *
from utils.offline_feature import *
import time
from replay_buffer import *
import json
import tensorflow.contrib.slim as slim
from graph import *
from scipy.special import softmax

flags = tf.app.flags
FLAGS = flags.FLAGS

id2class = json.load(open(os.path.join(cfg['codeDir'], 'Environment', 'id2class.json'), 'r'))
class2id = json.load(open(os.path.join(cfg['codeDir'], 'Environment', 'class2id.json'), 'r'))

cfg = json.load(open('../config.json', 'r'))


np.random.seed(12345)


class Worker():
    def __init__(self,
                 name,
                 envs,
                 scenes,
                 targets,
                 min_steps,
                 starting_points,
                 target_points,
                 highlevel_network_main,
                 highlevel_network_target,
                 lowlevel_network,
                 graph,
                 global_episodes,
                 global_frames):
        self.name = name
        self.envs = envs
        self.scenes = scenes
        self.targets = targets
        self.min_steps = min_steps
        self.starting_points = starting_points
        self.target_points = target_points
        self.highlevel_network_main = highlevel_network_main
        self.highlevel_network_target = highlevel_network_target
        self.lowlevel_network = lowlevel_network
        self.graph = graph

        self.global_episodes = global_episodes
        self.global_frames = global_frames

        self.episode_increment = self.global_episodes.assign_add(1)
        self.frame_increment = self.global_frames.assign_add(1)

        self.update_local_ops = update_multiple_target_graphs(from_scopes=['highlevel/global', 'lowlevel/global'],
                                                              to_scopes=['highlevel/local_%d' % self.name, 'lowlevel/local_%d' % self.name])
        self.update_target_ops = update_target_graph('highlevel/global/main', 'highlevel/global/target')

        self.saver = tf.train.Saver(max_to_keep=1)

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
                    # variable_to_restore = [val for val in variable_to_restore if 'termination' not in val.name]
                    var_list = {}
                    for var in variable_to_restore:
                        var_list[var.name.replace('lowlevel/', '').split(':')[0]] = var
                    temp_saver = tf.train.Saver(var_list)
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



    def _train_highlevel(self,
                         sess):
        # replay_buffer:
        # [0:vision_feature, 1:depth_feature, 2:plannig_results, 3:target_input, 4:subtarget, 5:reward,
        #  6:next_vision_feature, 7:next_depth_feature, 8:next_valid_targets, 9:done]
        with sess.as_default():
            batch = self.highlevel_replay_buffer.sample(FLAGS.batch_size)

            new_visions = [batch[i, 6][:, t] * batch[i, 2][t] for i in range(FLAGS.batch_size) for t in batch[i, 8]]
            new_visions = np.expand_dims(new_visions, -1)
            new_depths = [batch[i, 7] for i in range(FLAGS.batch_size) for _ in batch[i, 8]]

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
            target_q_values = batch[:, 5] + (1 - batch[:, 9]) * FLAGS.gamma*target_q_values
            highlevel_lr = self._get_learning_rate(FLAGS.highlevel_lr)

            qvalue_loss, _ = sess.run([self.highlevel_network_main.qvalue_loss,
                                       self.highlevel_network_main.highlevel_update],
                                      feed_dict={self.highlevel_network_main.visions:np.stack(batch[:, 0]),
                                                 self.highlevel_network_main.depths:np.stack(batch[:, 1]),
                                                 self.highlevel_network_main.chosen_objects:batch[:, 4],
                                                 self.highlevel_network_main.target_q_values:target_q_values,
                                                 self.highlevel_network_main.highlevel_lr:highlevel_lr})
            return qvalue_loss


    def _train_lowlevel(self,
                        sess,
                        bootstrap_value):
        # replay_buffer:
        # [0:vision_feature, 1:depth_feature, 2:target_input, 3:value, 4:action, 5:reward]
        with sess.as_default():
            batch = self.lowlevel_replay_buffer.get_buffer()
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
                                                                          self.lowlevel_network.subtargets: np.stack(
                                                                              batch[:, 2]),
                                                                          self.lowlevel_network.chosen_actions: batch[:,
                                                                                                                4],
                                                                          self.lowlevel_network.advantages: advantages,
                                                                          self.lowlevel_network.target_values: discounted_rewards,
                                                                          self.lowlevel_network.lowlevel_lr: lowlevel_lr,
                                                                          self.lowlevel_network.er: er})
            return loss, policy_loss, value_loss, entropy_loss


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
                              scene,
                              target,
                              min_steps,
                              starting_points,
                              target_points,
                              planning_results,
                              testing=False,
                              start_state=None):
        remove_background = np.ones(FLAGS.num_labels)
        remove_background[-1] = 0
        target_input = np.zeros(FLAGS.num_labels)
        target_input[int(class2id[target])] = 1


        if start_state is not None:
            state = env.start(start_state)
        else:
            num_starting_points = len(starting_points)
            if testing:
                state = env.start(starting_points[np.random.choice(num_starting_points)])
                # state = env.start((10,1,2))
            else:
                scope = max(int(num_starting_points * min(float(self.episode_count + 10) / 10000, 1)), 1) \
                    if FLAGS.curriculum_training and not testing else num_starting_points
                state = env.start(starting_points[np.random.choice(scope)])

        min_step = min_steps[str(state)][target]


        done = False
        sub_done = False

        subtarget = None
        subtarget_id = None
        subtarget_input = None


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

        losses = []
        qvalue_losses = []
        policy_losses = []
        value_losses = []
        entropy_losses = []

        termination = False
        seen_signal = np.zeros(FLAGS.num_labels)
        action = -1

        graph_plan = []

        (ep_start, anneal_steps, ep_end) = FLAGS.epsilon
        if testing:
            epsilon = ep_end
        else:
            ratio = max((anneal_steps - max(self.episode_count - FLAGS.replay_start_size, 0)) / float(anneal_steps), 0)
            epsilon = (ep_start - ep_end) * ratio + ep_end

        vision_feature, depth_feature = env.get_state_feature()
        vision_feature = [vision_feature for _ in range(FLAGS.history_steps)]
        depth_feature = [depth_feature for _ in range(FLAGS.history_steps)]
        visible_targets = env.get_visible_objects()
        list_visible_targets = [visible_targets for _ in range(FLAGS.history_steps)]
        valid_targets = get_distinct_list(list_visible_targets, add_on=FLAGS.num_labels-1)


        for _ in range(FLAGS.max_episode_steps):
            states_buffer.append(env.position)

            if not testing and lowlevel_steps != 0:
                for t in visible_targets:
                    if t != FLAGS.num_labels-1 and seen_signal[t] == 0: #and t != subtarget_id
                        seen_signal[t] = 1
                        self.exp_counts[subtarget_id, t, lowlevel_steps-1] += 1


            if highlevel_steps == 0 or sub_done or termination or lowlevel_steps == FLAGS.max_lowlevel_episode_steps:
                # print (sub_done, termination, lowlevel_steps == FLAGS.max_lowlevel_episode_steps)
                if not testing and highlevel_steps != 0 and not termination:
                    for i in range(FLAGS.num_labels-1):
                        if seen_signal[i] == 0:
                            self.exp_counts[subtarget_id, i, -1] += 1

                if highlevel_steps != 0:
                    avg_lowlevel_steps += lowlevel_steps
                    avg_lowlevel_sr += sub_done
                    disc_intrinsic_cumu_rewards += lowlevel_disc_rewards
                    disc_extrinsic_cumu_rewards += highlevel_gamma * step_extrinsic_cumu_rewards
                    highlevel_gamma *= FLAGS.gamma



                all_trajectories, all_rewards, distribution = self._plan_on_graph(valid_targets, planning_results)

                # print [id2class[str(o)] if o != FLAGS.num_labels-1 else 'background' for o in valid_targets]

                if np.random.rand() < epsilon:
                    # print "------------graph planning-------------------"
                    # print all_rewards
                    # print distribution
                    subtarget_id = np.random.choice(valid_targets, p=distribution)
                    #subtarget_id = np.random.choice(FLAGS.num_labels)
                else:
                    # print "-------------Q learning----------------------"
                    visions = [np.vstack(vision_feature)[:, t]*planning_results[1][t] for t in valid_targets]
                    visions = np.expand_dims(visions, -1)
                    depths = [np.vstack(depth_feature) for _ in valid_targets]
                    qvalues = sess.run(self.highlevel_network_main.q_values,
                                       feed_dict={self.highlevel_network_main.visions: np.stack(visions),
                                                  self.highlevel_network_main.depths: np.stack(depths)})
                    qvalues = qvalues.flatten()

                    # print qvalues
                    # print (max(qvalues[valid_targets]), max(qvalues))
                    # subtarget_id = np.argmax(qvalues)
                    # subtarget_value = max(qvalues[valid_targets])
                    # subtarget_id = list(qvalues).index(subtarget_value)
                    subtarget_id = valid_targets[np.argmax(qvalues)]

                graph_plan = planning_results[0][subtarget_id]
                # print [id2class[str(o)] if o != FLAGS.num_labels-1 else 'background' for o in graph_plan]

                highlevel_steps += 1

                subtarget = id2class[str(subtarget_id)]
                subtarget_input = np.zeros(FLAGS.num_labels)
                subtarget_input[subtarget_id] = 1

                sub_done = state in target_points[subtarget]
                self.lowlevel_replay_buffer.clear_buffer()
                seen_signal[:] = 0
                lowlevel_steps = 0
                lowlevel_disc_rewards = 0
                lowlevel_gamma = 1
                step_extrinsic_cumu_rewards = 0
                action = -1


            if not sub_done:
                action_policy, value = sess.run([self.lowlevel_network.policy,
                                                 self.lowlevel_network.value],
                                                feed_dict={self.lowlevel_network.visions: [np.vstack(vision_feature*remove_background)],
                                                           self.lowlevel_network.depths: [np.vstack(depth_feature)],
                                                           self.lowlevel_network.subtargets: [subtarget_input]})

                action = np.random.choice(FLAGS.a_size, p=action_policy[0])


                for _ in range(FLAGS.skip_frames):
                    new_state = env.action_step(action)
                    action_steps += 1
                    sub_done = new_state in target_points[subtarget]
                    done = new_state in target_points[target]# and subtarget == target
                    if sub_done or done:
                        break

            intrinsic_reward = 1 if sub_done else 0
            extrinsic_reward = 1 if done else 0

            disc_cumu_rewards += gamma * extrinsic_reward
            gamma *= FLAGS.gamma

            lowlevel_disc_rewards += lowlevel_gamma * intrinsic_reward
            lowlevel_gamma *= FLAGS.gamma

            step_extrinsic_cumu_rewards += extrinsic_reward
            subtargets_buffer.append(subtarget_id)
            actions_buffer.append(action)

            new_vision_feature, new_depth_feature = env.get_state_feature()
            new_vision_feature = vision_feature[1:] + [new_vision_feature]
            new_depth_feature = depth_feature[1:] + [new_depth_feature]
            visible_targets = env.get_visible_objects()
            list_visible_targets = list_visible_targets[1:] + [visible_targets]
            valid_targets = get_distinct_list(list_visible_targets, add_on=FLAGS.num_labels-1)

            termination = (len(graph_plan) > 1) and (graph_plan[1] in valid_targets)
            for g in graph_plan[1:]:
                termination = termination or (g in valid_targets)

            if not testing:
                if action != -1 and subtarget_id != FLAGS.num_labels-1:
                    self.lowlevel_replay_buffer.add(np.reshape(np.array([np.vstack(vision_feature),
                                                                         np.vstack(depth_feature),
                                                                         subtarget_input,
                                                                         value,
                                                                         action,
                                                                         intrinsic_reward]), [1, -1]))

                    if sub_done or done or lowlevel_steps == FLAGS.max_lowlevel_episode_steps or termination or\
                            (lowlevel_steps != 0 and lowlevel_steps % FLAGS.lowlevel_update_freq == 0):
                        if sub_done:
                            bootstrap_value = 0
                        else:
                            bootstrap_value = sess.run(self.lowlevel_network.value,
                                                       feed_dict={self.lowlevel_network.visions: [np.vstack(new_vision_feature)],
                                                                  self.lowlevel_network.depths: [np.vstack(new_depth_feature)],
                                                                  self.lowlevel_network.subtargets: [subtarget_input]})[0]

                        loss, policy_loss, value_loss, entropy_loss = self._train_lowlevel(sess=sess,
                                                                                           bootstrap_value=bootstrap_value)
                        self.lowlevel_replay_buffer.clear_buffer()
                        losses.append(loss)
                        policy_losses.append(policy_loss)
                        value_losses.append(value_loss)
                        entropy_losses.append(entropy_loss)

                visions = np.expand_dims(np.vstack(vision_feature)[:, subtarget_id] * planning_results[1][subtarget_id], -1)
                self.highlevel_replay_buffer.add(np.reshape(np.array([visions,
                                                                      np.vstack(depth_feature),
                                                                      planning_results[1],
                                                                      target_input,
                                                                      subtarget_id,
                                                                      step_extrinsic_cumu_rewards,
                                                                      np.vstack(new_vision_feature),
                                                                      np.vstack(new_depth_feature),
                                                                      valid_targets,
                                                                      done]), [1, -1]))

                if self.episode_count > FLAGS.replay_start_size and \
                        len(self.highlevel_replay_buffer.buffer) >= FLAGS.batch_size and \
                        self.frame_count % FLAGS.highlevel_update_freq == 0:
                    qvalue_loss = self._train_highlevel(sess=sess)
                    qvalue_losses.append(qvalue_loss)
                if self.episode_count > FLAGS.replay_start_size and self.frame_count % FLAGS.target_update_freq == 0:
                    sess.run(self.update_target_ops)


            episode_steps += 1
            lowlevel_steps += 1
            self.frame_count += 1
            if self.name == 0:
                sess.run(self.frame_increment)

            vision_feature = new_vision_feature
            depth_feature = new_depth_feature

            if done:
                states_buffer.append(env.position)
                disc_extrinsic_cumu_rewards += highlevel_gamma * step_extrinsic_cumu_rewards
                break

        if testing:
            left_step = min_steps[str(state)][target]

            return disc_cumu_rewards, episode_steps, min_step, done, left_step, states_buffer, subtargets_buffer, actions_buffer

        l = np.mean(losses) if len(losses) != 0 else 0
        pl = np.mean(policy_losses) if len(policy_losses) != 0 else 0
        vl = np.mean(value_losses) if len(value_losses) != 0 else 0
        el = np.mean(entropy_losses) if len(entropy_losses) != 0 else 0
        ql = np.mean(qvalue_losses) if len(qvalue_losses) != 0 else 0

        avg_lowlevel_steps /= highlevel_steps
        avg_lowlevel_sr /= highlevel_steps
        disc_intrinsic_cumu_rewards /= highlevel_steps

        return disc_cumu_rewards, episode_steps, min_step, done, l, pl, vl, el, ql, \
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
                         min_step,
                         states_buffer,
                         options_buffer,
                         actions_buffer):
        print 'len(states_buffer): ' + str(len(states_buffer))

        file_path = 'evaluate_%s.txt' % FLAGS.model_path.split('/')[-2]
        loc2idx = self._load_map_loc2idx(scene)

        n = len(states_buffer)
        with open(file_path, 'a') as f:
            f.write('%d / %d\n' % (min_step, len(states_buffer)))
            f.write('%s\n' % scene)
            f.write('%s\n' % target)
            for i in range(n - 1):
                lid = states_buffer[i]
                gid = self._local2global(loc2idx, lid)
                oid = options_buffer[i]
                olabel = id2class[str(oid)]
                f.write('%d %s %d %d %s %d\n' % (
                i, str(lid), gid, oid, olabel, actions_buffer[i]))
            lid = states_buffer[n - 1]
            gid = self._local2global(loc2idx, lid)
            f.write('%d %s %d \n' % (n - 1, str(lid), gid))
            f.write("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

    def test(self):
        np.random.seed(12345)
        with tf.Session() as sess:
            self._initialize_network(sess, testing=True)
            # sess.run(self.update_local_ops)
            self.episode_count = 0
            self.frame_count = 0

            # [t, s, s', o, a, r, done]
            self.lowlevel_replay_buffer = ReplayBuffer()
            self.highlevel_replay_buffer = ReplayBuffer()

            rewards = []
            steps = []
            min_steps = []
            is_success = []
            left_steps = []

            while self.episode_count < FLAGS.max_episodes:
                sid = np.random.choice(len(self.scenes))
                tid = np.random.choice(len(self.targets[sid]))
                scene = self.scenes[sid]
                target = self.targets[sid][tid]

                # scene = '5cf0e1e9493994e483e985c436b9d3bc'
                # target = 'television'
                # sid = self.scenes.index(scene)
                # tid = self.targets[sid].index(target)

                env = self.envs[sid]


                starting_points = self.starting_points[sid][tid]
                target_points = self.target_points[sid]
                scene_min_steps = self.min_steps[sid]

                planning_results = self.graph.dijkstra_plan(range(FLAGS.num_labels), int(class2id[target]))

                disc_cumu_rewards, episode_steps, min_step, done, left_step, \
                states_buffer, options_buffer, actions_buffer = self._run_training_episode(
                    sess=sess,
                    env=env,
                    scene=scene,
                    target=target,
                    min_steps=scene_min_steps,
                    starting_points=starting_points,
                    target_points=target_points,
                    planning_results=planning_results,
                    testing=True,
                    start_state=None)#(10, 1, 2))

                print (scene, target)
                print "min_step: " + str(min_step)
                print "episode_step: " + str(episode_steps)

                rewards.append(disc_cumu_rewards)
                steps.append(episode_steps)
                min_steps.append(min_step)
                is_success.append(done)
                left_steps.append(left_step)

                if done and float(min_step) / episode_steps > 0.4:
                    self._save_trajectory(scene=scene,
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

    def work(self,
             sess):
        print 'starting worker %s' % str(self.name)
        np.random.seed(self.name)
        with sess.as_default(), sess.graph.as_default():
            self._initialize_network(sess)
            self.episode_count = sess.run(self.global_episodes)
            self.frame_count = sess.run(self.global_frames)

            # [t, s, s', o, a, r, done]
            self.lowlevel_replay_buffer = ReplayBuffer()
            self.highlevel_replay_buffer = ReplayBuffer()

            self.exp_counts = np.zeros((FLAGS.num_labels, FLAGS.num_labels, FLAGS.max_lowlevel_episode_steps + 1))

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

            losses = np.zeros(num_record)
            policy_losses = np.zeros(num_record)
            value_losses = np.zeros(num_record)
            entropy_losses = np.zeros(num_record)
            qvalue_losses = np.zeros(num_record)

            if self.name == 0:
                self.summary_writer = tf.summary.FileWriter(
                    os.path.dirname(FLAGS.model_path) + '/' + str(self.name), graph=tf.get_default_graph())

            while self.episode_count < FLAGS.max_episodes:
                sess.run(self.update_local_ops)
                self.graph.update_graph(self.exp_counts)

                sid = np.random.choice(len(self.scenes))
                scene = self.scenes[sid]
                env = self.envs[sid]
                tid = np.random.choice(len(self.targets[sid]))
                target = self.targets[sid][tid]
                starting_points = self.starting_points[sid][tid]
                target_points = self.target_points[sid]
                scene_min_steps = self.min_steps[sid]

                planning_results = self.graph.dijkstra_plan(range(FLAGS.num_labels), int(class2id[target]))

                disc_cumu_rewards, action_steps, min_step, done, l, pl, vl, el, ql, \
                disc_extrinsic_cumu_rewards, highlevel_steps, disc_intrinsic_cumu_rewards, avg_lowlevel_steps, avg_lowlevel_sr \
                    = self._run_training_episode(sess=sess,
                                                 env=env,
                                                 scene=scene,
                                                 target=target,
                                                 min_steps=scene_min_steps,
                                                 starting_points=starting_points,
                                                 target_points=target_points,
                                                 planning_results=planning_results,
                                                 testing=False)
                if self.name == 0:
                    print 'episode:{:6}, scene:{} target:{:20} reward:{:5} steps:{:5}/{:5} done:{}'.format(
                        self.episode_count, scene, target, round(disc_cumu_rewards, 2), action_steps, min_step, done)
                    rewards[self.episode_count%num_record] = disc_cumu_rewards
                    highlevel_rewards[self.episode_count%num_record] = disc_extrinsic_cumu_rewards
                    lowlevel_rewards[self.episode_count%num_record] = disc_intrinsic_cumu_rewards
                    steps[self.episode_count%num_record] = action_steps
                    all_highlevel_steps[self.episode_count%num_record] = highlevel_steps
                    all_lowlevel_steps[self.episode_count%num_record] = avg_lowlevel_steps
                    lowlevel_sr[self.episode_count%num_record] = avg_lowlevel_sr
                    min_steps[self.episode_count%num_record] = min_step
                    is_success[self.episode_count%num_record] = done

                    losses[self.episode_count%num_record] = l
                    policy_losses[self.episode_count%num_record] = pl
                    value_losses[self.episode_count%num_record] = vl
                    entropy_losses[self.episode_count%num_record] = el
                    qvalue_losses[self.episode_count%num_record] = ql

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
                    summary.value.add(tag='Loss/loss', simple_value=np.mean(losses))
                    summary.value.add(tag='Loss/policy_loss', simple_value=np.mean(policy_losses))
                    summary.value.add(tag='Loss/value_loss', simple_value=np.mean(value_losses))
                    summary.value.add(tag='Loss/entropy_loss', simple_value=np.mean(entropy_losses))
                    summary.value.add(tag='Loss/qvalue_loss', simple_value=np.mean(qvalue_losses))

                    self.summary_writer.add_summary(summary, self.episode_count)
                    self.summary_writer.flush()

                    if self.episode_count % 1000 == 0 and self.episode_count != 0:
                        self.saver.save(sess, FLAGS.model_path + '/model' + str(self.episode_count) + '.cptk')

                        graph_path = FLAGS.model_path + '/graph' + str(self.episode_count) + '.mat'
                        self.graph.save(graph_path)
                        with open(FLAGS.model_path + '/graphcheckpoint.json', 'w') as f:
                            graphcheckpoint = {"graph_checkpoint_path": graph_path}
                            json.dump(graphcheckpoint, f)

                    # if self.episode_count % 10 == 0:
                        # self.graph.update_graph(self.exp_counts)
                        # self.exp_counts[:] = 0

                    sess.run(self.episode_increment)

                self.episode_count += 1


    def evaluate(self,
                 read_file='../random_method/1s1t.txt'):
        np.random.seed(12345)
        with tf.Session() as sess:
            self._initialize_network(sess, testing=True)
            self.episode_count = 0
            self.frame_count = 0

            self.lowlevel_replay_buffer = ReplayBuffer()
            self.highlevel_replay_buffer = ReplayBuffer()

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

                    env = self.envs[sid]

                    starting_points = self.starting_points[sid][tid]
                    target_points = self.target_points[sid]
                    scene_min_steps = self.min_steps[sid]

                    planning_results = self.graph.dijkstra_plan(range(FLAGS.num_labels), int(class2id[target]))

                    disc_cumu_rewards, episode_steps, min_step, done, left_step, \
                    _, _, _ = self._run_training_episode(sess=sess,
                                                         env=env,
                                                         scene=scene,
                                                         target=target,
                                                         min_steps=scene_min_steps,
                                                         starting_points=starting_points,
                                                         target_points=target_points,
                                                         testing=True,
                                                         planning_results=planning_results,
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
                           positions,
                           target_id):
        mode = 'gt' if FLAGS.use_gt else 'pred'
        semantic_dynamic = json.load(open('%s/%s_dynamic.json' % (env_dir, mode), 'r'))
        valid_options = [target_id]
        for pos in positions:
            transition = semantic_dynamic[str(pos)]
            options_str = transition.keys()
            valid_options += [int(o) for o in options_str if int(o) != FLAGS.num_labels-1]
        return set(valid_options)
        # return valid_options
























