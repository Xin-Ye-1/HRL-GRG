import scipy.io as sio
import json
import numpy as np

cfg = json.load(open('../config.json', 'r'))

class Environment():
    def __init__(self, env):
        self.env = '%s/utils/%s'%(cfg['codeDir'], env)
        self.map = sio.loadmat(self.env)['map']
        self.goals = json.load(open(self.env.replace('map_','goals_')+'.json', 'r'))
        self.goals_map = self._get_goals_map()
        self.actions = ['Up', 'Down', 'Left', 'Right']

    def get_sorted_start_positions(self, goal, start_from=0):
        sorted_positions = []
        distances = [int(s) for s in self.goals[str(goal)].keys()]
        for s in sorted(distances):
            if s > start_from:
                sorted_positions += self.goals[str(goal)][str(s)]
        return sorted_positions

    def get_sorted_start_positions_for_approaching(self, goal, around_size):
        all_start_positions = self.get_sorted_start_positions(goal)
        gx, gy = self.goals[str(goal)]['0'][0]
        return [(x, y) for (x, y) in all_start_positions if abs(x-gx) <= around_size and abs(y-gy) <= around_size]

    def _get_goals_map(self):
        goals_map = np.zeros((self.map.shape[0],
                              self.map.shape[1],
                              len(self.goals.keys())+1))
        goals_map[:, :, -1] = 1-self.map
        goals = [int(g) for g in self.goals.keys()]
        for goal in sorted(goals):
            gx, gy = self.goals[str(goal)]['0'][0]
            goals_map[gx, gy, int(goal)] = 1
            goals_map[gx, gy, -1] = 0
        return goals_map


    def start(self, position, goal):
        return self.reset(position, goal)


    def random_start(self, goal, start_from=0):
        valid_positions = self.get_sorted_start_positions(goal, start_from)
        random_position = valid_positions[np.random.choice(len(valid_positions))]
        return self.reset(random_position, goal)

    def reset(self, position, goal):
        min_step = 0
        for s, p in self.goals[str(goal)].items():
            if list(position) in p:
                min_step = int(s)
                break
        assert min_step != 0
        self.position = position
        self.goal = goal
        self.lastActionSuccess = True
        return self.position, min_step

    def goals_success(self):
        goal = -1
        x, y = self.position
        if np.sum(self.goals_map[x, y, :-1]) == 1:
            goal = np.argmax(self.goals_map[x, y, :-1])
        return int(goal)

    def action_success(self):
        return self.lastActionSuccess

    def get_around_state(self, around_size, goal=None, state_type='map', add_on=False):
        if goal is None:
            goal = self.goal
        pad_map = np.ones((self.map.shape[0]+2*around_size,
                           self.map.shape[1]+2*around_size))
        pad_map[around_size:self.map.shape[0]+around_size][:, around_size:self.map.shape[1]+around_size] = self.map
        x, y = self.position
        map_state = np.expand_dims(pad_map[x:x+2*around_size+1][:, y:y+2*around_size+1], axis=-1)
        if state_type == 'map':
            return map_state
        if state_type == 'map_goal':
            if int(goal) in range(self.goals_map.shape[2]):
                goal_map = np.zeros((self.map.shape[0]+2*around_size,
                                     self.map.shape[1]+2*around_size))
                goal_map[around_size:self.map.shape[0]+around_size][:, around_size:self.map.shape[1]+around_size] \
                    = self.goals_map[:,:, int(goal)]
                goal_state = np.expand_dims(goal_map[x:x+2*around_size+1][:, y:y+2*around_size+1], axis=-1)
            else:
                goal_state = np.zeros((2*around_size+1, 2*around_size+1, 1))
            map_goal_state = np.concatenate((map_state, goal_state), axis=-1)
            return map_goal_state
        if state_type == 'map_all_goals':
            add_on_size = 1 if add_on else 0
            goals_map = np.zeros((self.map.shape[0]+2*around_size,
                                  self.map.shape[1]+2*around_size,
                                  len(self.goals.keys()) + add_on_size))

            goals_map[around_size:self.map.shape[0]+around_size][:, around_size:self.map.shape[1]+around_size][:] \
                 = self.goals_map if add_on else self.goals_map[:, :, :-1]
            goals_state = goals_map[x:x+2*around_size+1][:, y:y+2*around_size+1][:]
            map_goals_state = np.concatenate((map_state, goals_state), axis=-1)
            return map_goals_state

    def get_full_state(self, goal=None):
        if goal is None:
            goal = self.goal
        position_state = np.zeros(self.map.shape)
        position_state[self.position] = 1

        return np.concatenate((np.expand_dims(self.map, axis=-1),
                               np.expand_dims(position_state, axis=-1),
                               np.expand_dims(self.goals_map[:, :, goal], axis=-1)), axis=-1)

    def get_goal_state(self, around_size, goal=None, state_type='single', add_on=False):
        if goal is None:
            goal = self.goal
        x, y = self.position
        if state_type == 'single':
            if str(goal) in self.goals:
                goal_map = np.zeros((self.map.shape[0] + 2 * around_size,
                                     self.map.shape[1] + 2 * around_size))
                goal_map[around_size:self.map.shape[0] + around_size][:, around_size:self.map.shape[1] + around_size] \
                    = self.goals_map[:, :, int(goal)]
                goal_state = np.expand_dims(goal_map[x:x + 2 * around_size + 1][:, y:y + 2 * around_size + 1], axis=-1)
            else:
                goal_state = np.zeros((2 * around_size + 1, 2 * around_size + 1, 1))
            return goal_state
        if state_type == 'all':
            add_on_size = 1 if add_on else 0
            goals_map = np.zeros((self.map.shape[0] + 2 * around_size,
                                  self.map.shape[1] + 2 * around_size,
                                  len(self.goals.keys())+add_on_size))

            goals_map[around_size:self.map.shape[0] + around_size][:, around_size:self.map.shape[1] + around_size][:] \
                = self.goals_map if add_on else self.goals_map[:, :, :-1]
            goals_state = goals_map[x:x + 2 * around_size + 1][:, y:y + 2 * around_size + 1][:]
            return goals_state


    def get_visible_goals(self, around_size):
        x, y = self.position
        min_x = max(x-around_size, 0)
        max_x = min(x+around_size, self.map.shape[0])
        min_y = max(y-around_size, 0)
        max_y = min(y+around_size, self.map.shape[1])
        local_goals = self.goals_map[min_x:max_x+1][:, min_y:max_y+1]
        visible_goals = []
        cur_goal = self.goals_success()
        for i in range(local_goals.shape[-1]):
            if np.sum(local_goals[:, :, i]) == 1: # and i != cur_goal
                visible_goals.append(i)
        return visible_goals




    def action_step(self, action):
        if isinstance(action, int):
            assert action in range(len(self.actions))
            action = self.actions[action]
        elif isinstance(action, str):
            assert action in self.actions
        self.position = self._get_current_position(action)
        done = list(self.position) == self.goals[str(self.goal)]['0'][0]
        reward = 1 if done else 0
        return self.position, reward, done


    def _get_current_position(self, action_name):
        (x, y) = self.position
        if action_name == 'Up':
            x -= 1
        elif action_name == 'Down':
            x += 1
        elif action_name == 'Left':
            y -= 1
        elif action_name == 'Right':
            y += 1
        if self.map[(x, y)] == 0:
            self.lastActionSuccess = True
        else:
            self.lastActionSuccess = False
            (x, y) = self.position
        return (x, y)






if __name__ == '__main__':
    path = 'maps_16X16_v6/train/map_0000'
    env = Environment(path)
    a = env.goals['0'].values()
    b = np.concatenate(env.goals['0'].values())
    env.start((7, 7), 0)
    print env.goals_success()
    # start_postions = env.get_sorted_start_positions_for_approaching(12, 3)
    # state = env.get_around_state(3, 16, state_type='map_goal')
    # print state.shape
    # list_state = [state for _ in range(3)]
    # list_state += [state]
    # print np.stack(list_state).shape

    # print env.get_around_state(3, 16, state_type='map_goal')[:,:,1]
    # print env.get_full_state(0)[:,:, 2]
    # for pos in start_postions:
    #     env.position = pos
    #     print pos
    #     print env.get_visible_goals(3)
    #     print '~~~~~~~~~~'
