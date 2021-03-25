import numpy as np
from utils.shortest_path import PriorityQueueWithFunction
import scipy.io as sio
import json

class Dirichlet_Multimodel_Graph():
    def __init__(self, num_nodes, num_categories, gamma):
        self.num_nodes = num_nodes
        self.num_categories = num_categories
        self.graph = np.zeros((num_nodes, num_nodes, num_categories)) # posterior
        self.params = np.zeros((num_nodes, num_nodes, num_categories)) # alpha
        self.gamma = gamma

    def set_params(self, params):
        assert params.shape == (self.num_nodes, self.num_nodes, self.num_categories)
        self.params = params
        self.graph = params


    def update_graph(self, exp_counts):
        assert exp_counts.shape == (self.num_nodes, self.num_nodes, self.num_categories)
        alpha0 = np.sum(self.params, axis=2, keepdims=True)
        N = np.sum(exp_counts, axis=2, keepdims=True)
        self.graph = (self.params + exp_counts)/(alpha0+N)


    def get_expectation(self, ni, nj):
        prob = self.graph[ni, nj, :]
        values = [self.gamma ** k for k in range(self.num_categories-1)]
        values.append(0)
        return sum(prob*values)


    def plan(self, ns, nt):
        if ns == nt:
            return [ns], 1 #self.get_expectation(ns, nt)

        def priorityFunction(item):
            return -item[-1][1]

        # queue item: a list of (node, accumulated discounted reward)
        queue = PriorityQueueWithFunction(priorityFunction)
        queue.push([(ns, 1)])
        visited_nodes = []
        while not queue.isEmpty():
            item = queue.pop()
            (nc, reward) = item[-1]
            if nc == nt:
                trajectory = [i[0] for i in item]
                return trajectory, reward #*self.get_expectation(nc, nt)
            if nc not in visited_nodes:
                visited_nodes.append(nc)
                for ni in range(self.num_nodes):
                    if ni != nc:
                        new_reward = reward*self.get_expectation(nc, ni)
                        new_item = item[:]
                        new_item.append((ni, new_reward))
                        queue.push(new_item)
        return None, None

    def dijkstra_plan(self, ns_list, nt):
        rewards = [-1 for _ in range(self.num_nodes)]
        previous_node = [None for _ in range(self.num_nodes)]
        rewards[nt] = 1  # self.get_expectation(nt, nt)
        seen = []
        while sum([ns in seen for ns in ns_list]) < len(ns_list):
            max_reward = -1
            nt = -1
            for i, r in enumerate(rewards):
                if i not in seen and max_reward < r:
                    max_reward = r
                    nt = i
            seen.append(nt)
            for ni in range(self.num_nodes):
                if ni not in seen and rewards[nt] * self.get_expectation(ni, nt) > rewards[ni]:
                    rewards[ni] = rewards[nt] * self.get_expectation(ni, nt)
                    previous_node[ni] = nt
        all_trajectories = []
        all_rewards = []
        for ns in ns_list:
            trajectory = []
            n = ns
            while n != None:
                trajectory.append(n)
                n = previous_node[n]
            all_trajectories.append(trajectory)
            all_rewards.append(rewards[ns])
        return all_trajectories, all_rewards

    def save(self,
             file_path):
        matdata = {'graph': self.graph}
        sio.savemat(file_path, matdata)



if __name__ == '__main__':
    pass







