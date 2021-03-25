import heapq

class PriorityQueue:
    def __init__(self):
        self.heap = []
        self.count = 0

    def push(self, item, priority):
        entry = (priority, self.count, item)
        heapq.heappush(self.heap, entry)
        self.count += 1

    def pop(self):
        (_,_,item) = heapq.heappop(self.heap)
        return item

    def isEmpty(self):
        return len(self.heap) == 0

class PriorityQueueWithFunction(PriorityQueue):
    def __init__(self, priorityFunction):
        self.priorityFunction = priorityFunction
        PriorityQueue.__init__(self)

    def push(self, item):
        PriorityQueue.push(self, item, self.priorityFunction(item))



def manhattanDistance(pos1, pos2):
    return abs(pos1[0]-pos2[0]) + abs(pos1[1]-pos2[1])


def heuristic(start,targets):
    min_dis = float('inf')
    for t in targets:
        dis = manhattanDistance(start[:2],t[:2]) + abs(start[2]-t[2])
        if dis < min_dis:
            min_dis = dis
    return min_dis



def aStarSearch(env, start, targets):
    if start in targets:
        return [start],0

    def priorityFunction(item):
        return item[-1][1]+item[-1][2]

    # queue item: a list of (state, accumulated cost, heuristic)
    queue = PriorityQueueWithFunction(priorityFunction)
    queue.push([(start, 0, heuristic(start, targets))])
    visited_pos = []
    while not queue.isEmpty():
        item = queue.pop()
        pos = item[-1][0]
        cost = item[-1][1]
        if pos not in visited_pos:
            visited_pos.append(pos)
            for a in range(len(env.actions)):
                env.start(pos)
                for _ in range(5):
                    new_pos = env.action_step(a)
                    if new_pos in targets:
                        trajectory = [i[0] for i in item]
                        trajectory.append(new_pos)
                        return trajectory, cost+1
                heu =  heuristic(new_pos, targets)
                new_item = item[:]
                new_item.append((new_pos, cost+1, heu))
                queue.push(new_item)



def uniformCostSearch(env, start, targets):
    if start in targets:
        return [start],0

    def priorityFunction(item):
        return item[-1][1]

    # queue item: a list of (state, accumulated cost)
    queue = PriorityQueueWithFunction(priorityFunction)
    queue.push([(start, 0)])
    visited_pos = []
    while not queue.isEmpty():
        item = queue.pop()
        pos = item[-1][0]
        cost = item[-1][1]
        if pos not in visited_pos:
            visited_pos.append(pos)
            for a in range(len(env.actions)):
                env.start(pos)
                for i in range(1):
                    new_pos = env.action_step(a)
                    if new_pos in targets:
                        trajectory = [i[0] for i in item]
                        trajectory.append(new_pos)
                        return trajectory, cost+1
                heu = heuristic(new_pos, targets)
                new_item = item[:]
                new_item.append((new_pos, cost+1, heu))
                queue.push(new_item)
    return None, None