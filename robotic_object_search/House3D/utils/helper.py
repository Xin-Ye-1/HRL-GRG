#!/usr/bin/env python
import tensorflow as tf
import numpy as np
import scipy.signal
from scipy import misc
import scipy.io
from PIL import Image
import json
import os
from offline_feature import *
from bbox_tool import *
import glob
from reward_function import *
from semantic_environment import *
from shortest_path import *

IMAGE_WIDTH = 600
IMAGE_HEIGHT = 450

# cfg = json.load(open('../config.json','r'))
cfg = json.load(open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.json'),'r'))

def get_distinct_list(inputs, add_on=None, remove=None):
    result = []
    if add_on is not None:
        result.append(add_on)
    for input in inputs:
        for element in input:
            if element != remove and element not in result:
                result.append(element)
    return result

def global2loc(global_id, map):
    orien = global_id % 4
    idx = global_id / 4
    (x, y) = map[idx]
    return (x, y, orien)


def load_map(scene_dir):
    map = {}
    map_path = '%s/map.txt'%scene_dir
    with open(map_path,'r') as f:
        for line in f:
            nums = line.split()
            if len(nums) == 3:
                idx = int(nums[0])
                pos = (int(nums[1]), int(nums[2]))
                map[idx] = pos
    return map


def get_starting_points_according_to_distance(scene, targets):
    starting_points = []
    all_starting_points = json.load(open('%s/Environment/houses/%s/starting_points_according_to_distance_1.json' %
                                         (cfg['codeDir'], scene), 'r'))

    def string2tuple(string):
        string = string.replace('(', '').replace(')','')
        x,y,orien = string.split(',')
        return(int(x),int(y), int(orien))

    for target in targets:
        str_starting_points = all_starting_points[target] if target in all_starting_points else []


        starting_points.append([string2tuple(s) for s in str_starting_points])
    return starting_points


def sort_starting_points_according_to_distance(scene, targets, starting_points):
    min_steps = json.load(open('%s/Environment/houses/%s/minimal_steps_1.json' %
                                         (cfg['codeDir'], scene), 'r'))
    sorted_starting_points = []
    for i, target in enumerate(targets):
        dis_starting_points = [(min_steps[str(pos)][target], pos) for pos in starting_points[i]]
        sorted_pos = sorted(dis_starting_points)
        sorted_pos = [item[-1] for item in sorted_pos]
        sorted_starting_points.append(sorted_pos)
    return sorted_starting_points

def get_starting_points(scene, targets, use_gt=True, use_semantic=False):
    if use_semantic:
        #print targets
        feature_tool = Feature_Tool(scene_name=scene, feature_pattern='_deeplab_depth_semantic_10')
        map = load_map(feature_tool.feature_dir)
        num_states = len(feature_tool.all_states_features)
        starting_points = [[] for i in range(len(targets))]
        class2id = json.load(open('%s/Environment/class2id.json' % cfg['codeDir'], 'r'))
        for global_id in range(num_states):
            semantic = feature_tool.get_state_feature(global_id)
            unique_labels, counts = np.unique(semantic, return_counts=True)
            for i, target in enumerate(targets):
                target_id = class2id[target]
                area = counts[unique_labels==target_id]
                _, done,_ = increasing_area_reward(scene, target, area, area)
                if area>0 and not done:
                    starting_point = global2loc(global_id, map)
                    # target_points = get_target_points(scene, [target])[0]
                    # min_steps, _ = get_minimal_steps(scene, [starting_point], target_points)
                    # if min_steps[0] is not None:
                    starting_points[i].append(starting_point)

        #print np.array(starting_points).shape
        return starting_points
    else:
        bbox_tool = Bbox_Tool(scene, use_gt=use_gt)
        map = load_map(bbox_tool.house_dir)
        starting_points = [[] for i in range(len(targets))]
        num_states = len(glob.glob(os.path.join(bbox_tool.bbox_dir, '*')))
        for global_id in range(num_states):
            for i, target in enumerate(targets):
                # threshold = get_threshod(scene,target)
                (x,y,w,h) = bbox_tool.get_gt_bbox(global_id, target)
                # x1 = x - (1.0 * w) / 2
                # y1 = y - (1.0 * h) / 2
                # x2 = x1 + w
                # y2 = y1 + h
                # if x1 != 0 and y1 != 0 and x2 != IMAGE_WIDTH - 1 and y2 != IMAGE_HEIGHT - 1:
                if w*h != 0: # and w*h < threshold:
                    starting_points[i].append(global2loc(global_id, map))
                    #starting_points[i].append(global_id)
        return starting_points

def get_target_points(scene, targets, use_gt=True):
    bbox_tool = Bbox_Tool(scene, use_gt=use_gt)
    map = load_map(bbox_tool.house_dir)
    # target_points = [[] for _ in range(len(targets))]
    target_points = {}
    for t in targets:
        target_points[t] = []
    num_states = len(glob.glob(os.path.join(bbox_tool.bbox_dir, '*')))
    for global_id in range(num_states):
        for i, target in enumerate(targets):
            threshod = get_threshod(scene, target, use_gt=use_gt)
            (x,y,w,h) = bbox_tool.get_gt_bbox(global_id, target)
            if w*h >= threshod:
                # target_points[i].append(global2loc(global_id, map))
                target_points[target].append(global2loc(global_id, map))

    # with open('all_target_positions.json','wb') as f:
    #     json.dump(target_points, f)

    return target_points

def get_minimal_steps(scene, starting_points, target_points):
    steps = []
    trajectories = []
    env = Semantic_Environment(scene)
    for starting_point in starting_points:
        trajectory, step = uniformCostSearch(env, starting_point, target_points)
        steps.append(step)
        trajectories.append(trajectory)

    return steps, trajectories


def update_target_graph(from_scope, to_scope, tau=1):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var.value()*tau + (1-tau)*to_var.value()))
    return op_holder

def update_multiple_target_graphs(from_scopes, to_scopes, tau=1):
    op_holder = []
    for from_scope, to_scope in zip(from_scopes, to_scopes):
        op_holder += update_target_graph(from_scope, to_scope, tau)
    return op_holder






