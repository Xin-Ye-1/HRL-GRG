import tensorflow as tf
from constant import *
import scipy.io as sio
import os
import numpy as np


def string2list(string):
    string = string.replace('[', '').replace(']', '')
    x, y, z, r, h = string.split(',')
    return [float(x), float(y), float(z), int(r), int(h)]


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


def normalize_depth_map():
    max_d = -np.inf
    min_d = np.inf

    for name in SCENE_NAMES:
        for i in range(1, 31):
            if name[-1].isdigit():
                scene_name = name + '%02d'%i
            else:
                scene_name = name + '%d'%i
            depth = sio.loadmat(os.path.join(ENV_DIR, scene_name, 'depth_%d'%HEIGHT))['feats']
            maximum = np.amax(depth)
            minimum = np.amin(depth)
            if maximum > max_d:
                max_d = maximum
            if minimum < min_d:
                min_d = minimum
    print (min_d, max_d)
    for name in SCENE_NAMES:
        for i in range(1, 31):
            if name[-1].isdigit():
                scene_name = name + '%02d'%i
            else:
                scene_name = name + '%d'%i
            depth = sio.loadmat(os.path.join(ENV_DIR, scene_name, 'depth_%d'%HEIGHT))['feats']
            norm_depth = (depth - min_d) / (max_d - min_d)
            matdata = {'feats': norm_depth}
            sio.savemat(os.path.join(ENV_DIR, scene_name, 'depth_norm_%d'%HEIGHT), matdata)

