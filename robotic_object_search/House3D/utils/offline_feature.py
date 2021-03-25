#!/usr/bin/env python

import numpy as np
import scipy.io
import os
import json
from semantic_environment import *
#from os.path import dirname



class Feature_Tool():

    def __init__(self, scene_name, model=None, feature_pattern=''):
        cfg = json.load(open('../config.json','r'))
        code_dir = cfg['codeDir']
        relative_feature_dir = 'Environment/houses'
        self.feature_dir = os.path.join(code_dir,relative_feature_dir,scene_name)
        self.scene_name = scene_name
        self.feature_pattern = feature_pattern
        self._load_features()
        self._load_map()

    def _load_features(self):
        state_feature_path = '%s/rgb_features%s.mat'%(self.feature_dir, self.feature_pattern)
        self.all_states_features = scipy.io.loadmat(state_feature_path)['feats']
        # self.semantic_dynamic = json.load(open('%s/pred_dynamic.json' % (self.feature_dir), 'r'))

    def _load_map(self):
        self.map = {}
        map_path = '%s/map.txt'%self.feature_dir
        with open(map_path,'r') as f:
            for line in f:
                nums = line.split()
                if len(nums) == 2:
                    self.abs_start_pos = (nums[0],nums[1])
                else:
                    idx = int(nums[0])
                    pos = (int(nums[1]), int(nums[2]))
                    self.map[pos] = idx


    def get_state_feature(self, state):
        if isinstance(state, tuple):
            (x,y,orien) = state
            global_state = self.map[(x,y)]*4 + orien           
            return self.all_states_features[global_state]
        if isinstance(state, int):
            return self.all_states_features[state]

    def get_all_history_features(self,states,steps = 4):    
        state_features = []
        for i,state in enumerate(states):
            features = []
            k = i-steps+1
            while k<=i:
                s = states[0] if k<0 else states[k]
                features.append(self.get_state_feature(s))
                k+=1
            state_features.append(np.vstack(features))
        return state_features

    def get_last_history_features(self,states,steps=4):
        features = []
        k = len(states)-steps
        while k<=len(states)-1:
            s = states[0] if k<0 else states[k]
            features.append(self.get_state_feature(s))
            k+=1
        return np.vstack(features)

    def get_target_feature(self,target_name):
        target_feature_path = '%s/targets_features%s/%s.mat'%(self.feature_dir,self.feature_pattern,target_name)
        if os.path.isfile(target_feature_path):
            feature = scipy.io.loadmat(target_feature_path)['feat']
            return np.squeeze(feature)
        return np.array([])
        


    def get_steps_target_features(self,target_name,steps=4):
        features = np.array([])
        target_feature = self.get_target_feature(target_name)
        for _ in range(steps):
            features = np.concatenate((features,target_feature))
        return features

    def get_states_valid_options(self, states, all_options=78):
        all_states_options=[]
        env = Semantic_Environment(self.scene_name)
        for state in states:
            env.start(state)
            valid_objects = env.get_options()
            if all_options - 1 not in valid_objects:
                valid_objects += [all_options- 1]
            valid_options = np.zeros(all_options)
            valid_options[valid_objects] = 1
            all_states_options.append(valid_options)
        return all_states_options


if __name__ == '__main__':
    pass





