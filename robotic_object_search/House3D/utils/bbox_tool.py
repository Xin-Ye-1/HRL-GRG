#!/usr/bin/env python

import os
import numpy as np
import json
import warnings
class Bbox_Tool():

    def __init__(self, scene_name, use_gt=True):
        cfg = json.load(open('../config.json','r'))
        code_dir = cfg['codeDir']
        self.environment_dir = os.path.join(code_dir,'Environment')
        self.house_dir = os.path.join(self.environment_dir, 'houses', scene_name)
        if use_gt:
            self.bbox_dir = os.path.join(self.house_dir, 'bbox')
        else:
            self.bbox_dir = os.path.join(self.house_dir, 'pred_bbox')
        self.scene_name = scene_name
        self._load_map()
        self._load_class_idx_mapping()
        self._load_gt_bboxes()

    def _load_map(self):
        self.map = {}
        map_path = '%s/map.txt'%self.house_dir
        with open(map_path,'r') as f:
            for line in f:
                nums = line.split()
                if len(nums) == 2:
                    self.abs_start_pos = (nums[0],nums[1])
                else:
                    idx = int(nums[0])
                    pos = (int(nums[1]), int(nums[2]))
                    self.map[pos] = idx

    def _load_class_idx_mapping(self):
        self.class_idx_mapping = json.load(open('%s/class2id.json'%self.environment_dir,'r'))
        


    def _load_gt_bboxes(self):
        self.all_gt_bboxes = []
        global_img_id = 0
        while True:
            bboxes_path = '%s/%08d.txt'%(self.bbox_dir,global_img_id)
            if not os.path.isfile(bboxes_path):
                break
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                bboxes = np.loadtxt(bboxes_path)
                self.all_gt_bboxes.append(bboxes)
                global_img_id += 1
        

                   
    def get_gt_bbox(self, state, target_name):
        global_state = state
        if isinstance(state,tuple):
            (x,y,orien)=state
            global_state = self.map[(x,y)]*4 + orien

        if target_name == 'background':
            return (0,0,0,0)
                   
        target_idx = int(self.class_idx_mapping[target_name])

        x_c = 0
        y_c = 0
        w = 0
        h = 0
        
        bboxes = self.all_gt_bboxes[global_state]
        if len(bboxes)!=0 and len(bboxes.shape) == 1:
            bboxes = np.expand_dims(bboxes,0)
        for bbox in bboxes:
            if bbox[0] == target_idx:
                x_c = bbox[1]
                y_c = bbox[2]
                w = bbox[3]
                h = bbox[4]

        return (x_c,y_c,w,h)



    
    
            
        
        
        
