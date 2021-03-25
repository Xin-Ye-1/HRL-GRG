import os
import json
import cv2
import numpy as np
import scipy.io
import warnings


class Semantic_Environment():
    def __init__(self,
                 env,
                 use_gt=False,
                 vision_feature='_deeplab_depth_logits_10',
                 depth_feature='_deeplab_depth_depth1_10'):
        self.env = env
        self.use_gt = use_gt
        cfg = json.load(open('../config.json','r'))
        code_dir = cfg['codeDir']
        self.dir = '%s/Environment/houses/%s'%(code_dir, self.env)
        self.actions = ['MoveAhead', 'RotateLeft', 'RotateRight', 'MoveBack', 'MoveLeft', 'MoveRight']
        # self.actions = ['MoveAhead', 'RotateLeft', 'RotateRight']
        self._load_map()
        self.vision_feature = self._load_features(vision_feature)
        self.depth_feature = self._load_features(depth_feature)
        if use_gt:
            self.bbox_dir = os.path.join(self.dir, 'bbox')
        else:
            self.bbox_dir = os.path.join(self.dir, 'pred_bbox')
        self._load_gt_bboxes()



    def _load_features(self, feature_pattern):
        feature_path = '%s/rgb_features%s.mat'%(self.dir, feature_pattern)
        return scipy.io.loadmat(feature_path)['feats']

    def _load_gt_bboxes(self):
        self.all_bboxes = {}
        global_img_id = 0
        while True:
            bboxes_path = '%s/%08d.txt'%(self.bbox_dir, global_img_id)
            if not os.path.isfile(bboxes_path):
                break
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                state_bboxes = {}
                bboxes = np.loadtxt(bboxes_path)
                if len(bboxes) != 0:
                    if len(bboxes.shape) == 1:
                        bboxes = np.expand_dims(bboxes, 0)
                    for bbox in bboxes:
                        state_bboxes[int(bbox[0])] = bbox[1:]
                self.all_bboxes[global_img_id] = state_bboxes
                global_img_id += 1


    def _load_map(self):
        self.map = {}
        map_path = '%s/map.txt'%self.dir
        with open(map_path,'r') as f:
            for line in f:
                nums = line.split()
                if len(nums) == 2:
                    self.abs_start_pos = (nums[0], nums[1])
                else:
                    idx = int(nums[0])
                    pos = (int(nums[1]), int(nums[2]))
                    self.map[pos] = idx


    def start(self, position=(0,0,0)):
        return self.reset(position)


    def random_start(self):
        x, y = self.map.keys()[np.random.choice(len(self.map))]
        orien = np.random.choice(4)
        return self.reset((x,y,orien))


    def reset(self, position):
        self.position = position
        self.lastActionSuccess = True
        self.lastOptionSuccess = True
        return self.position

    def action_success(self):
        return self.lastActionSuccess

    def option_success(self):
        return self.lastOptionSuccess

    def action_step(self, action_idx):
        self.position = self._get_current_position(int(action_idx))
        return self.position

    def random_move(self,max_steps = 500, rand = True):
        if rand:
            steps = np.random.randint(max_steps)
        else:
            steps = max_steps
        for i in range(steps):
            a = np.random.randint(len(self.actions))
            self.action_step(a)
        return self.position

    def get_global_state(self):
        (x,y,orien) = self.position
        global_state = self.map[(x,y)] * 4 + orien
        return global_state

    def get_state_image(self, mode):
        img_id = self.get_global_state()
        image_path = '%s/%s/%08d.png'%(self.dir, mode, img_id)
        image = cv2.imread(image_path)
        return image[:,:,::-1]

    def get_state_feature(self, state=None, feature_types=('vision', 'depth')):
        if state is None:
            state = self.position
        if isinstance(state, tuple):
            (x, y, orien) = state
            state = self.map[(x, y)] * 4 + orien
        features = []
        for ft in feature_types:
            if ft == 'vision':
                features.append(self.vision_feature[state])
            if ft == 'depth':
                features.append(self.depth_feature[state])
        return features

    def get_visible_objects(self, state=None):
        if state is None:
            state = self.position
        if isinstance(state, tuple):
            (x, y, orien) = state
            state = self.map[(x, y)] * 4 + orien
        return self.all_bboxes[state].keys()


    def _get_current_position(self, action_idx):
            action_name = self.actions[action_idx]
            (x, y, orien) = self.position
            if action_name == 'MoveLeft':
                if orien == 0:
                    x += 1
                elif orien == 1:
                    y += 1
                elif orien == 2:
                    x -= 1
                elif orien == 3:
                    y -= 1
            elif action_name == 'MoveRight':
                if orien == 0:
                    x -= 1
                elif orien == 1:
                    y -= 1
                elif orien == 2:
                    x += 1
                elif orien == 3:
                    y += 1
            elif action_name == 'MoveAhead':
                if orien == 0:
                    y += 1
                elif orien == 1:
                    x -= 1
                elif orien == 2:
                    y -= 1
                elif orien == 3:
                    x += 1
            elif action_name == 'MoveBack':
                if orien == 0:
                    y -= 1
                elif orien == 1:
                    x += 1
                elif orien == 2:
                    y += 1
                elif orien == 3:
                    x -= 1
            elif action_name == 'RotateRight':
                orien = (orien + 1) % 4
            elif action_name == 'RotateLeft':
                orien = (orien + 3) % 4

            self.lastActionSuccess = True
            if action_name in ['MoveLeft', 'MoveRight', 'MoveAhead', 'MoveBack']:
                if (x, y) not in self.map:
                    self.lastActionSuccess = False
                    (x, y, orien) = self.position

            return (x, y, orien)





if __name__ == '__main__':
    pass

