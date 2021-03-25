import numpy as np
import scipy.io as sio
from constant import *
from PIL import Image

np.random.seed(12345)


class Environment():
    def __init__(self,
                 scene_type,
                 scene_no,
                 window_size=30,
                 feature_pattern=('depth', 'depth_norm', 'semantic')
                 ):
        self.scene_type = scene_type
        self.scene_no = scene_no
        self.window_size = window_size
        self.scene_name = self._get_scene_name()
        self.scene_dir = os.path.join(ENV_DIR, self.scene_name)
        self.actions = ACTIONS
        self.feature_pattern = feature_pattern
        self.features = self._load_features()
        self.position = None
        self.lastActionSuccess = None

        self.pos2idx = None
        p = os.path.join(self.scene_dir, 'pos2idx.json')
        if os.path.exists(p):
            self.pos2idx = json.load(open(p, 'r'))

        self.idx2pos = None
        p = os.path.join(self.scene_dir, 'idx2pos.json')
        if os.path.exists(p):
            self.idx2pos = json.load(open(p, 'r'))

        self.minimal_steps = None
        p = os.path.join(self.scene_dir, 'minimal_steps.json')
        if os.path.exists(p):
            self.minimal_steps = json.load(open(p, 'r'))

        self.object_positions = None
        p = os.path.join(self.scene_dir, 'object_positions.json')
        if os.path.exists(p):
            self.object_positions = json.load(open(p, 'r'))

        self.train_positions_by_objects = None
        p = os.path.join(self.scene_dir, 'train_positions_by_objects.json')
        if os.path.exists(p):
            self.train_positions_by_objects = json.load(open(p, 'r'))

        self.visible_positions_by_objects = None
        p = os.path.join(self.scene_dir, 'visible_positions_by_objects_%d.json' % self.window_size)
        if os.path.exists(p):
            self.visible_positions_by_objects = json.load(open(p, 'r'))

    def _load_features(self):
        feature_path = []
        for fp in self.feature_pattern:
            if fp.startswith('resnet'):
                feature_path.append(os.path.join(self.scene_dir, '%s.mat'%fp))
            else:
                feature_path.append(os.path.join(self.scene_dir, '%s_%d.mat'%(fp, self.window_size)))
        # feature_path = [os.path.join(self.scene_dir, '%s_%d.mat'%(fp, self.window_size)) for fp in self.feature_pattern]
        return [sio.loadmat(fp)['feats'] for fp in feature_path if os.path.exists(fp)]

    def _get_scene_name(self):
        if isinstance(self.scene_type, str):
            self.scene_type = ALL_SCENES.index(self.scene_type)
        name = SCENE_NAMES[self.scene_type]
        if name[-1].isdigit():
            scene_name = name + '%02d' % self.scene_no
        else:
            scene_name = name + '%d' % self.scene_no
        return scene_name

    def get_scene_objects(self):
        return self.train_positions_by_objects.keys()

    def get_train_positions(self, obj):
        if isinstance(obj, int):
            obj = ALL_OBJECTS_LIST[obj]
        if obj in self.train_positions_by_objects:
            return self.train_positions_by_objects[obj]
        return []

    def get_visible_positions(self, obj):
        if isinstance(obj, int):
            obj = ALL_OBJECTS_LIST[obj]
        if obj in self.visible_positions_by_objects:
            return self.visible_positions_by_objects[obj]
        return []

    def start(self, position=None):
        self.lastActionSuccess = True
        if str(position) not in self.pos2idx:
            self.lastActionSuccess = False
        if position is None or not self.lastActionSuccess:
            all_positions = self.idx2pos.values()
            position = all_positions[np.random.choice(range(len(all_positions)))]
        self.position = position
        return self.position

    def reset(self, position=None):
        return self.start(position)

    def action_success(self):
        return self.lastActionSuccess

    def action_step(self, action_idx):
        if 0 <= action_idx < len(self.actions):
            action_name = self.actions[action_idx]
            self._get_current_position(action_name)
        return self.position

    def _get_current_position(self, action_name):
        x, y, z, r, h = self.position
        if action_name == 'MoveAhead':
            if r == 0:
                z += GRID_SIZE
            elif r == 90:
                x += GRID_SIZE
            elif r == 180:
                z -= GRID_SIZE
            elif r == 270:
                x -= GRID_SIZE
        elif action_name == 'MoveBack':
            if r == 0:
                z -= GRID_SIZE
            elif r == 90:
                x -= GRID_SIZE
            elif r == 180:
                z += GRID_SIZE
            elif r == 270:
                x += GRID_SIZE
        elif action_name == 'RotateLeft':
            r = (r + 270) % 360
        elif action_name == 'RotateRight':
            r = (r + 90) % 360
        new_pos = [x, y, z, r, h]
        if str(new_pos) in self.pos2idx:
            self.lastActionSuccess = True
            self.position = new_pos
        else:
            self.lastActionSuccess = False

    def get_state_feature(self, state=None, visibility=None):
        if state is None:
            state = self.position
        if isinstance(state, list):
            state = self.pos2idx[str(state)]
        if visibility is None:
            return [feature[state] for feature in self.features]
        depth, depth_norm, semantic = [feature[state] for feature in self.features]
        indices = (depth > visibility)
        semantic[indices] = -1
        return depth_norm, semantic

    def get_visible_objects(self, state=None):
        if state is None:
            state = self.position
        if isinstance(state, list):
            state = self.pos2idx[str(state)]
        feature_index = self.feature_pattern.index('semantic')
        return np.unique(self.features[feature_index][state])

    def get_state_image(self, state=None):
        if state is None:
            state = self.position
        if isinstance(state, list):
            state = self.pos2idx[str(state)]
        img = None
        img_path = os.path.join(self.scene_dir, 'rgb', '%08d.png'%state)
        if os.path.exists(img_path):
            img = Image.open(img_path)
        return img

    def is_done(self, obj, position=None):
        if position is None:
            position = self.position
        if isinstance(obj, int):
            if obj < 0 or obj >= NUM_OBJECTS:
                return False
            obj = ALL_OBJECTS_LIST[obj]
        return obj in self.object_positions and position in self.object_positions[obj]

    def get_minimal_steps(self, obj, position=None):
        if position is None:
            position = self.position
        if isinstance(obj, int):
            obj = ALL_OBJECTS_LIST[obj]
        if obj in self.minimal_steps[str(position)]:
            return self.minimal_steps[str(position)][obj]
        return np.inf



