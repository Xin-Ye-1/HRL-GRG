import json
import os
cfg = json.load(open('../config.json', 'r'))
ENV_DIR = os.path.join(cfg['codeDir'], 'env_info')

ALL_SCENES = ["kitchen", "living_room", "bedroom", "bathroom"]
SCENE_NAMES = ['FloorPlan', 'FloorPlan2', 'FloorPlan3', 'FloorPlan4']


# Scene Prior
TRAIN_OBJECTS = [["HousePlant", "StoveKnob", "Sink", "TableTop", "Potato", "Bread", "Tomato", "Knife", "Cabinet",
                  "Fridge", "Container", "ButterKnife", "Lettuce", "Pan", "Bowl", "CoffeeMachine", "StoveBurner",
                  "Plate"],
                 ["Television", "HousePlant", "Chair", "TableTop", "Box", "Cloth", "Newspaper", "KeyChain",
                  "WateringCan", "PaintingHanger"],
                 ["Painting", "HousePlant", "CellPhone", "LightSwitch", "Candle", "TableTop", "Bed", "Lamp", "Statue",
                  "Book", "CreditCard", "KeyChain", "Bowl", "Pen", "Box", "Pencil", "Blinds", "Laptop", "AlarmClock"],
                 ["SprayBottle", "Painting", "Candle", "LightSwitch", "Sink", "Cabinet", "TowelHolder", "Watch",
                  "ToiletPaper", "ShowerDoor", "SoapBottle"]]

TEST_OBJECTS = [["Mug", "Apple", "Microwave", "Toaster"],
                ["Painting", "Statue"],
                ['Television', 'Mirror', 'Cabinet'],
                ["SoapBar", "Towel"]]

ALL_OBJECTS = [TRAIN_OBJECTS[i] + TEST_OBJECTS[i] for i in range(len(ALL_SCENES))]

ALL_OBJECTS_LIST = ['AlarmClock', 'Apple', 'Bed', 'Blinds', 'Book', 'Bowl', 'Box', 'Bread', 'ButterKnife', 'Cabinet',
                    'Candle', 'CellPhone', 'Chair', 'Cloth', 'CoffeeMachine', 'Container', 'CreditCard', 'Fridge',
                    'HousePlant', 'KeyChain', 'Knife', 'Lamp', 'Laptop', 'Lettuce', 'LightSwitch', 'Microwave',
                    'Mirror', 'Mug', 'Newspaper', 'Painting', 'PaintingHanger', 'Pan', 'Pen', 'Pencil', 'Plate',
                    'Potato', 'ShowerDoor', 'Sink', 'SoapBar', 'SoapBottle', 'SprayBottle', 'Statue', 'StoveBurner',
                    'StoveKnob', 'TableTop', 'Television', 'Toaster', 'ToiletPaper', 'Tomato', 'Towel', 'TowelHolder',
                    'Watch', 'WateringCan']


NUM_OBJECTS = len(ALL_OBJECTS_LIST)


# ACTIONS = ["MoveAhead", "MoveBack", "RotateLeft", "RotateRight", "Done"]
ACTIONS = ["MoveAhead", "MoveBack", "RotateLeft", "RotateRight"]
NUM_ACTIONS = len(ACTIONS)

GRID_SIZE = 0.25
H_ROTATION = 90
V_ROTATION = 30

WIDTH = HEIGHT = 30

VISIBILITY_DISTANCE = 1

FOV = 90



