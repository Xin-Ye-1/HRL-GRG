
# Hierarchical and Partially Observable Goal-driven Policy Learning with Goals Relational Graph

<img src="https://github.com/Xin-Ye-1/HRL-GRG/blob/main/overview.png" width="80%" height="80%" align=center />

This is the source code for our HRL-GRG model and the baseline methods we mentioned in the paper. 

[paper](https://arxiv.org/pdf/2103.01350.pdf) | [project webpage](https://xin-ye-1.github.io/HRL-GRG/)

## Requirements

Our code is developed and tested under the following dependencies:

- python==2.7.15
- scipy==1.2.0
- numpy==1.15.4
- tensorflow==1.6.0
- tf Slim 
- opencv==3.2.0-dev

Before running the code, please specify the path to the code directory in the `config.json` that can be found in both the `grid_world`, the `robotic_object_search/House3D` and the `robotic_object_search/AI2-THOR` directory.

Before running the robotic object search code on AI2-THOR, please download [our pre-processed data](https://drive.google.com/file/d/141D4AtkXXsTVQLnzhIioVc9Lyv2j3loA/view?usp=sharing) sourced from [AI2-THOR](https://ai2thor.allenai.org/ithor/) and extract at the `robotic_object_search/AI2-THOR` directory.

Before running the robotic object search code on House3D, please download [our pre-processed data](https://drive.google.com/file/d/1sJwEkEGkeD2QoxaWCR3nD_Knsl-dW3__/view?usp=sharing) sourced from [House3D](https://github.com/facebookresearch/house3d) and extract at the `robotic_object_search/House3D` directory.

Download [our pre-trained models](https://drive.google.com/file/d/15AhLJh4J1uJAEy33PlaLVK0Yl1KWj4Vx/view?usp=sharing) and put them in the corresponding code directories for training and/or evaluating our method.

## Training

### Grid-world domain

To train our model `HRL-GRG` in the paper, run this command:
```bash
# From grid-world/HRL-GRG/
./train.sh
```
To train other baseline methods mentioned in the paper, run the same command from the corresponding directories.

### Robotic Object Search

#### AI2-THOR

To train our model `HRL-GRG` in the paper, run this command:

```bash
# Specify the parameters in robotic_object_search/AI2-THOR/HRL-GRG/train.sh, 
# and from robotic_object_search/AI2-THOR/HRL-GRG/
./train.sh
```
or

```bash
# From robotic_object_search/AI2-THOR/HRL-GRG/
python train.py \
    --pretrained_model_path=${PATH_TO_PRETRAINED_MODEL} \
    --model_path=${PATH_TO_MODEL} 
```

where the `pretrained_model_path` is `../A3C/result_pretrain/model`.


#### House3D

To train our model `HRL-GRG` in the paper, run this command:

```bash
# Specify the parameters in robotic_object_search/House3D/HRL-GRG/train.sh, 
# and from robotic_object_search/House3D/HRL-GRG/
./train.sh
```
or

```bash
# From robotic_object_search/House3D/HRL-GRG/
python train.py \
    --default_scenes=<enviroments_to_train> \
    --default_targets=<target_objects_to_train> \
    --pretrained_model_path=${PATH_TO_PRETRAINED_MODEL} \
    --model_path=${PATH_TO_MODEL} 
```

where the `pretrained_model_path` is `../A3C/result_se_for_pretrain/model` for the single environment setting, and `../A3C/result_me_for_pretrain/model` for the multiple environments setting.

To train other baseline methods mentioned in the paper, run the same command from the corresponding directories.



## Evaluation and Results

### Grid-world domain

To evaluate our method `HRL-GRG` on the grid-world domain, and reproduce the results of our method on the unseen grid-world maps for seen goals as follows,

| Unseen Envs Seen Goals  | SR    |    AS / MS   | SPL  |
| :-----------------------| :----:|:------------:|:----:|
| HRL-GRG                 | 0.57  | 28.71 / 9.03 | 0.33 |

run this command:

```bash
# From grid_world/HRL-GRG/
CUDA_VISIBLE_DEVICES=-1 python evaluate.py \
  --evaluate_file='../random_method/maps_16X16_v6_valid_seengoals.txt'
```
To reproduce the results for the `unseen goals` and the `overall goals`, specify the `evaluate_file` as `'../random_method/maps_16X16_v6_valid_unseengoals.txt'` and `'../random_method/maps_16X16_v6_valid_total.txt'` respectively.

To evaluate other baseline methods, run the same command from the corresponding directories.

### Robotic Object Search

#### AI2-THOR

To evaluate our method `HRL-GRG` for the robotic object search task on AI2-THOR, and reproduce the results of our method on the seen environments for seen goals as follows,

| Seen Env Seen Goals     | SR    | SPL  |
| :-----------------------| :----:|:----:|
| HRL-GRG                 | 0.74  | 0.34 |

run this command,

```bash
# From robotic_object_search/AI2-THOR/HRL-GRG/
CUDA_VISIBLE_DEVICES=-1 python evaluate.py \
  --model_path="result_pretrain/model" \
  --evaluate_file='../random_method/ssso.txt' 
```
To reproduce the results for the `seen environments unseen goals`, `unseen environments seen goals` and `unseen environments unseen goals`, specify the `evaluate_file` as `'../random_method/ssuo.txt'`, `'../random_method/usso.txt'` and `'../random_method/usuo.txt'` respectively.

To evaluate the corresponding `Random` method, run the following command with the `evaluate_file` being specified respectively.

```bash
# From robotic_object_search/AI2-THOR/random_method/
CUDA_VISIBLE_DEVICES=-1 python random_walk.py \
  --evaluate_file='../random_method/ssso.txt' 
```

#### House3D

To evaluate our method `HRL-GRG` for the robotic object search task on House3D,  

- run the command,
```bash
# From robotic_object_search/House3D/HRL-GRG/
CUDA_VISIBLE_DEVICES=-1 python evaluate.py \
  --model_path="result_se_pretrain/model" \
  --evaluate_file='../random_method/1s6t.txt' 
```
to reproduce the results of our method on the single environment for the seen goals as follows,

| Single Env Seen Goals   | SR    | SPL  |
| :-----------------------| :----:|:----:|
| HRL-GRG                 | 0.88  | 0.33 |

- run the command,
```bash
# From robotic_object_search/House3D/HRL-GRG/
CUDA_VISIBLE_DEVICES=-1 python evaluate.py \
  --model_path="result_se_pretrain/model" \
  --evaluate_file='../random_method/1s6t_test.txt' 
```
to reproduce the results of our method on the single environment for the unseen goals as follows,

| Single Env Unseen Goals | SR    | SPL  |
| :-----------------------| :----:|:----:|
| HRL-GRG                 | 0.79  | 0.21 |

- run the command,
```bash
# From robotic_object_search/House3D/HRL-GRG/
CUDA_VISIBLE_DEVICES=-1 python evaluate.py \
  --model_path="result_me_pretrain/model" \
  --evaluate_file='../random_method/4s6t.txt' 
```
to reproduce the results of our method on the multiple environments for the seen environments as follows,

| Multiple Envs Seen Envs | SR    | SPL  |
| :-----------------------| :----:|:----:|
| HRL-GRG                 | 0.76  | 0.20 |

- run the command,
```bash
# From robotic_object_search/House3D/HRL-GRG/
CUDA_VISIBLE_DEVICES=-1 python evaluate.py \
  --model_path="result_me_pretrain/model" \
  --evaluate_file='../random_method/4s6t_test.txt' 
```
to reproduce the results of our method on the multiple environments for the unseen environments as follows,

| Multiple Envs Unseen Envs| SR    | SPL  |
| :------------------------| :----:|:----:|
| HRL-GRG                  | 0.62  | 0.10 |

To evaluate other baseline methods, run the same command from the corresponding directories.
