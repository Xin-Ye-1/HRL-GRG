#### Scene Priors

This is a reimplementation of the [Scene Priors](https://arxiv.org/pdf/1810.06543.pdf) method.

Some minor differences are that we use [glove100d](https://nlp.stanford.edu/projects/glove/) instead as the word embeddings, and some hyperparameters are choosen empirically.

The folder `gcn` contains necessary data for the [Graph Convolutional Network (GCN)](https://arxiv.org/abs/1609.02907) in [Scene Priors](https://arxiv.org/pdf/1810.06543.pdf), including the adjacency matrix. The data is sourced from [savn](https://github.com/allenai/savn).

##### Evaluation and Results

To evaluate the method [Scene Priors](https://arxiv.org/pdf/1810.06543.pdf)  for the robotic object search task on [AI2-THOR](https://ai2thor.allenai.org/ithor/), and reproduce the results reported in the paper for the seen environments for seen goals as follows,

| Seen Env Seen Goals     | SR    | SPL  |
| :-----------------------| :----:|:----:|
| Scene Priors            | 0.62  | 0.26 |

run this command,

```bash
# From robotic_object_search/AI2-THOR/Scene_Priors/
CUDA_VISIBLE_DEVICES=-1 python evaluate.py \
  --model_path="result/model" \
  --evaluate_file='../random_method/ssso.txt' 
```
To reproduce the results for the `seen environments unseen goals`, `unseen environments seen goals` and `unseen environments unseen goals`, specify the `evaluate_file` as `'../random_method/ssuo.txt'`, `'../random_method/usso.txt'` and `'../random_method/usuo.txt'` respectively.

##### Training

To train the [Scene Priors](https://arxiv.org/pdf/1810.06543.pdf)  method, run this command:

```bash
# Specify the parameters in robotic_object_search/AI2-THOR/Scene_Priors/train.sh, 
# and from robotic_object_search/AI2-THOR/Scene_Priors/
./train.sh
```
