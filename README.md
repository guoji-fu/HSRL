# HSRL
This repository provides a reference implementation of *HSRL* as described in the [paper](https://arxiv.org/abs/1902.06684). 

### Basic Usage
```
$ python hsrl.py --input 'data/movielens/train_edges.txt' --output 'output/movielens/movielens_dw_hs_lp_embeddings.txt' --method deepwalk
```
>noted: your can just checkout hsrl.py to get what you want.
### Input
Your input graph data should be a **txt** file and be under **data** folder.
#### file format
The txt file should be **edgelist**.

#### txt file sample
	0 163
	0 359
	0 414
	...
	5297 4973

> noted: The nodeID start from 0.<br>
> noted: The graph should be an undirected graph, so if (I  J) exist in the Input file, (J  I) should not.
### Citing
If you find *HSRL* useful in your research, please cite our paper:
```
@inproceedings{fu2019learning,
  title={Learning topological representation for networks via hierarchical sampling},
  author={Fu, Guoji and Hou, Chengbin and Yao, Xin},
  booktitle={2019 International Joint Conference on Neural Networks (IJCNN)},
  pages={1--8},
  year={2019},
  organization={IEEE}
}
```