# CF_Rec


## Overview

A general framework for out-of-distribution recommendation.


## Run the Code

- We provide implementation for various baselines.

- To run the code, first run the following command to install tools used in evaluation:

```
python setup.py build_ext --inplace
```

### LightGCN backbone

- INFONCE Training:

```python
python main.py --train_norm --pred_norm --modeltype  INFONCE --dataset kuairec2 --n_layers 2 --batch_size 2048 --lr 3e-5 --neg_sample 128 --tau 2 --Ks 20 --dsc infonce
```

### MF backbone

- InfoNCE Training:

```python
python main.py --train_norm --pred_norm --modeltype INFONCE --dataset tencent_synthetic --n_layers 0 --tau 0.09 --neg_sample 128 --batch_size 2048 --lr 1e-3 --Ks 20 --dsc infonce
```


## Requirements

- python == 3.7.10

- pytorch == 1.12.1+cu102

- tensorflow == 1.14

- reckit == 0.2.4





