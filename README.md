# recvisfp2

Authors: Dana Aubakirova, Junior Cedric Tonga 

This project is based on the official implementation of the paper "Robust Learning with Progressive Data Expansion Against Spurious Correlation":

[[Paper](https://arxiv.org/abs/2306.04949)]

## Abstract

This study extends the analysis of spurious feature learning in supervised deep learning models to the unsupervised learning context. The most recent paper in spurious feature learning examined the learning process of a two-layer nonlinear convolutional neural network in the presence of spurious features. The analysis suggests that imbalanced data groups and easily learnable spurious features can lead to the dominance of spurious features during the learning process. In light of this, they propose a new training algorithm called PDE that efficiently enhances the model’s robustness for a better worst-group
performance. PDE begins with a group-balanced subset of training data and progressively expands it to facilitate the learning of the core features. On average, our method achieves a 2.8% improvement in worst-group accuracy compared with the state-of-the-art method, while enjoying up to 10× faster training efficiency. We perform similar set of experiments to analyze whether the findings hold for unsupervised models, such as Dino, Dinov2, MAE.

## Installation 
```
conda create -n myenv python=3.8
conda activate myenv

(CUDA>=11.1) 
conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=11.1 -c pytorch -c conda-forge
conda install pyg -c pyg
(CUDA==10.0) 
pip install torch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.8.0+cu102.html

pip install wilds
pip install transformers[torch]
pip install wandb

conda install -c conda-forge matplotlib
conda install -c anaconda seaborn

pip install opencv-python
```

## Wandb
Wandb is optional for logging and visualization.

To use wandb, you need to create an account at https://wandb.ai/ and login with
```
wandb login
```

Add `--use_wandb` to the command line to enable wandb logging.

## Usage
### Waterbirds Dataset
```
python run_expt.py --dataset waterbirds --download --algorithm ERM --model dino --root_dir data --log_dir ./logs --device 0 --lr 1e-2 --weight_decay 1e-2 --subsample --scheduler MultiStepLR --scheduler_multistep_gamma 0.01 --scheduler_kwargs verbose=True --add_start 140 --add_interval 10 --add_num 10 --batch_size 64 --seed 0
```

### CelebA Dataset
```
python run_expt.py --dataset celebA --download --algorithm ERM --model dino --root_dir data --log_dir ./logs --device 0 --lr 1e-2 --weight_decay 1e-4 --subsample --subsample_ref same_across_class --add_start 16 --add_interval 10 --add_num 50 --seed 0
```

### Synthetic Dataset
- [Case 1 ERM](synthetic/spurious_synthetic.ipynb)
- [Case 2 ERM](synthetic/spurious_synthetic_case2.ipynb)
- [Case 1 PDE](synthetic/spurious_PDE.ipynb)
- [Case 1 Warmup+All](synthetic/spurious_synthetic_warmup+all.ipynb)
  - [Case 1 Warmup+All, no momentum after warmup](synthetic/spurious_synthetic_warmup+all_no_momentum.ipynb)


## Acknowledgement
This repo is built upon [WILDS](https://github.com/p-lambda/wilds) and [PDE](https://uclaml.github.io/PDE/). We thank the authors for their great work.

## Contact
If you have any questions, please contact [Dana Aubakirova](mailto:danaaubakirova17@gmail.com) and [Junior Cedric Tonga](mailto:juniortonga2022@gmail.com).
