# FedAdamom: Adaptive Momentum for Improved Generalization in Federated  Optimization #

## Introduction ##
Federated learning has emerged as a widely adopted training paradigm for privacy-preserving machine learning. Despite the past success of SGD-based methods, they still suffer from severe data heterogeneity and the lack of adaptivity in practical applications. While several adaptive federated optimization methods (such as FedAdam) have been proposed and demonstrated to achieve faster convergence, they fail to show significant improvements in generalization performance under highly heterogeneous data distributions, and their optimization and generalization mechanisms remain insufficiently understood. To fill this gap, we introduce diffusion theory into the adaptive federated optimization framework and analyze the distinct effects ofadaptive learning rate and global momentum from the perspectives of saddle-point escaping and flatminima selection. Theoretical results show that although FedAdam outperforms FedAvg/FedAvgM in escaping saddle points, the latter escapes sharp minima more efficiently. The root cause lies in the fact that adaptive learning rate, while enhancing saddle-point escape, weaken the preference for flat minima. Motivated by these insights, we propose FedAdamom, a new adaptive federated optimization algorithm that adapts the momentum hyperparameter rather than the learning rate. FedAdamom maintains strong saddle-point escaping capability while enhancing flat-minima selection. We further establish its convergence guarantees under non-convex objectives. 

---

## Environment ##
All algorithms are implemented using PyTorch 2.0.0 with CUDA 11.8 on a GEFORCE RTX 4090 GPU.  
We recommend using a Conda environment to run the Python scripts for this project. Follow these commands to set up the environment and install the required libraries:

conda create -n fedadamom python=3.10

conda activate fedadamom

pip install -r requirements.txt

---
## Training ##
The learning rate decay is selected from the range of [0.99,0.998,0.9995,1.0]. 
The weight decay is selected from the range of [0.01, 0.001,0.0001, 0.00001]. 
The learning rate is selected from the range of [0.001,0.01,0.1,1.0]. 
 
## CIFAR-10 ##
python ./Code/fl/main_fed.py --seed 200 --gpu 0 --epochs 1000  --num_users 100 --frac 0.05 --dataset CIFAR10 --local_ep 5 --local_bs 50 --bs 50 --rule_arg 0.3 --lr 0.1 --globallr 1 --beta_ 0.1 --beta_0 0.1 --weigh_delay 0.001 --lr_decay 0.998 --method fedadamom  --model resnet18 

python ./Code/fl/main_fed.py --seed 200 --gpu 0 --epochs 1000  --num_users 500 --frac 0.02 --dataset CIFAR10 --local_ep 5 --local_bs 20 --bs 20 --rule_arg 0.3 --lr 0.1 --globallr 1 --beta_ 0.1 --beta_0 0.1 --weigh_delay 0.001 --lr_decay 0.998 --method fedadamom  --model resnet18 

## CIFAR-100 ##
python ./Code/fl/main_fed.py --seed 200 --gpu 0 --epochs 1000  --num_users 100 --frac 0.05 --dataset CIFAR100 --local_ep 5 --local_bs 50 --bs 50 --rule_arg 0.3 --lr 0.1 --globallr 1 --beta_ 0.1 --beta_0 0.1 --weigh_delay 0.001 --lr_decay 0.998 --method fedadamom  --model resnet18 

python ./Code/fl/main_fed.py --seed 200 --gpu 0 --epochs 1000  --num_users 500 --frac 0.02 --dataset CIFAR100 --local_ep 5 --local_bs 20 --bs 20 --rule_arg 0.3 --lr 0.1 --globallr 1 --beta_ 0.1 --beta_0 0.1 --weigh_delay 0.001 --lr_decay 0.998 --method fedadamom  --model resnet18 
