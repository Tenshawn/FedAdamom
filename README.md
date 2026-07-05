All algorithms are implemented using PyTorch 2.0.0 with CUDA 11.8 on a GEFORCE RTX 4090 GPU.  
The learning rate decay is selected from the range of [0.99,0.998,0.9995,1.0]. 
The weight decay is selected from the range of [0.01, 0.001,0.0001, 0.00001]. 
The learning rate is selected from the range of [0.001,0.01,0.1,1.0]. 
 
# CIFAR-10
python ./Code/fl/main_fed.py --seed 200 --gpu 0 --epochs 1000  --num_users 100 --frac 0.05 --dataset CIFAR10 --local_ep 5 --local_bs 50 --bs 50 --rule_arg 0.3 --lr 0.1 --globallr 1 --beta_ 0.1 --beta_0 0.1 --weigh_delay 0.001 --lr_decay 0.998 --method fedadamom  --model resnet18 

python ./Code/fl/main_fed.py --seed 200 --gpu 0 --epochs 1000  --num_users 500 --frac 0.02 --dataset CIFAR10 --local_ep 5 --local_bs 20 --bs 20 --rule_arg 0.3 --lr 0.1 --globallr 1 --beta_ 0.1 --beta_0 0.1 --weigh_delay 0.001 --lr_decay 0.998 --method fedadamom  --model resnet18 

# CIFAR-100
python ./Code/fl/main_fed.py --seed 200 --gpu 0 --epochs 1000  --num_users 100 --frac 0.05 --dataset CIFAR100 --local_ep 5 --local_bs 50 --bs 50 --rule_arg 0.3 --lr 0.1 --globallr 1 --beta_ 0.1 --beta_0 0.1 --weigh_delay 0.001 --lr_decay 0.998 --method fedadamom  --model resnet18 

python ./Code/fl/main_fed.py --seed 200 --gpu 0 --epochs 1000  --num_users 500 --frac 0.02 --dataset CIFAR100 --local_ep 5 --local_bs 20 --bs 20 --rule_arg 0.3 --lr 0.1 --globallr 1 --beta_ 0.1 --beta_0 0.1 --weigh_delay 0.001 --lr_decay 0.998 --method fedadamom  --model resnet18 
