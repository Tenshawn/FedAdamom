


import profile
import time
import matplotlib

matplotlib.use('Agg')
import ssl

import copy
import random
import os
import torch
import numpy as np
from utils.options import args_parser
from utils.seed import setup_seed
from utils.logg import get_logger
from models.Nets import client_model
from utils.dataset import DatasetObject, ShakespeareObjectCrop_noniid
from models.distributed_training_utils import Client, Server
from datetime import datetime
import time
import os, psutil, gc

torch.set_printoptions(
    precision=8,
    threshold=1000,
    edgeitems=3,
    linewidth=150,
    profile=None,
    sci_mode=False
)

def add_mome2(target, source1, source2, beta_1, beta_2):
    for name in target:
        target[name].data = beta_1 * source1[name].data.clone() + beta_2 * source2[name].data.clone()

proc = psutil.Process(os.getpid())

def cpu_tensor_mb():
    total = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and (not obj.is_cuda):
                total += obj.element_size() * obj.nelement()
        except Exception:
            pass
    return total / 1024 / 1024  


if __name__ == '__main__':
    ssl._create_default_https_context = ssl._create_unverified_context
    args = args_parser()
    dataset_key = args.dataset.lower()
    if dataset_key in {'shakespeare'}:
        args.dataset = dataset_key
    args.model = args.model.lower()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    setup_seed(args.seed)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    if args.dataset == 'shakespeare':
        data_path = os.path.join(base_dir, 'LEAF', 'shakespeare', 'data', '')
        data_obj = ShakespeareObjectCrop_noniid(data_path=data_path, dataset_prefix='dataset_prefix')
    else:
        data_path = './Folder/'
        data_obj = DatasetObject(dataset=args.dataset, n_client=args.num_users, seed=args.seed, rule=args.iid,
                                 rule_arg=args.rule_arg, data_path=data_path)
    

    if args.model == 'logistic' and args.dataset == 'emnist':
        net_glob = client_model('Linear',10, [1 * 28 * 28, 10]).to(args.device)
    elif args.model == 'rnn' and args.dataset == 'shakespeare':
        net_glob = client_model('shakes_LSTM',10).to(args.device)
    elif args.model == 'resnet18' and args.dataset == 'tinyimagenet':
        net_glob = client_model('Resnet18',200).to(args.device)
    elif args.model == 'resnet18' and args.dataset == 'CIFAR100':
        net_glob = client_model('Resnet18',100).to(args.device)
    elif args.model == 'resnet18' and args.dataset == 'CIFAR10':
        net_glob = client_model('Resnet18',10).to(args.device)
    else:
        exit('Error')

    clnt_x = data_obj.clnt_x
    clnt_y = data_obj.clnt_y
    total_clients = len(clnt_x)
    if args.num_users != total_clients:
        args.num_users = total_clients
    clnt_x_list = [np.asarray(x) for x in clnt_x]
    clnt_y_list = [np.asarray(y) for y in clnt_y]
    cent_x = np.concatenate(clnt_x_list, axis=0)
    cent_y = np.concatenate(clnt_y_list, axis=0)

    server = Server((net_glob).to(args.device), args)
    args.filepath = ("./result/" + str(args.method) +  ".txt")
    logger = get_logger(args.filepath)

    maxtest = maxiter = 0

    for iter in range(args.epochs):  
        net_glob.train()
        m = max(int(args.frac * args.num_users), 1)
        selected_ids = random.sample(range(args.num_users), m)
        participating_clients = [Client(model=net_glob.to(args.device), args=args, trn_x=data_obj.clnt_x[i],
                                        trn_y=data_obj.clnt_y[i], dataset_name=data_obj.dataset, id_num=i) for i in
                                 selected_ids]
        process = psutil.Process(os.getpid())
        for client in participating_clients:
            client.synchronize_with_server(server)  
            client.compute_weight_update(server, iter=iter)
        server.aggregate_weight_updates(clients=participating_clients, iter=iter)
        server.computer_weight_update_down_dw(clients=participating_clients, iter=iter)
        results_test, loss_test1 = server.evaluate(data_x=data_obj.tst_x, data_y=data_obj.tst_y,
                                                    dataset_name=data_obj.dataset)
        logger.info('epoch:{}\tlr =\t{:.5f}\tloss2=\t{:.5f}\tacc_test=\t{:.5f}'.
                        format(iter, args.lr, loss_test1, results_test))
        args.lr = args.lr * (args.lr_decay)
        if results_test > maxtest:
            maxiter = iter
            maxtest = results_test

    logger.info('maxiter={}'.format(maxiter))
    logger.info('maxtest={}'.format(maxtest))
