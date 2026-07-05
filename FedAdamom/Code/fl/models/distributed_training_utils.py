import torch
import copy
import math
from torch import nn, autograd
import numpy as np
from torch.utils.data import DataLoader
from utils.dataset import Dataset


max_norm = 10

def add(target, source):
    for name in target:
        target[name].data += source[name].data.clone()


def add_mome(target, source, beta_):
    for name in target:
        target[name].data = (beta_ * target[name].data + source[name].data.clone())


def add_mome2(target, source1, source2, beta_1, beta_2):
    for name in target:
        target[name].data = beta_1 * source1[name].data.clone() + beta_2 * source2[name].data.clone()


def scale(target, scaling):
    for name in target:
        target[name].data = scaling * target[name].data.clone()


def subtract_(target, minuend, subtrahend):
    for name in target:
        target[name].data = minuend[name].data.clone() - subtrahend[name].data.clone()


def average(target, sources):
    for name in target:
        target[name].data = torch.mean(torch.stack([source[name].data for source in sources]), dim=0).clone()


def get_mdl_params(model_list, n_par=None):
    if n_par == None:
        exp_mdl = model_list[0]
        n_par = 0
        for name, param in exp_mdl.named_parameters():
            n_par += len(param.data.reshape(-1))

    param_mat = np.zeros((len(model_list), n_par)).astype('float32')
    for i, mdl in enumerate(model_list):
        idx = 0
        for name, param in mdl.named_parameters():
            temp = param.data.cpu().numpy().reshape(-1)
            param_mat[i, idx:idx + len(temp)] = temp
            idx += len(temp)
    return np.copy(param_mat)


class DistributedTrainingDevice(object):
    def __init__(self, model, args):
        self.model = model
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()

class Client(DistributedTrainingDevice):
    def __init__(self, model, args, trn_x, trn_y, dataset_name, id_num=0):
        super().__init__(model, args)

        self.trn_gen = DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name),
                                  batch_size=self.args.local_bs, shuffle=True)

        self.id = id_num
        self.local_epoch = int(np.ceil(trn_x.shape[0] / self.args.local_bs))

        self.W = {name: value for name, value in self.model.named_parameters()}
        self.W_old = {name: torch.zeros(value.shape).to(self.args.device) for name, value in self.W.items()}
        self.dW = {name: torch.zeros(value.shape).to(self.args.device) for name, value in self.W.items()}
        self.mt = {name: torch.zeros(value.shape).to(self.args.device) for name, value in self.W.items()}
        self.vt = {name: torch.zeros(value.shape).to(self.args.device) for name, value in self.W.items()}
        self.nt = {name: torch.zeros(value.shape).to(self.args.device) for name, value in self.W.items()}
        self.dt = {name: torch.zeros(value.shape).to(self.args.device) for name, value in self.W.items()}
        self.gt = {name: torch.zeros(value.shape).to(self.args.device) for name, value in self.W.items()}
        if self.args.method == 'scaffold' or self.args.method == 'our_sd':
            self.ci = {name: torch.zeros(value.shape).to(self.args.device) for name, value in self.W.items()}
            self.delta_ci = {name: torch.zeros(value.shape).to(self.args.device) for name, value in self.W.items()}
            self.c_plus = {name: torch.zeros(value.shape).to(self.args.device) for name, value in self.W.items()}
            self.unbias = {name: torch.zeros(value.shape).to(self.args.device) for name, value in self.W.items()}
            self.dg = {name: torch.zeros(value.shape).to(self.args.device) for name, value in self.W.items()}

        if self.args.method == 'feddyn':
            self.histi = {name: torch.zeros(value.shape).to(self.args.device) for name, value in self.W.items()}
            self.unbias = {name: torch.zeros(value.shape).to(self.args.device) for name, value in self.W.items()}

        if self.args.method == 'fed_localnesterov' or self.args.method == 'fed_localnesterov_nocom' or self.args.method == 'fed_localmome':
            self.bias = {name: torch.zeros(value.shape).to(self.args.device) for name, value in self.W.items()}
            self.buffer_momentum = {name: torch.zeros(value.shape).to(self.args.device) for name, value in self.W.items()}
            self.buffer_nestrov = {name: torch.zeros(value.shape).to(self.args.device) for name, value in self.W.items()}
        if self.args.method == 'fastslowmo':
            self.yt = {name: torch.zeros(value.shape).to(self.args.device) for name, value in self.W.items()}
        if self.args.method == 'fed_nesterov':
            self.bias = {name: torch.zeros(value.shape).to(self.args.device) for name, value in self.W.items()}
        if self.args.method == 'ours':
            self.bias = {name: torch.zeros(value.shape).to(self.args.device) for name, value in self.W.items()}

        if self.args.method == 'mime':
            self.bias = {name: torch.zeros(value.shape).to(self.args.device) for name, value in self.W.items()}
            self.buffer_m = {name: torch.zeros(value.shape).to(self.args.device) for name, value in self.W.items()}

        self.state_params_diff = 0.0
        self.train_loss = 0.0
        self.n_par = get_mdl_params([self.model]).shape[1]

    def synchronize_with_server(self, server):
        self.model = copy.deepcopy(server.model)
        self.W = {name: value for name, value in self.model.named_parameters()}

    def train_cnn(self, server, iter):
        self.model.train()

        if (self.args.method == 'fedavg' or self.args.method == 'fedadamom'
                or self.args.method == 'fedadagrad' or self.args.method == 'fedyogi'
                or self.args.method == 'fedavgm' or self.args.method == 'fedadam'):
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=self.args.momentum,
                                       weight_decay=self.args.weigh_delay)






        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            trn_gen_iter = self.trn_gen.__iter__()
            batch_loss = []
            for i in range(self.local_epoch):
                images, labels = trn_gen_iter.__next__()
                images, labels = images.to(self.args.device), labels.to(self.args.device)

                optimizer.zero_grad()
                log_probs = self.model(images)
                loss_f_i = self.loss_func(log_probs, labels.reshape(-1).long())

                local_par_list = None
                for param in self.model.parameters():
                    if not isinstance(local_par_list, torch.Tensor):
                        local_par_list = param.reshape(-1)
                    else:
                        local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)

                loss_algo = torch.sum(local_par_list * self.state_params_diff)


                if (self.args.method == 'fedavg' or self.args.method == 'fedadam' or self.args.method == 'fedadamom'
                        or self.args.method == 'fedadagrad' or self.args.method == 'fedyogi'
                        or self.args.method == 'fedavgm' ):
                    loss = loss_f_i
                else:
                    exit('Error')


                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=max_norm)

                if (self.args.method == 'fedavg' or self.args.method == 'fedadam' or self.args.method == 'fedadamom' 
                    or self.args.method == 'fedadagrad' or self.args.method == 'fedyogi' or self.args.method == 'fedavgm' ):
                     optimizer.step()

                else:
                    exit('Error')

                batch_loss.append(loss.item())

            scheduler.step()
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return sum(epoch_loss) / len(epoch_loss)

    def compute_weight_update(self, server, iter):
        self.model.train()
        self.W_old = copy.deepcopy(self.W)
        self.train_loss = self.train_cnn(server, iter)
        subtract_(target=self.dW, minuend=self.W, subtrahend=self.W_old)


class Server(DistributedTrainingDevice):
    def __init__(self, model, args):
        super().__init__(model, args)

        self.W = {name: value for name, value in self.model.named_parameters()}
        self.dW = {name: torch.zeros(value.shape).to(self.args.device) for name, value in self.W.items()}
        self.mome = {name: torch.zeros(value.shape).to(self.args.device) for name, value in self.W.items()}
        self.all_model = copy.deepcopy(model).to(args.device)
        self.all_W = {name: value for name, value in self.all_model.named_parameters()}
        self.mt = {name: torch.zeros(value.shape).to(self.args.device) for name, value in self.W.items()}
        self.vt = {name: torch.zeros(value.shape).to(self.args.device) for name, value in self.W.items()}
        self.nt = {name: torch.zeros(value.shape).to(self.args.device) for name, value in self.W.items()}
        self.local_epoch = 0

    def aggregate_weight_updates(self, clients, iter, aggregation="mean"):
        self.local_epoch = clients[0].local_epoch
        if aggregation == "mean":
            average(target=self.dW, sources=[client.dW for client in clients])
        for client in clients:
            client = None

    def computer_weight_update_down_dw(self, clients, iter):
        self.local_epoch = clients[0].local_epoch

        if self.args.method == 'fedavgm':
            add_mome(target = self.mome, source =self.dW, beta_ = self.args.beta_)
            add_mome2(target=self.W, source1=self.W, source2=self.mome, beta_1=1, beta_2=self.args.globallr)

        elif self.args.method == 'fedadam':
            add_mome2(target=self.mt, source1=self.mt, source2=self.dW, beta_1=self.args.beta_, beta_2=1-self.args.beta_)
            add_mome2(
                target=self.vt,
                source1=self.vt,
                source2={k: v**2 for k, v in self.dW.items()},
                beta_1=self.args.beta_0,
                beta_2=1-self.args.beta_0
            )
            with torch.no_grad():
                for k in self.W.keys():
                    update = (self.mt[k] * self.args.globallr) / (self.vt[k] ** 0.5 + self.args.mu)
                    self.W[k].add_(update)
        elif self.args.method == 'fedadagrad':
            add_mome2(target=self.mt, source1=self.mt, source2=self.dW, beta_1=self.args.beta_, beta_2=1-self.args.beta_)
            dW_square = {key: value ** 2 for key, value in self.dW.items()}
            add_mome2(target=self.vt, source1=self.vt, source2= dW_square, beta_1=1, beta_2=1)
            adapt_dw = {k: (self.mt[k] * self.args.globallr) / (self.vt[k] ** 0.5 + self.args.mu) for k in self.mt}
            add_mome2(target=self.W, source1=self.W, source2=adapt_dw, beta_1=1, beta_2=1)
        elif self.args.method == 'fedyogi':
            add_mome2(target=self.mt, source1=self.mt, source2=self.dW, beta_1=self.args.beta_, beta_2=1-self.args.beta_)
            dW_square = {key: value ** 2 for key, value in self.dW.items()}
            add_mome2(target=self.nt, source1=self.vt, source2=dW_square, beta_1=1,beta_2=-1)
            for name in self.nt:
                self.nt[name] = torch.sign(self.nt[name])*dW_square[name]
            add_mome2(target=self.vt, source1=self.vt, source2= self.nt, beta_1=1, beta_2=-(1-self.args.beta_0))
            adapt_dw = {k: (self.mt[k] * self.args.globallr) / (self.vt[k] ** 0.5 + self.args.mu) for k in self.mt}
            add_mome2(target=self.W, source1=self.W, source2=adapt_dw, beta_1=1, beta_2=1)
        elif self.args.method == 'fedadamom':
            add_mome2(
                target=self.vt,
                source1=self.vt,
                source2={k: v**2 for k, v in self.dW.items()},
                beta_1=self.args.beta_0,
                beta_2=1-self.args.beta_0
            )
            size_sum = sum(v.numel() for v in self.vt.values())
            value_sum = sum(v.sum().item() for v in self.vt.values())
            mean_v = value_sum/size_sum

            eps = self.args.eps
            beta1_t = {
                k: (1.0 - v*(1/ mean_v)).clip(0.0, 1.0 - eps)
                for k, v in self.vt.items()
            }
            adapt_mt = {k: (self.mt[k] * beta1_t[k] + (1 - beta1_t[k]) * self.dW[k]) for k in self.mt}
            self.mt = adapt_mt
            add_mome2(target=self.W, source1=self.W, source2=self.mt,
                    beta_1=1, beta_2=self.args.globallr)
        else:
            add_mome2(target=self.W, source1=self.W, source2=self.dW, beta_1=1, beta_2=self.args.globallr)

    @torch.no_grad()
    def evaluate(self, data_x, data_y, dataset_name):
        self.model.eval()

        test_loss = 0
        acc_overall = 0
        n_tst = data_x.shape[0]
        tst_gen = DataLoader(Dataset(data_x, data_y, dataset_name=dataset_name), batch_size=self.args.bs, shuffle=False)
        tst_gen_iter = tst_gen.__iter__()
        for i in range(int(np.ceil(n_tst / self.args.bs))):
            data, target = tst_gen_iter.__next__()
            data, target = data.to(self.args.device), target.to(self.args.device)
            log_probs = self.model(data)

            test_loss += nn.CrossEntropyLoss(reduction='sum')(log_probs, target.reshape(-1).long()).item()

            log_probs = log_probs.cpu().detach().numpy()
            log_probs = np.argmax(log_probs, axis=1).reshape(-1)
            target = target.cpu().numpy().reshape(-1).astype(np.int32)
            batch_correct = np.sum(log_probs == target)
            acc_overall += batch_correct
        test_loss /= n_tst
        accuracy = 100.00 * acc_overall / n_tst
        return accuracy, test_loss

    @torch.no_grad()
    def all_model_evaluate(self, clients, data_x, data_y, dataset_name):
        average(target=self.all_W, sources=[client.W for client in clients])
        self.all_model.eval()

        test_loss = 0
        acc_overall = 0
        n_tst = data_x.shape[0]
        tst_gen = DataLoader(Dataset(data_x, data_y, dataset_name=dataset_name), batch_size=self.args.bs, shuffle=False)
        tst_gen_iter = tst_gen.__iter__()
        for i in range(int(np.ceil(n_tst / self.args.bs))):
            data, target = tst_gen_iter.__next__()
            data, target = data.to(self.args.device), target.to(self.args.device)
            log_probs = self.all_model(data)

            test_loss += nn.CrossEntropyLoss(reduction='sum')(log_probs, target.reshape(-1).long()).item()

            log_probs = log_probs.cpu().detach().numpy()
            log_probs = np.argmax(log_probs, axis=1).reshape(-1)
            target = target.cpu().numpy().reshape(-1).astype(np.int32)
            batch_correct = np.sum(log_probs == target)
            acc_overall += batch_correct

        test_loss /= n_tst
        accuracy = 100.00 * acc_overall / n_tst
        return accuracy, test_loss
