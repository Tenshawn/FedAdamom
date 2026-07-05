import json
from scipy import io
import numpy as np
import torchvision
from torchvision import transforms
import torch
from torch.utils import data
import os
from PIL import Image

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from torchvision import datasets, transforms

class DatasetObject:
    def __init__(self, dataset, n_client, seed, rule, rule_arg='', data_path=''):
        self.dataset = dataset
        self.n_client = n_client
        self.rule = rule
        self.rule_arg = rule_arg
        self.seed = seed
        rule_arg_str = rule_arg if isinstance(rule_arg, str) else '%.3f' % rule_arg
        self.name = "%s_%d_%d_%s_%s" % (self.dataset, self.n_client, self.seed, self.rule, rule_arg_str)
        self.data_path = data_path
        self.set_data()

    def set_data(self):
        
        if self.dataset in ('celeba', 'femnist'):
            self._set_leaf_vision_data()
            return
        if not os.path.exists('%sData/%s' % (self.data_path, self.name)):
            
            if self.dataset == 'mnist':
                transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
                trnset = torchvision.datasets.MNIST(root='%sData/Raw' % self.data_path,
                                                    train=True, download=True, transform=transform)
                tstset = torchvision.datasets.MNIST(root='%sData/Raw' % self.data_path,
                                                    train=False, download=True, transform=transform)

                trn_load = torch.utils.data.DataLoader(trnset, batch_size=60000, shuffle=False, num_workers=1)
                tst_load = torch.utils.data.DataLoader(tstset, batch_size=10000, shuffle=False, num_workers=1)
                self.channels = 1
                self.width = 28
                self.height = 28
                self.n_cls = 10

            if self.dataset == 'CIFAR10':
                transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.491, 0.482, 0.447],
                                                                     std=[0.247, 0.243, 0.262])])

                trnset = torchvision.datasets.CIFAR10(root='%sData/Raw' % self.data_path,
                                                      train=True, download=True, transform=transform)
                tstset = torchvision.datasets.CIFAR10(root='%sData/Raw' % self.data_path,
                                                      train=False, download=True, transform=transform)

                trn_load = torch.utils.data.DataLoader(trnset, batch_size=50000, shuffle=False, num_workers=1)
                tst_load = torch.utils.data.DataLoader(tstset, batch_size=10000, shuffle=False, num_workers=1)
                self.channels = 3;
                self.width = 32;
                self.height = 32;
                self.n_cls = 10;

            if self.dataset == 'CIFAR100':
                
                transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                                                     std=[0.2675, 0.2565, 0.2761])])
                trnset = torchvision.datasets.CIFAR100(root='%sData/Raw' % self.data_path,
                                                       train=True, download=True, transform=transform)
                tstset = torchvision.datasets.CIFAR100(root='%sData/Raw' % self.data_path,
                                                       train=False, download=True, transform=transform)
                trn_load = torch.utils.data.DataLoader(trnset, batch_size=50000, shuffle=False, num_workers=0)
                tst_load = torch.utils.data.DataLoader(tstset, batch_size=10000, shuffle=False, num_workers=0)
                self.channels = 3;
                self.width = 32;
                self.height = 32;
                self.n_cls = 100;

            if self.dataset == 'tinyimagenet':
                print(self.dataset)
                transform = transforms.Compose([  
                    transforms.ToTensor(),
                    
                    
                    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.5, 0.5, 0.5])])
                
                root_dir = self.data_path + "Data/Raw/tiny-imagenet-200/"
                print("root_dir", root_dir)
                trn_img_list, trn_lbl_list, tst_img_list, tst_lbl_list = [], [], [], []
                trn_file = os.path.join(root_dir, 'train_list.txt')
                tst_file = os.path.join(root_dir, 'val_list.txt')
                with open(trn_file) as f:
                    line_list = f.readlines()
                    for line in line_list:
                        img, lbl = line.strip().split()
                        img = img.replace('\\', '/')
                        trn_img_list.append(img)
                        trn_lbl_list.append(int(lbl))
                with open(tst_file) as f:
                    line_list = f.readlines()
                    for line in line_list:
                        img, lbl = line.strip().split()
                        img = img.replace('\\', '/')
                        tst_img_list.append(img)
                        tst_lbl_list.append(int(lbl))
                trainset = DatasetFromDir(img_root=root_dir, img_list=trn_img_list, label_list=trn_lbl_list,
                                          transformer=transform)
                testset = DatasetFromDir(img_root=root_dir, img_list=tst_img_list, label_list=tst_lbl_list,
                                         transformer=transform)
                trn_load = torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=False,
                                                       num_workers=0)
                tst_load = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=False, num_workers=0)
                self.channels = 3;
                self.width = 64;
                self.height = 64;
                self.n_cls = 200;
                print("tect...", self.rule)

            if self.dataset != 'emnist':
                trn_itr = trn_load.__iter__();
                tst_itr = tst_load.__iter__()
                
                trn_x, trn_y = trn_itr.__next__()
                tst_x, tst_y = tst_itr.__next__()
                trn_x = trn_x.numpy()
                trn_y = trn_y.numpy().reshape(-1, 1)
                tst_x = tst_x.numpy()
                tst_y = tst_y.numpy().reshape(-1, 1)

            if self.dataset == 'emnist':
                emnist = io.loadmat(self.data_path + "Data/Raw/matlab/emnist-letters.mat")
                
                x_train = emnist["dataset"][0][0][0][0][0][0]
                x_train = x_train.astype(np.float32)
                y_train = emnist["dataset"][0][0][0][0][0][1] - 1  
                trn_idx = np.where(y_train < 10)[0]
                y_train = y_train[trn_idx]
                x_train = x_train[trn_idx]
                mean_x = np.mean(x_train)
                std_x = np.std(x_train)

                
                x_test = emnist["dataset"][0][0][1][0][0][0]
                x_test = x_test.astype(np.float32)
                y_test = emnist["dataset"][0][0][1][0][0][1] - 1  
                tst_idx = np.where(y_test < 10)[0]
                y_test = y_test[tst_idx]
                x_test = x_test[tst_idx]
                x_train = x_train.reshape((-1, 1, 28, 28))
                x_test = x_test.reshape((-1, 1, 28, 28))

                trn_x = (x_train - mean_x) / std_x
                trn_y = y_train
                tst_x = (x_test - mean_x) / std_x
                tst_y = y_test

                self.channels = 1;
                self.width = 28;
                self.height = 28;
                self.n_cls = 10;

            
            np.random.seed(self.seed)
            rand_perm = np.random.permutation(len(trn_y))
            trn_x = trn_x[rand_perm]
            trn_y = trn_y[rand_perm]
            self.trn_x = trn_x
            self.trn_y = trn_y
            self.tst_x = tst_x
            self.tst_y = tst_y

            
            n_data_per_clnt = int((len(trn_y)) / self.n_client)
            clnt_data_list = [n_data_per_clnt] * self.n_client

            if self.rule == 'noniid':
                print("noniid...start")
                dict_users = {i: np.array([], dtype='int64') for i in range(self.n_client)}
                idx_list = [np.where(trn_y == i)[0] for i in range(self.n_cls)]
                clnt_x = [np.zeros((clnt_data_list[clnt__], self.channels, self.height, self.width)).astype(np.float32)
                          for
                          clnt__ in range(self.n_client)]
                clnt_y = [np.zeros((clnt_data_list[clnt__], 1)).astype(np.int64) for clnt__ in range(self.n_client)]

                for clnt_idx_ in range(self.n_client):
                    n_data_per_clnt_per_class = clnt_data_list[clnt_idx_] / self.rule_arg
                    curr_class = np.random.randint(self.n_cls)
                    budget = n_data_per_clnt
                    while budget > 0:
                        take = int(min(budget, n_data_per_clnt_per_class, len(idx_list[curr_class])))
                        dict_users[clnt_idx_] = np.concatenate((dict_users[clnt_idx_], idx_list[curr_class][:take]),
                                                               axis=0)
                        idx_list[curr_class] = idx_list[curr_class][take:]

                        budget -= take
                        curr_class = (curr_class + 1) % self.n_cls

                    clnt_x[clnt_idx_] = trn_x[dict_users[clnt_idx_]]
                    clnt_y[clnt_idx_] = trn_y[dict_users[clnt_idx_]]

                clnt_x = np.asarray(clnt_x)
                clnt_y = np.asarray(clnt_y)

            if self.rule == 'Dirichlet':
                print("Dirichlet...start")
                cls_priors = np.random.dirichlet(alpha=[self.rule_arg] * self.n_cls, size=self.n_client)
                prior_cumsum = np.cumsum(cls_priors, axis=1)

                idx_list = [np.where(trn_y == i)[0] for i in range(self.n_cls)]
                cls_amount = [len(idx_list[i]) for i in range(self.n_cls)]
                true_sample = [0 for i in range(self.n_cls)]
                
                clnt_x = [
                    np.zeros((clnt_data_list[client__], self.channels, self.height, self.width)).astype(np.float32)
                    for client__ in range(self.n_client)]
                clnt_y = [np.zeros((clnt_data_list[client__], 1)).astype(np.int64) for client__ in
                          range(self.n_client)]

                while (np.sum(clnt_data_list) != 0):
                    curr_client = np.random.randint(self.n_client)
                    
                    
                    if clnt_data_list[curr_client] <= 0:
                        continue
                    clnt_data_list[curr_client] -= 1
                    curr_prior = prior_cumsum[curr_client]
                    while True:
                        cls_label = np.argmax(np.random.uniform() <= curr_prior)
                        
                        if cls_amount[cls_label] <= 0:
                            cls_amount[cls_label] = len(idx_list[cls_label])
                            continue
                        cls_amount[cls_label] -= 1
                        true_sample[cls_label] += 1

                        clnt_x[curr_client][clnt_data_list[curr_client]] = trn_x[
                            idx_list[cls_label][cls_amount[cls_label]]]
                        clnt_y[curr_client][clnt_data_list[curr_client]] = trn_y[
                            idx_list[cls_label][cls_amount[cls_label]]]

                        break
                print(true_sample)
                clnt_x = np.asarray(clnt_x)
                clnt_y = np.asarray(clnt_y)

            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            

            elif self.rule == 'iid' and self.dataset == 'CIFAR100':
                assert len(trn_y) // 100 % self.n_client == 0

                
                idx = np.argsort(trn_y[:, 0])
                n_data_per_clnt = len(trn_y) // self.n_client
                
                clnt_x = np.zeros((self.n_client, n_data_per_clnt, 3, 32, 32), dtype=np.float32)
                clnt_y = np.zeros((self.n_client, n_data_per_clnt, 1), dtype=np.float32)
                trn_x = trn_x[idx]  
                trn_y = trn_y[idx]
                n_cls_sample_per_device = n_data_per_clnt // 100
                for i in range(self.n_client):  
                    for j in range(100):  
                        clnt_x[i, n_cls_sample_per_device * j:n_cls_sample_per_device * (j + 1), :, :, :] = trn_x[
                                                                                                            500 * j + n_cls_sample_per_device * i:500 * j + n_cls_sample_per_device * (
                                                                                                                        i + 1),
                                                                                                            :, :, :]
                        clnt_y[i, n_cls_sample_per_device * j:n_cls_sample_per_device * (j + 1), :] = trn_y[
                                                                                                      500 * j + n_cls_sample_per_device * i:500 * j + n_cls_sample_per_device * (
                                                                                                                  i + 1),
                                                                                                      :]


            elif self.rule == 'iid':
                print("iid...start")
                clnt_x = [np.zeros((clnt_data_list[clnt__], self.channels, self.height, self.width)).astype(np.float32)
                          for clnt__ in range(self.n_client)]
                clnt_y = [np.zeros((clnt_data_list[clnt__], 1)).astype(np.int64) for clnt__ in range(self.n_client)]

                clnt_data_list_cum_sum = np.concatenate(([0], np.cumsum(clnt_data_list)))
                for clnt_idx_ in range(self.n_client):
                    clnt_x[clnt_idx_] = trn_x[clnt_data_list_cum_sum[clnt_idx_]:clnt_data_list_cum_sum[clnt_idx_ + 1]]
                    clnt_y[clnt_idx_] = trn_y[clnt_data_list_cum_sum[clnt_idx_]:clnt_data_list_cum_sum[clnt_idx_ + 1]]

                clnt_x = np.asarray(clnt_x)
                clnt_y = np.asarray(clnt_y)
                print("iid...end")

            self.clnt_x = clnt_x;
            self.clnt_y = clnt_y

            self.tst_x = tst_x;
            self.tst_y = tst_y

            
            os.mkdir('%sData/%s' % (self.data_path, self.name))

            np.save('%sData/%s/clnt_x.npy' % (self.data_path, self.name), clnt_x)
            np.save('%sData/%s/clnt_y.npy' % (self.data_path, self.name), clnt_y)

            np.save('%sData/%s/tst_x.npy' % (self.data_path, self.name), tst_x)
            np.save('%sData/%s/tst_y.npy' % (self.data_path, self.name), tst_y)
        else:
            print("Data is already downloaded")
            self.clnt_x = np.load('%sData/%s/clnt_x.npy' % (self.data_path, self.name), allow_pickle=True)
            self.clnt_y = np.load('%sData/%s/clnt_y.npy' % (self.data_path, self.name), allow_pickle=True)
            self.n_client = len(self.clnt_x)

            self.tst_x = np.load('%sData/%s/tst_x.npy' % (self.data_path, self.name), allow_pickle=True)
            self.tst_y = np.load('%sData/%s/tst_y.npy' % (self.data_path, self.name), allow_pickle=True)

            if self.dataset == 'mnist':
                self.channels = 1;
                self.width = 28;
                self.height = 28;
                self.n_cls = 10;
            if self.dataset == 'CIFAR10':
                self.channels = 3;
                self.width = 32;
                self.height = 32;
                self.n_cls = 10;
            if self.dataset == 'CIFAR100':
                self.channels = 3;
                self.width = 32;
                self.height = 32;
                self.n_cls = 100;
            if self.dataset == 'fashion_mnist':
                self.channels = 1;
                self.width = 28;
                self.height = 28;
                self.n_cls = 10;
            if self.dataset == 'emnist':
                self.channels = 1;
                self.width = 28;
                self.height = 28;
                self.n_cls = 10;
            if self.dataset == 'tinyimagenet':
                self.channels = 3;
                self.width = 64;
                self.height = 64;
                self.n_cls = 200;

        
        count = 0
        for clnt in range(self.n_client):
            
            
            
            count += self.clnt_y[clnt].shape[0]

        print('Total Amount:%d' % count)
        print('--------')

        print("      Test: " +
              ', '.join(["%.3f" % np.mean(self.tst_y == cls) for cls in range(self.n_cls)]) +
              ', Amount:%d' % self.tst_y.shape[0])

    def _set_leaf_vision_data(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        candidate_roots = []
        raw_candidate = os.path.normpath(os.path.join(self.data_path, self.dataset))
        candidate_roots.append(raw_candidate)
        candidate_roots.append(os.path.normpath(os.path.join(os.path.abspath(self.data_path), self.dataset)))
        candidate_roots.append(os.path.normpath(os.path.join(base_dir, '..', 'LEAF', self.dataset)))
        candidate_roots.append(os.path.normpath(os.path.join(base_dir, '..', '..', 'LEAF', self.dataset)))
        candidate_roots.append(os.path.normpath(os.path.join(base_dir, '..', '..', 'leaf-master', 'data', self.dataset)))
        candidate_roots.append(os.path.normpath(os.path.join(base_dir, '..', '..', 'leaf-master', self.dataset)))

        seen = set()
        resolved_root = None
        for cand in candidate_roots:
            if cand in seen:
                continue
            seen.add(cand)
            train_dir = os.path.join(cand, 'data', 'train')
            test_dir = os.path.join(cand, 'data', 'test')
            if os.path.isdir(train_dir) and os.path.isdir(test_dir):
                resolved_root = cand
                break

        if resolved_root is None:
            msg = (
                "Could not locate LEAF data for dataset '{}'. "
                "Checked the following directories:\n  - {}"
            ).format(self.dataset, '\n  - '.join(seen))
            raise FileNotFoundError(msg)

        leaf_root = resolved_root
        train_dir = os.path.join(leaf_root, 'data', 'train')
        test_dir = os.path.join(leaf_root, 'data', 'test')

        users, groups, train_data, test_data = read_data(train_dir, test_dir)
        total_users = len(users)
        if total_users == 0:
            fallback = self._load_leaf_all_data(leaf_root)
            if fallback is None:
                raise ValueError("No clients found for dataset '{}' under '{}'.".format(self.dataset, leaf_root))
            users, groups, train_data, test_data = fallback
            total_users = len(users)

        rng = np.random.RandomState(self.seed)
        if self.n_client <= 0 or self.n_client > total_users:
            selected_indices = np.arange(total_users)
        else:
            selected_indices = np.sort(rng.choice(total_users, size=self.n_client, replace=False))
        selected_users = [users[idx] for idx in selected_indices]

        self.users = selected_users
        self.user_idx = np.arange(len(selected_users))

        if self.dataset == 'celeba':
            self.channels = 3
            self.width = 84
            self.height = 84
            self.n_cls = 2
            images_dir = os.path.join(leaf_root, 'data', 'raw', 'img_align_celeba')
            if not os.path.isdir(images_dir):
                raise FileNotFoundError("CelebA image directory not found at '{}'.".format(images_dir))
        else:
            self.channels = 1
            self.width = 28
            self.height = 28
            self.n_cls = 62
            images_dir = None

        client_x = []
        client_y = []
        test_x_parts = []
        test_y_parts = []

        for user in selected_users:
            train_x_raw = train_data[user]['x']
            train_y_raw = train_data[user]['y']
            proc_train_x, proc_train_y = self._convert_leaf_partition(train_x_raw, train_y_raw, images_dir)
            if proc_train_x.shape[0] == 0:
                continue

            client_x.append(proc_train_x)
            client_y.append(proc_train_y)

            test_x_raw = test_data[user]['x']
            test_y_raw = test_data[user]['y']
            proc_test_x, proc_test_y = self._convert_leaf_partition(test_x_raw, test_y_raw, images_dir)
            if proc_test_x.shape[0] > 0:
                test_x_parts.append(proc_test_x)
                test_y_parts.append(proc_test_y)

        if not client_x:
            raise ValueError("No training samples found for selected clients in dataset '{}'.".format(self.dataset))

        self.n_client = len(client_x)
        self.clnt_x = np.asarray(client_x, dtype=object)
        self.clnt_y = np.asarray(client_y, dtype=object)

        if test_x_parts:
            self.tst_x = np.concatenate(test_x_parts, axis=0)
            self.tst_y = np.concatenate(test_y_parts, axis=0)
        else:
            self.tst_x = np.zeros((0, self.channels, self.height, self.width), dtype=np.float32)
            self.tst_y = np.zeros((0, 1), dtype=np.int64)

        print("Loaded {} dataset with {} clients.".format(self.dataset, self.n_client))

    def _convert_leaf_partition(self, samples_x, samples_y, images_dir):
        if len(samples_x) == 0:
            empty_x = np.zeros((0, self.channels, self.height, self.width), dtype=np.float32)
            empty_y = np.zeros((0, 1), dtype=np.int64)
            return empty_x, empty_y

        if self.dataset == 'femnist':
            x_arr = np.asarray(samples_x, dtype=np.float32)
            if x_arr.ndim == 2:
                x_arr = x_arr.reshape((-1, self.height, self.width))
            if x_arr.ndim == 3:
                x_arr = x_arr[:, None, :, :]
            elif x_arr.ndim == 4:
                pass
            else:
                raise ValueError("Unexpected FEMNIST sample shape: {}".format(x_arr.shape))
            x_arr = np.ascontiguousarray(x_arr, dtype=np.float32)
        else:
            processed = [self._load_celeba_image(name, images_dir) for name in samples_x]
            if processed:
                x_arr = np.stack(processed, axis=0).astype(np.float32)
                x_arr = np.ascontiguousarray(x_arr, dtype=np.float32)
            else:
                x_arr = np.zeros((0, self.channels, self.height, self.width), dtype=np.float32)

        y_arr = np.asarray(samples_y, dtype=np.int64).reshape(-1, 1)
        return x_arr, y_arr

    def _load_celeba_image(self, img_name, images_dir):
        candidate_names = [img_name, os.path.basename(img_name)]
        for name in candidate_names:
            img_path = os.path.join(images_dir, name)
            if os.path.exists(img_path):
                with Image.open(img_path) as img:
                    img = img.resize((self.width, self.height)).convert('RGB')
                    arr = np.asarray(img, dtype=np.float32)
                arr = np.transpose(arr, (2, 0, 1)).astype(np.float32)
                return arr
        raise FileNotFoundError("Image '{}' not found under '{}'.".format(img_name, images_dir))

    def _load_leaf_all_data(self, leaf_root):
        all_data_file = os.path.join(leaf_root, 'data', 'all_data', 'all_data.json')
        if not os.path.isfile(all_data_file):
            return None

        with open(all_data_file, 'r') as f:
            raw = json.load(f)

        users = raw.get('users', [])
        if not users:
            return None

        groups = raw.get('hierarchies', [])
        user_data = raw.get('user_data', {})
        rng = np.random.RandomState(self.seed)
        train_data = {}
        test_data = {}

        for user in users:
            data_entry = user_data.get(user, {'x': [], 'y': []})
            data_x = list(data_entry.get('x', []))
            data_y = list(data_entry.get('y', []))
            count = len(data_x)
            if count == 0:
                train_data[user] = {'x': [], 'y': []}
                test_data[user] = {'x': [], 'y': []}
                continue

            indices = np.arange(count)
            rng.shuffle(indices)

            if count == 1:
                split = 1
            else:
                split = int(round(0.9 * count))
                split = max(1, min(split, count - 1))

            train_idx = indices[:split]
            test_idx = indices[split:]
            train_data[user] = {
                'x': [data_x[i] for i in train_idx],
                'y': [data_y[i] for i in train_idx],
            }
            if len(test_idx) == 0:
                test_data[user] = {
                    'x': [data_x[i] for i in train_idx],
                    'y': [data_y[i] for i in train_idx],
                }
            else:
                test_data[user] = {
                    'x': [data_x[i] for i in test_idx],
                    'y': [data_y[i] for i in test_idx],
                }

        return users, groups, train_data, test_data


class DatasetSynthetic:
    def __init__(self, alpha, beta, theta, iid_scale, iid_data, n_dim, n_clnt, n_cls, avg_data, data_path, name_prefix):
        self.dataset = 'synt'
        self.name = name_prefix + '_'
        self.name += '%d_%d_%d_%d_%f_%f_%f_%f_%s' % (n_dim, n_clnt, n_cls, avg_data,
                                                     alpha, beta, theta, iid_scale, iid_data)

        if (not os.path.exists('%sData/%s/' % (data_path, self.name))):
            
            print('Sythetize')
            data_x, data_y = generate_syn_logistic(dimension=n_dim, n_clnt=n_clnt, n_cls=n_cls, avg_data=avg_data,
                                                   alpha=alpha, beta=beta, theta=theta,
                                                   iid_scale=iid_scale, iid_dat=iid_data)

            os.makedirs('%sData/%s/' % (data_path, self.name))
            os.makedirs('%sModel/%s/' % (data_path, self.name))
            np.save('%sData/%s/data_x.npy' % (data_path, self.name), data_x)
            np.save('%sData/%s/data_y.npy' % (data_path, self.name), data_y)
        else:
            
            print('Load')
            data_x = np.load('%sData/%s/data_x.npy' % (data_path, self.name))
            data_y = np.load('%sData/%s/data_y.npy' % (data_path, self.name))









        for clnt in range(n_clnt):
            print(', '.join(['%.4f' % np.mean(data_y[clnt] == t) for t in range(n_cls)]))

        self.clnt_x = data_x
        self.clnt_y = data_y

        self.tst_x = np.concatenate(self.clnt_x, axis=0)
        self.tst_y = np.concatenate(self.clnt_y, axis=0)
        self.n_client = len(data_x)
        print(self.clnt_x.shape)


class DatasetFromDir(data.Dataset):

    def __init__(self, img_root, img_list, label_list, transformer):
        super(DatasetFromDir, self).__init__()
        self.root_dir = img_root
        self.img_list = img_list
        self.label_list = label_list
        self.size = len(self.img_list)
        self.transform = transformer

    def __getitem__(self, index):
        img_name = self.img_list[index % self.size]
        
        img_path = os.path.join(self.root_dir, img_name)
        img_id = self.label_list[index % self.size]

        img_raw = Image.open(img_path).convert('RGB')
        img = self.transform(img_raw)
        return img, img_id

    def __len__(self):
        return len(self.img_list)








class ShakespeareObjectCrop:
    def __init__(self, data_path, dataset_prefix, crop_amount=2000, tst_ratio=5, rand_seed=0):
        self.dataset = 'shakespeare'
        self.name = dataset_prefix
        users, groups, train_data, test_data = read_data(data_path + 'train/', data_path + 'test/')

        self.users = users

        self.n_client = len(users)
        self.user_idx = np.asarray(list(range(self.n_client)))
        self.clnt_x = list(range(self.n_client))
        self.clnt_y = list(range(self.n_client))

        tst_data_count = 0

        for clnt in range(self.n_client):
            np.random.seed(rand_seed + clnt)
            start = np.random.randint(len(train_data[users[clnt]]['x']) - crop_amount)
            self.clnt_x[clnt] = np.asarray(train_data[users[clnt]]['x'])[start:start + crop_amount]
            self.clnt_y[clnt] = np.asarray(train_data[users[clnt]]['y'])[start:start + crop_amount]

        tst_data_count = (crop_amount // tst_ratio) * self.n_client
        self.tst_x = list(range(tst_data_count))
        self.tst_y = list(range(tst_data_count))

        tst_data_count = 0
        for clnt in range(self.n_client):
            curr_amount = (crop_amount // tst_ratio)
            np.random.seed(rand_seed + clnt)
            start = np.random.randint(len(test_data[users[clnt]]['x']) - curr_amount)
            self.tst_x[tst_data_count: tst_data_count + curr_amount] = np.asarray(test_data[users[clnt]]['x'])[
                                                                       start:start + curr_amount]
            self.tst_y[tst_data_count: tst_data_count + curr_amount] = np.asarray(test_data[users[clnt]]['y'])[
                                                                       start:start + curr_amount]

            tst_data_count += curr_amount

        self.clnt_x = np.asarray(self.clnt_x)
        self.clnt_y = np.asarray(self.clnt_y)

        self.tst_x = np.asarray(self.tst_x)
        self.tst_y = np.asarray(self.tst_y)

        

        self.clnt_x_char = np.copy(self.clnt_x)
        self.clnt_y_char = np.copy(self.clnt_y)

        self.tst_x_char = np.copy(self.tst_x)
        self.tst_y_char = np.copy(self.tst_y)

        self.clnt_x = list(range(len(self.clnt_x_char)))
        self.clnt_y = list(range(len(self.clnt_x_char)))

        for clnt in range(len(self.clnt_x_char)):
            clnt_list_x = list(range(len(self.clnt_x_char[clnt])))
            clnt_list_y = list(range(len(self.clnt_x_char[clnt])))

            for idx in range(len(self.clnt_x_char[clnt])):
                clnt_list_x[idx] = np.asarray(word_to_indices(self.clnt_x_char[clnt][idx]))
                clnt_list_y[idx] = np.argmax(np.asarray(letter_to_vec(self.clnt_y_char[clnt][idx]))).reshape(-1)

            self.clnt_x[clnt] = np.asarray(clnt_list_x)
            self.clnt_y[clnt] = np.asarray(clnt_list_y)

        self.clnt_x = np.asarray(self.clnt_x)
        self.clnt_y = np.asarray(self.clnt_y)

        self.tst_x = list(range(len(self.tst_x_char)))
        self.tst_y = list(range(len(self.tst_x_char)))

        for idx in range(len(self.tst_x_char)):
            self.tst_x[idx] = np.asarray(word_to_indices(self.tst_x_char[idx]))
            self.tst_y[idx] = np.argmax(np.asarray(letter_to_vec(self.tst_y_char[idx]))).reshape(-1)

        self.tst_x = np.asarray(self.tst_x)
        self.tst_y = np.asarray(self.tst_y)


class ShakespeareObjectCrop_noniid:
    def __init__(self, data_path, dataset_prefix, n_client=100, crop_amount=2000, tst_ratio=5, rand_seed=0):
        self.dataset = 'shakespeare'
        self.name = dataset_prefix
        users, groups, train_data, test_data = read_data(data_path + 'train/', data_path + 'test/')

        
        
        
        
        
        

        

        self.users = users

        tst_data_count_per_clnt = (crop_amount // tst_ratio)
        
        arr = []
        for clnt in range(len(users)):
            if (len(np.asarray(train_data[users[clnt]]['y'])) > crop_amount
                    and len(np.asarray(test_data[users[clnt]]['y'])) > tst_data_count_per_clnt):
                arr.append(clnt)

        
        self.n_client = n_client
        np.random.seed(rand_seed)
        np.random.shuffle(arr)
        self.user_idx = arr[:self.n_client]

        self.clnt_x = list(range(self.n_client))
        self.clnt_y = list(range(self.n_client))
        
        
        tst_data_count = 0

        for clnt, idx in enumerate(self.user_idx):
            np.random.seed(rand_seed + clnt)
            start = np.random.randint(len(train_data[users[idx]]['x']) - crop_amount)
            self.clnt_x[clnt] = np.asarray(train_data[users[idx]]['x'])[start:start + crop_amount]
            self.clnt_y[clnt] = np.asarray(train_data[users[idx]]['y'])[start:start + crop_amount]
        
        
        tst_data_count = (crop_amount // tst_ratio) * self.n_client
        self.tst_x = list(range(tst_data_count))
        self.tst_y = list(range(tst_data_count))

        tst_data_count = 0

        for clnt, idx in enumerate(self.user_idx):
            curr_amount = (crop_amount // tst_ratio)
            np.random.seed(rand_seed + clnt)
            start = np.random.randint(len(test_data[users[idx]]['x']) - curr_amount)
            self.tst_x[tst_data_count: tst_data_count + curr_amount] = np.asarray(test_data[users[idx]]['x'])[
                                                                       start:start + curr_amount]
            self.tst_y[tst_data_count: tst_data_count + curr_amount] = np.asarray(test_data[users[idx]]['y'])[
                                                                       start:start + curr_amount]
            tst_data_count += curr_amount

        self.clnt_x = np.asarray(self.clnt_x)
        self.clnt_y = np.asarray(self.clnt_y)

        self.tst_x = np.asarray(self.tst_x)
        self.tst_y = np.asarray(self.tst_y)

        

        self.clnt_x_char = np.copy(self.clnt_x)
        self.clnt_y_char = np.copy(self.clnt_y)
        
        
        self.tst_x_char = np.copy(self.tst_x)
        self.tst_y_char = np.copy(self.tst_y)

        self.clnt_x = list(range(len(self.clnt_x_char)))
        self.clnt_y = list(range(len(self.clnt_x_char)))
        
        for clnt in range(len(self.clnt_x_char)):
            
            clnt_list_x = list(range(len(self.clnt_x_char[clnt])))
            clnt_list_y = list(range(len(self.clnt_x_char[clnt])))
            

            for idx in range(len(self.clnt_x_char[clnt])):
                clnt_list_x[idx] = np.asarray(word_to_indices(self.clnt_x_char[clnt][idx]))
                clnt_list_y[idx] = np.argmax(np.asarray(letter_to_vec(self.clnt_y_char[clnt][idx]))).reshape(-1)

            self.clnt_x[clnt] = np.asarray(clnt_list_x)
            self.clnt_y[clnt] = np.asarray(clnt_list_y)

        self.clnt_x = np.asarray(self.clnt_x)
        self.clnt_y = np.asarray(self.clnt_y)

        self.tst_x = list(range(len(self.tst_x_char)))
        self.tst_y = list(range(len(self.tst_x_char)))

        for idx in range(len(self.tst_x_char)):
            self.tst_x[idx] = np.asarray(word_to_indices(self.tst_x_char[idx]))
            self.tst_y[idx] = np.argmax(np.asarray(letter_to_vec(self.tst_y_char[idx]))).reshape(-1)

        self.tst_x = np.asarray(self.tst_x)
        self.tst_y = np.asarray(self.tst_y)


class Dataset(torch.utils.data.Dataset):

    def __init__(self, data_x, data_y=True, train=False, dataset_name=''):
        self.name = dataset_name
        if self.name in ('mnist', 'synt', 'emnist', 'femnist'):
            self.X_data = torch.tensor(data_x).float()
            self.y_data = data_y
            if not isinstance(data_y, bool):
                self.y_data = torch.tensor(data_y).long()

        elif self.name == 'CIFAR10' or self.name == 'CIFAR100' or self.name == "tinyimagenet":
            self.train = train
            self.transform = transforms.Compose([transforms.ToTensor()])

            self.X_data = data_x
            self.y_data = data_y
            if not isinstance(data_y, bool):
                self.y_data = data_y.astype('float32')

        elif self.name == 'celeba':
            self.train = train
            self.X_data = torch.tensor(data_x).float()
            self.y_data = data_y
            if not isinstance(data_y, bool):
                self.y_data = torch.tensor(data_y).long()

        elif self.name == 'shakespeare':

            self.X_data = data_x
            self.y_data = data_y

            self.X_data = torch.tensor(self.X_data).long()
            if not isinstance(data_y, bool):
                self.y_data = torch.tensor(self.y_data).float()

    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, idx):
        if self.name in ('mnist', 'synt', 'emnist', 'femnist'):
            X = self.X_data[idx, :]
            if isinstance(self.y_data, bool):
                return X
            else:
                y = self.y_data[idx]
                return X, y

        elif self.name == 'CIFAR10' or self.name == 'CIFAR100':
            img = self.X_data[idx]
            if self.train:
                img = np.flip(img, axis=2).copy() if (np.random.rand() > .5) else img  
                if (np.random.rand() > .5):
                    
                    pad = 4
                    extended_img = np.zeros((3, 32 + pad * 2, 32 + pad * 2)).astype(np.float32)
                    extended_img[:, pad:-pad, pad:-pad] = img
                    dim_1, dim_2 = np.random.randint(pad * 2 + 1, size=2)
                    img = extended_img[:, dim_1:dim_1 + 32, dim_2:dim_2 + 32]
            img = np.moveaxis(img, 0, -1)
            img = self.transform(img)
            if isinstance(self.y_data, bool):
                return img
            else:
                y = self.y_data[idx]

                return img, y
        elif self.name == 'tinyimagenet':
            img = self.X_data[idx]
            if self.train:
                img = np.flip(img, axis=2).copy() if (np.random.rand() > .5) else img  
                if np.random.rand() > .5:
                    
                    pad = 8
                    extended_img = np.zeros((3, 64 + pad * 2, 64 + pad * 2)).astype(np.float32)
                    extended_img[:, pad:-pad, pad:-pad] = img
                    dim_1, dim_2 = np.random.randint(pad * 2 + 1, size=2)
                    img = extended_img[:, dim_1:dim_1 + 64, dim_2:dim_2 + 64]
            img = np.moveaxis(img, 0, -1)
            img = self.transform(img)
            if isinstance(self.y_data, bool):
                return img
            else:
                y = self.y_data[idx]
                return img, y
        elif self.name == 'celeba':
            img = self.X_data[idx].clone()
            if isinstance(self.y_data, bool):
                return img
            else:
                y = self.y_data[idx]
                return img, y

        elif self.name == 'shakespeare':
            x = self.X_data[idx]
            y = self.y_data[idx]
            return x, y
