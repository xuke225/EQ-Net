import os
import json
import numpy as np
from tqdm import tqdm
import torch
import torch.utils.data
import sys
import random


def list_sum(x):
    return x[0] if len(x) == 1 else x[0] + list_sum(x[1:])


def list_mean(x):
    return list_sum(x) / len(x)


__all__ = ['net_setting2id', 'net_id2setting', 'AccuracyDataset']


def net_setting2id(net_setting):
    return json.dumps(net_setting)


def net_id2setting(net_id):
    return json.loads(net_id)


def sample_helper(bw, m):
    if bw not in m.keys():
        near_list = [abs(i - bw) for i in m.keys()]
        bw = list(m.keys())[near_list.index(min(near_list))]

    keys = list(m[bw].keys())
    probs = list(m[bw].values())

    return random.choices(keys, weights=probs)[0]  # probmap


class RegDataset(torch.utils.data.Dataset):

    def __init__(self, inputs, targets):
        super(RegDataset, self).__init__()
        self.inputs = inputs
        self.targets = targets

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]

    def __len__(self):
        return self.inputs.size(0)


class AccuracyDataset:

    def __init__(self, path, bw_dataset=None):
        self.w_sym = None
        self.a_sym = None
        self.channel_wise = None
        self.path = path
        self.bw_dataset = bw_dataset
        self.prob_map = self.bw_dataset.build_trasition_prob_matrix(0.2)
        os.makedirs(self.path, exist_ok=True)

    @property
    def net_id_path(self):
        return os.path.join(self.path, 'net_id_c{}_w{}_a{}.dict'.format(self.channel_wise, self.w_sym, self.a_sym))

    @property
    def acc_src_folder(self):
        return os.path.join(self.path, 'src')

    @property
    def acc_dict_path(self):
        return os.path.join(self.path, 'acc_c{}_w{}_a{}.dict'.format(self.channel_wise, self.w_sym, self.a_sym))

    def random_sample_arch(self, constraint):
        archs = {}
        con_id = 0
        for k in ['w_bit_list', 'a_bit_list']:
            temp_list = []
            for idx in sorted(list(self.prob_map[k].keys())):
                temp_list.append(sample_helper(constraint[con_id], self.prob_map[k][idx]))
            con_id += 1
            archs[k] = temp_list
        return archs

    # TODO: support parallel building
    def build_acc_dataset(self, ofa_network, valid_loader, train_loader, test_func, min_bw=2.4, max_bw=7.6, n_arch=1000, channel_wise=True, w_sym=True, a_sym=True, num_samples=256):
        # load net_id_list, random sample if not exist
        self.channel_wise = channel_wise
        self.w_sym = w_sym
        self.a_sym = a_sym

        if os.path.isfile(self.net_id_path):
            net_id_list = json.load(open(self.net_id_path))
        else:
            net_id_list = set()
            while len(net_id_list) < n_arch:
                target_bw_weights = random.uniform(min_bw, max_bw)
                target_bw_fm = random.uniform(min_bw, max_bw)
                net_setting = self.random_sample_arch((target_bw_weights, target_bw_fm))
                net_id = net_setting2id(net_setting)
                net_id_list.add(net_id)
            net_id_list = list(net_id_list)
            net_id_list.sort()
            json.dump(net_id_list, open(self.net_id_path, 'w'), indent=4)

        with tqdm(total=len(net_id_list), desc='Building Acc Dataset') as t:
            # load val dataset into memory
            # save path
            os.makedirs(self.acc_src_folder, exist_ok=True)
            acc_save_path = os.path.join(self.acc_src_folder, 'acc_c{}_w{}_a{}.dict'.format(channel_wise, w_sym, a_sym))
            bw_save_path = os.path.join(self.acc_src_folder, 'acc_bw_c{}_w{}_a{}.dict'.format(channel_wise, w_sym, a_sym))
            acc_dict = {}
            bw_dict = {}
            # load existing acc dict
            if os.path.isfile(acc_save_path):
                existing_acc_dict = json.load(open(acc_save_path, 'r'))
                existing_bw_dict = json.load(open(bw_save_path, 'r'))
            else:
                existing_acc_dict = {}
                existing_bw_dict = {}
            for net_id in net_id_list:
                net_setting = net_id2setting(net_id)
                key = net_setting2id({**net_setting})
                if key in existing_acc_dict:
                    acc_dict[key] = existing_acc_dict[key]
                    bw_dict[key] = existing_bw_dict[key]

                    t.set_postfix({
                        # 'net_id': net_id,
                        'info_val': acc_dict[key],
                        'avg_bw': bw_dict[key],
                        'status': 'loading',
                    })
                    t.update()
                    continue

                ofa_network.set_quantization_params(channel_wise=channel_wise, w_sym=w_sym, a_sym=a_sym, **net_setting)

                top1, top5 = test_func(ofa_network, train_loader, valid_loader, num_samples)
                avg_w, avg_fm = self.bw_dataset.Bitwidth_estimator.get_efficiency(net_setting)
                info_val = top1

                t.set_postfix({
                    # 'net_id': net_id,
                    'info_val': info_val,
                    'avg_bw': (avg_w, avg_fm)
                })
                t.update()

                acc_dict.update({
                    key: info_val
                })
                bw_dict.update({
                    key: (avg_w, avg_fm)
                })
            json.dump(acc_dict, open(acc_save_path, 'w'), indent=4)
            json.dump(bw_dict, open(bw_save_path, 'w'), indent=4)

    def build_acc_data_loader(self, arch_encoder, n_training_sample=None, batch_size=256, n_workers=2):
        # load data
        X_all = []
        Y_all = []
        for channel_wise in [True, False]:
            for w_sym in [True, False]:
                for a_sym in [True, False]:
        # for channel_wise in [True]:
        #     for w_sym in [True, False]:
        #         for a_sym in [True, False]:
                    self.channel_wise = channel_wise
                    self.w_sym = w_sym
                    self.a_sym = a_sym
                    acc_dict = json.load(open(self.acc_dict_path))
                    with tqdm(total=len(acc_dict), desc='Loading data') as t:
                        for k, v in acc_dict.items():
                            dic = json.loads(k)
                            X_all.append(arch_encoder.arch2feature(dic, channel_wise, w_sym, a_sym))
                            Y_all.append(v / 100.)  # range: 0 - 1
                            t.update()

        base_acc = np.mean(Y_all)
        # convert to torch tensor
        X_all = torch.tensor(X_all, dtype=torch.float)
        Y_all = torch.tensor(Y_all)

        # random shuffle
        shuffle_idx = torch.randperm(len(X_all))
        X_all = X_all[shuffle_idx]
        Y_all = Y_all[shuffle_idx]

        # split data
        idx = X_all.size(0) // 5 * 4 if n_training_sample is None else n_training_sample
        val_idx = X_all.size(0) // 5 * 4
        X_train, Y_train = X_all[:idx], Y_all[:idx]
        X_test, Y_test = X_all[val_idx:], Y_all[val_idx:]
        print('Train Size: %d,' % len(X_train), 'Valid Size: %d' % len(X_test))

        # build data loader
        train_dataset = RegDataset(X_train, Y_train)
        val_dataset = RegDataset(X_test, Y_test)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=n_workers
        )
        valid_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=n_workers
        )

        return train_loader, valid_loader, base_acc
