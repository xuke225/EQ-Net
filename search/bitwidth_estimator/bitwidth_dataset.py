# bitwidth probs
import os
import json
import numpy as np
from tqdm import tqdm
import sys
import random


def list_sum(x):
    return x[0] if len(x) == 1 else x[0] + list_sum(x[1:])


def list_mean(x):
    return list_sum(x) / len(x)


__all__ = ['net_setting2id', 'net_id2setting', 'BitwidthDataset']


def net_setting2id(net_setting):
    return json.dumps(net_setting)


def net_id2setting(net_id):
    return json.loads(net_id)


def round_bw(bw, step):
    return round(round(bw / step) * step, 2)


# return int(round(bw / step) * step)


# prob
def convert_count_to_prob(m):
    if isinstance(m[list(m.keys())[0]], dict):
        for k in m:
            convert_count_to_prob(m[k])
    else:
        t = sum(m.values())
        for k in m:
            m[k] = 1.0 * m[k] / t


def count_helper(v, bw, m):
    if bw not in m:
        m[bw] = {}
    if v not in m[bw]:
        m[bw][v] = 0
    m[bw][v] += 1


class BitwidthDataset:

    def __init__(self, path, Bitwidth_estimator, bw_weights_list=None, bw_fm_list=None, module_nums=None):
        self.path = path

        os.makedirs(self.path, exist_ok=True)
        self.bw_weights_list = [2, 3, 4, 5, 6, 7, 8] if bw_weights_list is None else bw_weights_list
        self.bw_fm_list = [2, 3, 4, 5, 6, 7, 8] if bw_fm_list is None else bw_fm_list
        self.module_nums = 22 if module_nums is None else module_nums
        self.Bitwidth_estimator = Bitwidth_estimator

    @property
    def net_id_path(self):
        return os.path.join(self.path, 'Bitwidth_net_id.dict')

    @property
    def bw_src_folder(self):
        return os.path.join(self.path, 'src')

    @property
    def bw_dict_path(self):
        return os.path.join(self.path, 'Bitwidth.dict')

    def random_sample_arch(self):
        return {
            'w_bit_list': random.choices(self.bw_weights_list, k=self.module_nums),
            'a_bit_list': random.choices(self.bw_fm_list, k=self.module_nums)
        }

    # TODO: support parallel building
    def build_bw_dataset(self, n_arch=50000):
        # load net_id_list, random sample if not exist
        if os.path.isfile(self.net_id_path):
            net_id_list = json.load(open(self.net_id_path))
        else:
            net_id_list = set()
            while len(net_id_list) < n_arch:
                net_setting = self.random_sample_arch()
                net_id = net_setting2id(net_setting)
                net_id_list.add(net_id)
            net_id_list = list(net_id_list)
            net_id_list.sort()
            json.dump(net_id_list, open(self.net_id_path, 'w'), indent=4)

        with tqdm(total=len(net_id_list), desc='Building bitwidth Dataset') as t:

            # load val dataset into memory

            # save path
            os.makedirs(self.bw_src_folder, exist_ok=True)
            bw_save_path = os.path.join(self.bw_src_folder, 'Bitwidth.dict')
            bw_dict = {}
            # load existing bw dict
            if os.path.isfile(bw_save_path):
                existing_bw_dict = json.load(open(bw_save_path, 'r'))
            else:
                existing_bw_dict = {}

            for net_id in net_id_list:
                net_setting = net_id2setting(net_id)
                key = net_setting2id({**net_setting})
                if key in existing_bw_dict:
                    bw_dict[key] = existing_bw_dict[key]
                    t.set_postfix({
                        'net_id': net_id,
                        'info_val': bw_dict[key],
                        'status': 'loading',
                    })
                    t.update()
                    continue

                # print(net_setting)

                net_setting_str = ','.join(['%s_%s' % (
                    key, '%.1f' % list_mean(val) if isinstance(val, list) else val
                ) for key, val in net_setting.items()])
                avg_w, avg_fm = self.Bitwidth_estimator.get_efficiency(net_setting)
                info_val = (avg_w, avg_fm)
                # print(info_val)

                t.set_postfix({
                    'net_id': net_id,
                    'info_val': info_val,
                })
                t.update()

                bw_dict.update({
                    key: info_val
                })
            json.dump(bw_dict, open(bw_save_path, 'w'), indent=4)

    def build_trasition_prob_matrix(self, step=None):

        # initlizie
        prob_map = {}
        prob_map['discretize_step'] = 0.2 if step is None else step
        for k in ['Avg_w', 'Avg_a', 'w_bit_list', 'a_bit_list']:
            prob_map[k] = {}

        os.makedirs(self.bw_src_folder, exist_ok=True)
        bw_save_path = os.path.join(self.bw_src_folder, 'Bitwidth.dict')
        bw_dict = json.load(open(bw_save_path))

        cc = 0

        for k, v in bw_dict.items():
            dic = json.loads(k)

            # discretize
            bw_weight_value = round_bw(v[0], step)
            prob_map['Avg_w'][bw_weight_value] = prob_map['Avg_w'].get(bw_weight_value, 0) + 1

            bw_fm_value = round_bw(v[1], step)
            prob_map['Avg_a'][bw_fm_value] = prob_map['Avg_a'].get(bw_fm_value, 0) + 1

            for idx, v in enumerate(dic['w_bit_list']):
                if idx not in prob_map['w_bit_list']:
                    prob_map['w_bit_list'][idx] = {}
                count_helper(v, bw_weight_value, prob_map['w_bit_list'][idx])

            for idx, v in enumerate(dic['a_bit_list']):
                if idx not in prob_map['a_bit_list']:
                    prob_map['a_bit_list'][idx] = {}
                count_helper(v, bw_fm_value, prob_map['a_bit_list'][idx])

            cc += 1

        for k in ['Avg_w', 'Avg_a', 'w_bit_list', 'a_bit_list']:
            convert_count_to_prob(prob_map[k])
        prob_map['n_observations'] = cc
        # return bw_dict
        return prob_map
