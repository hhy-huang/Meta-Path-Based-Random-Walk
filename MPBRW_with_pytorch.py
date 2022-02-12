"""
Task:           Meta-Path Based Random Walk
Coder:          Haoyu Huang
Source Data:    Foursquare
"""
import time
import datetime
import argparse
import numpy as np
import pickle as pickle
from collections import Counter
import collections
from math import radians, cos, sin, asin, sqrt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class MPBasedRandomWalk:
    def __init__(self, meta_path, walk_length, walk_num, data):
        self.meta_path = meta_path
        self.walk_length = walk_length
        self.walk_num = walk_num
        self.data = data
        # 数据路径
        self.tmp_path = "../data"
        self.triple_pc_path = self.tmp_path + "/triple_pc.txt"
        self.triple_ptp_path = self.tmp_path + "/triple_ptp.txt"
        self.triple_utp_path = self.tmp_path + "/triple_utp.txt"
        self.loc_path = self.tmp_path + "/entity_loc_dict.txt"
        self.user_path = self.tmp_path + "/entity_user_dict.txt"
        self.word_path = self.tmp_path + "/entity_category_name_id.txt"

        self.loc_dict = {}
        self.user_dict = {}
        self.word_dict = {}

        self.word_loc_adlist = {}
        self.loc_word_adlist = {}  # for convenience
        self.user_loc_adlist = {}
        self.loc_user_adlist = {}  # for convenience
        self.loc_loc_adlist = {}  # dict的dict
        self.loc2neighbor_LL_dict = {}
        self.loc2neighbor_LUL_dict = {}
        self.loc2neighbor_LVL_dict = {}
        self.loc2neighbor_dict = {}  # 上面三者的merge
        self.loc2paths_LL_dict = {}  # 这里path只存放pid
        self.loc2paths_LUL_dict = {}
        self.loc2paths_LVL_dict = {}
        self.v = {}
        self.tensor_ll = torch.zeros(len(self.loc_dict.keys()), len(self.loc_dict.keys()))
        self.tensor_lu = torch.zeros(len(self.loc_dict.keys()), len(self.user_dict.keys()))
        self.tensor_ul = torch.zeros(len(self.user_dict.keys()), len(self.loc_dict.keys()))
        self.tensor_lv = torch.zeros(len(self.loc_dict.keys()), len(self.word_dict.keys()))
        self.tensor_vl = torch.zeros(len(self.word_dict.keys()), len(self.loc_dict.keys()))

    # load data from three triple file,或者可以理解为生成各自的转移矩阵
    def make_adlist(self):
        print('Reading ptp file...')
        with open(self.triple_ptp_path, 'r') as f:
            for line in f:
                line = line.strip('\n').split('\t')
                pid1, _, pid2 = line  # 无向图，pid1->pid2 and pid2->pid1
                if pid1 not in self.loc_loc_adlist:
                    self.loc_loc_adlist.update({pid1: {pid2: 1}})
                    if pid2 not in self.loc_loc_adlist:
                        self.loc_loc_adlist.update({pid2: {pid1: 1}})
                    else:
                        if pid1 not in self.loc_loc_adlist[pid2]:
                            self.loc_loc_adlist[pid2].update({pid1: 1})
                        else:
                            self.loc_loc_adlist[pid2][pid1] += 1
                else:
                    if pid2 not in self.loc_loc_adlist[pid1]:
                        self.loc_loc_adlist[pid1].update({pid2: 1})
                    else:
                        self.loc_loc_adlist[pid1][pid2] += 1

        print("Reading utp file...")
        with open(self.triple_utp_path, 'r') as f:
            for line in f:
                line = line.strip('\n').split('\t')
                uid, _, pid = line
                if uid not in self.user_loc_adlist:
                    self.user_loc_adlist.update({uid: {pid: 1}})
                    if pid not in self.loc_user_adlist:
                        self.loc_user_adlist.update({pid: {uid: 1}})
                    else:
                        if uid not in self.loc_user_adlist[pid]:
                            self.loc_user_adlist[pid].update({uid: 1})
                        else:
                            self.loc_user_adlist[pid][uid] += 1
                else:
                    if pid not in self.user_loc_adlist[uid]:
                        self.user_loc_adlist[uid].update({pid: 1})
                    else:
                        self.user_loc_adlist[uid][pid] += 1

        print("Reading pc file...")
        with open(self.triple_pc_path, 'r') as f:
            for line in f:
                line = line.strip('\n').split('\t')
                pid, _, wid = line
                if pid not in self.loc_word_adlist:
                    self.loc_word_adlist.update({pid: {wid: 1}})
                    if wid not in self.word_loc_adlist:
                        self.word_loc_adlist.update({wid: {pid: 1}})
                    else:
                        if pid not in self.word_loc_adlist[wid]:
                            self.word_loc_adlist[wid].update({pid: 1})
                        else:
                            self.word_loc_adlist[wid][pid] += 1
                else:
                    if wid not in self.loc_word_adlist[pid]:
                        self.loc_word_adlist[pid].update({pid: 1})
                    else:
                        self.loc_word_adlist[pid][wid] += 1  # 逻辑上讲不可能出现这样的情况

        print('Reading loc file...')
        with open(self.loc_path, 'r') as f:
            for line in f:
                line = line.strip('\n').split('\t')
                pid, loc_name = line
                self.loc_dict[pid] = loc_name
        self.loc_dict = {k: v for k, v in
                         sorted(self.loc_dict.items(), key=lambda items: int(items[0]))}  # 保证顺序与tensor一致

        print('Reading user file...')
        with open(self.user_path, 'r') as f:
            for line in f:
                line = line.strip('\n').split('\t')
                uid, user_name = line
                self.user_dict[uid] = user_name
        self.user_dict = {k: v for k, v in sorted(self.user_dict.items(), key=lambda items: int(items[0]))}

        print('Reading word file...')
        with open(self.word_path, 'r') as f:
            for line in f:
                line = line.strip('\n').split('\t')
                wid, word_name = line
                self.word_dict[wid] = word_name
        self.word_dict = {k: v for k, v in sorted(self.word_dict.items(), key=lambda items: int(items[0]))}

    # 生成一个单一类型元素的dict
    def make_vec(self, keys):
        result = dict()
        for key in keys:
            result.update({key: 1})
        return result

    # 起始的向量，只有一个起始点loc，也就是只有这一个元素为1     <too slow>
    # def begin_vec(self, vec, begin):
    #     tmp_vec = vec.copy()
    #     for key in tmp_vec.keys():
    #         if key == begin:
    #             tmp_vec[key] = float(1)
    #         else:
    #             tmp_vec[key] = float(0)
    #     return tmp_vec

    # 对adlist进行平均分布，也就是概率
    def nom_adlist(self, adlist):
        tmp_adlist = adlist.copy()  # value->key,为什么以有向图的方式存，因为要方便不同种类的entity进行转移

        for key1 in tqdm(tmp_adlist.keys()):  # 遍历所有的node（5000）
            count = 0
            for key2 in tmp_adlist.keys():  # 再次遍历所有的node, 计算key1的出度count
                if key1 in tmp_adlist[key2]:
                    count += 1
            for key2 in tmp_adlist.keys():  # key1->key2，1/key1的出度
                if key1 in tmp_adlist[key2]:
                    tmp_adlist[key2][key1] = float(tmp_adlist[key2][key1]) / count

        return tmp_adlist  # 返回的结果是转移矩阵

    # 矩阵dict与向量dict相乘 (n,n)·(n,1)=(n,1),        <too slow>
    # def adlist_vec_multiply(self, adlist, vec):
    #     result = dict()
    #     for key1 in adlist.keys():
    #         tmp1 = adlist[key1]                         # 是所有指向key1的node
    #         tmp_sum = 0
    #         for key2 in vec.keys():                     # 遍历向量的每一个元素
    #             if key2 in tmp1:                        # 如果说向量的这个位置所对应的node，在矩阵的这一行中存在，就讲这两个值相乘
    #                 tmp_sum += tmp1[key2] * vec[key2]   # 矩阵这一行的乘积加起来，没有的就是0，得到结果向量的某个位置的值
    #         result.update({key1: tmp_sum})
    #     return result

    # 初始化v
    def init_vec(self, length, idx):  # idx是pid - 1，其它id模除
        tmp_tensor = torch.zeros((length, 1))
        tmp_tensor[idx] = 1
        return tmp_tensor.cuda()

    # dict转成tensor
    def dict2tensor(self):
        self.loc_loc_adlist = self.nom_adlist(self.loc_loc_adlist)
        self.user_loc_adlist = self.nom_adlist(self.user_loc_adlist)
        self.loc_user_adlist = self.nom_adlist(self.loc_user_adlist)
        self.loc_word_adlist = self.nom_adlist(self.loc_word_adlist)
        self.word_loc_adlist = self.nom_adlist(self.word_loc_adlist)

        print('loc_loc_adlist to tensor...')
        tmp_tensor_ll = torch.zeros(len(self.loc_dict.keys()), len(self.loc_dict.keys()))  # 注意index+1为id
        for pid1, row in zip(self.loc_loc_adlist.keys(), self.loc_loc_adlist.values()):
            for pid, w in zip(row.keys(), row.values()):
                tmp_tensor_ll[int(pid1) - 1][int(pid) - 1] = w

        print('loc_user_adlist to tensor...')
        tmp_tensor_lu = torch.zeros(len(self.loc_dict.keys()), len(self.user_dict.keys()))
        for pid, row in zip(self.loc_user_adlist.keys(), self.loc_user_adlist.values()):
            for uid, w in zip(row.keys(), row.values()):
                tmp_tensor_lu[int(pid) - 1][int(uid) % len(self.user_dict.keys())] = w  # 因为uid不是从1编码的

        print('user_loc_adlist to tensor...')
        tmp_tensor_ul = torch.zeros(len(self.user_dict.keys()), len(self.loc_dict.keys()))
        for uid, row in zip(self.user_loc_adlist.keys(), self.user_loc_adlist.values()):
            for pid, w in zip(row.keys(), row.values()):
                tmp_tensor_ul[int(uid) % len(self.user_dict.keys())][int(pid) - 1] = w

        print('loc_word_adlist to tensor...')
        tmp_tensor_lv = torch.zeros(len(self.loc_dict.keys()), len(self.word_dict.keys()))
        for pid, row in zip(self.loc_word_adlist.keys(), self.loc_word_adlist.values()):
            for wid, w in zip(row.keys(), row.values()):
                tmp_tensor_lv[int(pid) - 1][int(wid) % len(self.word_dict.keys())] = w

        print('word_loc_adlist to tensor...')
        tmp_tensor_vl = torch.zeros(len(self.word_dict.keys()), len(self.loc_dict.keys()))
        for wid, row in zip(self.word_loc_adlist.keys(), self.word_loc_adlist.values()):
            for pid, w in zip(row.keys(), row.values()):
                tmp_tensor_vl[int(wid) % len(self.word_dict.keys())][int(pid) - 1] = w

        return [tmp_tensor_ll, tmp_tensor_lu, tmp_tensor_ul, tmp_tensor_lv, tmp_tensor_vl]

    # 决定元路径
    def method(self):
        print('Convert to tensor...')
        self.tensor_ll, self.tensor_lu, self.tensor_ul, \
        self.tensor_lv, self.tensor_vl = self.dict2tensor()
        self.tensor_ll = self.tensor_ll.cuda()
        self.tensor_lu = self.tensor_lu.cuda()
        self.tensor_ul = self.tensor_ul.cuda()
        self.tensor_lv = self.tensor_lv.cuda()
        self.tensor_vl = self.tensor_vl.cuda()
        if self.meta_path == 'LL':
            print('LL based random walk...')
            for pid in tqdm(self.loc_loc_adlist):
                total_paths = self.run_with_LL(pid)
                self.loc2paths_LL_dict[pid] = total_paths
        # elif self.meta_path == 'LVL':
            print('LUL based random walk...')
            for pid in tqdm(self.loc_loc_adlist):
                total_paths = self.run_with_LVL(pid)
                self.loc2paths_LVL_dict[pid] = total_paths
        # elif self.meta_path == 'LUL':
            print('LVL based random walk...')
            for pid in tqdm(self.loc_loc_adlist):
                total_paths = self.run_with_LUL(pid)
                self.loc2paths_LUL_dict[pid] = total_paths

    # 三类随机游走的实现
    def run_with_LL(self, pid):
        total_paths = []
        for i in range(self.walk_num):
            # self.v = self.nom_vec(self.make_vec(self.loc_loc_adlist.keys()))
            # self.v = self.begin_vec(self.make_vec(self.loc_loc_adlist.keys()), pid)
            # self.v = self.vec2tensor(self.v, 'LL')
            self.v = self.init_vec(len(self.loc_dict), int(pid) - 1)
            path = [pid]
            for j in range(self.walk_length):  # walk_length长度的path
                # self.v = self.adlist_vec_multiply(self.loc_loc_adlist, self.v)
                self.v = torch.mm(self.tensor_ll, self.v)
                # tmp_vec = np.array(list(self.nom_vec(self.v).values()))
                tmp_vec = self.v / torch.sum(self.v)  # 以当前的权重向量为权重进行random walk下一跳的选择
                tmp_vec = tmp_vec.cpu()
                tmp_vec = tmp_vec.numpy()
                next_pid = np.random.choice(list(self.loc_dict.keys()), p=tmp_vec.ravel())
                path.append(next_pid)
                # self.v = self.begin_vec(self.make_vec(self.loc_loc_adlist.keys()), next_pid)
                # self.v = self.vec2tensor(self.v, 'LL')
                self.v = self.init_vec(len(self.loc_dict), int(next_pid) - 1)  # 重置起点
            total_paths.append(path)
        return total_paths

    def run_with_LUL(self, pid):
        total_paths = []
        for i in range(self.walk_num):
            # self.v = self.begin_vec(self.make_vec(self.loc_loc_adlist.keys()), pid)
            # self.v = self.vec2tensor(self.v, 'LL')
            self.v = self.init_vec(len(self.loc_dict), int(pid) - 1)  # 依然将pid作为起点,loc的起点权重向量
            path = [pid]
            for j in range(self.walk_length):
                # self.v = self.adlist_vec_multiply(self.user_loc_adlist, self.v)
                self.v = torch.mm(self.tensor_ul, self.v)  # L->U
                # tmp_vec = np.array(list(self.nom_vec(self.v).values()))
                tmp_vec = self.v / torch.sum(self.v)  # 得到下一个uid的分布权重
                tmp_vec = (tmp_vec.cpu()).numpy()
                next_uid = np.random.choice(list(self.user_dict.keys()), p=tmp_vec.ravel())  # 找到置信uid
                # next_uid = np.random.choice(list(self.v.keys()), p=tmp_vec.ravel())
                path.append(next_uid)
                # self.v = self.begin_vec(self.make_vec(self.user_loc_adlist.keys()), next_uid)
                self.v = self.init_vec(len(self.user_dict), int(next_uid) % len(self.user_dict))  # user的起点权重向量
                # self.v = self.adlist_vec_multiply(self.loc_user_adlist, self.v)
                self.v = torch.mm(self.tensor_lu, self.v)  # U->L
                # tmp_vec = np.array(list(self.nom_vec(self.v).values()))
                tmp_vec = self.v / torch.sum(self.v)  # 得到下一个pid的分布权重
                tmp_vec = (tmp_vec.cpu()).numpy()
                next_pid = np.random.choice(list(self.loc_dict.keys()), p=tmp_vec.ravel())  # 找到置信pid
                path.append(next_pid)
                # self.v = self.begin_vec(self.make_vec(self.loc_user_adlist.keys()), next_uid)
                self.v = self.init_vec(len(self.loc_dict), int(next_pid) - 1)  # loc的起点权重向量
            total_paths.append(path)
        return total_paths

    def run_with_LVL(self, pid):
        total_paths = []
        for i in range(self.walk_length):
            # self.v = self.begin_vec(self.make_vec(self.loc_loc_adlist.keys()), pid)
            self.v = self.init_vec(len(self.loc_dict), int(pid) - 1)
            path = [pid]
            for j in range(self.walk_length):
                # self.v = self.adlist_vec_multiply(self.word_loc_adlist, self.v)
                self.v = torch.mm(self.tensor_vl, self.v)  # L->V
                # tmp_vec = np.array(list(self.nom_vec(self.v).values()))
                tmp_vec = self.v / torch.sum(self.v)
                tmp_vec = (tmp_vec.cpu()).numpy()
                next_wid = np.random.choice(list(self.word_dict.keys()), p=tmp_vec.ravel())
                # next_wid = np.random.choice(list(self.v.keys()), p=tmp_vec.ravel())
                path.append(next_wid)
                # self.v = self.begin_vec(self.make_vec(self.word_loc_adlist.keys()), next_wid)
                self.v = self.init_vec(len(self.word_dict), int(next_wid) % len(self.word_dict))
                # self.v = self.adlist_vec_multiply(self.loc_word_adlist, self.v)
                self.v = torch.mm(self.tensor_lv, self.v)  # V->L
                # tmp_vec = np.array(list(self.nom_vec(self.v).values()))
                tmp_vec = self.v / torch.sum(self.v)
                tmp_vec = (tmp_vec.cpu()).numpy()
                next_pid = np.random.choice(list(self.loc_dict.keys()), p=tmp_vec.ravel())
                path.append(next_pid)
                # self.v = self.begin_vec(self.make_vec(self.loc_word_adlist.keys()), next_wid)
                self.v = self.init_vec(len(self.loc_dict), int(next_pid) - 1)
            total_paths.append(path)
        return total_paths

    def save_variables(self):
        paths = {'LL': self.loc2paths_LL_dict, 'LUL': self.loc2paths_LUL_dict, 'LVL': self.loc2paths_LVL_dict}
        pickle.dump(paths, open(self.tmp_path + '/paths' + '.pkl', 'wb'))


if __name__ == '__main__':
    generator = MPBasedRandomWalk(meta_path='LL', walk_length=5, walk_num=10, data=[])
    generator.make_adlist()
    generator.method()
    generator.save_variables()
