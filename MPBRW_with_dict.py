"""
Task:           Meta-Path Based Random Walk
Coder:          Haoyu Huang
Source Data:    Foursquare
the slow version
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


class MPBasedRandomWalk:
    def __init__(self, meta_path, walk_length, walk_num, data):
        self.meta_path = meta_path
        self.walk_length = walk_length
        self.walk_num = walk_num
        self.data = data

        self.tmp_path = "../data"
        self.triple_pc_path = self.tmp_path + "/triple_pc.txt"
        self.triple_ptp_path = self.tmp_path + "/triple_ptp.txt"
        self.triple_utp_path = self.tmp_path + "/triple_utp.txt"

        self.word_loc_adlist = {}
        self.loc_word_adlist = {}  # for convenience
        self.user_loc_adlist = {}
        self.loc_user_adlist = {}  # for convenience
        self.loc_loc_adlist = {}  # dict的dict
        self.loc2neighbor_LL_dict = {}
        self.loc2neighbor_LUL_dict = {}
        self.loc2neighbor_LVL_dict = {}
        self.loc2neighbor_dict = {}  # 上面三者的merge
        self.loc2paths_LL_dict = {}
        self.loc2paths_LUL_dict = {}
        self.loc2paths_LVL_dict = {}
        self.v = {}

    # load data from three triple file,或者可以理解为生成各自的转移矩阵
    def make_adlist(self):
        print('Reading ptp file:\n')
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

        print("Reading utp file:\n")
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

        print("Reading pc file:\n")
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

    # 生成一个单一类型元素的dict
    def make_vec(self, keys):
        result = dict()
        for key in keys:
            result.update({key: 1})
        return result

    # 对单一类型的元素dict进行平均分布
    def nom_vec(self, vec):
        tmp_vec = vec.copy()  # 不改变原本数据
        total = sum(tmp_vec.values())
        for key in tmp_vec.keys():
            tmp_vec[key] = float(tmp_vec[key]) / total
        return tmp_vec

    # 起始的向量，只有一个起始点loc，也就是只有这一个元素为1
    def begin_vec(self, vec, begin):
        tmp_vec = vec.copy()
        for key in tmp_vec.keys():
            if key == begin:
                tmp_vec[key] = float(1)
            else:
                tmp_vec[key] = float(0)
        return tmp_vec

    # 对adlist进行平均分布，也就是概率
    def nom_adlist(self, adlist):
        tmp_adlist = adlist.copy()                      # value->key,为什么以有向图的方式存，因为要方便不同种类的entity进行转移

        for key1 in tqdm(tmp_adlist.keys()):            # 遍历所有的node（5000）
            count = 0
            for key2 in tmp_adlist.keys():              # 再次遍历所有的node, 计算key1的出度count
                if key1 in tmp_adlist[key2]:
                    count += 1
            for key2 in tmp_adlist.keys():              # key1->key2，1/key1的出度
                if key1 in tmp_adlist[key2]:
                    tmp_adlist[key2][key1] = float(tmp_adlist[key2][key1]) / count

        return tmp_adlist                               # 返回的结果是转移矩阵

    # 矩阵与向量相乘 (n,n)·(n,1)=(n,1),<too slow>
    def adlist_vec_multiply(self, adlist, vec):
        result = dict()
        for key1 in adlist.keys():
            tmp1 = adlist[key1]                         # 是所有指向key1的node
            tmp_sum = 0
            for key2 in vec.keys():                     # 遍历向量的每一个元素
                if key2 in tmp1:                        # 如果说向量的这个位置所对应的node，在矩阵的这一行中存在，就讲这两个值相乘
                    tmp_sum += tmp1[key2] * vec[key2]   # 矩阵这一行的乘积加起来，没有的就是0，得到结果向量的某个位置的值
            result.update({key1: tmp_sum})
        return result

    # 决定元路径
    def method(self):
        if self.meta_path == 'LL':
            for pid in tqdm(self.loc_loc_adlist):
                total_paths = self.run_with_LL(pid)
                self.loc2paths_LL_dict[pid] = total_paths
        # elif self.meta_path == 'LVL':
            for pid in tqdm(self.loc_loc_adlist):
                total_paths = self.run_with_LVL(pid)
                self.loc2paths_LVL_dict[pid] = total_paths
        # elif self.meta_path == 'LUL':
            for pid in tqdm(self.loc_loc_adlist):
                total_paths = self.run_with_LUL(pid)
                self.loc2paths_LUL_dict[pid] = total_paths

    # 三类随机游走的实现
    def run_with_LL(self, pid):
        print('LL based random walk:')
        self.loc_loc_adlist = self.nom_adlist(self.loc_loc_adlist)
        # self.v = self.nom_vec(self.make_vec(self.loc_loc_adlist.keys()))
        total_paths = []
        for i in range(self.walk_num):
            self.v = self.begin_vec(self.make_vec(self.loc_loc_adlist.keys()), pid)
            path = [pid]
            for j in range(self.walk_length):                                                 # walk_length长度的path
                self.v = self.adlist_vec_multiply(self.loc_loc_adlist, self.v)
                tmp_vec = np.array(list(self.nom_vec(self.v).values()))                       # 以当前的权重向量为权重进行random walk下一跳的选择
                next_pid = np.random.choice(list(self.v.keys()), p=tmp_vec.ravel())
                path.append(next_pid)
                self.v = self.begin_vec(self.make_vec(self.loc_loc_adlist.keys()), next_pid)  # 重置起点
            total_paths.append(path)
        return total_paths

    def run_with_LUL(self, pid):
        print('LUL based random walk:')
        self.user_loc_adlist = self.nom_adlist(self.user_loc_adlist)
        self.loc_user_adlist = self.nom_adlist(self.loc_user_adlist)
        total_paths = []
        for i in range(self.walk_num):
            self.v = self.begin_vec(self.make_vec(self.loc_loc_adlist.keys()), pid)                # 依然将pid作为起点,loc的起点权重向量
            path = [pid]
            for j in range(self.walk_length):
                self.v = self.adlist_vec_multiply(self.user_loc_adlist, self.v)                # L->U
                tmp_vec = np.array(list(self.nom_vec(self.v).values()))                        # 得到下一个uid的分布权重
                next_uid = np.random.choice(list(self.v.keys()), p=tmp_vec.ravel())            # 找到置信uid
                path.append(next_uid)
                self.v = self.begin_vec(self.make_vec(self.user_loc_adlist.keys()), next_uid)  # user的起点权重向量
                self.v = self.adlist_vec_multiply(self.loc_user_adlist, self.v)                # U->L
                tmp_vec = np.array(list(self.nom_vec(self.v).values()))                        # 得到下一个pid的分布权重
                next_pid = np.random.choice(list(self.v.keys()), p=tmp_vec.ravel())            # 找到置信pid
                path.append(next_pid)
                self.v = self.begin_vec(self.make_vec(self.loc_user_adlist.keys()), next_uid)  # loc的起点权重向量
            total_paths.append(path)
        return total_paths

    def run_with_LVL(self, pid):
        print('LVL based random walk:')
        self.loc_word_adlist = self.nom_adlist(self.loc_word_adlist)
        self.word_loc_adlist = self.nom_adlist(self.word_loc_adlist)
        total_paths = []
        for i in range(self.walk_length):
            self.v = self.begin_vec(self.make_vec(self.loc_loc_adlist.keys()), pid)
            path = [pid]
            for j in range(self.walk_length):
                self.v = self.adlist_vec_multiply(self.word_loc_adlist, self.v)                # L->V
                tmp_vec = np.array(list(self.nom_vec(self.v).values()))
                next_wid = np.random.choice(list(self.v.keys()), p=tmp_vec.ravel())
                path.append(next_wid)
                self.v = self.begin_vec(self.make_vec(self.word_loc_adlist.keys()), next_wid)
                self.v = self.adlist_vec_multiply(self.loc_word_adlist, self.v)                # V->L
                tmp_vec = np.array(list(self.nom_vec(self.v).values()))
                next_pid = np.random.choice(list(self.v.keys()), p=tmp_vec.ravel())
                path.append(next_pid)
                self.v = self.begin_vec(self.make_vec(self.loc_word_adlist.keys()), next_wid)
            total_paths.append(path)
        return total_paths

    def save_variables(self):
        paths = {'LL': self.loc2paths_LL_dict, 'LUL': self.loc2paths_LUL_dict, 'LVL': self.loc2paths_LVL_dict}
        pickle.dump(paths, open(self.tmp_path + 'paths' + '.pkl', 'wb'))


if __name__ == '__main__':
    generator = MPBasedRandomWalk(meta_path='LL', walk_length=10, walk_num=50, data=[])
    generator.make_adlist()
    generator.method()
