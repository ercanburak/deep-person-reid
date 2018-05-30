from __future__ import absolute_import
from collections import defaultdict
import numpy as np

import torch
from torch.utils.data.sampler import Sampler


class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/data/sampler.py.

    Args:
        data_source (Dataset): dataset to sample from.
        num_instances (int): number of instances per identity.
    """

    def __init__(self, data_source, num_instances=4):
        self.data_source = data_source
        self.num_instances = num_instances
        self.index_dic = defaultdict(list)
        for index, (_, pid, _) in enumerate(data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.num_identities = len(self.pids)

    def __iter__(self):
        indices = torch.randperm(self.num_identities)
        ret = []
        for i in indices:
            pid = self.pids[i]
            t = self.index_dic[pid]
            replace = False if len(t) >= self.num_instances else True
            t = np.random.choice(t, size=self.num_instances, replace=replace)
            ret.extend(t)
        return iter(ret)

    def __len__(self):
        return self.num_identities * self.num_instances


class RandomSamplerDoneRight(Sampler):
    """
    Select N random examples (Same N examples are used for k iterations)

    Args:
        data_source (Dataset): dataset to sample from.
        k: Same N examples are used for k iterations
    """

    def __init__(self, data_source, k=16):
        self.data_source = data_source
        self.num_examples = len(data_source)
        # self.k = k
        self.index_dic = defaultdict(list)
        for index, (_, pid, _) in enumerate(data_source):
            self.index_dic[pid].append(index)
        # self.pids = list(self.index_dic.keys())
        # self.num_identities = len(self.pids)

    # self.iter_counter = 0 # Count the iterations, since same N examples will be used for k iterations

    def __iter__(self):
        # if self.iter_counter % self.k == 0:
        self.indices = torch.randperm(self.num_examples)
        # self.iter_counter += 1
        ret = self.index_dic[self.indices]
        return iter(ret)

    def __len__(self):
        return self.num_examples
