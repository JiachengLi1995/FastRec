from .negative_samplers import negative_sampler_factory

import os
import json
import random

class ItemDataset(object):
    def __init__(self, args):
        self.args = args
        self.path = args.data_path
        self.train = self.read_json(os.path.join(self.path, "train.json"), True)
        self.val = self.read_json(os.path.join(self.path, "val.json"), True)
        self.test = self.read_json(os.path.join(self.path, "test.json"), True)
        self.data = self.merge(self.train, self.val, self.test)
        self.umap = self.read_json(os.path.join(self.path, "umap.json"))
        self.smap = self.read_json(os.path.join(self.path, "smap.json"))
        negative_sampler = negative_sampler_factory(args.test_negative_sampler_code, self.train, self.val, self.test,
                                                         len(self.umap), len(self.smap),
                                                         args.test_negative_sample_size,
                                                         args.test_negative_sampling_seed,
                                                         self.path)
        self.negative_samples = negative_sampler.get_negative_samples()

    def merge(self, a, b, c):
        data = {}
        for i in c:
            data[i] = a[i] + b[i] + c[i]
        return data

    def read_json(self, path, as_int=False):
        with open(path, 'r') as f:
            raw = json.load(f)
            if as_int:
                data = dict((int(key), value) for (key, value) in raw.items())
            else:
                data = dict((key, value) for (key, value) in raw.items())
            del raw
            return data
    
    @classmethod
    def code(cls):
        return "item"

    def subdataset(self, k=100000):
        assert len(self.smap) > k
        # sample
        user_shuffle = list(self.train.keys())
        random.shuffle(user_shuffle)
        items, users = set([]), []
        for u in user_shuffle:
            for i in self.train[u]:
                items.add(i)
            users.append(u)
            if len(items) > k:
                break
        # user mapping
        umap = dict(zip(users, range(len(users))))
        train = {umap[u]:self.train[u] for u in users}
        val = {umap[u]:self.val[u] for u in users if u in self.val}
        test = {umap[u]:self.test[u] for u in users if u in self.test}
        # item mapping
        items = []
        for temp in [train, val, test]:
            for i in temp:
                items.extend(temp[i])
        items = list(set(items))
        smap = dict(zip(items, range(len(items))))
        for temp in [train, val, test]:
            for i in temp:
                temp[i] = [smap[j] for j in temp[i]]
        # sampling on the fly
        negative_sampler = negative_sampler_factory(self.args.test_negative_sampler_code, train, val, test,
                                                         len(umap), len(smap),
                                                         self.args.test_negative_sample_size,
                                                         self.args.test_negative_sampling_seed,
                                                         self.path)
        negative_samples = negative_sampler.get_negative_samples(save=False)

        return SubItemDataset(self.path, train, val, test, umap, smap, negative_samples)

class SubItemDataset(object):
    def __init__(self, path, train, val, test, umap, smap, negative_samples):
        self.path = path
        self.train = train
        self.val = val
        self.test = test
        self.umap = umap
        self.smap = smap
        self.negative_samples = negative_samples
        self.data = self.merge(self.train, self.val, self.test)
        
    def merge(self, a, b, c):
        data = {}
        for i in c:
            data[i] = a[i] + b[i] + c[i]
        return data

    @classmethod
    def code(cls):
        return "item"
