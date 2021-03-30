from abc import *
import random
import numpy as np

class AbstractDataloader(metaclass=ABCMeta):
    def __init__(self, args, dataset):
        self.args = args
        seed = args.dataloader_random_seed
        self.rng = random.Random(seed)
        self.save_folder = args.data_path
        self.train = dataset.train
        self.val = dataset.val
        self.test = dataset.test
        self.umap = dataset.umap
        self.smap = dataset.smap
        self.user_count = len(self.umap)
        self.item_count = len(self.smap)
        if 'pop' in self.args.model_code:
            self.args.item_freq = self.pop_rec(self.train)

    def pop_rec(self, data):
        from collections import Counter
        item_freq = []
        for u in data:
            item_freq.extend(data[u])
        item_freq = Counter(item_freq)
        item_freq_vec = np.zeros(self.item_count)
        for i in item_freq:
            item_freq_vec[i] = item_freq[i]
        return item_freq_vec

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    @abstractmethod
    def get_pytorch_dataloaders(self):
        pass
