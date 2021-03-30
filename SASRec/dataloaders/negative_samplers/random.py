from .base import AbstractNegativeSampler

from tqdm import trange
from tqdm import tqdm

import numpy as np
import random

TEST_MAX = 100000

class RandomNegativeSampler(AbstractNegativeSampler):
    @classmethod
    def code(cls):
        return 'random'

    def generate_negative_samples(self):
        assert self.seed is not None, 'Specify seed for random sampling'
        np.random.seed(self.seed)
        negative_samples = {}
        
        # print('Sampling negative items')
        # for user in tqdm(self.test):
        #     if isinstance(self.train[user][0], tuple):
        #         seen = set(x[0] for x in self.train[user])
        #         seen.update(x[0] for x in self.val[user])
        #         seen.update(x[0] for x in self.test[user])
        #     else:
        #         seen = set(self.train[user])
        #         seen.update(self.val[user])
        #         seen.update(self.test[user])

        #     samples = []
        #     for _ in range(self.sample_size):
        #         item = np.random.choice(self.item_count)
        #         while item in seen or item in samples:
        #             item = np.random.choice(self.item_count)
        #         samples.append(item)

        #     negative_samples[user] = samples

        candidates = range(self.item_count)
        test = list(self.test.keys())
        random.shuffle(test)

        for i, user in tqdm(enumerate(test)):
            if i > TEST_MAX: 
                break
            negative_samples[user] = random.choices(candidates, k=self.sample_size)

        return negative_samples
