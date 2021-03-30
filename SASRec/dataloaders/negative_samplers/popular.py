from .base import AbstractNegativeSampler

from tqdm import trange
from tqdm import tqdm

from collections import Counter

import numpy as np


class PopularNegativeSampler(AbstractNegativeSampler):
    @classmethod
    def code(cls):
        return 'popular'

    def generate_negative_samples(self):
        popularity = self.items_by_popularity()

        keys = list(popularity.keys())
        values = [popularity[k] for k in keys]
        sum_value = np.sum(values)
        p = [value / sum_value for value in values]

        negative_samples = {}
        print('Sampling negative items')
        for user in tqdm(self.test):
            seen = set(self.train[user])
            seen.update(self.val[user])
            seen.update(self.test[user])

            samples = []
            while len(samples) < self.sample_size:
                sampled_ids = np.random.choice(keys, self.sample_size, replace=False, p=p).tolist()
                sampled_ids = [x for x in sampled_ids if x not in seen and x not in samples]
                samples.extend(sampled_ids)
            samples = samples[:self.sample_size]
            negative_samples[user] = samples

        return negative_samples

    def items_by_popularity(self):
        popularity = Counter()
        for user in tqdm(self.test):
            popularity.update(self.train[user])
            popularity.update(self.val[user])
            popularity.update(self.test[user])
        return popularity
