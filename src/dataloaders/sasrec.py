import random
import torch
import torch.utils.data as data_utils
import numpy as np
import json

class SASRecDataloader(object):
    def __init__(self, args, dataset):
        self.args = args
        seed = args.dataloader_random_seed
        self.rng = random.Random(seed)
        self.train = dataset.train
        self.val = dataset.val
        self.test = dataset.test
        self.umap = dataset.umap
        self.smap = dataset.smap
        self.num_users = args.num_users = len(self.umap)
        self.num_items = args.num_items = len(self.smap)

        self.max_len = args.trm_max_len
        self.CLOZE_MASK_TOKEN = self.num_items
        self.PAD_TOKEN = self.CLOZE_MASK_TOKEN + 1

        self.test_negative_samples = dataset.negative_samples
        

    @classmethod
    def code(cls):
        return 'sasrec'

    def get_pytorch_dataloaders(self):
        train_loader = self._get_train_loader()
        val_loader = self._get_eval_loader(mode='val')
        test_loader = self._get_eval_loader(mode='test')
        return train_loader, val_loader, test_loader

    def _get_train_loader(self):
        dataset = SASRecTrainDataset(self.train, self.max_len, 
                    self.CLOZE_MASK_TOKEN, self.num_items, self.rng, self.PAD_TOKEN)
        dataloader = data_utils.DataLoader(dataset, batch_size=self.args.train_batch_size, 
                    drop_last=False, shuffle=True, pin_memory=True)
        dataloader.pad_token = self.PAD_TOKEN
        return dataloader

    def _get_eval_loader(self, mode):
        batch_size = self.args.val_batch_size if mode == 'val' else self.args.test_batch_size
        answers = self.val if mode == 'val' else self.test
        dataset = SASRecEvalDataset(self.train, answers, self.max_len,
             self.CLOZE_MASK_TOKEN, self.test_negative_samples, self.PAD_TOKEN, 
                mode=mode, val=self.val, is_all=(self.args.trainer_code=="sasrec_all"))
        dataloader = data_utils.DataLoader(dataset, batch_size=batch_size, drop_last=False,
                shuffle=True, pin_memory=True)
        return dataloader


class SASRecTrainDataset(data_utils.Dataset):
    def __init__(self, u2seq, max_len, mask_token, num_items, rng, pad_token):
        self.u2seq = u2seq
        self.users = sorted(self.u2seq.keys())
        self.max_len = max_len
        self.rng = rng
        self.num_items = num_items
        self.pad_token = pad_token

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self.u2seq[user]

        labels = seq[-self.max_len-1:]
        if len(labels) > 1:
            tokens = labels[:-1]
            labels = labels[1:]
        else:
            tokens = [self.pad_token]
        length = len(tokens)


        item_negative = []
        while len(item_negative) < length:
            item_negative_tmp = self.rng.randint(0, self.num_items-1)
            while item_negative_tmp in self.u2seq[user]:
                item_negative_tmp = self.rng.randint(0, self.num_items-1)
            item_negative.append(item_negative_tmp)

        padding_len = self.max_len - length

        tokens = tokens + [self.pad_token] * padding_len
        labels = torch.LongTensor(labels + [-100] * padding_len).unsqueeze(-1)
        negs = torch.LongTensor(item_negative + [-100] * padding_len).unsqueeze(-1)

        return torch.LongTensor([user]), torch.LongTensor(tokens), torch.cat((labels, negs), dim=-1)


class SASRecEvalDataset(data_utils.Dataset):
    def __init__(self, u2seq, u2answer, max_len, mask_token, negative_samples, pad_token, mode, val, is_all):
        self.u2seq = u2seq
        self.negative_samples = negative_samples
        self.users = {i:u for i,u in enumerate(negative_samples)}
        self.u2answer = u2answer
        self.max_len = max_len
        self.mask_token = mask_token
        self.pad_token = pad_token
        self.mode = mode
        self.val = val
        self.is_all = is_all

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        return self.all(index) if self.is_all else self.sample(index)

    def all(self, index):
        user = self.users[index]
        seq = self.u2seq[user] if self.mode == "val" else self.u2seq[user] + self.val[user]
        answer = self.u2answer[user]

        seq = seq[-self.max_len:]
        length = len(seq)
        padding_len = self.max_len - length
        seq = seq + [self.pad_token] * padding_len

        return torch.LongTensor([user]), torch.LongTensor(seq), torch.LongTensor(answer), torch.LongTensor([length-1])

    def sample(self, index):
        user = self.users[index]
        seq = self.u2seq[user] if self.mode == "val" else self.u2seq[user] + self.val[user]
        answer = self.u2answer[user]
        negs = self.negative_samples[user]
        # TODO: test all ranking
        # negs = set(range(self.mask_token+2))
        # negs = list(negs - set(seq + answer))

        candidates = answer + negs
        labels = [1] * len(answer) + [0] * len(negs)

        seq = seq[-self.max_len:]
        length = len(seq)
        padding_len = self.max_len - length
        seq = seq + [self.pad_token] * padding_len

        return torch.LongTensor([user]), torch.LongTensor(seq), torch.LongTensor(candidates), torch.LongTensor(labels), torch.LongTensor([length-1])
