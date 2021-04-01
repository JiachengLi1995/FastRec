from .base import AbstractTrainer
from .utils import Ranker, SampleRanker

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SASRecSampleTrainer(AbstractTrainer):
    def __init__(self, args, model, train_loader, val_loader, test_loader, ckpt_root, user2seq, embed_only=False):
        super().__init__(args, model, train_loader, val_loader, test_loader, ckpt_root, user2seq, embed_only)
        self.ce = nn.CrossEntropyLoss()
        self.ranker = SampleRanker(self.metric_ks, self.user2seq)

    @classmethod
    def code(cls):
        return 'sasrec_sample'

    def calculate_loss(self, batch):
        users, tokens, candidates = batch
        x, loss = self.model(tokens, candidates=candidates, mode="train")  # scores, loss
        return loss

    def calculate_metrics(self, batch, mode):
        users, seqs, candidates, labels, length = batch
        scores = self.model(seqs, candidates=candidates, length=length, mode="sample")  # B x T x C
        res = self.ranker(scores)
        metrics = {}
        for i, k in enumerate(self.args.metric_ks):
            metrics["NDCG@%d" % k] = res[2*i]
            metrics["Recall@%d" % k] = res[2*i+1]
        metrics["MRR"] = res[-3]
        metrics["AUC"] = res[-2]
        return metrics
