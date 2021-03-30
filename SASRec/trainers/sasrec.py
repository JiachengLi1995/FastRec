from .base import AbstractTrainer
from .utils import recalls_and_ndcgs_for_ks, Ranker, SampleRanker
from .parallel import DataParallelCriterion, DataParallelMetric

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

CASE_LEN = 100

class SASRecAllTrainer(AbstractTrainer):
    def __init__(self, args, model, train_loader, val_loader, test_loader, ckpt_root, user2seq):
        super().__init__(args, model, train_loader, val_loader, test_loader, ckpt_root, user2seq)
        if self.is_parallel:
            self.ce_train = DataParallelCriterion(nn.CrossEntropyLoss())
            self.ce_test = nn.CrossEntropyLoss()
            self.ranker = DataParallelMetric(Ranker(self.metric_ks, self.user2seq))
        else:
            self.ce = nn.CrossEntropyLoss()
            self.ranker = Ranker(self.metric_ks, self.user2seq)

    @classmethod
    def code(cls):
        return 'sasrec_all'

    def calculate_loss(self, batch):
        if self.is_parallel:
            return self.calculate_loss_parallel(batch)
        else:
            return self.calculate_loss_single(batch)

    def calculate_metrics(self, batch, mode):
        if self.is_parallel:
            return self.calculate_metrics_parallel(batch, mode)
        else:
            return self.calculate_metrics_single(batch, mode)

    def calculate_loss_parallel(self, batch):
        users, tokens, candidates = batch
        x = self.model(tokens, candidates=candidates, mode="train")  # scores, loss
        device = x[0][0].device
        loss = [x_[-1].to(device) for x_ in x]
        loss = sum(loss) / len(loss)
        return loss

    def calculate_metrics_parallel(self, batch, mode):
        users, seqs, labels, lengths = batch
        scores = self.model(seqs, length=lengths, mode="all")  # B x T x C
        res = np.array(self.ranker(scores, labels, users=users)).mean(axis=0)
        metrics = {}
        for i, k in enumerate(self.args.metric_ks):
            metrics["NDCG@%d" % k] = res[2*i]
            metrics["Recall@%d" % k] = res[2*i+1]
        metrics["MRR"] = res[-3]
        metrics["AUC"] = res[-2]
        metrics["loss"] = res[-1]
        return metrics

    def calculate_loss_single(self, batch):
        users, tokens, candidates = batch
        x, loss = self.model(tokens, candidates=candidates, mode="train")  # scores, loss
        return loss

    def calculate_metrics_single(self, batch, mode):
        users, seqs, labels, lengths = batch
        scores = self.model(seqs, length=lengths, mode="all")  # B x T x C
        res = self.ranker(scores, labels, users=users)
        metrics = {}
        for i, k in enumerate(self.args.metric_ks):
            metrics["NDCG@%d" % k] = res[2*i]
            metrics["Recall@%d" % k] = res[2*i+1]
        metrics["MRR"] = res[-3]
        metrics["AUC"] = res[-2]
        metrics["loss"] = res[-1]
        return metrics

    def add_extra_loggers(self):
        pass

    def log_extra_train_info(self, log_data):
        pass

    def log_extra_val_info(self, log_data):
        pass


class SASRecSampleTrainer(AbstractTrainer):
    def __init__(self, args, model, train_loader, val_loader, test_loader, ckpt_root, user2seq):
        super().__init__(args, model, train_loader, val_loader, test_loader, ckpt_root, user2seq)
        if self.is_parallel:
            self.ce_train = DataParallelCriterion(nn.CrossEntropyLoss())
            self.ce_test = nn.CrossEntropyLoss()
            self.ranker = DataParallelMetric(SampleRanker(self.metric_ks, self.user2seq))
        else:
            self.ce = nn.CrossEntropyLoss()
            self.ranker = SampleRanker(self.metric_ks, self.user2seq)

    @classmethod
    def code(cls):
        return 'sasrec_sample'

    def calculate_loss(self, batch):
        if self.is_parallel:
            return self.calculate_loss_parallel(batch)
        else:
            return self.calculate_loss_single(batch)

    def calculate_metrics(self, batch, mode):
        if self.is_parallel:
            return self.calculate_metrics_parallel(batch, mode)
        else:
            return self.calculate_metrics_single(batch, mode)

    def calculate_loss_parallel(self, batch):
        users, tokens, candidates = batch
        x = self.model(tokens, candidates=candidates, mode="train")  # scores, loss
        device = x[0][0].device
        loss = [x_[-1].to(device) for x_ in x]
        loss = sum(loss) / len(loss)
        return loss

    def calculate_metrics_parallel(self, batch, mode):
        users, seqs, candidates, labels, length = batch
        scores = self.model(seqs, candidates=candidates, length=length, mode="sample")  # B x T x C
        scores = torch.cat(scores, 0) # concat scores from gpus
        res = self.ranker(scores)
        metrics = {}
        for i, k in enumerate(self.args.metric_ks):
            metrics["NDCG@%d" % k] = res[2*i]
            metrics["Recall@%d" % k] = res[2*i+1]
        metrics["MRR"] = res[-3]
        metrics["AUC"] = res[-2]
        metrics["loss"] = res[-1]
        return metrics

    def calculate_loss_single(self, batch):
        users, tokens, candidates = batch
        x, loss = self.model(tokens, candidates=candidates, mode="train")  # scores, loss
        return loss

    def calculate_metrics_single(self, batch, mode):
        users, seqs, candidates, labels, length = batch
        scores = self.model(seqs, candidates=candidates, length=length, mode="sample")  # B x T x C
        res = self.ranker(scores)
        metrics = {}
        for i, k in enumerate(self.args.metric_ks):
            metrics["NDCG@%d" % k] = res[2*i]
            metrics["Recall@%d" % k] = res[2*i+1]
        metrics["MRR"] = res[-3]
        metrics["AUC"] = res[-2]
        metrics["loss"] = res[-1]
        return metrics

    def add_extra_loggers(self):
        pass

    def log_extra_train_info(self, log_data):
        pass

    def log_extra_val_info(self, log_data):
        pass