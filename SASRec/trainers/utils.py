import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score

MAX_VAL = 1e4
THRESHOLD = 0.5

def more_metrics(scores, labels):
    nums = labels.sum(dim=-1).cpu().numpy()
    labels = labels.cpu().numpy()
    _, topk = scores.topk(50)
    topk = topk.cpu().numpy()
    scores = scores.zero_().cpu().numpy()
    for i, (j, n) in enumerate(zip(topk, nums)):
        for i_, j_ in enumerate(j):
            scores[i, j_] = 1
            if i_ >= n-1:
                break
    assert nums[0] == scores[0].sum()
    metrics = {
        'f1': f1_score(y_true=labels, y_pred=scores, average='samples'),
        'prec': precision_score(y_true=labels, y_pred=scores, average='samples'),
        'recall': recall_score(y_true=labels, y_pred=scores, average='samples')
    }
    return metrics

def recall(scores, labels, k):
    scores = scores
    labels = labels
    rank = (-scores).argsort(dim=1)
    cut = rank[:, :k]
    hit = labels.gather(1, cut)
    return (hit.sum(1).float() / torch.min(torch.Tensor([k]).to(hit.device), labels.sum(1).float())).mean().cpu().item()


def ndcg(scores, labels, k):
    scores = scores.cpu()
    labels = labels.cpu()
    rank = (-scores).argsort(dim=1)
    cut = rank[:, :k]
    hits = labels.gather(1, cut)
    position = torch.arange(2, 2+k)
    weights = 1 / torch.log2(position.float())
    dcg = (hits.float() * weights).sum(1)
    idcg = torch.Tensor([weights[:min(int(n), k)].sum() for n in labels.sum(1)])
    ndcg = dcg / idcg
    return ndcg.mean()


def recalls_and_ndcgs_for_ks(scores, labels, ks):
    metrics = {}

    scores = scores
    labels = labels
    answer_count = labels.sum(1)

    labels_float = labels.float()
    rank = (-scores).argsort(dim=1)
    cut = rank
    for k in sorted(ks, reverse=True):
       cut = cut[:, :k]
       hits = labels_float.gather(1, cut)
       metrics['Recall@%d' % k] = \
           (hits.sum(1) / torch.min(torch.Tensor([k]).to(labels.device), labels.sum(1).float())).mean().cpu().item()

       position = torch.arange(2, 2+k)
       weights = 1 / torch.log2(position.float())
       dcg = (hits * weights.to(hits.device)).sum(1)
       idcg = torch.Tensor([weights[:min(int(n), k)].sum() for n in answer_count]).to(dcg.device)
       ndcg = (dcg / idcg).mean()
       metrics['NDCG@%d' % k] = ndcg.cpu().item()

    return metrics


class Ranker(nn.Module):
    def __init__(self, metrics_ks, user2seq):
        super().__init__()
        self.ks = metrics_ks
        self.ce = nn.CrossEntropyLoss()
        self.user2seq = user2seq

    def forward(self, scores, labels, lengths=None, seqs=None, users=None):
        labels = labels.squeeze(-1)
        loss = self.ce(scores, labels)
        predicts = scores[torch.arange(scores.size(0)), labels].unsqueeze(-1) # gather perdicted values
        if seqs is not None:
            scores[torch.arange(scores.size(0)).unsqueeze(-1), seqs] = -MAX_VAL # mask the rated items
        if users is not None:
            for i in range(len(users)):
                scores[i][self.user2seq[users[i].item()]] = -MAX_VAL
        valid_length = (scores > -MAX_VAL).sum(-1).float()
        rank = (predicts < scores).sum(-1).float()
        res = []
        for k in self.ks:
            indicator = (rank < k).float()
            res.append(
                ((1 / torch.log2(rank+2)) * indicator).mean().item() # ndcg@k
            ) 
            res.append(
                indicator.mean().item() # hr@k
            )
        res.append((1 / (rank+1)).mean().item()) # MRR
        res.append((1 - (rank/valid_length)).mean().item()) # AUC
        # res.append((1 - (rank/valid_length)).mean().item()) # AUC
        return res + [loss.item()]

class SampleRanker(nn.Module):
    def __init__(self, metrics_ks, user2seq):
        super().__init__()
        self.ks = metrics_ks
        self.ce = nn.CrossEntropyLoss()
        self.user2seq = user2seq

    def forward(self, scores):
        predicts = scores[:, 0].unsqueeze(-1) # gather perdicted values
        valid_length = scores.size()[-1] - 1
        rank = (predicts < scores).sum(-1).float()
        res = []
        for k in self.ks:
            indicator = (rank < k).float()
            res.append(
                ((1 / torch.log2(rank+2)) * indicator).mean().item() # ndcg@k
            ) 
            res.append(
                indicator.mean().item() # hr@k
            )
        res.append((1 / (rank+1)).mean().item()) # MRR
        res.append((1 - (rank/valid_length)).mean().item()) # AUC
        return res + [0]
