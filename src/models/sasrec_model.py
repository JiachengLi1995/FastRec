import os
import math
import numpy as np

import torch
from torch import nn as nn

from src.utils.utils import fix_random_seed_as


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs


class SASRecModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        fix_random_seed_as(args.model_init_seed)

        self.loss = nn.BCEWithLogitsLoss()

        self.num_items = args.num_items

        self.item_emb = torch.nn.Embedding(self.num_items+2, args.trm_hidden_dim, padding_idx=-1)
        self.pad_token = self.item_emb.padding_idx
        self.pos_emb = torch.nn.Embedding(args.trm_max_len, args.trm_hidden_dim) # TO IMPROVE
        self.emb_dropout = torch.nn.Dropout(p=args.trm_dropout)

        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.trm_hidden_dim, eps=1e-8)

        for _ in range(args.trm_num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.trm_hidden_dim, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(args.trm_hidden_dim,
                                                            args.trm_num_heads,
                                                            args.trm_dropout)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.trm_hidden_dim, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.trm_hidden_dim, args.trm_dropout)
            self.forward_layers.append(new_fwd_layer)
        
        # weights initialization
        self.init_weights()

    def log2feats(self, log_seqs):
        seqs = self.lookup(log_seqs)
        seqs *= self.args.trm_hidden_dim ** 0.5
        positions = torch.arange(log_seqs.shape[1]).long().unsqueeze(0).repeat([log_seqs.shape[0], 1])
        seqs = seqs + self.pos_emb(positions.to(seqs.device))
        seqs = self.emb_dropout(seqs)

        timeline_mask = (log_seqs == 0).bool()
        seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim

        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=seqs.device))

        attn_output_weights = []

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, attn_output_weight = self.attention_layers[i](Q, seqs, seqs, 
                                            attn_mask=attention_mask)
                                            # key_padding_mask=timeline_mask
                                            # need_weights=False) this arg do not work?
            
            attn_output_weights.append(attn_output_weight)
            
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *=  ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)

        return log_feats, attn_output_weights


    def forward(self, x, candidates=None, length=None, save_name=None, mode="train", users=None, need_weights=False):

        idx1, idx2 = self.select_predict_index(x) if mode == "train" else (torch.arange(x.size(0)), length.squeeze())
        log_feats, attn_weights = self.log2feats(x) # user_ids hasn't been used yet

        log_feats = log_feats[idx1, idx2]

        if mode == "serving":
            if need_weights:
                return log_feats, attn_weights
            return log_feats

        elif mode == "train":
            candidates = candidates[idx1, idx2]
            pos_seqs = candidates[:, 0]
            neg_seqs = candidates[:, -1]

            # import pdb; pdb.set_trace()

            pos_embs = self.lookup(pos_seqs)
            neg_embs = self.lookup(neg_seqs)

            pos_logits = (log_feats * pos_embs).sum(dim=-1)
            neg_logits = (log_feats * neg_embs).sum(dim=-1)

            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=pos_logits.device), torch.zeros(neg_logits.shape, device=neg_logits.device)
            loss = self.loss(pos_logits, pos_labels)
            loss += self.loss(neg_logits, neg_labels)

            return [pos_logits, neg_logits], loss

        else:
            if candidates is not None:
                log_feats = log_feats.unsqueeze(1) # x is (batch_size, 1, embed_size)
                w = self.lookup(candidates).transpose(2,1) # (batch_size, embed_size, candidates)
                logits = torch.bmm(log_feats, w).squeeze(1) # (batch_size, candidates)
            else:
                logits = self.all_predict(log_feats)
            if need_weights:
                return logits, attn_weights
            return logits
        

    def select_predict_index(self, x):
        return (x!=self.pad_token).nonzero(as_tuple=True)            

    def init_weights(self, mean=0, std=0.02, lower=-0.04, upper=0.04):
        with torch.no_grad():
            for n, p in self.named_parameters():
                if ('norm' not in n) and ('bias' not in n):
                    try:
                        torch.nn.init.xavier_uniform_(p.data)
                    except:
                        pass # just ignore those failed init layers

    def all_predict(self, log_feats):
        if self.args.emb_device_idx is None:
            w = self.item_emb.weight.transpose(1,0)
            return torch.matmul(log_feats, w)
        elif type(self.args.emb_device_idx) == str and self.args.emb_device_idx.lower() == 'cpu':
            w = self.item_emb.weight.transpose(1,0)
            return torch.matmul(log_feats.to('cpu'), w).to(self.args.device)
        else:
            res = 0
            for i, emb_device in enumerate(self.args.emb_device_idx):
                b, e = self.args.emb_device_idx[emb_device]
                x = log_feats[..., b:e].to(emb_device)
                res += torch.matmul(x, self.item_emb_list[i].weight.transpose(1,0)).to(self.args.device)
            return res

    def lookup(self, x):
        if self.args.emb_device_idx is None:
            return self.item_emb(x)
        elif type(self.args.emb_device_idx) == str and self.args.emb_device_idx.lower() == 'cpu':
            return self.item_emb(x.to('cpu')).to(self.args.device)
        else:
            res = []
            for emb_layer in self.item_emb_list:
                device = emb_layer.weight.device
                res.append(emb_layer(x.to(device)).to(self.args.device))
            return torch.cat(res, dim=-1)

    def to_device(self, device):
        if self.args.emb_device_idx is None:
            return self.to(device)
        elif self.args.emb_device_idx.lower() == 'cpu':
            temp = self.item_emb
            self.item_emb = None
            self.to(device)
            self.item_emb = temp
            print('move embedding layer to:', self.item_emb.weight.device)
            return self
        try:
            self.args.emb_device_idx = eval(self.args.emb_device_idx)
            temp = self.item_emb.weight.data.detach()
            self.item_emb = None
            self.to(device)
            self.item_emb_list = []
            for emb_device in self.args.emb_device_idx:
                b, e = self.args.emb_device_idx[emb_device]
                partial_emb = nn.Embedding.from_pretrained(temp[..., b:e], freeze=False, padding_idx=-1).to(emb_device)
                self.item_emb_list.append(partial_emb)
            self.item_emb_list = nn.ModuleList(self.item_emb_list)
            print('move embedding layer to:', self.args.emb_device_idx)
            return self
        except:
            print("ERROR: please follow this rule to set emb_device_idx: None / cpu / {'cpu':(0,16), 'cuda:0':(16,50)}")
            exit()

    def device_state_dict(self):
        if type(self.args.emb_device_idx) != dict:
            return self.state_dict()
        else:
            params = self.state_dict()
            name_from, name_to = 'item_emb_list', 'item_emb'
            temp = []
            for i in params.keys():
                j = i.split('.')
                if j[0] == name_from:
                    temp.append((int(j[1]), i, params[i].to('cpu')))
            for _, i, _ in temp:
                del params[i]
            temp = [t[-1] for t in sorted(temp)]
            params[name_to+".weight"] = torch.cat(temp, dim=-1)
            return params