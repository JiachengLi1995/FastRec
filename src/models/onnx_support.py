
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import math 

MAX_VAL = 1e4 # considering fp16

########
# Tril #
########


def tril_mask_onnx(inputs: torch.BoolTensor,
              diagonal: Optional[int] = 0) -> torch.FloatTensor:
    """Caveat to export an tril-based mask with ONNX.

    Args:
        inputs: Input tensor.
        diagonal: Value of diagonal.

    Returns:
        (torch.FloatTensor): Output tensor.

    """

    arange = torch.arange(inputs.size(0), device=inputs.device)
    arange2 = torch.arange(inputs.size(1), device=inputs.device)

    mask = arange.unsqueeze(-1).expand(-1, inputs.size(1)) >= (arange2 - diagonal)

    return mask

class MultiHeadAttention(nn.Module):
    "Take in model size and number of heads."
    def __init__(self, d_model, h, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h
        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, attn_mask=None):
        batch_size = query.size(0)

        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))

        if attn_mask is not None:
            # scores = scores.masked_fill(mask, -MAX_VAL) 
            # TensorRT TODO: `masked_fill` cannot be supported by TensorRT 8.2.0
            if len(attn_mask.size()) == 2:
                mask = attn_mask.unsqueeze(0).unsqueeze(0) * MAX_VAL
            scores = scores - mask

        p_attn = F.softmax(scores, dim=-1)
        if self.dropout is not None:
            p_attn = self.dropout(p_attn)

        x = torch.matmul(p_attn, value)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x), p_attn