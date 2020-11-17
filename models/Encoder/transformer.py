# -*- coding: utf-8 -*-
# AUTHOR: Shun Zheng
# DATE: 19-9-19


import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class PositionalEncoding(nn.Module):
    """Implement the PE function."""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        input x: B x L x d_model
        output: x: B x L x d_model(增加了postion embedding)
        """
        x = x + self.pe[:, :x.size(1)].to(device=x.device)
        return self.dropout(x)

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, d_model, nhead, num_layers, drop_out=0.1):
        super(TransformerEncoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.proj_linear_before = nn.Linear(input_dim, d_model)
        self.position_emb = PositionalEncoding(d_model, dropout=drop_out)
        self.proj_linear_after = nn.Linear(d_model, output_dim)
        self._init_weight()

    def _init_weight(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, src_key_padding_mask=None):
        """
        x : B x L x input_dim
        """
        x = self.proj_linear_before(x) # B x L x d_model
        x = self.position_emb(x)
        x = x.permute(1, 0 ,2).contiguous() # L x B x d_model
        if src_key_padding_mask is None:
            x = self.transformer_encoder(x)
        else:
            x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)  # L x B x d_model
        x = x.permute(1, 0, 2).contiguous() # B x L x d_model
        x = self.proj_linear_after(x) # B x L x out_dim
        return x
