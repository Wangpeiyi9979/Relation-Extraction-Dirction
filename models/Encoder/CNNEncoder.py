#encoding:utf-8
"""
@Time: 2020/11/17 17:26
@Author: Wang Peiyi
@Site : 
@File : CNNEncoder.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNEncoder(nn.Module):
    def __init__(self, filters_num, filters, din):
        super(CNNEncoder, self).__init__()
        self.convs = nn.ModuleList([nn.Conv2d(1, filters_num, (k, din), padding=(int(k / 2), 0)) for k in filters])
        self.init_model_weight()

    def init_model_weight(self):
        for conv in self.convs:
            nn.init.xavier_uniform_(conv.weight)
            nn.init.constant_(conv.bias, 0.0)

    def sequence_mask(self, sequence_length, max_len=None):
        if max_len is None:
            max_len = sequence_length.data.max()
        batch_size = sequence_length.size(0)
        seq_range = torch.arange(0, max_len)
        seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
        if sequence_length.is_cuda:
            seq_range_expand = seq_range_expand.cuda()
        seq_length_expand = (sequence_length.unsqueeze(1).expand_as(seq_range_expand))
        return seq_range_expand < seq_length_expand

    def Mask(self, inputs, sqe_len=None):
        if sqe_len is None:
            return inputs
        mask = self.sequence_mask(sqe_len, inputs.size(1))  # (B, L)
        mask = mask.unsqueeze(-1)      # (B, L, 1)
        outputs = inputs * mask.float()
        return outputs

    def forward(self, inputs, lengths=None):
        """

        Args:
            inputs: B x L x din
            lengths:

        Returns: B x L x filter_num*len(filgers)

        """
        if not lengths is None:
            inputs = self.Mask(inputs, lengths)
        x = inputs.unsqueeze(1)
        x = [conv(x).relu().squeeze(3).permute(0, 2, 1) for conv in self.convs]
        x = torch.cat(x, 2)
        return x