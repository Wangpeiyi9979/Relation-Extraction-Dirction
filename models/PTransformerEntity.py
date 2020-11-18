#encoding:utf-8
"""
@Time: 2020/11/17 12:55
@Author: Wang Peiyi
@Site :
@File : PCNN.py
"""
import torch
import torch.nn as nn
from .Embedding.word2vec import Word2vec_Embedder
from .BasicModule import BasicModule
from .Encoder.TransformerEncoder import TransformerEncoder
class PTransformerEntity(BasicModule):
    def __init__(self, opt):
        super(PTransformerEntity, self).__init__()
        self.model_name = opt.model
        self.opt = opt
        self.word_emb = Word2vec_Embedder(word_file=opt.vocab_txt_path,
                                          word2vec_file=opt.word2vec_txt_path,
                                          use_gpu=opt.use_gpu,
                                          word_dim=opt.d_model,
                                          npy_file=opt.npy_path)
        self.pos1_emb = nn.Embedding(self.opt.sen_max_length * 2, opt.d_model)
        self.pos2_emb = nn.Embedding(self.opt.sen_max_length * 2, opt.d_model)
        self.dropout = nn.Dropout(opt.dropout)
        self.sen_encoder = TransformerEncoder(d_model=opt.d_model,
                                              nhead=opt.nhead,
                                              num_layers=opt.num_layers)
        self.linear = nn.Linear(opt.d_model*3, opt.class_num)
        self.linear = nn.Linear(opt.d_model*2, opt.class_num)

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

    def get_sen_feature(self, input):
        max_sen_length = input['num:length'].max().item()
        embeddings = self.word_emb(input['str:token'], max_sen_length)
        pos1_emb = self.pos1_emb(input['num:pos1'])[:, :max_sen_length, :]
        pos2_emb = self.pos2_emb(input['num:pos2'])[:, :max_sen_length, :]
        embeddings = embeddings + pos1_emb + pos2_emb
        input_mask = self.sequence_mask(input['num:length'], max_len=max_sen_length)
        tout = self.sen_encoder(embeddings, input_mask.logical_not_())
        tout = self.dropout(tout)  # B  x L x dim
        return tout

    def forward(self, inputs):
        x = self.get_sen_feature(inputs)
        # sen_feature, _ = torch.max(x, 1)
        head_feature = []
        tail_feature = []
        head_spans = inputs['var:h_span']
        tail_spans = inputs['var:t_span']
        for idx, (head_span, tail_span) in enumerate(zip(head_spans, tail_spans)):
            head_rep = torch.mean(x[idx][head_span[0]: head_span[1]], 0)
            tail_rep = torch.mean(x[idx][tail_span[0]: tail_span[1]], 0)
            head_feature.append(head_rep)
            tail_feature.append(tail_rep)
        head_feature = torch.stack(head_feature, 0)
        tail_feature = torch.stack(tail_feature, 0)
        # feature = torch.cat([sen_feature, head_feature, tail_feature], -1)
        feature = torch.cat([head_feature, tail_feature], -1)

        logit = self.linear(feature)
        _, pred = torch.max(logit, 1)
        return logit, pred