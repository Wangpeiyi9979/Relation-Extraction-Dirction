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
from .Encoder.basic_encoder import Encoder

class PCNNEntity(BasicModule):
    def __init__(self, opt):
        super(PCNNEntity, self).__init__()
        self.model_name = opt.model
        self.opt = opt
        self.word_emb = Word2vec_Embedder(word_file=opt.vocab_txt_path,
                                          word2vec_file=opt.word2vec_txt_path,
                                          use_gpu=opt.use_gpu)
        self.pos1_emb = nn.Embedding(self.opt.sen_max_length * 2, opt.pos_dim)
        self.pos2_emb = nn.Embedding(self.opt.sen_max_length * 2, opt.pos_dim)
        self.dropout = nn.Dropout(opt.dropout)
        self.sen_encoder = Encoder(
            enc_method='cnn',
            filters_num=opt.filter_num,
            filters=opt.filters,
            f_dim=self.word_emb.word_dim + opt.pos_dim * 2)
        self.linear = nn.Linear(3*opt.filter_num * len(opt.filters), opt.class_num)
        self.apply(self.weights_init)

    def get_sen_feature(self, input):
        max_sen_length = input['num:length'].max().item()
        embeddings = self.word_emb(input['str:token'], max_sen_length)
        pos1_emb = self.pos1_emb(input['num:pos1'])[:, :max_sen_length, :]
        pos2_emb = self.pos2_emb(input['num:pos2'])[:, :max_sen_length, :]
        embeddings = torch.cat([embeddings, pos1_emb, pos2_emb], 2)
        tout = self.sen_encoder(embeddings, input['num:length'])
        tout = self.dropout(tout)  # B * N / Q x L x dim
        return tout

    def forward(self, inputs):
        x = self.get_sen_feature(inputs)
        sen_feature, _ = torch.max(x, 1)
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
        feature = torch.cat([sen_feature, head_feature, tail_feature], -1)

        logit = self.linear(feature)
        _, pred = torch.max(logit, 1)
        return logit, pred
