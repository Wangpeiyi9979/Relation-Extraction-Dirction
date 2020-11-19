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
from .Encoder.CNNEncoder import CNNEncoder
class CNN(BasicModule):
    def __init__(self, opt):
        super(CNN, self).__init__()
        self.model_name = opt.model
        self.opt = opt
        self.word_emb = Word2vec_Embedder(word_file=opt.vocab_txt_path,
                                          word2vec_file=opt.word2vec_txt_path,
                                          use_gpu=opt.use_gpu)
        self.dropout = nn.Dropout(opt.dropout)
        self.sen_encoder = CNNEncoder(
            filters_num=opt.filter_num,
            filters=opt.filters,
            din=opt.word_dim)
        self.linear = nn.Linear(opt.filter_num * len(opt.filters), opt.class_num)

    def get_sen_feature(self, input):
        max_sen_length = input['num:length'].max().item()
        embeddings = self.word_emb(input['str:token'], max_sen_length)
        tout = self.sen_encoder(embeddings, input['num:length'])
        tout = self.dropout(tout)  # B * N / Q x L x dim
        return tout

    def forward(self, input):
        x = self.get_sen_feature(input)
        x, _ = torch.max(x, 1)
        logit = self.linear(x)
        _, pred = torch.max(logit, 1)
        return logit, pred
