import torch
import torch.nn as nn
from transformers import AutoModel
from .BasicModule import BasicModule
import torch.nn.functional as F

class BertEntity(BasicModule):
    def __init__(self, opt):
        super(BertEntity, self).__init__()
        self.model_name = 'BertEntity'
        self.opt = opt
        self.bert = AutoModel.from_pretrained(opt.roberta_model_path)
        self.linear = nn.Linear(3*opt.input_feature, opt.class_num)

    def forward(self, inputs):
        x = inputs['num:bert_token_id']
        encoder_padding_mask = x.eq(self.opt.padding_idx).logical_not_()
        x, cls = self.bert(x, attention_mask=encoder_padding_mask)
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
        feature = torch.cat([cls, head_feature, tail_feature], -1)
        logit = self.linear(feature)
        _, pred = torch.max(logit, 1)
        return logit, pred
