import torch
import torch.nn as nn
from transformers import AutoModel
from .crf import CRF, get_crf_constraint
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from .BasicModule import BasicModule
from torchcrf import CRF
import torch.nn.functional as F

class Bert(BasicModule):
    def __init__(self, opt):
        super().__init__()
        self.model_name = 'Bert'
        self.opt = opt
        self.bert = AutoModel.from_pretrained(opt.roberta_model_path)
        self.hidden2tag = nn.Linear(opt.input_features, opt.tag_num)


    def forward(self, sample, train=True):
        features, encoder_padding_mask = self.extract_features(sample)
        logits = features[:,1:-1,].contiguous()
        if train:
            no_pad_logits = []
            no_pad_labels = []
            length = sample['length']
            for idx in range(len(logits)):
                no_pad_logits.append(logits[idx][:length[idx]])
                no_pad_labels.append(sample['label_id'][idx][:length[idx]])
            no_pad_logits = torch.cat(no_pad_logits, 0)
            no_pad_labels = torch.cat(no_pad_labels, 0)
            loss = F.cross_entropy(no_pad_logits.view(-1, no_pad_logits.size(-1)), no_pad_labels.view(-1))
            pred = torch.max(logits, -1)[1]
            return loss, pred
        else:
            pred = torch.max(logits, -1)[1]
            return pred

        # return self.nll_loss(features[1:-1,:,], sample['label_id'], encoder_padding_mask[:-2,:], 'sum')

    def extract_features(self, sample):
        x = sample['token_id']
        encoder_padding_mask = x.eq(self.opt.padding_idx).logical_not_()
        x = self.bert(x, attention_mask=encoder_padding_mask)[0]
        x = self.hidden2tag(x)
        return x, encoder_padding_mask

    def decode(self, sample):
        features, encoder_padding_mask = self.extract_features(sample)
        return self.crf.decode(features, mask=encoder_padding_mask)

    # def nll_loss(self, features, tags, encoder_padding_mask, reduction):
    #     return -self.crf(features, tags.transpose(0, 1), encoder_padding_mask, reduction=reduction)

