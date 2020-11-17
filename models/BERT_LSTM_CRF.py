import torch
import torch.nn as nn
from transformers import AutoModel
from .crf import CRF, get_crf_constraint
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from .BasicModule import BasicModule
# from torchcrf import CRF

class BERT_LSTM_CRF(BasicModule):
    def __init__(self, opt):
        super().__init__()
        self.model_name = 'Bert_Lstm_Crf'
        self.opt = opt
        self.bert = AutoModel.from_pretrained(opt.roberta_model_path)
        # for p in self.parameters():
        #     p.requires_grad = False
        self.lstm = nn.LSTM(opt.input_features, opt.hidden_features // 2,
                            num_layers=1, bidirectional=True)
        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(opt.hidden_features, opt.tag_num)
        start_tags, constraints = get_crf_constraint(opt.tags)
        self.crf = CRF(opt.tag_num, constraints=constraints, start_tags=start_tags)
        # self.crf = CRF(opt.tag_num, batch_first=True)

    def forward(self, sample, train=True):
        features, encoder_padding_mask = self.extract_features(sample)
        logits = features[1:-1, :, :]
        encoder_padding_mask = encoder_padding_mask[1:-1, :]
        if train:
            loss = -self.crf(logits, sample['label_id'].transpose(0, 1), encoder_padding_mask)
            print(loss.size())
            pred = self.crf.decode(logits, encoder_padding_mask[:, 1:-1])
            return loss, pred
        else:
            return self.crf.decode(logits, encoder_padding_mask)
        # return self.nll_loss(features[1:-1,:,], sample['label_id'], encoder_padding_mask[:-2,:], 'sum')

    def extract_features(self, sample):
        x = sample['token_id']
        length = torch.tensor(sample['length']).long() + 2
        encoder_padding_mask = x.eq(self.opt.padding_idx).logical_not_()
        x = self.bert(x, attention_mask=encoder_padding_mask)[0]
        x = pack_padded_sequence(x, length, True, False)
        x, _ = self.lstm(x)
        x, length = pad_packed_sequence(x)
        x = self.hidden2tag(x)
        return x, encoder_padding_mask.transpose(0, 1)

    def decode(self, sample):
        features, encoder_padding_mask = self.extract_features(sample)
        return self.crf.decode(features, mask=encoder_padding_mask)

    # def nll_loss(self, features, tags, encoder_padding_mask, reduction):
    #     return -self.crf(features, tags.transpose(0, 1), encoder_padding_mask, reduction=reduction)
