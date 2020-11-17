#encoding:utf-8
"""
@Time: 2020/3/4 16:35
@Author: Wang Peiyi
@Site : 
@File : bert.py
"""
from overrides import overrides

import numpy as np
import torch
import torch.nn as nn
from pytorch_pretrained_bert.modeling import BertModel
from pytorch_pretrained_bert.tokenization import BertTokenizer

class My_Bert_Tokenizer(BertTokenizer):

    def __init__(self, *args, **kwargs):
        super(My_Bert_Tokenizer, self).__init__(*args, **kwargs)

    @overrides
    def tokenize(self, tokens, split=False):
        """
        @param tokens: 一个token列表，一般是原始句子通过分词器分开的，比如以空格分隔，这个tokens和glove等词向量的输入共享
        @param split: 是否用Bert原始的切割token，比如会把trainyou->train, ##you, 估计是原作者为了处理一些书写上的连词错误
        @return:
            token_id(n):numpy: 原始tokens转换后的bert 词表id，添加了CLS和SEP两个词的id
            gather_indexs(n, max_length):numpy/None: 每一行代表原始一个单词在token_id中所占的位置
        """
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        if not split:
            split_tokens = [t if t in self.vocab else '[UNK]' for t in tokens]
            gather_indexes = None
        else:
            split_tokens, _gather_indexes = [], []
            for token in tokens:
                indexes = []
                for i, sub_token in enumerate(self.wordpiece_tokenizer.tokenize(token)):
                    indexes.append(len(split_tokens))
                    split_tokens.append(sub_token)
                _gather_indexes.append(indexes)

            max_index_list_len = max(len(indexes) for indexes in _gather_indexes)
            gather_indexes = np.zeros((len(_gather_indexes), max_index_list_len))
            for i, indexes in enumerate(_gather_indexes):
                for j, index in enumerate(indexes):
                    gather_indexes[i, j] = index
        token_ids = np.array(self.convert_tokens_to_ids(split_tokens))
        return token_ids, gather_indexes


class My_Bert_Encoder(BertModel):
    """
    对Bert Model进行了再封装，应为Bert切词可能将一个单词切为多个，
    比如himself -> him, ##self, 按照原始的bert，对himself会输出两个向量
    这里根据是否给出token_subword_index来合并这两个单词
    """
    def __init__(self, config):
        super(My_Bert_Encoder, self).__init__(config)

    def forward(self, input_ids,
                token_type_ids=None,
                attention_mask=None,
                output_all_encoded_layers=True,
                token_subword_index=None):
        """
        :param input_ids: same as it in BertModel
        :param token_type_ids: same as it in BertModel
        :param attention_mask: same as it in BertModel
        :param output_all_encoded_layers: same as it in BertModel
        :param token_subword_index: [batch_size, num_tokens, num_subwords]
        :return:
        """
        # encoded_layers: [batch_size, num_subword_pieces, hidden_size]

        encoded_layers, pooled_output = super(My_Bert_Encoder, self).forward(
            input_ids, token_type_ids, attention_mask, output_all_encoded_layers)

        if token_subword_index is None:
            return encoded_layers, pooled_output
        else:
            if output_all_encoded_layers is False:
                return self.average_pooling(encoded_layers, token_subword_index), pooled_output
            else:
                average_pools = []
                for encoded_layer in encoded_layers:
                    average_pool = self.average_pooling(encoded_layer, token_subword_index)
                    average_pools.append(average_pool)
                return average_pools, pooled_output

    def average_pooling(self, encoded_layers, token_subword_index):

        batch_size, num_tokens, num_subwords = token_subword_index.size()
        batch_index = torch.arange(batch_size).view(-1, 1, 1).type_as(token_subword_index)
        token_index = torch.arange(num_tokens).view(1, -1, 1).type_as(token_subword_index)
        _, num_total_subwords, hidden_size = encoded_layers.size()
        expanded_encoded_layers = encoded_layers.unsqueeze(1).expand(
            batch_size, num_tokens, num_total_subwords, hidden_size)
        # [batch_size, num_tokens, num_subwords, hidden_size]
        token_reprs = expanded_encoded_layers[batch_index, token_index, token_subword_index]
        subword_pad_mask = token_subword_index.eq(0).unsqueeze(3).expand(
            batch_size, num_tokens, num_subwords, hidden_size)
        token_reprs.masked_fill_(subword_pad_mask, 0)
        # [batch_size, num_tokens, hidden_size]
        sum_token_reprs = torch.sum(token_reprs, dim=2)
        # [batch_size, num_tokens]
        num_valid_subwords = token_subword_index.ne(0).sum(dim=2)
        pad_mask = num_valid_subwords.eq(0).long()
        # Add ones to arrays where there is no valid subword.
        divisor = (num_valid_subwords + pad_mask).unsqueeze(2).type_as(sum_token_reprs)
        # [batch_size, num_tokens, hidden_size]
        avg_token_reprs = sum_token_reprs / divisor
        return avg_token_reprs

class Bert_Embedder(nn.Module):
    def __init__(self, vocab_dir :str, bert_model_dir:str, output_all_encoder_layers=False, split=False, use_gpu=True):
        """
        @param vocab_dir: bert的词表地址
        @param bert_model_dir: bert预训练模型地址
        @param output_all_encoded_layers: False: 只输出最后一层
        @param split: 是否对输入进来的单词列表中的单词再次进行bert切分。 比如：
            如果再切分，那么输入['trainyour', 'model'] -> ['train' ,'##you','##r', 'model']，
            并且最后trainyour的词向量相当于'train', '##you', '##r'3个词向量的平均
            如果不再切分，那么trainyour将会被看做[UNK]处理
        @use_gpu
        """
        super(Bert_Embedder, self).__init__()
        self.output_all_encoder_layers = output_all_encoder_layers
        self.split = split
        self.use_gpu = use_gpu
        self.tokenizer = My_Bert_Tokenizer.from_pretrained(vocab_dir)
        self.bert_encoder = My_Bert_Encoder.from_pretrained(bert_model_dir)
        print("InFo: bert embeder构建完成")
    def forward(self, tokens_lists_no_cls_sep, token_type_ids=None):
        """
        @param tokens_lists_no_cls_sep: tokens(n*list:str): n个句子, 输入的相当于一个batch, 长度不需要相等, 为了和其他词向量使用保持一致，
                                                 输入句子不需要有[CLS]和[SEP], 并且输入句子不应该进行PADDING
                attention_mask自动根据输入padding构建，token type以SEP为分隔自动构建

               token_type_ids: 当希望不连续地赋值segment_id为1时给定
        @return:
            embeddings:
                split为False：将不在bert词表中的词替换为，[UNK], 首尾添加了Bert特有的[CLS]和[SEP]，然后转换为id，送入bert
                    输出：Tensor(n, max_length+2, word_dim): n个句子中token的embedding, tokenize的的时候先添加了[CLS]和[SEP]，
                     注意，原始bert输出的词向量，padding的词的输出词向量不为0
                split为True: 将不再bert词表中的词切分为更小的词单元
                    输出：Tensor(n, max_length+2, word_dim):n个句子中token的embedding, L长的句子中，其embedding只有前L个向量不为0，
                    tokenize的的时候先添加了[CLS]和[SEP]
            pooled_out: Tensor(n, hidden_size): 每个句子最后一层encoder的第一个词[CLS]经过Linear层和激活函数Tanh()后的Tensor. 其代表了句子信息
        """
        tokens_id_lists_with_cls_sep = []
        tokens_subword_index_lists = []
        for tokens in tokens_lists_no_cls_sep:
            tokens_id, tokens_subword_index = self.tokenizer.tokenize(tokens, split=self.split)
            tokens_id_lists_with_cls_sep.append(tokens_id.tolist())
            tokens_subword_index_lists.append(tokens_subword_index)

        max_len = max(map(lambda x: len(x), tokens_id_lists_with_cls_sep))
        tokens_id_padding_lists_with_cls_sep = list(
            map(lambda x: x + [self.tokenizer.vocab['[PAD]']] * (max_len - len(x)), tokens_id_lists_with_cls_sep))

        tokens_id_padding_lists_with_cls_sep = np.array(tokens_id_padding_lists_with_cls_sep)
        tokens_id_padding_lists_with_cls_sep = torch.LongTensor(tokens_id_padding_lists_with_cls_sep)

        if self.split is True:
            max_token_len = max(map(lambda x: len(x[0]), tokens_subword_index_lists))
            rel_max_len = max(map(lambda x: len(x), tokens_lists_no_cls_sep)) + 2 # 这里需要添加上CLS和SEP的长度
            for idx, tokens_subword_index in enumerate(tokens_subword_index_lists):
                padding_token_num = rel_max_len - len(tokens_subword_index)
                padding_index_num = max_token_len - len(tokens_subword_index[0])
                padding_token = np.zeros([padding_token_num, len(tokens_subword_index[0])])
                tokens_subword_index = np.concatenate([tokens_subword_index, padding_token], 0)
                padding_subword_index = np.zeros([rel_max_len, padding_index_num])
                tokens_subword_index = np.concatenate([tokens_subword_index, padding_subword_index], 1)
                tokens_subword_index_lists[idx] = tokens_subword_index

            tokens_subword_index_lists = np.array(tokens_subword_index_lists)
            tokens_subword_index_lists = torch.LongTensor(tokens_subword_index_lists)
        else:
            tokens_subword_index_lists = None
        attention_mask = tokens_id_padding_lists_with_cls_sep.ne(0)


        token_type_ids_create = []
        if token_type_ids is None:
            for token_ids in tokens_id_padding_lists_with_cls_sep:
                type_id = 0
                type_id_list = []
                for token_id in token_ids:
                    if token_id == self.tokenizer.convert_tokens_to_ids(['[SEP]']):
                        type_id = 1
                    type_id_list.append(type_id)
                token_type_ids_create.append(type_id_list)
            token_type_ids = torch.LongTensor(token_type_ids_create)

        elif self.split is True:
            for idx, token_type_id in enumerate(token_type_ids):
                type_id_list = []
                sub_word_index = tokens_subword_index_lists[idx]
                for idy, single_type_id in enumerate(token_type_id):
                    single_sub_word_index = sub_word_index[idy]
                    type_id_list.append(single_type_id)
                    for sub_word_unint in single_sub_word_index[1:]:
                        if sub_word_unint != 0:
                            type_id_list.append(single_type_id)
                        else:
                            break
                if len(type_id_list) > max_len:
                    type_id_list = type_id_list[:max_len]
                else:
                    type_id_list = type_id_list + [0] * (max_len - len(type_id_list))
                token_type_ids_create.append(type_id_list)
            token_type_ids = torch.LongTensor(token_type_ids_create)

        if self.use_gpu:
            tokens_id_padding_lists_with_cls_sep = tokens_id_padding_lists_with_cls_sep.cuda()
            attention_mask = attention_mask.cuda()
            token_type_ids = token_type_ids.cuda()
            if self.split is True:
                tokens_subword_index_lists = tokens_subword_index_lists.cuda()
        embeddings, pooled_out = self.bert_encoder(tokens_id_padding_lists_with_cls_sep, token_type_ids, attention_mask,
                                                   self.output_all_encoder_layers, tokens_subword_index_lists)
        return embeddings, pooled_out
