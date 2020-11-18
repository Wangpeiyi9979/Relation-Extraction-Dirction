# encoding:utf-8
"""
@Time: 2020/3/4 11:09
@Author: Wang Peiyi
@Site : 
@File : word2vec.py
"""
import math
import torch
import os
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import multiprocessing
import json

class Word2vec_Embedder(nn.Module):

    def __init__(self, word_file: str, word2vec_file=None, static=False, use_gpu=True, UNKNOW_TOKEN='[UNK]',
                 PADDING_TOKEN='[PAD]', npy_file=None, word_dim=None):
        """
        @param word_file: 存储具体任务单词的txt文件，每一行是一个单词
        @param word2vec_file: 原始的word2vec文件，如"word2vec.840B.300d.txt"
        @param static: 表示是否更新word2vec embedding的参数，默认更新
        @param use_gpu: 是否使用gpu
        @param UNKNOW_TOKEN: 代表UNKONW单词
        @param PADDING_TOKEN: 代表PADDING单词
        @param npy_file: 每次从word2vec中加载太慢了，第一次加载后存入npy文件，最后再读取
        @param word_dim: 当word2vec_file和npy_file都没有给定时，使用embedding_size初始化
        """
        super(Word2vec_Embedder, self).__init__()
        self.UNKNOW_TOKEN = UNKNOW_TOKEN
        self.PADDING_TOKEN = PADDING_TOKEN
        self.static = static
        self.use_gpu = use_gpu
        self.vocab_size, self.word2id, self.id2word = self._get_vocab_size_and_word2id(word_file)
        self.npy_file = npy_file

        try:
            self.word_dim, self.embedder = self._get_word_dim_and_embedder_from_npy(npy_file)
        except:
            if word2vec_file is not None:
                self.left_word_num = 0  # 用来word2vec中丢失的原始vocab单词数
                self.word_dim, self.embedder = self._get_word_dim_and_embedder_from_txt(word2vec_file)
            else:
                if word_dim is None:
                    raise EnvironmentError("word2vec_file、npy_file and word_dim are can not be all None")
                print('no pre-trained wordvec, random init')
                self.left_word_num = self.vocab_size
                self.word_dim = word_dim
                self.embedder = nn.Embedding(self.vocab_size, self.word_dim)
        self.report_info()

    def report_info(self):
        print(
            "InFo: word2vec embedder构建完成, word2vec丢失单词数:{}/{}, 是否更新word2vec embedding: {}".format(
                self.left_word_num, self.vocab_size, not self.static))

    def _get_vocab_size_and_word2id(self, word_file: str):
        """
        @param word_file: 见__init__参数word_file
        @return:
            vocab_size: int, 词典大小
            word2id: dict[str: int], 映射单词到id的字典
            id2word: dict[int: str]
        """
        word2id = {}
        with open(word_file, 'r') as f:
            words = f.readlines()
            for idx, word in enumerate(words):
                word2id[word.strip()] = idx
        id2word = {k: v for v, k in word2id.items()}
        return len(word2id), word2id, id2word

    def _parse_word2vec_lines(self, lines):
        """
        提取出word2vec的词向量, 单独写一个函数，是为了多进程读取
        @param lines:
        @return: word2vec
        """
        word2vec = {}
        for line in lines:
            word, vec = line.split(' ', 1)
            vec = vec.strip().split(' ')
            vec = np.array(list(map(lambda x: float(x), vec)))
            word2vec[word] = vec
        return word2vec

    def _get_word_dim_and_embedder_from_txt(self, word2vec_file: str):
        """
        @param word2vec_file: 见__init__参数word2vec_file
        @return:
            word_dim(int): 单词的embedding维数
            embedder:(nn.Embedding), 这里就是用word2vec_file去初始化nn.Embedding的look up table
        """

        if word2vec_file.endswith('txt'):
            with open(word2vec_file, 'r') as f:
                lines = f.readlines()

            import multiprocessing
            cpu_count = multiprocessing.cpu_count()
            p = multiprocessing.Pool(cpu_count)
            process_reses = []

            size = math.ceil(len(lines) / cpu_count)
            for index in range(cpu_count):
                start = index * size
                end = min((index + 1) * size, len(lines))
                sub_data = lines[start:end]
                process_reses.append(p.apply_async(self._parse_word2vec_lines, args=(sub_data,)))
            p.close()
            p.join()
            word2vec = {}
            for res in process_reses:
                word2vec.update(res.get())
        else:
            word2vec = {}
            word2vec_json_datas = json.load(open(word2vec_file))
            for data in word2vec_json_datas:
                word = data['word']
                vec = data['vec']
                vec = np.array(list(map(lambda x: float(x), vec)))
                word2vec[word] = vec

        word_dim = len(list(word2vec.values())[0])
        look_up_table = []
        for id in range(self.vocab_size):
            word = self.id2word[id]
            vec = word2vec.get(word, None)  # 单词如果不在word2vec中，随机初始化一个向量
            if vec is None:
                vec = np.random.randn(word_dim)
                self.left_word_num += 1
            look_up_table.append(vec)

        look_up_table = np.array(look_up_table)
        if self.npy_file is not None:
            if not os.path.exists(self.npy_file):
                np.save(self.npy_file, look_up_table)

        look_up_table = torch.from_numpy(look_up_table)
        if self.use_gpu:
            look_up_table.cuda()
        embedder = nn.Embedding(self.vocab_size, word_dim)
        embedder.weight.data.copy_(look_up_table)

        if self.static is True:
            embedder.weight.requires_grad = False

        return word_dim, embedder

    def _get_word_dim_and_embedder_from_npy(self, npy_file):
        self.left_word_num = 'x'  # 从npy中读取，不知道丢失了多少
        look_up_table = np.load(npy_file)
        look_up_table = torch.from_numpy(look_up_table)
        word_dim = len(look_up_table[0])
        if self.use_gpu:
            look_up_table.cuda()
        embedder = nn.Embedding(self.vocab_size, word_dim)
        embedder.weight.data.copy_(look_up_table)
        if self.static is True:
            embedder.weight.requires_grad = False
        return word_dim, embedder

    def forward(self, tokens_lists, max_len=None):
        """
        @param tokenss(n*list:str): n个句子, 输入的相当于一个batch
        @return:
            embeddings:Tensor(n, max_length, word_dim), 输出n个句子中token的embedding
        """
        if max_len is None:
            max_len = max(map(lambda x: len(x), tokens_lists))
        # 将句子单词装换为单词id表示
        tokens_id_lists = list(
            map(lambda x: list(map(lambda w: self.word2id.get(w, self.word2id[self.UNKNOW_TOKEN]), x)),
                tokens_lists))

        # 按最长句子进行padding
        tokens_padding_id_lists = list(
            map(lambda x: x + [self.word2id[self.PADDING_TOKEN]] * (max_len - len(x)), tokens_id_lists))
        tokens_padding_id_lists = torch.tensor(tokens_padding_id_lists).long()
        if self.use_gpu is True:
            tokens_padding_id_lists = tokens_padding_id_lists.cuda()
        embeddings = self.embedder(tokens_padding_id_lists)
        return embeddings
