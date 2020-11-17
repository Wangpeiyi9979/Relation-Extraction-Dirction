# -*- coding: utf-8 -*-

import torch
import time
from torch.autograd import Variable
import numpy as np
import torch.nn as nn


class BasicModule(torch.nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name=str(type(self))  # model name
        self.cost = nn.CrossEntropyLoss()

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            self.init_linear(m)
        elif classname.find('LSTM') != -1:
            self.init_lstm(m)

    def init_linear(self, input_linear):
        """
        Initialize linear transformation
        """
        bias = np.sqrt(6.0 / (input_linear.weight.size(0) + input_linear.weight.size(1)))
        nn.init.uniform_(input_linear.weight, -bias, bias)
        if input_linear.bias is not None:
            input_linear.bias.data.zero_()

    def init_lstm(self, input_lstm):
        """
        Initialize lstm
        """
        for ind in range(0, input_lstm.num_layers):
            weight = eval('input_lstm.weight_ih_l' + str(ind))
            bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
            nn.init.uniform_(weight, -bias, bias)
            weight = eval('input_lstm.weight_hh_l' + str(ind))
            bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
            nn.init.uniform_(weight, -bias, bias)
        if input_lstm.bias:
            for ind in range(0, input_lstm.num_layers):
                weight = eval('input_lstm.bias_ih_l' + str(ind))
                weight.data.zero_()
                weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1
                weight = eval('input_lstm.bias_hh_l' + str(ind))
                weight.data.zero_()
                weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1

    def init_cnn(self, input_cnn):
        n = input_cnn.in_channels
        for k in input_cnn.kernel_size:
            n *= k
        stdv = np.sqrt(6. / n)
        input_cnn.weight.data.uniform_(-stdv, stdv)
        if input_cnn.bias is not None:
            input_cnn.bias.data.uniform_(-stdv, stdv)

    def cross_entopy_loss(self, logits, label):
        '''
        logits: Logits with the size (..., class_num)
        label: Label with whatever size.
        return: [Loss] (A single value)
        '''
        N = logits.size(-1)
        return self.cost(logits.view(-1, N), label.view(-1))

    def accuracy(self, pred, label):
        '''
        pred: Prediction results with whatever size
        label: Label with whatever size
        return: [Accuracy] (A single value)
        '''
        return torch.mean((pred.view(-1) == label.view(-1)).float()) * 100