#encoding:utf-8
from .BaseConfig import BaseConfig1
class PLSTMConfig(BaseConfig1):
    """
    这里参数：格式
    model = 'model'
    """
    model = 'PLSTM'
    pos_dim = 5
    lstm_dout=512
    num_layers=2
    word_dim = 50
