#encoding:utf-8
from .BaseConfig import BaseConfig1
class PCNNConfig(BaseConfig1):
    """
    这里参数：格式
    model = 'model'
    """
    model = 'PCNN'
    pos_dim = 5
    filter_num = 300
    filters = [3]
    word_dim = 50

