#encoding:utf-8
from .BaseConfig import BaseConfig1
class PCNNEntityConfig(BaseConfig1):
    """
    这里参数：格式
    model = 'model'
    """
    model = 'PCNNEntity'
    pos_dim = 5
    filter_num = 300
    filters = [3,5,7]
    word_dim = 50

