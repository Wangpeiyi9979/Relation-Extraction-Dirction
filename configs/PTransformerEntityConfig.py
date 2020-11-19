#encoding:utf-8
from .BaseConfig import BaseConfig1
class PTransformerEntityConfig(BaseConfig1):
    """
    这里参数：格式
    model = 'model'
    """
    model = 'PTransformerEntity'
    word2vec_txt_path = None
    npy_path = None
    pos_dim = 5
    d_model=512
    num_layers = 2
    nhead=8
    word_dim = 50
    lr = 1e-4