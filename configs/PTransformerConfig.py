#encoding:utf-8
from .BaseConfig import BaseConfig1
class PTransformerConfig(BaseConfig1):
    """
    这里参数：格式
    model = 'model'
    """
    model = 'PTransformer'
    word2vec_txt_path = None
    npy_path = None
    d_model=300
    num_layers = 2
    nhead=5
    tout_dim = 200
    word_dim = 50
    lr = 1e-4
