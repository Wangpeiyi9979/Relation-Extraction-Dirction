from .BaseConfig import BaseConfig1
class BertEntityConfig(BaseConfig1):
    """
    这里参数：格式
    model = 'model'
    """
    model = 'BertEntity'
    roberta_model_path = './bert-base-uncased'
    train_batch_size = 8
    input_feature = 768
    lr = 3e-5

