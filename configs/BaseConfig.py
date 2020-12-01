#encoding:utf-8
"""
@Time: 2020/11/19 9:21
@Author: Wang Peiyi
@Site : 
@File : BaseConfig.py
"""
class BaseConfig1(object):
    optimizer = 'adam'
    ckpt_dir = './checkpoints'
    data_model = 'DataModel1'
    data_dir = './dataset/data1_processed_data'
    vocab_txt_path = './tool_data/vocab.txt'
    word2vec_txt_path = './tool_data/glove.6B.50d.json'
    class_num = 12
    clip_grad = 10
    use_gpu = True
    gpu_id = 0
    padding_idx = 0
    continue_training = False
    load_checkpoint = None
    train_batch_size = 32
    val_batch_size = 64
    num_epochs = 50
    lr = 5e-3
    weight_decay = 1e-5
    seed = 123
    dropout = 0.5
    sen_max_length = 128

    def parse(self, kwargs):
        '''
        user can update the default hyperparamter
        '''


        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise Exception('opt has No key: {}'.format(k))
            setattr(self, k, v)

    def __repr__(self):
        info = "*"*10 + 'model config' + "*" * 10 + '\n'
        info += '\n'.join(['{}: {}'.format(k, getattr(self, k)) for k in dir(self) if k[0] != '_' and k != 'parse'])
        info += '\n' + ''"*"*34
        return info
