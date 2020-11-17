#encoding:utf-8
class PCNNEntityConfig(object):
    """
    这里参数：格式
    model = 'model'
    """
    model = 'PCNNEntity'
    vocab_txt_path = './tool_data/vocab.txt'
    word2vec_txt_path = './tool_data/glove.6B.50d.json'
    data_dir = './dataset/pcnn_processed_data'
    optimizer = 'adam'
    ckpt_dir = './checkpoints'
    data_model = 'PCNNDataModel'
    continue_training = False
    load_checkpoint = None
    seed = 9979
    use_gpu = True
    weight_decay = 1e-5
    gpu_id = 3
    dropout = 0.5
    pos_dim = 5
    filter_num = 300
    class_num = 12
    clip_grad = 10
    filters = [3]
    word_dim = 50
    sen_max_length = 128
    train_batch_size = 16
    val_batch_size = 64
    lr = 1e-3
    num_epochs = 50

    def parse(self, kwargs, print_info=True):
        '''
        user can update the default hyperparamter
        '''


        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise Exception('opt has No key: {}'.format(k))
            setattr(self, k, v)
        if print_info:
            print("**************************model config **************************")
            for k, v in self.__class__.__dict__.items():
                if not (k.startswith('__') or k == 'parse'):
                    print("{}:{}".format(str(k), str(getattr(self, k))))
            print("*****************************************************************")


