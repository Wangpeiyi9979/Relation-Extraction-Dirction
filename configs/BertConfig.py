
class BertConfig(object):
    """
    这里参数：格式
    model = 'model'
    """
    model = 'Bert'
    roberta_model_path = 'chinese-roberta-wwm-ext'
    input_features:int = 768
    tag_num = 21
    clip_grad: int = 10
    use_gpu: bool = True
    gpu_id: int = 0
    padding_idx = 0
    continue_training = False
    load_checkpoint = None

    train_batch_size = 8
    test_batch_size = 64
    num_epochs = 15

    lr = 3e-5
    seed = 123
    data_dir = 'dataset/'


    def parse(self, kwargs):
        '''
        user can update the default hyperparamter
        '''
        print("**************************model config **************************")
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise Exception('opt has No key: {}'.format(k))
            setattr(self, k, v)
        for k, v in self.__class__.__dict__.items():
            if not (k.startswith('__') or k == 'parse'):
                print("{}:{}".format(str(k), str(getattr(self, k))))
        print("*****************************************************************")
