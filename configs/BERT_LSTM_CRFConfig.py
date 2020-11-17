


class BERT_LSTM_CRFConfig(object):
    """
    这里参数：格式
    model = 'model'
    """
    model = 'BERT_LSTM_CRF'
    roberta_model_path = 'chinese-roberta-wwm-ext'
    tags_path = 'dataset/tool_data/label.txt'
    tags = {}
    with open(tags_path, 'r') as f:
        for i, line in enumerate(f.readlines()):
            if line != '':
                tags.update({line.strip(): i})
    input_features = 768
    tag_num = 21
    hidden_features = 768 * 2
    padding_idx = 0
    clip_grad = 10
    continue_training = False
    load_checkpoint = None

    use_gpu = True
    gpu_id = 0
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
