
class BertEntityConfig(object):
    """
    这里参数：格式
    model = 'model'
    """
    model = 'BertEntity'
    roberta_model_path = './bert-base-uncased'
    optimizer = 'adam'
    ckpt_dir = './checkpoints'
    data_model = 'DataModel1'
    data_dir = './dataset/pcnn_processed_data'

    input_feature = 768
    tag_num = 21
    class_num = 12
    clip_grad = 10
    use_gpu = True
    gpu_id = 0
    padding_idx = 0
    continue_training = False
    load_checkpoint = None

    train_batch_size = 32
    val_batch_size = 64
    num_epochs = 10

    lr = 3e-5
    seed = 123



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
