import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import sys
from utils import create_label_dict
class PCNNDataModel(Dataset):
    def __init__(self, opt, case='train'):
        self.token = np.load(os.path.join(opt.data_dir, "{}_{}.npy".format(case, 'token')), allow_pickle=True)
        self.label = np.load(os.path.join(opt.data_dir, "{}_{}.npy".format(case, 'label')), allow_pickle=True)
        self.label_id = np.load(os.path.join(opt.data_dir, "{}_{}.npy".format(case, 'label_id')), allow_pickle=True)
        self.length = np.load(os.path.join(opt.data_dir, "{}_{}.npy".format(case, 'length')), allow_pickle=True)
        self.pos1 = np.load(os.path.join(opt.data_dir, "{}_{}.npy".format(case, 'pos2')), allow_pickle=True)
        self.pos2 = np.load(os.path.join(opt.data_dir, "{}_{}.npy".format(case, 'pos1')), allow_pickle=True)
        self.h = np.load(os.path.join(opt.data_dir, "{}_{}.npy".format(case, 'h')), allow_pickle=True)
        self.t = np.load(os.path.join(opt.data_dir, "{}_{}.npy".format(case, 't')), allow_pickle=True)
    def __getitem__(self, idx):
        return self.token[idx], self.pos1[idx], self.pos2[idx], self.length[idx], self.label[idx], self.label_id[idx], self.h[idx], self.t[idx]
    def __len__(self):
        return len(self.token)
        # return 200
def collate_fn(datas):
    batch_data = {
        'str:token': [],
        'num:pos1': [],
        'num:pos2': [],
        'str:label': [],
        'num:label_id': [],
        'num:length': [],
        'str:h':[],
        'str:t':[]
    }
    token, pos1, pos2, length, label, label_id, h, t = zip(*datas)
    batch_data['str:token'].extend(token)
    batch_data['num:pos1'].extend(pos1)
    batch_data['num:pos2'].extend(pos2)
    batch_data['str:label'].extend(label)
    batch_data['num:length'].extend(length)
    batch_data['num:label_id'].extend(label_id)
    batch_data['str:h'].extend(h)
    batch_data['str:t'].extend(t)
    for key in batch_data:
        if 'num' in key:
            batch_data[key] = torch.tensor(batch_data[key]).long()
    return batch_data

class DataProcessor(object):
    def __init__(self, data_dir, label_path, sen_max_length):
        self.data_dir = data_dir
        self.label2id = create_label_dict(label_path)
        self.id2label = {j: i for i, j in self.label2id.items()}
        self.sen_max_length = sen_max_length
    def create_examples(self, case='train'):
        """Creates examples for the training and dev sets."""
        json_datas = []
        origin_datas = open('./dataset/reverse_data_{}.json'.format(case))
        for data in origin_datas:
            data = json.loads(data)
            json_datas.append(data)
        all_data = {
            'str:token': [],
            'num:pos1': [],
            'num:pos2': [],
            'str:label': [],
            'num:label_id': [],
            'num:length': [],
            'str:h':[],
            'str:t':[]
        }
        for data in tqdm(json_datas):
            return_data = self._create_single_example(data)
            for key in all_data.keys():
                all_data[key].append(return_data[key])
        for file, data in all_data.items():
            _, file = file.split(':')
            np.save(os.path.join(self.data_dir, '{}_{}.npy'.format(case, file)), data)
    def _create_single_example(self, data):

        sen_token_list = []
        token_pos1 = np.zeros(self.sen_max_length, dtype=np.int32)
        token_pos2 = np.zeros(self.sen_max_length, dtype=np.int32)
        pos1 = data['h'][2][0][0]
        pos2 = data['t'][2][0][0]
        for idx, token in enumerate(data['tokens']):
            token = token.lower()
            sen_token_list.append(token)
            token_pos1[idx] = idx - pos1 + self.sen_max_length
            token_pos2[idx] = idx - pos2 + self.sen_max_length  # 加入max_length统一pos
        sen_length = len(sen_token_list)
        return_data = {}
        return_data['str:token'] = sen_token_list
        return_data['num:length'] = sen_length
        return_data['num:pos1'] = token_pos1
        return_data['num:pos2'] = token_pos2
        return_data['str:label'] = data['label']
        return_data['num:label_id'] = self.label2id[data['label']]
        return_data['str:h'] = data['h'][0]
        return_data['str:t'] = data['t'][0]
        return return_data

if __name__ == '__main__':
    data_processor = DataProcessor(data_dir='dataset/pcnn_processed_data',
                                   label_path='tool_data/label.txt',
                                   sen_max_length=128)
    data_processor.create_examples('train')
    data_processor.create_examples('val')
