import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(__file__) + os.sep + '../')
# from .. import utils
from utils import create_label_dict


class DataModel(Dataset):
    def __init__(self, opt, case='train'):
        self.token = np.load(os.path.join(opt.data_dir, "{}_{}.npy".format(case, 'token')), allow_pickle=True)
        self.token_id = np.load(os.path.join(opt.data_dir, "{}_{}.npy".format(case, 'token_id')), allow_pickle=True)
        self.label = np.load(os.path.join(opt.data_dir, "{}_{}.npy".format(case, 'label')), allow_pickle=True)
        self.label_id = np.load(os.path.join(opt.data_dir, "{}_{}.npy".format(case, 'label_id')), allow_pickle=True)
        self.length = np.load(os.path.join(opt.data_dir, "{}_{}.npy".format(case, 'length')), allow_pickle=True)

    def __getitem__(self, idx):
        return self.token[idx], self.token_id[idx], self.label[idx], self.label_id[idx], self.length[idx]

    def __len__(self):
        return len(self.token)
        # return 200
def collate_fn(data):
    token, token_id, label, label_id, length = zip(*data)
    max_length = max(length) + 2 # [CLS] and [SEP]
    token_id_pad = []
    label_id_pad = []
    for idx in range(len(token_id)):
        pad_num = max_length - len(token_id[idx])
        tid = token_id[idx] + [0] * pad_num
        lid = label_id[idx].tolist() + [0] * pad_num
        token_id_pad.append(tid)
        label_id_pad.append(lid)
    token_id_pad = torch.tensor(token_id_pad).long()
    label_id_pad = torch.tensor(label_id_pad).long()
    batch_data = {
        'token': token,
        'token_id': token_id_pad,
        'label': label,
        'label_id':label_id_pad,
        'length': length
    }
    return batch_data

class DataProcessor(object):
    def __init__(self, data_dir, label_path):
        self.data_dir = data_dir
        self.label2id = create_label_dict(label_path)
        self.id2label = {j: i for i, j in self.label2id.items()}
        self.tokenizer = AutoTokenizer.from_pretrained("chinese-roberta-wwm-ext")
        self.bad_entitys = 0
    def create_examples(self, case='train'):
        """Creates examples for the training and dev sets."""
        json_datas = []
        origin_datas = open('./dataset/{}.json'.format(case))
        for data in origin_datas:
            data = json.loads(data)
            json_datas.append(data)
        all_data = {
            'token': [],
            'token_id':[],
            'label':[],
            'label_id':[],
            'length': []
        }
        for data in tqdm(json_datas):
            return_data = self._create_single_example(data)
            for key in all_data.keys():
                all_data[key].append(return_data[key])
        for file, data in all_data.items():
            np.save(os.path.join(self.data_dir, '{}_{}.npy'.format(case, file)), data)
        print("有:{}个实体误标记".format(self.bad_entitys))
        self.bad_entitys = 0
    def _create_single_example(self, data):
        text = data['text']
        label_dict = data['label']

        token = self.tokenizer.tokenize(text)
        token_id = self.tokenizer.encode(text)
        label_id = np.zeros(len(token)).astype(int)
        have_taged_entity = set()
        for label, pos_dict in label_dict.items():
            b_id = self.label2id['B-{}'.format(label)]
            i_id = self.label2id['I-{}'.format(label)]
            poses = []
            entitys = pos_dict.keys()
            # bert切词与原始句子实体位置标记保持一致，pos直接用
            if len(text) == len(token):
                for entity in entitys:
                    poses.extend(pos_dict[entity])
            else:
                # bert切词与原始句子实体位置标记不一致，需要修正pos
                # todo： 这里有问题，比如这样的句子 【斗罗大陆是根据小说斗罗大陆改编的手游】,这里重新处理会把所有的标成一种type，使用bad_entitys记录有多少这样的数据。
                # 最后在dev集上是没有错误标记的，trian上有两个, 因此不影响评测结果
                for entity in entitys:
                    entity = "".join(self.tokenizer.tokenize(entity))
                    if entity in have_taged_entity:
                        self.bad_entitys += len(self.get_all_span(token, entity))
                    else:
                        have_taged_entity.add(entity)
                    poses.extend(self.get_all_span(token, entity))
            for pos in poses:
                label_id[pos[0]] = b_id
                label_id[pos[0]+1:pos[-1]+1] = i_id

        label = [self.id2label[idx] for idx in label_id]
        assert len(token) + 2 == len(token_id) # +2是因为多了Cls和Sep
        assert len(label_id) + 2 == len(token_id)
        return_data = {
            'token': token,
            'token_id': token_id,
            'label': label,
            'label_id':label_id,
            'length': len(token)
        }
        return  return_data

    def get_positions(self, data_list, map_str):
        """
        返回实体在单词列表中的位置
        sample:
        >> input: ['球','星','姚'，'明', ...., ], '姚明'
        >> return: (2, 3)
        """
        map_str = map_str.strip().replace(' ', '$')
        map_str =  self.tokenizer.tokenize(map_str)
        map_str = [i.replace('#', '') for i in map_str]
        map_str = ''.join(map_str)
        data_list = [i.replace('#', '') for i in data_list]
        # 如果只由一个词组成
        for word in data_list:
            if map_str.lower() in word.lower():
                start_id = end_id = data_list.index(word)
                return start_id, end_id

        start_id = -1
        end_id = -1
        for idx, word in enumerate(data_list):
            if start_id != - 1 and end_id != -1:
                return start_id, end_id
            if map_str.startswith(word):
                start_id = end_id = idx
                while end_id+1 < len(data_list) and data_list[end_id+1] in map_str:
                    if "".join(data_list[start_id:end_id+2]) == map_str:
                        # print("".join(data_list[start_id:end_id+3]))
                        return start_id, end_id+1
                    end_id += 1
                find_str = ""
                for idx in range(start_id, end_id+1):
                    find_str = find_str + data_list[idx]
                if find_str != map_str:
                    pre_extend = (data_list[start_id-1] if start_id > 0 else "") + find_str
                    last_extend = find_str + (data_list[end_id+1] if end_id < len(data_list)-1 else "")
                    pre_last_extend = (data_list[start_id-1] if start_id > 0 else "")+ find_str + (data_list[end_id+1] if end_id < len(data_list)-1 else "")
                    if map_str in pre_extend:
                        start_id -= 1
                    elif map_str in last_extend:
                        end_id += 1
                    elif map_str in pre_last_extend:
                        start_id -= 1
                        end_id += 1
                    else:
                        start_id = -1
                        end_id = -1
        if start_id != -1 and end_id != -1:
            return start_id, end_id
        for idx, word in enumerate(data_list[:-1]):
            if map_str in (word+data_list[idx+1]):
                return idx,idx+1
        # print("word_list{}  map_str {} loss".format(data_list, map_str))
        return start_id, end_id

    def get_all_span(self, data_list, entity):
        tem_data_list = data_list[:]
        res_list = []
        while True:
            sid, eid = self.get_positions(tem_data_list, entity)
            if sid == -1 and eid == -1:
                break
            res_list.append([sid, eid])
            for idx in range(sid, eid + 1):
                tem_data_list[idx] = '~'
        return res_list

if __name__ == '__main__':
    data_processor = DataProcessor(data_dir='dataset/',
                                  label_path='dataset/tool_data/label.txt')
    data_processor.create_examples('train')
    data_processor.create_examples('test')
