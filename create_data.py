#encoding:utf-8
"""
@Time: 2020/11/13 15:28
@Author: Wang Peiyi
@Site : 
@File : create_data.py
"""

# 创建label.txt
import json
import numpy
import sys
chose_id2name = json.load(open('dataset/FewRel/raw/chose_id2name.json'))
all_chose_data = []
all_labels = []
reverse_data_train_path = 'dataset/reverse_data_train.json'
reverse_data_val_path = 'dataset/reverse_data_val.json'
label_path = './tool_data/label.txt'
per_class_num = 700 / 2
import torch

def add_chose_data(data_json):
    data_all = json.load(open(data_json))
    for k in data_all:
        if k not in chose_id2name:
            continue
        label_name = chose_id2name[k]
        all_labels.append(label_name)
        all_labels.append('r-'+label_name)
        datas4label = data_all[k]
        for idx, data in enumerate(datas4label):
            data_unit = {}
            data_unit['tokens'] = data['tokens']
            h = data['h']
            t = data['t']
            label = label_name
            if idx >= len(datas4label) / 2:
                h = data['t']
                t = data['h']
                label = 'r-' + label_name
            data_unit['h'] = h
            data_unit['t'] = t
            data_unit['label'] = label
            all_chose_data.append(data_unit)

if __name__ == '__main__':
    tp = 0.8 # 训练集数量占比
    add_chose_data('dataset/FewRel/raw/train.json')
    add_chose_data('dataset/FewRel/raw/val.json')
    add_chose_data('dataset/FewRel/raw/test.json')

    index = torch.arange(per_class_num)
    train_index = index[:int(per_class_num*float(tp))]
    val_index = index[int(per_class_num*float(tp)):]
    train_index = train_index.unsqueeze(0).expand(len(all_labels), -1)
    shift = torch.arange(len(all_labels)) * per_class_num
    train_index = (train_index + shift.unsqueeze(1)).contiguous().view(-1).tolist()
    val_index = val_index.unsqueeze(0).expand(len(all_labels), -1)
    val_index = (val_index + shift.unsqueeze(1)).contiguous().view(-1).tolist()

    train_data = []
    val_data = []

    for idx , data in enumerate(all_chose_data):
        assert (idx in train_index and idx in val_index) is False
        if idx in train_index:
            train_data.append(data)
        else:
            val_data.append(data)
    print(len(all_chose_data))
    print(len(train_data))
    print(len(val_data))

    with open(reverse_data_train_path, 'w') as f:
        for data in train_data:
            f.write(json.dumps(data))
            f.write('\n')
    with open(reverse_data_val_path, 'w') as f:
        for data in val_data:
            f.write(json.dumps(data))
            f.write('\n')
    with open(label_path, 'w') as f:
        for data in all_labels:
            f.write(data + '\n')
