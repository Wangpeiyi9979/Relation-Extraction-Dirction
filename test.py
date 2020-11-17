#encoding:utf-8
"""
@Time: 2020/11/13 14:00
@Author: Wang Peiyi
@Site : 
@File : val.py
"""


import argparse
import sys

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
from texttable import Texttable
from sklearn.metrics import confusion_matrix

import datamodels
import metric
import models
import utils
from datamodels.PCNNDataModel import collate_fn
from configs import PCNNConfig

def cut_name(data_list, max_length=5):
    res = []
    for i in data_list:
        if len(i) > max_length:
            i = i[:max_length]
        res.append(i)
    return res
id2label = utils.create_label_dict('./tool_data/label.txt', reverse=True)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='PCNN')
    args = parser.parse_args()
    args.path = 'checkpoints/{}_best.pt'.format(args.model)
    checkpoint = torch.load(args.path, map_location='cpu')
    try:
        opt = checkpoint['opt']
    except KeyError:
        opt = PCNNConfig()
    model: nn.Module = getattr(models, opt.model)(opt)
    try:
        model.load_state_dict(checkpoint['parameters'])
    except KeyError:
        model.load_state_dict(checkpoint)
    print("record acc: {}".format(checkpoint['best_acc']))
    if opt.use_gpu:
        model.cuda()
    DataModel = getattr(datamodels, opt.data_model)
    val_data = DataModel(opt, case='val')
    val_data_loader = DataLoader(val_data, opt.val_batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)
    model.eval()
    golden_label = []
    pred_label = []
    output_file = open('./badcase/{}_badcase.txt'.format(args.model), 'w')
    with torch.no_grad():
        for idy, data in enumerate(val_data_loader):
            for key in data.keys():
                if 'num' in key and opt.use_gpu:
                    data[key] = data[key].cuda()
            _, batch_pred = model(data)
            golden_label.extend(data['num:label_id'].tolist())
            pred_label.extend(batch_pred.tolist())
            for idx, label in enumerate(zip(data['str:label'], batch_pred)):
                label, pred = label
                pred = id2label[pred.item()]
                if label != pred:
                    output_file.write(" ".join(data['str:token'][idx]) + '\n')
                    output_file.write('[h]: {}; [t]: {}\n'.format(data['str:h'][idx], data['str:t'][idx]))
                    output_file.write("[Pred]: {}\n".format(pred))
                    output_file.write("[Gold]: {}\n\n".format(label))
            sys.stdout.flush()
    acc = model.accuracy(torch.tensor(pred_label), torch.tensor(golden_label))
    print(acc.item())
    output_file.write('acc: {}\n'.format(acc))
    golden_label = [id2label[idx] for idx in golden_label]
    pred_label = [id2label[idx] for idx in pred_label]
    labels = list(id2label.values())
    cf_matrix = confusion_matrix(golden_label, pred_label, labels=labels)
    table = Texttable()
    table.add_row([" "] + [i[:4] for i in labels])
    table.set_max_width(2000)
    for idx, r in enumerate(cf_matrix):
        table.add_row([labels[idx][:6]] + [str(i) for i in cf_matrix[idx]])
    output_file.write(table.draw())
    output_file.close()
