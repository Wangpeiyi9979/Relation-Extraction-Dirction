import fire
import json
import torch
import os
import numpy as np
import torch.optim as optim

import sys
from tqdm import trange
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torch.nn as nn
from datamodels.DataModel1 import collate_fn
import datamodels
import models
import utils
import configs
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def run(**keward):
    opt = getattr(configs, keward.get('model', 'PCNN') + 'Config')()
    opt.parse(keward)
    print(opt)
    if opt.use_gpu:
        torch.cuda.set_device(opt.gpu_id)
    # setup_seed(opt.seed)

    DataModel = getattr(datamodels, opt.data_model)
    train_data = DataModel(opt, case='train')
    train_data_loader = DataLoader(train_data, opt.train_batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_data = DataModel(opt, case='val')
    val_data_loader = DataLoader(val_data, opt.val_batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)

    print("train data size:{}; val data size:{}".format(len(train_data), len(val_data)))

    checkpoint = None
    if opt.continue_training:
        checkpoint = 'checkpoints/{}_last.pt'.format(opt.model)
        checkpoint = torch.load(checkpoint, map_location='cpu')
    elif opt.load_checkpoint is not None:
        checkpoint = 'checkpoints/{}_{}.pt'.format(opt.model, opt.load_checkpoint)
        checkpoint = torch.load(checkpoint, map_location='cpu')
    opt = opt if checkpoint is None else checkpoint['opt']
    model = getattr(models, opt.model)(opt)
    if opt.use_gpu:
        model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1 / (1 + 0.95 * epoch))
    train_steps = (len(train_data) + opt.train_batch_size - 1) // opt.train_batch_size
    val_steps = (len(val_data) + opt.val_batch_size - 1) // opt.val_batch_size
    best_acc = 0
    if checkpoint is not None:
        model.load_state_dict(checkpoint['parameters'])
        scheduler.load_state_dict(checkpoint['optimizer'])
        best_acc = checkpoint['best_acc']

    print("start training...")
    for epoch in range(opt.num_epochs):
        print("{}; epoch:{}/{}:".format(utils.now(), epoch, opt.num_epochs))
        train(model, train_data_loader, scheduler, optimizer, train_steps, opt)
        acc = eval(model, val_data_loader, val_steps, opt)
        if best_acc < acc:
            best_acc = acc
            print('[Save]: {}_{:.3f}'.format(model.model_name, acc))
            torch.save({'opt': opt, 'parameters': model.state_dict(), 'optimizer': scheduler.state_dict(),
                        'current_acc': acc, 'best_acc': acc}, 'checkpoints/{}_best.pt'
                       .format(model.model_name))
        # torch.save({'opt': opt, 'parameters': model.state_dict(), 'optimizer': scheduler.state_dict(),
        #             'current_acc': acc, 'best_acc': best_acc}, 'checkpoints/{}_{}.pt'
        #            .format(model.model_name, epoch))
        torch.save({'opt': opt, 'parameters': model.state_dict(), 'optimizer': scheduler.state_dict(),
                    'current_acc': acc, 'best_acc': best_acc}, 'checkpoints/{}_last.pt'
                   .format(model.model_name))
        print("[Result] acc:{:.3f}%".format(acc))
        print('[Best] val acc:{:.3f}\n'.format(best_acc))


def train(model, dataLoader, scheduler, optimizer, steps, opt):
    model.train()
    lossAll = utils.RunningAverage()
    Acc = utils.RunningAverage()
    for it, data in enumerate(dataLoader):
        for key in data.keys():
            if 'num' in key and opt.use_gpu:
                data[key] = data[key].cuda()
        batch_logit, batch_pred = model(data)
        loss = model.cross_entopy_loss(batch_logit, data['num:label_id'])
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=opt.clip_grad)
        optimizer.step()
        acc = model.accuracy(batch_pred, data['num:label_id'])
        lossAll.update(loss.item())
        Acc.update(acc.item())
        sys.stdout.write(
            '[Train] step: {}/{} | loss: {:.6f} acc:{:.3f}%'.format(it + 1,
                                                                   steps,
                                                                   lossAll(),
                                                                   Acc()) + '\r')
        sys.stdout.flush()
    print()
    scheduler.step()

def eval(model, dataLoader, steps, opt):
    model.eval()
    golden_label = []
    pred_label = []
    with torch.no_grad():
        for it, data in enumerate(dataLoader):
            for key in data.keys():
                if 'num' in key and opt.use_gpu:
                    data[key] = data[key].cuda()
            batch_logit, batch_pred = model(data)
            golden_label.extend(data['num:label_id'].tolist())
            pred_label.extend(batch_pred.tolist())
            sys.stdout.write(
                '[Eval] step: {}/{}'.format(it + 1,
                                            steps)
                + '\r')
            sys.stdout.flush()
    print("")
    acc = model.accuracy(torch.tensor(pred_label), torch.tensor(golden_label))
    return acc.item()


if __name__ == '__main__':
    fire.Fire()
