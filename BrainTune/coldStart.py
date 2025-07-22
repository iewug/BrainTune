'''
After obtaining the pre-trained backbone and classifier using pretrain_s1/2.py,
finetune the classifier with few-shot supervised data, freezing the backbone network.
python coldStart.py --test-subject 1~15 --save-dir coldStart/xx
'''
import argparse
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from model import DownTask
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from gnn.graphInfo.load_graph import load_edge
from torch import optim
from datasets.seed3.dataset import dataset_single as Seed3DatasetSingle
from datasets.mixed import MixedDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import Accuracy,ConfusionMatrix


##########
# parser #
##########
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default=0, type=int, help='gpu')
parser.add_argument('--batch-size', default=32, type=int, help='batch-size')
# finetune parameters
parser.add_argument('--epochs', default=3, type=int, help='epochs')
parser.add_argument('--val-cycle', default=1, type=int, help='val-cycle')
parser.add_argument('--save-dir', default='tmp', type=str, help='where to save model')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--scratch', action='store_true', help='whether train cls from scatch')
parser.add_argument('--pre-bb', default='pretrain_s1', type=str, help='where to load pretrained backbone model')
parser.add_argument('--pre-cls', default='pretrain_s2', type=str, help='where to load pretrained cls model')
parser.add_argument('--test-subject', required=True, type=int, help='1~15 leave one out')
args = parser.parse_args()


# dataset
print("Loading data...")
means = np.load(f'checkpoints/{args.test_subject}/{args.pre_bb}/means.npy')
stds = np.load(f'checkpoints/{args.test_subject}/{args.pre_bb}/stds.npy')
neigh_matrix = load_edge(file_path='../gnn/graphInfo/seed/edge.csv',node_num=67,retmat=True)
neigh_matrix = torch.from_numpy(neigh_matrix).float().to(args.gpu)
# train
mode = args.save_dir.split('/')[-1]
if mode[0] == '1':
    videoLen = int(mode[1])*3
    trainset = Seed3DatasetSingle(subjectName=f'{args.test_subject}_1',videoIdx=[i for i in range(videoLen)],means=means,stds=stds)
elif mode[0] == '2': 
    videoLen = int(mode[1])*3
    trainset1 = Seed3DatasetSingle(subjectName=f'{args.test_subject}_1',videoIdx=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14],means=means,stds=stds)
    trainset2 = Seed3DatasetSingle(subjectName=f'{args.test_subject}_2',videoIdx=[i for i in range(videoLen)],means=means,stds=stds)
    trainset = MixedDataset(trainset1, trainset2)
elif mode[0] == '3':
    trainset1 = Seed3DatasetSingle(subjectName=f'{args.test_subject}_1',videoIdx=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14],means=means,stds=stds)
    trainset2 = Seed3DatasetSingle(subjectName=f'{args.test_subject}_2',videoIdx=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14],means=means,stds=stds)
    trainset3 = Seed3DatasetSingle(subjectName=f'{args.test_subject}_3',videoIdx=[0,1,2],means=means,stds=stds)
    trainset12 = MixedDataset(trainset1, trainset2)
    trainset = MixedDataset(trainset12,trainset3)
# test
testset = Seed3DatasetSingle(subjectName=f'{args.test_subject}_3',videoIdx=[3,4,5,6,7,8,9,10,11,12,13,14],means=means,stds=stds)
# dataloader
trainloader = DataLoader(trainset, shuffle=True, batch_size=args.batch_size, num_workers=4, pin_memory=True)
valloader = DataLoader(testset, shuffle=False, batch_size=args.batch_size, num_workers=4, pin_memory=True)
testloader = DataLoader(testset, shuffle=False, batch_size=args.batch_size, num_workers=4, pin_memory=True)

# log
saveDir = os.path.join('checkpoints',str(args.test_subject),args.save_dir)
writer = SummaryWriter(log_dir=saveDir)
with open(f'{saveDir}/cmd.txt', 'w') as f: # save hyperparameters
    json.dump(args.__dict__, f, indent=2)
logPath = os.path.join(saveDir,'log.txt') # print log


# net & optimizer & criterion
net = DownTask(neigh_matrix).to(args.gpu)
optimizer = optim.AdamW(net.fc.parameters(), lr=args.lr, amsgrad=True) # only update fc
# optimizer = optim.AdamW(net.parameters(), lr=args.lr, amsgrad=True)
criterion = nn.CrossEntropyLoss()
# load pretrained weight
if args.scratch:
    # load backbone
    pretrainModel_bb = torch.load(f"checkpoints/{args.test_subject}/{args.pre_bb}/model.pkl")
    print(f"Loading model-{args.pre_bb} (Acc_task:{pretrainModel_bb['acc_task']}, Acc_subject:{pretrainModel_bb['acc_subject']}, Acc_exp:{pretrainModel_bb['acc_exp']}, Epoch:{pretrainModel_bb['epoch']})")
    with open(logPath,'a') as f:
        f.write(f"Loading model-{args.pre_bb} (Acc_task:{pretrainModel_bb['acc_task']}, Acc_subject:{pretrainModel_bb['acc_subject']}, Acc_exp:{pretrainModel_bb['acc_exp']}, Epoch:{pretrainModel_bb['epoch']})\n")
    net.load_state_dict(pretrainModel_bb['state'], strict=False)
else:
    # load backbone+fc
    pretrainModel_cls = torch.load(f"checkpoints/{args.test_subject}/{args.pre_cls}/model.pkl")
    print(f"Loading model-{args.pre_cls} (Acc:{pretrainModel_cls['acc']}, Epoch:{pretrainModel_cls['epoch']})")
    with open(logPath,'a') as f:
        f.write(f"Loading model-{args.pre_cls} (Acc:{pretrainModel_cls['acc']}, Epoch:{pretrainModel_cls['epoch']})\n")
    net.load_state_dict(pretrainModel_cls['state'], strict=True)

for name, param in net.named_parameters(): # only update fc
    if name.startswith('fc'):
        param.requires_grad = True
    else:
        param.requires_grad = False

# find best model
bestModel = {
    'state': None,
    'acc': 0,
    'epoch': 0
}


# metric
accMetric = Accuracy(task="multiclass", num_classes=3).to(args.gpu)
cmMetric = ConfusionMatrix(task="multiclass", num_classes=3).to(args.gpu)


# train and val
print("Start training...")
for epoch in range(args.epochs):
    # train
    net.train()
    epochloss = 0
    for data,mask,_,label,_ in trainloader:
        data = data.to(args.gpu)
        mask = mask.to(args.gpu)
        label = label.to(args.gpu)
        label_pred = net(data,mask)
        prediction = torch.max(label_pred, 1)[1] # (bs,)
        loss = criterion(label_pred,label)
        epochloss += loss.item()
        # update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # metric
        accMetric(prediction,label)
    # log
    avgSampleLoss = epochloss/len(trainloader)
    acc = accMetric.compute()
    print(f'Train Epoch: {epoch}, Acc: {acc:.4f}, Loss: {avgSampleLoss:.4f}')
    with open(logPath,'a') as f:
        f.write(f'Train Epoch: {epoch}, Acc: {acc:.4f}, Loss: {avgSampleLoss:.4f}\n')
    writer.add_scalar('Train/Loss', avgSampleLoss, epoch)
    writer.add_scalar('Train/Acc',acc,epoch)
    accMetric.reset()

    # val
    if (epoch+1) % args.val_cycle == 0:
        with torch.no_grad():
            net.eval()
            for data,mask,_,label,_ in valloader:
                data = data.to(args.gpu) # (bs,32,5)
                mask = mask.to(args.gpu) # (bs,32)
                label = label.to(args.gpu) # (bs,)
                label_pred = net(data,mask)
                prediction = torch.max(label_pred, 1)[1]
                # metric
                accMetric(prediction,label)
                cmMetric(prediction,label)
            # log     
            acc = accMetric.compute()
            cm = cmMetric.compute()
            writer.add_scalar('Val/Acc',acc,epoch)
            print(f'Val   Epoch: {epoch}, Acc: {acc:.4f}')
            print(f'Confusion Matrix:\n{cm}')
            with open(logPath,'a') as f:
                f.write(f'Val   Epoch: {epoch}, Acc: {acc:.4f}\n')
                f.write(f'Confusion Matrix:\n{cm}\n')
            accMetric.reset()
            cmMetric.reset()

            # update best model by acc
            # if acc > bestModel['acc']:
            if epoch == args.epochs - 1:
                bestModel['acc'] = acc
                bestModel['state'] = deepcopy(net.state_dict())
                bestModel['epoch'] = epoch
writer.close()

# testing best model
print(f'Testing best model (Acc:{bestModel["acc"]},Epoch:{bestModel["epoch"]})')
with open(logPath,'a') as f:
    f.write(f'Testing best model (Acc:{bestModel["acc"]},Epoch:{bestModel["epoch"]})\n')
with torch.no_grad():
    net.load_state_dict(bestModel['state'])
    net.eval()
    for data,mask,_,label,_ in testloader:
        data = data.to(args.gpu) # (bs,32,5)
        mask = mask.to(args.gpu) # (bs,32)
        label = label.to(args.gpu) # (bs,)
        label_pred = net(data,mask) # (bs,2)
        prediction = torch.max(label_pred, 1)[1]
        # metric
        accMetric(prediction,label)
        cmMetric(prediction,label)
    acc = accMetric.compute()
    cm = cmMetric.compute()
    print(f'Test, Acc: {acc:.4f}')
    print(f'Confusion Matrix:\n{cm}')
    with open(logPath,'a') as f:
        f.write(f'Test, Acc: {acc:.4f}\n')
        f.write(f'Confusion Matrix:\n{cm}\n')
    accMetric.reset()
    cmMetric.reset()

# saving best model
savePath = os.path.join(saveDir,"model.pkl")
print(f'Saving best model to {savePath}...')
torch.save(bestModel,savePath)

print("*"*80)
with open(logPath,'a') as f:
    f.write("*"*80)
    f.write('\n')


'''
0.40337+0.13124+0.164865
0.102426+0.113714+0.368811
0.429461+ 0.136922+ 0.129313

'''