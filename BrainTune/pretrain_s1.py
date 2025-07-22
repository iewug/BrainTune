'''
Pre-training Phase 1:
Trains the backbone network using three-layer contrastive learning as a general feature extractor.
Afterward, a shared classification head needs to be trained with pretrain_s2.py to initialize parameters.
Run with: python pretrain_s1.py --save-dir pretrain_s1 --test-subject 1~15
Models will be saved in checkpoints/<test-subject>/<save-dir>.
'''
import argparse
import os
import json
import torch
import torch.nn as nn
from copy import deepcopy
from model import BrainTune, infoNCELoss, accuracy
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from gnn.graphInfo.load_graph import load_edge
from torch import optim
from datasets.seed345.dataset import dataset_contrast_v2 as Seed345DatasetCont
from datasets.seed3.dataset import dataset_contrast_v2 as Seed3DatasetCont
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np


##########
# Parser #
##########
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default=0, type=int, help='GPU ID to use')
parser.add_argument('--batch-size', default=4096, type=int, help='Batch size for training')
parser.add_argument('--epochs', default=50, type=int, help='Number of training epochs')
parser.add_argument('--val-cycle', default=1, type=int, help='Validation cycle (perform validation every N epochs)')
parser.add_argument('--test-subject', type=int, required=True, help='Subject ID for leave-one-out testing (1~15)')
parser.add_argument('--save-dir', default='tmp', type=str, help='Directory to save the trained model')
parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
parser.add_argument('--alpha', default=0.1, type=float, help='Weight for subject loss: loss = loss_exp + alpha * loss_subject + beta * loss_task')
parser.add_argument('--beta', default=1.0, type=float, help='Weight for task loss: loss = loss_exp + alpha * loss_subject + beta * loss_task')
parser.add_argument('--aug', action='store_true', help='Enable training data augmentation (seems not necessary)')
parser.add_argument('--fakeMask', action='store_true', help='Disable artifact module')
args = parser.parse_args()


# Dataset loading
print("Loading data...")
# SEED345
nameList = [[f"{i}_{j}" for i in range(1, 16) if i != args.test_subject for j in range(1, 4)]] # seed 3
nameList.append([f"{i}_{j}" for i in range(1, 16) for j in range(1, 4)]) # seed 4
nameList.append([f"{i}_{j}" for i in range(1, 17) for j in range(1, 4)]) # seed 5
trainset = Seed345DatasetCont(train=True,nameList=nameList,aug=args.aug,fakeMask=args.fakeMask)
means,stds = trainset.getMeanStd()
valset = Seed345DatasetCont(val=True,means=means,stds=stds,nameList=nameList,fakeMask=args.fakeMask)
# SEED3
# nameList_train = [f"{i}_{j}" for i in range(1, 16) if i != args.test_subject for j in range(1, 4)]
# trainset = Seed3DatasetCont(train=True,nameList=nameList_train,aug=args.aug)
# means,stds = trainset.getMeanStd()
# valset = Seed3DatasetCont(val=True,means=means,stds=stds,nameList=nameList_train)
nameList_test = [f'{args.test_subject}_{j}' for j in range(1,4)]
testset = Seed3DatasetCont(test=True,means=means,stds=stds,nameList=nameList_test,fakeMask=args.fakeMask)
trainloader = DataLoader(trainset, shuffle=True, batch_size=args.batch_size, num_workers=4, pin_memory=True, drop_last=True)
valloader = DataLoader(valset, shuffle=True, batch_size=args.batch_size, num_workers=4, pin_memory=True, drop_last=True)
testloader = DataLoader(testset, shuffle=True, batch_size=args.batch_size, num_workers=4, pin_memory=True, drop_last=True)
neigh_matrix = load_edge(file_path='../gnn/graphInfo/seed/edge.csv',node_num=67,retmat=True)
neigh_matrix = torch.from_numpy(neigh_matrix).float().to(args.gpu)


# Logging setup
saveDir = os.path.join('checkpoints',str(args.test_subject),args.save_dir)
writer = SummaryWriter(log_dir=saveDir)
with open(f'{saveDir}/cmd.txt', 'w') as f: # Save hyperparameters
    json.dump(args.__dict__, f, indent=2)
logPath = os.path.join(saveDir,'log.txt') # Log file path
np.save(f'{saveDir}/means.npy', means.data)
np.save(f'{saveDir}/stds.npy', stds.data)

# Network, optimizer, and criterion initialization
net = BrainTune(neigh_matrix).to(args.gpu)
optimizer = optim.AdamW(net.parameters(), lr=args.lr, amsgrad=True)
crit_exp = nn.CrossEntropyLoss()

# Best model tracking
bestModel = {
    'state': None,
    'acc_exp': 0,
    'acc_task': 0,
    'acc_subject': 0,
    'epoch': 0
}
# Final model tracking
finalModel = {
    'state': None,
    'acc_exp': 0,
    'acc_task': 0,
    'acc_subject': 0,
    'epoch': 0
}

# For early stopping
acclist = []
def earlyStop(accList) -> bool:
    # if max(accList) - min(accList) < 0.005:
    #     accList.clear()
    #     return True
    if max(accList) > 0.9: # If task accuracy exceeds 0.9, stop early
        accList.clear()
        return True
    else:
        accList.clear()
        return False
    

# Training and validation loop
print("Start training...")
for epoch in range(args.epochs):
    # Train
    cnt = len(trainloader)
    net.train()
    epochloss_exp = 0
    epochloss_subject = 0
    epochloss_task = 0
    acc_exp = 0
    acc_subject = 0
    acc_task = 0
    for data,mask,name,task,exp in trainloader:
        data = data.to(args.gpu)
        mask = mask.to(args.gpu)
        name = name.to(args.gpu)
        task = task.to(args.gpu)
        exp = exp.to(args.gpu)
        similarity_matrix_exp, labels_exp, similarity_matrix_subject, labels_subject, similarity_matrix_task, labels_task = net(data,mask,name,exp,task)

        loss_exp = infoNCELoss(similarity_matrix_exp,labels_exp)
        loss_subject = infoNCELoss(similarity_matrix_subject,labels_subject)
        loss_task = infoNCELoss(similarity_matrix_task,labels_task)

        # Update weights
        loss = loss_exp + loss_subject*args.alpha + loss_task*args.beta
        epochloss_exp += loss_exp.item()
        epochloss_subject += loss_subject.item()
        epochloss_task += loss_task.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate metrics
        acc_exp += accuracy(similarity_matrix_exp,labels_exp).item()
        acc_subject += accuracy(similarity_matrix_subject,labels_subject).item()
        acc_task += accuracy(similarity_matrix_task,labels_task).item()

    # Log metrics for training
    avgloss_exp = epochloss_exp/cnt
    avgloss_subject = epochloss_subject/cnt
    avgloss_task = epochloss_task/cnt
    avgacc_exp = acc_exp/cnt
    avgacc_subject = acc_subject/cnt
    avgacc_task = acc_task/cnt
    acclist.append(avgacc_task)
    print(f'Train Epoch: {epoch}, Acc_exp: {avgacc_exp:.4f}, Acc_subject: {avgacc_subject:.4f}, Acc_task: {avgacc_task:.4f}, Loss_exp: {avgloss_exp:.4f}, Loss_subject: {avgloss_subject:.4f}, Loss_task: {avgloss_task:.4f}')
    with open(logPath,'a') as f:
        f.write(f'Train Epoch: {epoch}, Acc_exp: {avgacc_exp:.4f}, Acc_subject: {avgacc_subject:.4f}, Acc_task: {avgacc_task:.4f}, Loss_exp: {avgloss_exp:.4f}, Loss_subject: {avgloss_subject:.4f}, Loss_task: {avgloss_task:.4f}\n')
    writer.add_scalar('Train/Loss_exp', avgloss_exp, epoch)
    writer.add_scalar('Train/Loss_subject', avgloss_subject, epoch)
    writer.add_scalar('Train/Loss_task', avgloss_task, epoch)
    writer.add_scalar('Train/Acc_exp',avgacc_exp,epoch)
    writer.add_scalar('Train/Acc_subject',avgacc_subject,epoch)
    writer.add_scalar('Train/Acc_task',avgacc_task,epoch)

    # Validate
    if (epoch+1) % args.val_cycle == 0:
        with torch.no_grad():
            net.eval()
            acc_exp = 0
            acc_subject = 0
            acc_task = 0
            # for data,mask,name,task,exp,seedidx in valloader:
            for data,mask,name,task,exp in valloader:
                data = data.to(args.gpu)
                mask = mask.to(args.gpu)
                name = name.to(args.gpu)
                task = task.to(args.gpu)
                exp = exp.to(args.gpu)
                # seedidx = seedidx.to(args.gpu)
                similarity_matrix_exp, labels_exp, similarity_matrix_subject, labels_subject, similarity_matrix_task, labels_task= net(data,mask,name,exp,task)
                
                # Metric calculation
                acc_exp += accuracy(similarity_matrix_exp,labels_exp).item()
                acc_subject += accuracy(similarity_matrix_subject,labels_subject).item()
                acc_task += accuracy(similarity_matrix_task,labels_task).item()

            # Log metrics for validation
            cnt = len(valloader)
            avgacc_exp = acc_exp/cnt
            avgacc_subject = acc_subject/cnt
            avgacc_task = acc_task/cnt
            print(f'Val   Epoch: {epoch}, Acc_exp: {avgacc_exp:.4f}, Acc_subject: {avgacc_subject:.4f}, Acc_task: {avgacc_task:.4f}')
            with open(logPath,'a') as f:
                f.write(f'Val   Epoch: {epoch}, Acc_exp: {avgacc_exp:.4f}, Acc_subject: {avgacc_subject:.4f}, Acc_task: {avgacc_task:.4f}\n')
            writer.add_scalar('Val/Acc_exp',avgacc_exp,epoch)
            writer.add_scalar('Val/Acc_subject',avgacc_subject,epoch)
            writer.add_scalar('Val/Acc_task',avgacc_task,epoch)

            # Update best model based on task accuracy
            if avgacc_task > bestModel['acc_task']:
                bestModel['acc_task'] = avgacc_task
                bestModel['acc_subject'] = avgacc_subject
                bestModel['acc_exp'] = avgacc_exp
                bestModel['state'] = deepcopy(net.state_dict())
                bestModel['epoch'] = epoch

        if earlyStop(acclist):
            print('Early Stop!')
            break
# Also store the final model
finalModel['acc_task'] = avgacc_task
finalModel['acc_exp'] = avgacc_exp
finalModel['acc_subject'] = avgacc_subject
finalModel['state'] = deepcopy(net.state_dict())
finalModel['epoch'] = epoch
writer.close()

# Testing the best model
print(f'Testing best model (Acc_task:{bestModel["acc_task"]:.4f}, Acc_subject:{bestModel["acc_subject"]:.4f}, Acc_exp:{bestModel["acc_exp"]:.4f}, Epoch:{bestModel["epoch"]})')
with open(logPath,'a') as f:
    f.write(f'Testing best model (Acc_task:{bestModel["acc_task"]:.4f}, Acc_subject:{bestModel["acc_subject"]:.4f}, Acc_exp:{bestModel["acc_exp"]:.4f}, Epoch:{bestModel["epoch"]})\n')
with torch.no_grad():
    net.load_state_dict(bestModel['state'])
    net.eval()
    acc_exp = 0
    acc_subject = 0
    acc_task = 0
    for data,mask,name,task,exp in testloader:
        data = data.to(args.gpu)
        mask = mask.to(args.gpu)
        name = name.to(args.gpu)
        task = task.to(args.gpu)
        exp = exp.to(args.gpu)
        similarity_matrix_exp, labels_exp, similarity_matrix_subject, labels_subject, similarity_matrix_task, labels_task= net(data,mask,name,exp,task)
        # Metric calculation
        acc_exp += accuracy(similarity_matrix_exp,labels_exp).item()
        acc_subject += accuracy(similarity_matrix_subject,labels_subject).item()
        acc_task += accuracy(similarity_matrix_task,labels_task).item()
    cnt = len(testloader)
    avgacc_exp = acc_exp/cnt
    avgacc_subject = acc_subject/cnt
    avgacc_task = acc_task/cnt
    print(f'Test, Acc_exp: {avgacc_exp:.4f}, Acc_subject: {avgacc_subject:.4f}, Acc_task: {avgacc_task:.4f}')
    with open(logPath,'a') as f:
        f.write(f'Test, Acc_exp: {avgacc_exp:.4f}, Acc_subject: {avgacc_subject:.4f}, Acc_task: {avgacc_task:.4f}\n')

# Testing the final model
print(f'Testing final model (Acc_task:{finalModel["acc_task"]:.4f}, Acc_subject: {finalModel["acc_subject"]:.4f}, Acc_exp:{finalModel["acc_exp"]:.4f}, Epoch:{finalModel["epoch"]})')
with open(logPath,'a') as f:
    f.write(f'Testing final model (Acc_task:{finalModel["acc_task"]:.4f}, Acc_subject: {finalModel["acc_subject"]:.4f}, Acc_exp:{finalModel["acc_exp"]:.4f}, Epoch:{finalModel["epoch"]})\n')
with torch.no_grad():
    net.load_state_dict(finalModel['state'])
    net.eval()
    acc_exp = 0
    acc_subject = 0
    acc_task = 0
    for data,mask,name,task,exp in testloader:
        data = data.to(args.gpu)
        mask = mask.to(args.gpu)
        name = name.to(args.gpu)
        task = task.to(args.gpu)
        exp = exp.to(args.gpu)
        similarity_matrix_exp, labels_exp, similarity_matrix_subject, labels_subject, similarity_matrix_task, labels_task = net(data,mask,name,exp,task)
        # Metric calculation
        acc_exp += accuracy(similarity_matrix_exp,labels_exp).item()
        acc_subject += accuracy(similarity_matrix_subject,labels_subject).item()
        acc_task += accuracy(similarity_matrix_task,labels_task).item()
    cnt = len(testloader)
    avgacc_exp = acc_exp/cnt
    avgacc_subject = acc_subject/cnt
    avgacc_task = acc_task/cnt
    print(f'Test, Acc_exp: {avgacc_exp:.4f}, Acc_subject: {avgacc_subject:.4f}, Acc_task: {avgacc_task:.4f}')
    with open(logPath,'a') as f:
        f.write(f'Test, Acc_exp: {avgacc_exp:.4f}, Acc_subject: {avgacc_subject:.4f}, Acc_task: {avgacc_task:.4f}\n')

# Saving best model
savePath = os.path.join(saveDir,"model.pkl")
print(f'Saving best model to {savePath}...')
torch.save(bestModel,savePath)
# Saving final model
savePath = os.path.join(saveDir,"model_final.pkl")
print(f'Saving final model to {savePath}...')
torch.save(finalModel,savePath)

print("*"*80)
with open(logPath,'a') as f:
    f.write("*"*80)
    f.write('\n')