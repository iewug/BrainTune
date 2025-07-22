'''
Pre-training Phase 2:
Trains a shared classification head using MAML to initialize parameters.
This follows the backbone network trained in pretrain_s1.py.
Run with: python pretrain_s2.py --save-dir pretrain_s2 --test-subject 1~15
Loads the model from checkpoints/<test-subject>/<pre-bb> and saves to checkpoints/<test-subject>/<save-dir>.
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
from datasets.seed345.dataset import dataset_MAML as Seed345DatasetMAML
from datasets.seed3.dataset import dataset_MAML as Seed3DatasetMAML
from torch.utils.tensorboard import SummaryWriter
import higher


##########
# Parser #
##########
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default=0, type=int, help='GPU ID to use')
parser.add_argument('--k-spt', default=32, type=int, help='Support set size') # 64 for seed3
parser.add_argument('--k-qry', default=32, type=int, help='Query set size')
parser.add_argument('--epochs', default=5, type=int, help='Number of meta-training epochs')
parser.add_argument('--save-dir', default='tmp', type=str, help='Directory to save the trained model')
parser.add_argument('--inner-lr', default=1e-2, type=float, help='Task-level (inner loop) learning rate') # 1e-3 for seed3
parser.add_argument('--outer-lr', default=1e-3, type=float, help='Meta-level (outer loop) learning rate') # 1e-4 for seed3
parser.add_argument('--update-step', default=3, type=int, help='Task-level inner update steps')
parser.add_argument('--pre-bb', default='pretrain_s1', type=str, help='Directory to load the pretrained backbone model from')
parser.add_argument('--test-subject', type=int, required=True, help='Subject ID for leave-one-out testing (1~15)')
parser.add_argument('--fakeMask', action='store_true', help='Disable artifact module')
args = parser.parse_args()


# Dataset loading
print("Loading data...")
means = np.load(f'checkpoints/{args.test_subject}/{args.pre_bb}/means.npy')
stds = np.load(f'checkpoints/{args.test_subject}/{args.pre_bb}/stds.npy')
# SEED345 dataset configuration
pretrainNameList = [[f"{i}_{j}" for i in range(1, 16) if i != args.test_subject for j in range(1, 4)]] # seed 3
pretrainNameList.append([f"{i}_{j}" for i in range(1, 16) for j in range(1, 4)]) # seed 4
pretrainNameList.append([f"{i}_{j}" for i in range(1, 17) for j in range(1, 4)]) # seed 5
datasetMAML = Seed345DatasetMAML(means=means,stds=stds,nameList=pretrainNameList,k_spt=args.k_spt,k_qry=args.k_qry,fakeMask=args.fakeMask)
# SEED3 dataset configuration (commented out, for reference)
# pretrainNameList = [f"{i}_{j}" for i in range(1, 16) if i != args.test_subject for j in range(1, 4)]
# datasetMAML = Seed3DatasetMAML(means=means,stds=stds,nameList=pretrainNameList,k_spt=args.k_spt,k_qry=args.k_qry)
neigh_matrix = load_edge(file_path='../gnn/graphInfo/seed/edge.csv',node_num=67,retmat=True)
neigh_matrix = torch.from_numpy(neigh_matrix).float().to(args.gpu)


# Logging setup
saveDir = os.path.join('checkpoints',str(args.test_subject),args.save_dir)
writer = SummaryWriter(log_dir=saveDir)
with open(f'{saveDir}/cmd.txt', 'w') as f: # Save hyperparameters
    json.dump(args.__dict__, f, indent=2)
logPath = os.path.join(saveDir,'log.txt') # Log file path


# Network, optimizer, and criterion initialization
net = DownTask(neigh_matrix).to(args.gpu)
meta_opt = optim.AdamW(net.fc.parameters(), lr=args.outer_lr, amsgrad=True) # Only optimize the fully connected layer (fc)
criterion = nn.CrossEntropyLoss()

# Load pretrained backbone weights
bestModel = torch.load(f"checkpoints/{args.test_subject}/{args.pre_bb}/model.pkl")
print(f"Loading model from {args.pre_bb} (Acc_task:{bestModel['acc_task']:.4f}, Acc_subject:{bestModel['acc_subject']:.4f}, Acc_exp:{bestModel['acc_exp']:.4f}, Epoch:{bestModel['epoch']})")
with open(logPath,'a') as f:
    f.write(f"Loading model from {args.pre_bb} (Acc_task:{bestModel['acc_task']:.4f}, Acc_subject:{bestModel['acc_subject']:.4f}, Acc_exp:{bestModel['acc_exp']:.4f}, Epoch:{bestModel['epoch']})\n")
net.load_state_dict(bestModel['state'], strict=False)
# print([name for name, param in net.named_parameters()]) # Print model parameter names
# Set requires_grad for parameters: only the 'fc' layer parameters will be updated
for name, param in net.named_parameters(): 
    if name.startswith('fc'):
        param.requires_grad = True
    else:
        param.requires_grad = False


# Best model tracking
bestModel = {
    'state': None,
    'acc': 0,
    'epoch': 0
}

# Training and validation loop
print("Start training...")
# Meta-training loop
for epoch in range(args.epochs):
    net.train()
    n_train_iter = datasetMAML.maxIter()
    epoch_accs = []
    epoch_losses = []
    datasetMAML.reset() # Reset dataset iterator for a new epoch

    for batch_idx in range(n_train_iter): # Iterate through all batches in an epoch
        de_spt, mask_spt, y_spt, de_qry, mask_qry, y_qry = datasetMAML.next()
        de_spt, mask_spt, y_spt, de_qry, mask_qry, y_qry = de_spt.to(args.gpu), mask_spt.to(args.gpu), \
            y_spt.to(args.gpu), de_qry.to(args.gpu), mask_qry.to(args.gpu), y_qry.to(args.gpu)
        task_num = de_spt.size(0)
        inner_opt = optim.SGD(net.fc.parameters(), lr=args.inner_lr) # Inner loop optimizer for fc layer

        qry_losses = []
        qry_accs = []
        meta_opt.zero_grad() # Zero gradients for meta-optimizer

        for i in range(task_num): # Iterate through all tasks in the current batch
            with higher.innerloop_ctx(
                net, inner_opt, copy_initial_weights=False # Use higher to enable differentiable inner loop optimization
            ) as (fnet, diffopt):
                # Inner loop updates
                for _ in range(args.update_step):
                    spt_logits = fnet(de_spt[i],mask_spt[i])
                    spt_loss = criterion(spt_logits, y_spt[i])
                    diffopt.step(spt_loss) # Perform one inner loop gradient step

                # Evaluate on query set after inner loop updates
                qry_logits = fnet(de_qry[i],mask_qry[i])
                qry_loss = criterion(qry_logits, y_qry[i])
                qry_losses.append(qry_loss.detach()) # Detach loss for meta-loss accumulation
                qry_acc = (qry_logits.argmax(dim=1) == y_qry[i]).sum().item() / args.k_qry
                qry_accs.append(qry_acc)
                qry_loss.backward() # Backpropagate query loss for meta-gradient

        meta_opt.step() # Perform one outer loop (meta-level) gradient step
        qry_losses = sum(qry_losses) / task_num # Average query loss across tasks
        qry_accs = 100. * sum(qry_accs) / task_num # Average query accuracy across tasks
        i = epoch + float(batch_idx) / n_train_iter
        writer.add_scalar('Loss', qry_losses, batch_idx+epoch*n_train_iter)
        writer.add_scalar('Acc', qry_accs, batch_idx+epoch*n_train_iter)
        epoch_accs.append(qry_accs)
        epoch_losses.append(qry_losses)

        if batch_idx % 4 == 0: # Print loss and accuracy for this batch
            print(f'[Epoch {i:.2f}] Train Loss: {qry_losses:.2f} | Acc: {qry_accs:.2f}')
            with open(logPath,'a') as f:
                f.write(f'[Epoch {i:.2f}] Train Loss: {qry_losses:.2f} | Acc: {qry_accs:.2f}\n')
    
    epoch_losses = sum(epoch_losses) / n_train_iter # Average epoch loss
    epoch_accs = sum(epoch_accs) / n_train_iter # Average epoch accuracy
    print(f'[AvgEpoch {epoch}] Train Loss: {epoch_losses:.2f} | Acc: {epoch_accs:.2f}')
    with open(logPath,'a') as f:
        f.write(f'[AvgEpoch {epoch}] Train Loss: {epoch_losses:.2f} | Acc: {epoch_accs:.2f}\n')
    
    # Update best model if current epoch's accuracy is higher
    if epoch_accs > bestModel['acc']:
        bestModel['acc'] = epoch_accs
        bestModel['state'] = deepcopy(net.state_dict())
        bestModel['epoch'] = epoch

# Saving the best model
savePath = os.path.join(saveDir,"model.pkl")
print(f"Saving best model (acc:{bestModel['acc']:.4f}, epoch:{bestModel['epoch']}) to {savePath}...")
torch.save(bestModel,savePath)

print("*"*80)
with open(logPath,'a') as f:
    f.write("*"*80)
    f.write('\n')