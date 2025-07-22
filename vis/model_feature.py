'''
seed345
load dataset and pretrianed model + get features + tsne visualization
one color per subject, three experiments three color depths, three labels three shapes
'''
import argparse
import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from BrainTune.model import BrainTune
from gnn.graphInfo.load_graph import load_edge
from datasets.seed345.dataset import dataset_contrast_v2 as Seed345DatasetCont
from datasets.mixed import BalancedSampler
from torch.utils.data import DataLoader
from cuml import TSNE
import numpy as np
import matplotlib.pyplot as plt


##########
# parser #
##########
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default=0, type=int, help='gpu')
parser.add_argument('--batch-size', default=32, type=int, help='batch-size')
parser.add_argument('--idn', default='pretrain_s1', type=str, help='input directory name')
parser.add_argument('--ofn', default='out', type=str, help='output file name')
parser.add_argument('--test-subject', required=True, type=int, help='1~15 leave one out')
parser.add_argument('--load', action='store_true', help='load precalculated position')
parser.add_argument('--de', action='store_true', help='use original de')
parser.add_argument('--feat', choices=['exp','sub','tsk'], default='tsk', type=str, help='visualize experiment or subject or task features')
args = parser.parse_args()


############
# settings #
############
test_subject = args.test_subject
label_meanings = {
    0: 'happy',
    1: 'neutral',
    2: 'sad'
}


###########
# dataset #
###########
if not args.load:
    print("Loading data...")
    nameList = [[f"{i}_{j}" for i in range(1, 16) for j in range(1, 4)]] # seed 3
    nameList.append([f"{i}_{j}" for i in range(1, 16) for j in range(1, 4)]) # seed 4
    nameList.append([f"{i}_{j}" for i in range(1, 17) for j in range(1, 4)]) # seed 5
    # CHECK MEANS AND STDS!!!
    means = np.load(f'checkpoints/{args.test_subject}/pretrain_s1/means.npy')
    stds = np.load(f'checkpoints/{args.test_subject}/pretrain_s1/stds.npy')
    dset = Seed345DatasetCont(train=True,nameList=nameList,means=means,stds=stds)
    # sampler
    sampler = BalancedSampler(
            dataset=dset,
            group_indices=[2, 3, 4],  # group by (name, task, experiment) 
            samples_per_group=10
        )
    dloader = DataLoader(dset, batch_sampler=sampler)
    neigh_matrix = load_edge(file_path='../gnn/graphInfo/seed/edge.csv',node_num=67,retmat=True)
    neigh_matrix = torch.from_numpy(neigh_matrix).float().to(args.gpu)


    ################
    # get features #
    ################
    net = BrainTune(neigh_matrix).to(args.gpu)
    bestModel = torch.load(f"checkpoints/{args.test_subject}/{args.idn}/model.pkl")
    print(f"Loading model (Acc_task:{bestModel['acc_task']},Acc_subject:{bestModel['acc_subject']},Acc_exp:{bestModel['acc_exp']},Epoch:{bestModel['epoch']})")
    net.load_state_dict(bestModel['state'])
    net.eval()


    # only need the first batch
    data,mask,name,task,exp = next(iter(dloader)) # bs = name_num * exp_num * task_num * samples_per_group
    with torch.no_grad():
        data = data.to(args.gpu) # (bs,10,67,5)
        mask = mask.to(args.gpu)
        name = name.to(args.gpu)
        task = task.to(args.gpu)
        exp = exp.to(args.gpu)
        features_exp, features_subject, features_task = net(data,mask,name,exp,task,True)
    # In order to keep consistency with previous code
    if args.de:
        featureList = data[:,0,:-5,:].flatten(1).cpu().numpy()
        print(featureList.shape)
    else:
        if args.feat == 'exp':
            featureList = features_exp.cpu().numpy() # (bs,128)
        elif args.feat == 'sub':
            featureList = features_subject.cpu().numpy() # (bs,128)
        elif args.feat == 'tsk':
            featureList = features_task.cpu().numpy() # (bs,128)
    nameList = name.cpu().numpy() # (bs,)
    taskList = task.cpu().numpy() # (bs,)
    expList = exp.cpu().numpy() # (bs,)
    np.save('result/paper/name.npy',nameList)
    np.save('result/paper/task.npy',taskList)
    np.save('result/paper/exp.npy',expList)

    ########
    # TSNE #
    ########
    tsne = TSNE(n_components=2, random_state=42)
    X_embedded = tsne.fit_transform(featureList) # (bs,2)
    np.save('result/paper/pos.npy',X_embedded)

else:
    nameList = np.load('result/paper/name.npy')
    taskList = np.load('result/paper/task.npy')
    expList = np.load('result/paper/exp.npy')
    X_embedded = np.load('result/paper/pos.npy')

#################
# visualization #
#################
marker_dict = {0: '^', 1: 'o', 2: '*'}

color_map = np.array([
    [0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255],  # Basic colors
    [255, 255, 0], [0, 255, 255], [255, 0, 255], [128, 128, 128], [192, 192, 192],  # More basics
    [255, 165, 0], [0, 128, 128], [128, 0, 128], [0, 128, 255], [255, 105, 180],  # Oranges, teals, pinks
    [139, 69, 19], [0, 191, 255], [30, 144, 255], [147, 112, 219], [50, 205, 50],  # Browns, blues, violets
    [255, 20, 147], [64, 224, 208], [210, 105, 30], [255, 140, 0], [154, 205, 50],  # Diverse shades
    [0, 250, 154], [72, 209, 204], [238, 130, 238], [221, 160, 221], [176, 224, 230],  # Pastels
    [123, 104, 238], [255, 69, 0], [34, 139, 34], [255, 182, 193], [135, 206, 250],  # Strong contrasts
    [75, 0, 130], [255, 228, 181], [244, 164, 96], [255, 222, 173], [70, 130, 180],  # Warm and cold tones
    [240, 128, 128], [173, 216, 230], [244, 164, 96], [0, 206, 209], [32, 178, 170],  # Aquas
    [250, 128, 114], [255, 248, 220], [220, 20, 60], [169, 169, 169], [105, 105, 105]  # Distinct finishes
]) / 255.0  # Normalize to [0, 1] for matplotlib

colors = np.array([
    color_map[nameList[i]]
    for i in range(len(nameList))
])


fig,ax = plt.subplots(figsize=(4,3.5))
for task, marker in marker_dict.items():
    mask = (taskList == task) & (nameList != (test_subject-1))
    plt.scatter(X_embedded[mask, 0], X_embedded[mask, 1], 
                c=colors[mask], marker=marker,label=label_meanings[task],s=5,alpha=0.7,edgecolors='none')
for task, marker in marker_dict.items():
    mask = (taskList == task) & (nameList == (test_subject-1))
    plt.scatter(X_embedded[mask, 0], X_embedded[mask, 1], 
                c=colors[mask], marker=marker,label=label_meanings[task],s=20,alpha=1,edgecolors='black')

plt.tight_layout()
plt.xticks([])
plt.yticks([])

plt.savefig(f'{args.ofn}.pdf')