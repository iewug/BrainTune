import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from typing import Optional
from einops import repeat, rearrange
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from gnn.graphsage import GraphSage
    

class BrainTune(nn.Module):
    '''
    Three-layer contrastive learning.
    Shallow layer: same subject, same experiment. Middle layer: same subject, mixed experiments. Deep layer: same task.
    '''
    def __init__(self, neigh_mat:torch.Tensor) -> None:
        super().__init__()
        # gnn
        self.gnn = GraphSage(neigh_mat, out_dim=128, hid_dims=[5,16,32,32])

        # transformer
        # encoder1: segregates by subject
        self.encoder1 = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=512, dropout=0.3, batch_first=True) for _ in range(1)
        ])
        # encoder2: segregates by task
        self.encoder2 = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=512, dropout=0.3, batch_first=True) for _ in range(1)
        ])
        self.pos_embedding = nn.Parameter(torch.randn(10, 128)*0.1)

        # projection head
        self.proj_head_exp = nn.Sequential(
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,128)
        )
        self.proj_head_subject = nn.Sequential(
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,128)
        )
        self.proj_head_task = nn.Sequential(
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,128)
        )

    def forward(self, x:torch.Tensor, mask_bad:torch.Tensor, name:torch.Tensor, exp:torch.Tensor, task:torch.Tensor, retFeatures=False):
        '''
        Parameters
        ----------
        - x: (batch_size,time_len,node_num,feature_dim)
        - mask_bad: (batch_size,time_len,node_num), dtype=bool, True for bad channels
        - name: (batch_size,)
        - exp: (batch_size,)
        - task: (batch_size,) nan indicates no unlabeled data, which should not occur
        - seedidx: (batch_size) source (seed3, 4, or 5) (deprecated)

        Returns
        -------
        if retFeatures:
        - features_exp (batch_size,feature_dim')
        - features_subject (batch_size,feature_dim')
        - features_task (batch_size,feature_dim')
        
        else:
        - similarity_matrix_exp: (batch_size,batch_size-1) similarity of each sample in the batch with others (excluding itself)
        - labels_exp: (batch_size,batch_size-1) whether other samples (excluding itself) are positive (True) or negative (False)
        - similarity_matrix_subject: (batch_size,batch_size-1)
        - labels_subject: (batch_size,batch_size-1)
        - similarity_matrix_task: (batch_size,batch_size-1)
        - labels_task: (batch_size,batch_size-1)
        '''
        # gnn
        batch_size = x.size()[0]
        x = rearrange(x,'b t n d -> (b t) n d')
        mask_bad = rearrange(mask_bad,'b t n -> (b t) n')
        features_exp = self.gnn(x,mask_bad) # (batch_size*time_len,dim)
        features_exp = rearrange(features_exp, '(b t) d -> b t d', b=batch_size) # (batch_size,time_len,dim)

        # transformer I
        features_subject = features_exp
        features_subject += self.pos_embedding
        for layer in self.encoder1:
            features_subject = layer(features_subject) # (bs,tl,dim)

        # transformer II
        features_task = features_subject
        for layer in self.encoder2:
            features_task = layer(features_task) # (bs,tl,dim)

        # projection head & normalization
        features_exp = features_exp[:,0,:] # (batch_size,dim) only take the first feature of the time series
        features_exp = self.proj_head_exp(features_exp)
        features_exp = F.normalize(features_exp, dim=1) # (batch_size,dim)
        features_subject = features_subject.mean(dim=1) # clstoken (batch_size,dim)
        features_subject = self.proj_head_subject(features_subject)
        features_subject = F.normalize(features_subject, dim=1) # (batch_size,dim)
        features_task = features_task.mean(dim=1) # clstoken (batch_size,dim)
        features_task = self.proj_head_task(features_task)
        features_task = F.normalize(features_task, dim=1) # (batch_size,dim)
        
        if retFeatures:
            return features_exp, features_subject, features_task

        # Similarity matrix
        similarity_matrix_exp = torch.matmul(features_exp, features_exp.T) # (bs, bs) similarity_matrix[i][j] is equivalent to the dot product of feature i and feature j
        similarity_matrix_subject = torch.matmul(features_subject, features_subject.T) # (bs, bs)
        similarity_matrix_task = torch.matmul(features_task, features_task.T) # (bs, bs)
        # Positive/negative sample labels: True for positive, False for negative
        nameLabels = (name.unsqueeze(0) == name.unsqueeze(1)) # (bs,bs)
        expLabels = (exp.unsqueeze(0) == exp.unsqueeze(1)) # (bs,bs)
        # Designed for unsupervised calibration; otherwise, labels_task = (task.unsqueeze(0) == task.unsqueeze(1)) could replace the following three lines
        labels_task = (task.unsqueeze(0) == task.unsqueeze(1)).to(torch.float32) # (bs,bs)
        nan_mask = torch.isnan(task.unsqueeze(0)) | torch.isnan(task.unsqueeze(1))
        labels_task[nan_mask] = float('nan') # set relation with unlabeled data to nan
        # labels_exp = nameLabels & expLabels # same subject, same experiment
        labels_exp = torch.zeros_like(nameLabels, dtype=torch.float32)
        labels_exp[nameLabels & expLabels] = True # True for same subject, same experiment
        labels_exp[nameLabels & ~expLabels] = float('nan') # NaN for same subject, different experiment
        labels_exp[~nameLabels] = False # False for different subjects
        labels_subject = torch.zeros_like(nameLabels, dtype=torch.float32)
        labels_subject[nameLabels & expLabels] = float('nan') # NaN for same subject, same experiment
        labels_subject[nameLabels & ~expLabels] = True # True for same subject, different experiment
        labels_subject[~nameLabels] = False # False for different subjects

        # Remove diagonal elements
        mask = torch.eye(labels_exp.shape[0], dtype=torch.bool).to(x.device) # (bs,bs) identity matrix
        labels_exp = labels_exp[~mask].view(labels_exp.shape[0],-1) # (bs,bs-1)
        labels_subject = labels_subject[~mask].view(labels_subject.shape[0],-1) # (bs,bs-1)
        labels_task = labels_task[~mask].view(labels_task.shape[0],-1) # (bs,bs-1)
        similarity_matrix_exp = similarity_matrix_exp[~mask].view(similarity_matrix_exp.shape[0], -1) # (bs,bs-1)
        similarity_matrix_subject = similarity_matrix_subject[~mask].view(similarity_matrix_subject.shape[0], -1) # (bs,bs-1)
        similarity_matrix_task = similarity_matrix_task[~mask].view(similarity_matrix_task.shape[0], -1) # (bs,bs-1)

        return similarity_matrix_exp, labels_exp, similarity_matrix_subject, labels_subject, similarity_matrix_task, labels_task


def infoNCELoss(similarity_matrix: torch.Tensor, labels: torch.Tensor):
    '''
    Output version: -sum(log(exp(positive sample similarity/temperature))/sum(exp(all sample similarity/temperature)))/number of positive samples
    Supports torch.nan in incoming labels, which will be ignored during calculation; should not occur.
    Note: The values in similarity_matrix will be changed (divided by temperature).

    Parameters
    ----------
    - similarity_matrix: (batch_size, batch_size-1) similarity of each sample in the batch with others (excluding itself)
    - labels: (batch_size, batch_size-1) whether other samples (excluding itself) are positive (True) or negative (False)

    Returns
    -------
    loss: scalar
    '''
    similarity_matrix /= 0.07  # Divide by temperature
    exp_similarity = torch.exp(similarity_matrix)  # (bs, bs-1)
    # Use isnan function to mark irrelevant elements and set them to 0 to ignore
    valid_mask = ~torch.isnan(labels)  # (bs, bs-1) True for non-NaN positions
    labels = labels.clone()
    labels[~valid_mask] = 0  # Replace NaN with 0 for calculation
    posCnt = labels.sum(dim=1)  # (bs,) number of valid positive samples for each sample
    hasPos = (posCnt != 0)  # (bs,) possible that no positive sample was drawn
    numerator = exp_similarity * labels.float() # positive sample exp (bs, bs-1), negative samples are zero
    denominator = (exp_similarity * valid_mask.float()).sum(dim=1, keepdim=True)  # sum of exp of valid samples (bs,1)
    division = numerator[hasPos] / denominator[hasPos]  # (<=bs, bs-1)
    mask = labels[hasPos].bool() # (<=bs, bs-1) corresponding positive/negative sample flags for division
    losses = torch.zeros_like(division)  # (<=bs, bs-1)
    losses[mask] = -torch.log(division[mask])  # (<=bs, bs-1) -log for positive samples, negative and irrelevant samples remain 0
    losses = losses.sum(dim=1) / posCnt[hasPos]  # (<=bs,) average of valid positive samples
    return losses.mean()  # return the batch's average loss
    

def accuracy(similarity_matrix: torch.Tensor, labels: torch.Tensor):
    '''
    Contrastive learning accuracy.
    Supports torch.nan in incoming labels, which will be ignored during calculation; should not occur.

    Parameters
    ----------
    - similarity_matrix: (batch_size, batch_size-1) similarity of each sample in the batch with others (excluding itself)
    - labels: (batch_size, batch_size-1) whether other samples (excluding itself) are positive (True) or negative (False)

    Returns
    -------
    acc: scalar
    '''
    with torch.no_grad():
        # Use isnan function to mark irrelevant elements and set them to 0 to ignore
        valid_mask = ~torch.isnan(labels)  # (batch_size, batch_size-1)
        labels = labels.clone()
        labels[~valid_mask] = 0
        # Sort similarity_matrix by descending order of indices
        sorted_indices = torch.argsort(similarity_matrix, dim=1, descending=True)
        # Labels and valid mask are also sorted accordingly
        sorted_labels = torch.gather(labels, 1, sorted_indices)
        sorted_valid_mask = torch.gather(valid_mask, 1, sorted_indices)  # valid positions after sorting
        # Accumulate number of non-NaN labels
        cumulative_valid_mask = torch.cumsum(sorted_valid_mask.float(), dim=1)
        # Accumulate number of True labels
        cumulative_true_positives = torch.cumsum(sorted_labels.float(), dim=1)
        # Calculate precision at each position, only for valid positions
        precision_at_k = cumulative_true_positives / cumulative_valid_mask
        precision_at_k[torch.isnan(precision_at_k)] = 0 # NaN in precision can occur if cumulative_valid_mask starts with 0
        # Calculate average precision for each row
        ap_scores = (precision_at_k * sorted_labels.float()).sum(dim=1) / (sorted_labels).sum(dim=1)
        filtered_ap = ap_scores[~torch.isnan(ap_scores)] # possible that no positive sample was drawn
        # Return the average precision of all samples
        return filtered_ap.mean()
    

class DownTask(nn.Module):
    '''
    Adds a fully connected layer after the backbone.
    '''
    def __init__(self, neigh_mat:torch.Tensor, class_num=3) -> None:
        super().__init__()
        # gnn
        self.gnn = GraphSage(neigh_mat, out_dim=128, hid_dims=[5,16,32,32])

        # transformer
        # encoder1: segregates by subject
        self.encoder1 = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=512, dropout=0.3, batch_first=True) for _ in range(1)
        ])
        # encoder2: segregates by task
        self.encoder2 = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=512, dropout=0.3, batch_first=True) for _ in range(1)
        ])
        self.pos_embedding = nn.Parameter(torch.randn(10, 128)*0.1)

        # Additional classifier
        self.fc = nn.Sequential(
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,class_num)
        )

    def forward(self, x:torch.Tensor, mask_bad:torch.Tensor):
        '''
        Parameters
        ----------
        - x: (batch_size,time_len,node_num,feature_dim)
        - mask_bad: (batch_size,time_len,node_num), dtype=bool, True for bad channels

        Returns
        -------
        - x (batch_size, class_num)
        - feature (batch_size, feature_dim')
        '''
        batch_size = x.size()[0]
        x = rearrange(x,'b t n d -> (b t) n d')
        mask_bad = rearrange(mask_bad,'b t n -> (b t) n')
        x = self.gnn(x,mask_bad) # (batch_size*time_len,dim)
        x = rearrange(x, '(b t) d -> b t d', b=batch_size)
        x += self.pos_embedding
        for layer in self.encoder1:
            x = layer(x) # (bs,tl,dim)
        for layer in self.encoder2:
            x = layer(x) # (bs,tl,dim)
        x = x.mean(dim=1) # clstoken (batch_size,dim)
        return self.fc(x)



if __name__ == '__main__':
    ##################
    # Test BrainTune #
    ##################
    neigh_mat = torch.randint(0, 2, (38, 38)).to(0)
    model = BrainTune(neigh_mat).to(0)
    x = torch.randn((328,10,38,5)).to(0)
    name = torch.randint(0,15,(328,)).to(0)
    task = torch.randint(0,3,(328,)).to(0)
    # task = torch.randint(0,3,(328,)).float().to(0)
    # task[0] = float('nan')
    exp = torch.randint(0,3,(328,)).to(0)
    mask = torch.randint(0, 2, (328, 10, 38), dtype=torch.bool).to(0)
    similarity_matrix_exp, labels_exp, similarity_matrix_subject, labels_subject, similarity_matrix_task, labels_task = model(x,mask,name,exp,task)
    print(similarity_matrix_exp.size())
    print(labels_exp.size())
    print(similarity_matrix_subject.size())
    print(labels_subject.size())
    print(similarity_matrix_task.size())
    print(labels_task.size())
    loss = infoNCELoss(similarity_matrix_exp,labels_exp)
    print(loss)
    acc = accuracy(similarity_matrix_exp,labels_exp)
    print(acc)
    loss = infoNCELoss(similarity_matrix_subject,labels_subject)
    print(loss)
    acc = accuracy(similarity_matrix_subject,labels_subject)
    print(acc)
    loss = infoNCELoss(similarity_matrix_task,labels_task)
    print(loss)
    acc = accuracy(similarity_matrix_task,labels_task)
    print(acc)