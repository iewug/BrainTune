'''
Merge seed3, seed4, and seed5 datasets
'''
from pathlib import Path
import numpy as np
import h5py
import torch
from torch.utils import data
from typing import Optional, Tuple, List
from einops import rearrange
from tqdm import tqdm
from collections import defaultdict

class dataset_contrast_v2(data.Dataset):
    '''
    Multi-subject SEED345 dataset.
    Labels are happy:0, neutral:1, sad:2. Fear:3 and disgust:4 are ignored.
    Using torch.concatenate excessively leads to inefficient dataset loading.
    '''
    def __init__(self, train=False, val=False, test=False, window_size=10, means:Optional[np.ndarray]=None, stds:Optional[np.ndarray]=None, nameList:Optional[List[List[str]]]=None, aug=False, fakeMask=False):
        '''
        Parameters
        ----------
        - train, val, test: select between training, validation, and test sets.
        - window_size: set how many DE features are concatenated for an output sample.
        - If it is val or test, the means and stds calculated from the training set must be passed.
        - If it is train and means and stds are also passed, the passed values are used; otherwise, they are calculated from the train set itself.
        - nameList: select corresponding dataset files. nameList[0],[1],[2] correspond to seed3, 4, 5 respectively.
        - fakeMask: if True, the output masks are all False, i.e., no artifacts are assumed.
        '''
        '''
        Logically, we could call a pre-written dataset for seed345,
        but that would make it inconvenient to calculate the standard deviation for all data.
        So we are writing it from scratch.
        '''
        if train:
            print('Building Train Dataset')
        elif val:
            print('Building Val Dataset')
        elif test:
            print('Building Test Dataset')
        self.data = []
        self.aug = aug
        self.nameList = [[] for _ in range(3)]
        splitIdx = [0]
        ch_num = 62
        datasetDE = np.empty((0,5,ch_num))
        datasetMask = np.empty((0,ch_num),dtype=bool)
        datasetTask = np.empty(0,dtype=np.int64)
        datasetName = np.empty(0,dtype=np.int64) # seed3 start from 0, seed4 start from 15, seed5 start from 30
        datasetExp =  np.empty(0,dtype=np.int64)
        root_dir = str(Path(__file__).resolve().parent.parent.parent)
        if nameList == None:
            self.nameList = [[f'{i}_{j}' for i in range(1,16) for j in range(1,4)] for _ in range(3)]
            self.nameList[2].extend([f'16_{j}' for j in range(1,4)])
        else:
            self.nameList = nameList

        ##########
        # SEED 3 #
        ##########
        print('Loading SEED3 Dataset')
        if self.nameList[0] is not None:
            # Default12 train + 3 validation = 15 test
            if train:
                videoIdx = [i for i in range(12)]
            elif val:
                videoIdx = [12,13,14]
            elif test:
                videoIdx = [i for i in range(15)]
            # Labels need to be unified
            LABELS = [2,1,0,0,1,2,0,1,2,2,1,0,1,2,0]
            for i,v in enumerate(LABELS):
                if v == 0:
                    LABELS[i] = 2
                elif v == 2:
                    LABELS[i] = 0
            for subjectName in tqdm(self.nameList[0]):
                name = int(subjectName.split('_')[0])-1
                experiment = int(subjectName.split('_')[1])-1
                # load data
                dataFolder = f'{root_dir}/data/seed3/{subjectName}'
                # eeg data
                h5f = h5py.File(f'{dataFolder}/de.h5','r')
                de = h5f[subjectName]['de'][:] # (sample_num_all,feature_dim,ch_num)
                trialStart = h5f[subjectName].attrs['trialStart']
                trialEnd = h5f[subjectName].attrs['trialEnd']
                # mask: True for bad channels
                mask = np.load(f'{dataFolder}/artifact/all.npy').T # (sample_num_all,ch_num)
                if fakeMask:
                    mask = np.full_like(mask,False,dtype=bool)

                # Construct the dataset
                for i in videoIdx: # Only need data from trials
                    videoDE = de[trialStart[i]:trialEnd[i]]
                    videoMask = mask[trialStart[i]:trialEnd[i]]
                    videoLabel = np.full(videoMask.shape[0],LABELS[i])
                    videoName = np.full(videoMask.shape[0],name)
                    videoExp = np.full(videoMask.shape[0],experiment)
                    datasetDE = np.concatenate((datasetDE,videoDE),axis=0) # (sample_num,feature_dim,ch_num)
                    datasetMask = np.concatenate((datasetMask,videoMask),axis=0) # (sample_num,ch_num)
                    datasetTask = np.concatenate((datasetTask,videoLabel),axis=0) # (sample_num,)
                    datasetName = np.concatenate((datasetName,videoName),axis=0) # (sample_num,)
                    datasetExp = np.concatenate((datasetExp,videoExp),axis=0) # (sample_num,)
                    splitIdx.append(datasetDE.shape[0])
        else:
            print('Skip SEED3 Dataset')
        

        ##########
        # SEED 4 #
        ##########
        print('Loading SEED4 Dataset')
        if self.nameList[1] is not None:
            # The label order of the SEED4 dataset is a bit strange
            videoIdx = [
                [[0,2,3,5,6,7,8,9,11,15,18,19,12,20,21],[13,22,23],[0,2,3,5,6,7,8,9,11,15,18,19,12,20,21,13,22,23]],
                [[1,2,3,4,6,8,9,11,13,14,15,17,18,19,21],[20,22,23],[1,2,3,4,6,8,9,11,13,14,15,17,18,19,21,20,22,23]],
                [[0,3,4,5,6,7,8,11,13,15,18,19,10,14,21],[17,22,23],[0,3,4,5,6,7,8,11,13,15,18,19,10,14,21,17,22,23]]
            ]
            if train:
                dsetIdx = 0
            elif val:
                dsetIdx = 1
            elif test:
                dsetIdx = 2

            for subjectName in tqdm(self.nameList[1]):
                name = int(subjectName.split('_')[0])-1 + 15 # seed4 start from 15
                experiment = int(subjectName.split('_')[1])-1
                # load data
                dataFolder = f'{root_dir}/data/seed4/{subjectName}'
                # eeg data
                h5f = h5py.File(f'{dataFolder}/de.h5','r')
                de = h5f[subjectName]['de'][:] # (sample_num_all,feature_dim,ch_num)
                trialStart = h5f[subjectName].attrs['trialStart']
                trialEnd = h5f[subjectName].attrs['trialEnd']
                LABELS = h5f[subjectName].attrs['label']
                for i,v in enumerate(LABELS):
                    if v == 0:
                        LABELS[i] = 1
                    elif v == 1:
                        LABELS[i] = 2
                    elif v == 2:
                        LABELS[i] = 3
                    elif v == 3:
                        LABELS[i] = 0
                # mask: True for bad channels
                mask = np.load(f'{dataFolder}/artifact/all.npy').T # (sample_num_all,ch_num)
                if fakeMask:
                    mask = np.full_like(mask,False,dtype=bool)
                    
                # Construct the dataset
                for i in videoIdx[experiment][dsetIdx]: # Only need data from trials
                    videoDE = de[trialStart[i]:trialEnd[i]]
                    videoMask = mask[trialStart[i]:trialEnd[i]]
                    videoLabel = np.full(videoMask.shape[0],LABELS[i])
                    videoName = np.full(videoMask.shape[0],name)
                    videoExp = np.full(videoMask.shape[0],experiment)
                    datasetDE = np.concatenate((datasetDE,videoDE),axis=0) # (sample_num,feature_dim,ch_num)
                    datasetMask = np.concatenate((datasetMask,videoMask),axis=0) # (sample_num,ch_num)
                    datasetTask = np.concatenate((datasetTask,videoLabel),axis=0) # (sample_num,)
                    datasetName = np.concatenate((datasetName,videoName),axis=0) # (sample_num,)
                    datasetExp = np.concatenate((datasetExp,videoExp),axis=0) # (sample_num,)
                    splitIdx.append(datasetDE.shape[0])
        else:
            print('Skip SEED4 Dataset')


        ##########
        # SEED 5 #
        ##########
        print('Loading SEED5 Dataset')
        if self.nameList[2] is not None:
            videoIdx = [
                [[0,2,3,5,7,8],[10,12,13],[0,2,3,5,7,8,10,12,13]],
                [[0,2,4,5,7,8],[10,11,13],[0,2,4,5,7,8,10,11,13]],
                [[0,2,4,5,7,8],[10,11,13],[0,2,4,5,7,8,10,11,13]]
            ]
            if train:
                dsetIdx = 0
            elif val:
                dsetIdx = 1
            elif test:
                dsetIdx = 2
            for subjectName in tqdm(self.nameList[2]):
                name = int(subjectName.split('_')[0])-1 + 30 # seed5 start from 30
                experiment = int(subjectName.split('_')[1])-1
                # load data
                dataFolder = f'{root_dir}/data/seed5/{subjectName}'
                # eeg data
                h5f = h5py.File(f'{dataFolder}/de.h5','r')
                de = h5f[subjectName]['de'][:] # (sample_num_all,feature_dim,ch_num)
                trialStart = h5f[subjectName].attrs['trialStart']
                trialEnd = h5f[subjectName].attrs['trialEnd']
                LABELS = h5f[subjectName].attrs['label']
                for i,v in enumerate(LABELS):
                    if v == 0:
                        LABELS[i] = 4
                    elif v == 1:
                        LABELS[i] = 3
                    elif v == 3:
                        LABELS[i] = 1
                    elif v == 4:
                        LABELS[i] = 0
                # mask: True for bad channels
                mask = np.load(f'{dataFolder}/artifact/all.npy').T # (sample_num_all,ch_num)
                if fakeMask:
                    mask = np.full_like(mask,False,dtype=bool)

                # Construct the dataset
                for i in videoIdx[experiment][dsetIdx]: # Only need data from trials
                    videoDE = de[trialStart[i]:trialEnd[i]]
                    videoMask = mask[trialStart[i]:trialEnd[i]]
                    videoLabel = np.full(videoMask.shape[0],LABELS[i])
                    videoName = np.full(videoMask.shape[0],name)
                    videoExp = np.full(videoMask.shape[0],experiment)
                    datasetDE = np.concatenate((datasetDE,videoDE),axis=0) # (sample_num,feature_dim,ch_num)
                    datasetMask = np.concatenate((datasetMask,videoMask),axis=0) # (sample_num,ch_num)
                    datasetTask = np.concatenate((datasetTask,videoLabel),axis=0) # (sample_num,)
                    datasetName = np.concatenate((datasetName,videoName),axis=0) # (sample_num,)
                    datasetExp = np.concatenate((datasetExp,videoExp),axis=0) # (sample_num,)
                    splitIdx.append(datasetDE.shape[0])
        else:
            print('Skip SEED5 Dataset')
        

        # calculate mean and std
        if train and means is None:
            datasetMask_expanded = np.expand_dims(datasetMask, axis=1) # (sample_num,1,ch_num)
            datasetMask_expanded = np.broadcast_to(datasetMask_expanded, datasetDE.shape)  # (sample_num,feature_dim,ch_num)
            datasetDE_masked = np.ma.masked_array(datasetDE, mask=datasetMask_expanded)
            self.means = datasetDE_masked.mean(axis=(0,2)) # (feature_dim,)
            self.stds = datasetDE_masked.std(axis=(0,2)) # (feature_dim,)
        else:
            self.means = means
            self.stds = stds
        # norm
        datasetDE = (datasetDE-self.means[:,np.newaxis])/self.stds[:,np.newaxis] # (sample_num,feature_dim,ch_num)

        # convert to tensor
        datasetDE = torch.from_numpy(datasetDE).float() # (sample_num,feature_dim,ch_num)
        datasetDE = rearrange(datasetDE,'n f c -> n c f') # (sample_num,ch_num,feature_dim)
        sample_num,ch_num,feature_dim = datasetDE.size()
        datasetMask = torch.from_numpy(datasetMask) # (sample_num,ch_num)
        datasetTask = torch.from_numpy(datasetTask) # (sample_num,)
        datasetName = torch.from_numpy(datasetName) # (sample_num,)
        datasetExp = torch.from_numpy(datasetExp) # (sample_num,)
        # Finally, add 5 virtual nodes
        DEPad = torch.zeros((sample_num,5,feature_dim),dtype=torch.float32)
        datasetDE = torch.cat((datasetDE,DEPad),dim=1)
        maskPad = torch.full((sample_num,5),True,dtype=bool)
        datasetMask = torch.cat((datasetMask,maskPad),dim=1)

        # Create the final dataset, where each sequence consists of window_size DE features
        if window_size == 1:
            for tupleData in zip(datasetDE,datasetMask,datasetName,datasetTask,datasetExp):
                self.data.append(tupleData)
        else:
            for i in range(len(splitIdx)-1):
                for j in range(splitIdx[i],splitIdx[i+1]-window_size+1):
                    self.data.append((datasetDE[j:j+window_size],datasetMask[j:j+window_size],datasetName[j],datasetTask[j],datasetExp[j]))


    def getMeanStd(self) -> Tuple[np.ndarray,np.ndarray]:
        '''
        (means,stds) (feature_dim,)
        '''
        return self.means, self.stds

    def __getitem__(self, index):
        '''
        (de,mask,name,task,experiment)
        - de (window_size, node_num, feature_dim) if window_size != 1, else (node_num, feature_dim)
        - mask (window_size, node_num) if window_size != 1, else (node_num,)
        - name scaler
        - task scaler
        - experiment scaler
        '''
        if self.aug:
            mask = self.data[index][1]
            window_size, node_num = mask.size()
            random_tensor = torch.rand((window_size,node_num))
            augmask = random_tensor < 0.1 # 每个通道都有10%的概率认为是坏的
            return self.data[index][0], mask | augmask, self.data[index][2], self.data[index][3], self.data[index][4]
        else:
            return self.data[index]
    
    def __len__(self):
        return len(self.data)


class dataset_MAML:
    '''
    SEED345 MAML dataset
    - The next() method returns de_spt, mask_spt, y_spt, de_qry, mask_qry, y_qry
        - de_spt/de_qry has a size of (subject_num, k_spt/k_qry, window_size, node_num, feature_dim)
        - mask_spt/mask_qry has a size of (subject_num, k_spt/k_qry, window_size, node_num)
        - y_spt/y_qry has a size of (subject_num, k_spt/k_qry)
    - The reset() method needs to be called at the beginning of each epoch to ensure random sampling in the next() method.
    '''
    def __init__(
            self,
            means:np.ndarray,
            stds:np.ndarray,
            nameList:List[List[str]],        
            k_spt = 64,
            k_qry = 16,
            window_size = 10,
            fakeMask = False
            ):
        '''
        Parameters
        ----------
        - means, stds: parameters for DE feature normalization
        - nameList: nameList[i] corresponds to the dataset files for seed{i+3}
        - k_spt/k_qry: the size of the support and query set for each subject in a batch
        - window_size: set how many DE features are concatenated for an output sample, must be > 1
        - fakeMask: if True, the output masks are all False, i.e., no artifacts are assumed
        '''
        # for next() method
        self.k_spt = k_spt
        self.k_qry = k_qry

        data1 = defaultdict(list) # {'1_1':[(de,mask,label)*15]} helper dict; de(videolen,ch,dim), mask(videolen,ch), label scaler
        
        ##########
        # SEED 3 #
        ##########
        print('Loading SEED3...')
        LABELS = [2,1,0,0,1,2,0,1,2,2,1,0,1,2,0]
        for i,v in enumerate(LABELS):
            if v == 0:
                LABELS[i] = 2
            elif v == 2:
                LABELS[i] = 0
        root_dir = str(Path(__file__).resolve().parent.parent.parent)

        for subjectName in tqdm(nameList[0]):
            dataFolder = f'{root_dir}/data/seed3/{subjectName}'
            # eeg data
            h5f = h5py.File(f'{dataFolder}/de.h5','r')
            de = h5f[subjectName]['de'][:] # (sample_num,feature_dim,ch_num)
            de = (de-means[:,np.newaxis])/stds[:,np.newaxis] # normalize
            de = torch.from_numpy(de).float()
            de = rearrange(de,'n f c -> n c f') # (sample_num,ch_num,feature_dim)
            sample_num,ch_num,feature_dim = de.size()
            DEPad = torch.zeros((sample_num,5,feature_dim),dtype=torch.float32)
            de = torch.cat((de,DEPad),dim=1) # (sample_num,ch_num+5,feature_dim)
            trialStart = h5f[subjectName].attrs['trialStart']
            trialEnd = h5f[subjectName].attrs['trialEnd']
            # mask: True for bad channels
            mask = np.load(f'{dataFolder}/artifact/all.npy').T # (sample_num,ch_num)
            if fakeMask:
                mask = np.full_like(mask,False,dtype=bool)
            mask = torch.from_numpy(mask)
            maskPad = torch.full((sample_num,5),True,dtype=bool)
            mask = torch.cat((mask,maskPad),dim=1) # (sample_num,ch_num+5)
            # create data1
            for i in range(15):
                videoDE = de[trialStart[i]:trialEnd[i]]
                videoMask = mask[trialStart[i]:trialEnd[i]]
                data1[subjectName].append((videoDE,videoMask,LABELS[i]))
        
        ##########
        # SEED 4 #
        ##########
        print('Loading SEED4...')
        # Select videos with target emotions
        videoIdx = [
            [0,2,3,5,6,7,8,9,11,15,18,19,12,20,21,13,22,23],
            [1,2,3,4,6,8,9,11,13,14,15,17,18,19,21,20,22,23],
            [0,3,4,5,6,7,8,11,13,15,18,19,10,14,21,17,22,23]
        ]
        for subjectName in tqdm(nameList[1]):
            dataFolder = f'{root_dir}/data/seed4/{subjectName}'
            # eeg data
            h5f = h5py.File(f'{dataFolder}/de.h5','r')
            de = h5f[subjectName]['de'][:] # (sample_num,feature_dim,ch_num)
            de = (de-means[:,np.newaxis])/stds[:,np.newaxis] # normalize
            de = torch.from_numpy(de).float()
            de = rearrange(de,'n f c -> n c f') # (sample_num,ch_num,feature_dim)
            sample_num,ch_num,feature_dim = de.size()
            DEPad = torch.zeros((sample_num,5,feature_dim),dtype=torch.float32)
            de = torch.cat((de,DEPad),dim=1) # (sample_num,ch_num+5,feature_dim)
            trialStart = h5f[subjectName].attrs['trialStart']
            trialEnd = h5f[subjectName].attrs['trialEnd']
            LABELS = h5f[subjectName].attrs['label']
            for i,v in enumerate(LABELS):
                if v == 0:
                    LABELS[i] = 1
                elif v == 1:
                    LABELS[i] = 2
                elif v == 2:
                    LABELS[i] = 3
                elif v == 3:
                    LABELS[i] = 0
            # mask 坏通道为True
            mask = np.load(f'{dataFolder}/artifact/all.npy').T # (sample_num,ch_num)
            if fakeMask:
                mask = np.full_like(mask,False,dtype=bool)
            mask = torch.from_numpy(mask)
            maskPad = torch.full((sample_num,5),True,dtype=bool)
            mask = torch.cat((mask,maskPad),dim=1) # (sample_num,ch_num+5)
            name, exp = subjectName.split('_')
            name = int(name)
            exp = int(exp)
            # create data1
            for i in videoIdx[exp-1]:
                videoDE = de[trialStart[i]:trialEnd[i]]
                videoMask = mask[trialStart[i]:trialEnd[i]]
                data1[f'{name+15}_{exp}'].append((videoDE,videoMask,LABELS[i]))

        ##########
        # SEED 5 #
        ##########
        print('Loading SEED5...')
        # Select videos with target emotions
        videoIdx = [
            [0,2,3,5,7,8,10,12,13],
            [0,2,4,5,7,8,10,11,13],
            [0,2,4,5,7,8,10,11,13]
        ]
        for subjectName in tqdm(nameList[2]):
            dataFolder = f'{root_dir}/data/seed5/{subjectName}'
            # eeg data
            h5f = h5py.File(f'{dataFolder}/de.h5','r')
            de = h5f[subjectName]['de'][:] # (sample_num,feature_dim,ch_num)
            de = (de-means[:,np.newaxis])/stds[:,np.newaxis] # normalize
            de = torch.from_numpy(de).float()
            de = rearrange(de,'n f c -> n c f') # (sample_num,ch_num,feature_dim)
            sample_num,ch_num,feature_dim = de.size()
            DEPad = torch.zeros((sample_num,5,feature_dim),dtype=torch.float32)
            de = torch.cat((de,DEPad),dim=1) # (sample_num,ch_num+5,feature_dim)
            trialStart = h5f[subjectName].attrs['trialStart']
            trialEnd = h5f[subjectName].attrs['trialEnd']
            LABELS = h5f[subjectName].attrs['label']
            for i,v in enumerate(LABELS):
                if v == 0:
                    LABELS[i] = 4
                elif v == 1:
                    LABELS[i] = 3
                elif v == 3:
                    LABELS[i] = 1
                elif v == 4:
                    LABELS[i] = 0
            # mask: True for bad channels
            mask = np.load(f'{dataFolder}/artifact/all.npy').T # (sample_num,ch_num)
            if fakeMask:
                mask = np.full_like(mask,False,dtype=bool)
            mask = torch.from_numpy(mask)
            maskPad = torch.full((sample_num,5),True,dtype=bool)
            mask = torch.cat((mask,maskPad),dim=1) # (sample_num,ch_num+5)
            name, exp = subjectName.split('_')
            name = int(name)
            exp = int(exp)
            # create data1
            for i in videoIdx[exp-1]:
                videoDE = de[trialStart[i]:trialEnd[i]]
                videoMask = mask[trialStart[i]:trialEnd[i]]
                data1[f'{name+30}_{exp}'].append((videoDE,videoMask,LABELS[i]))  

        print('Creating Dataset...')
        # Use other experiments for qry
        self.data = {}  # {'1_1':{'spt':[(de_sequence,mask_sequence,label),...],'qry':[(de_sequence,mask_sequence,label),...]},...}
                        # de_sequence (winsz,ch,dim), mask_sequence (winsz,ch), label scaler
        name2otherExp = {} # {'1_1': ['1_2','1_3'], '1_2': ['1_1','1_3'],...}
        name2exp = {} # {'1': ['1_1','1_2','1_3'], '2': ['2_1','2_2','2_3'],...}
        for subjectName in data1.keys():
            name, _ = subjectName.split('_')
            if name not in name2exp:
                name2exp[name] = []
            name2exp[name].append(subjectName)
        for subjectName in data1.keys():
            name, _ = subjectName.split('_')
            name2otherExp[subjectName] = [other for other in name2exp[name] if other != subjectName]
        for subjectName in data1.keys():
            if subjectName not in self.data:
                self.data[subjectName] = {'spt':[],'qry':[]}
            if int(subjectName.split('_')[0]) < 16:
                sptVideoIdx = [i for i in range(12)]
                qryVideoIdx = [i for i in range(12,15)]
            elif int(subjectName.split('_')[0]) < 31:
                sptVideoIdx = [i for i in range(15)]
                qryVideoIdx = [i for i in range(15,18)]
            else:
                sptVideoIdx = [i for i in range(6)]
                qryVideoIdx = [i for i in range(6,9)]
            for i in sptVideoIdx: # support用自己的
                de,mask,label = data1[subjectName][i]
                for j in range(de.size(0)-window_size+1):
                    self.data[subjectName]['spt'].append((de[j:j+window_size],mask[j:j+window_size],label))
            for otherExp in name2otherExp[subjectName]: # Use other experiments of the same subject for query
                for i in qryVideoIdx:
                    de,mask,label = data1[otherExp][i]
                    for j in range(de.size(0)-window_size+1):
                        self.data[subjectName]['qry'].append((de[j:j+window_size],mask[j:j+window_size],label))
        # set indices
        self.reset()

    def maxIter(self):
        '''
        Returns the total number of batches in one epoch
        '''
        ret = float('inf')
        for name in self.data.keys():
            ret = min(len(self.data[name]['qry'])//self.k_qry, ret)
            ret = min(len(self.data[name]['spt'])//self.k_spt, ret)
        return ret

    def reset(self):
        self.indices = {
            subject: {
                'spt': torch.randperm(len(self.data[subject]['spt'])).tolist(),
                'qry': torch.randperm(len(self.data[subject]['qry'])).tolist()
            }
            for subject in self.data.keys()
        }

    def next(self):
        '''
        Returns
        -------
        de_spt, mask_spt, y_spt, de_qry, mask_qry, y_qry
        - de_spt/de_qry (subject_num,k_spt/k_qry,window_size,node_num,feature_dim)
        - mask_spt/mask_qry (subject_num,k_spt/k_qry,window_size,node_num)
        - y_spt/y_qry (subject_num,k_spt/k_qry)
        '''
        spt_de_batch = []
        spt_mask_batch = []
        spt_label_batch = []
        qry_de_batch = []
        qry_mask_batch = []
        qry_label_batch = []

        for subject in self.data.keys():
            spt_de_sub = []
            spt_mask_sub = []
            spt_label_sub = []
            qry_de_sub = []
            qry_mask_sub = []
            qry_label_sub = []

            # Get the random indices for the current subject
            spt_indices = self.indices[subject]['spt'][:self.k_spt]
            qry_indices = self.indices[subject]['qry'][:self.k_qry]

            # Get the corresponding samples from spt and qry
            for idx in spt_indices:
                spt_de,spt_mask,spt_label = self.data[subject]['spt'][idx]
                spt_de_sub.append(spt_de) # [(winsz,ch,dim),...]
                spt_mask_sub.append(spt_mask) # [(winsz,ch),...]
                spt_label_sub.append(spt_label) # [int,..]
            for idx in qry_indices:
                qry_de,qry_mask,qry_label = self.data[subject]['qry'][idx]
                qry_de_sub.append(qry_de) # [(winsz,ch,dim),...]
                qry_mask_sub.append(qry_mask) # [(winsz,ch),...]
                qry_label_sub.append(qry_label) # [int,...]
            spt_de_sub = torch.stack(spt_de_sub) # (k_spt,winsz,ch,dim)
            spt_mask_sub = torch.stack(spt_mask_sub) # (k_spt,winsz,ch)
            spt_label_sub = torch.tensor(spt_label_sub) #(k_spt,)
            qry_de_sub = torch.stack(qry_de_sub) # (k_qry,winsz,ch,dim)
            qry_mask_sub = torch.stack(qry_mask_sub) # (k_qry,winsz,ch)
            qry_label_sub = torch.tensor(qry_label_sub) #(k_qry,)

            spt_de_batch.append(spt_de_sub)
            spt_mask_batch.append(spt_mask_sub)
            spt_label_batch.append(spt_label_sub)
            qry_de_batch.append(qry_de_sub)
            qry_mask_batch.append(qry_mask_sub)
            qry_label_batch.append(qry_label_sub)

            # Update indices, removing the drawn samples
            self.indices[subject]['spt'] = self.indices[subject]['spt'][self.k_spt:]
            self.indices[subject]['qry'] = self.indices[subject]['qry'][self.k_qry:]

        spt_de_batch = torch.stack(spt_de_batch) # (subject_num,k_spt,winsz,ch,dim)
        spt_mask_batch = torch.stack(spt_mask_batch) # (subject_num,k_spt,winsz,ch)
        spt_label_batch = torch.stack(spt_label_batch) # (subject_num,k_spt)
        qry_de_batch = torch.stack(qry_de_batch) # (subject_num,k_qry,winsz,ch,dim)
        qry_mask_batch = torch.stack(qry_mask_batch) # (subject_num,k_qry,winsz,ch)
        qry_label_batch = torch.stack(qry_label_batch) # (subject_num,k_qry)
        return spt_de_batch, spt_mask_batch, spt_label_batch, qry_de_batch, qry_mask_batch, qry_label_batch
    
    

if __name__ == '__main__':
    # test dataset MAML
    test_subject = 2
    nameList = [[f"{i}_{j}" for i in range(1, 16) if i != test_subject for j in range(1, 4)]] # seed 3
    nameList.append([f"{i}_{j}" for i in range(1, 16) for j in range(1, 4)]) # seed 4
    nameList.append([f"{i}_{j}" for i in range(1, 17) for j in range(1, 4)]) # seed 5
    means = np.array([-17.275, -19.73, -20.64, -21.92, -23.38])
    stds = np.array([2.099, 1.638, 1.553, 1.493, 1.823])
    dset = dataset_MAML(means=means,stds=stds,nameList=nameList,k_qry=64,k_spt=64)
    print(dset.maxIter())
    de_spt, mask_spt, y_spt, de_qry, mask_qry, y_qry = dset.next()
    print(de_spt.size())
    print(mask_spt.size())
    print(y_spt.size())
    print(de_qry.size())
    print(mask_qry.size())
    print(y_qry.size())
