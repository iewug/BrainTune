'''
SEED3 Dataset
'''
from pathlib import Path
import numpy as np
import h5py
import torch
from torch.utils import data
from typing import Optional, Tuple, List
from einops import rearrange

class dataset_contrast_v2(data.Dataset):
    '''
    SimCLR SEED3 Dataset.
    By default, 12*3 (12 train + 3 validation) + 3*3 (test) subjects/sessions.
    Labels are originally negative 0, neutral 1, and positive 2.
    They are remapped to positive (happy) 0, neutral 1, and negative (sad) 2.
    '''
    def __init__(self, train=False, val=False, test=False, window_size=10, means:Optional[np.ndarray]=None, stds:Optional[np.ndarray]=None, nameList:Optional[List[str]]=None, floatTaskLabel=False, aug=False, fakeMask=False):
        '''
        Parameters
        ----------
        - train, val, test: Selects training, validation, or test set.
        - window_size: Sets the number of concatenated DE features for each output sample.
        - means, stds: If val or test, these should be passed from the mean and std calculated during training.
          If train and means/stds are also provided, they will be used; otherwise, they will be calculated from the training data itself.
        - nameList: Optional list to specify dataset files, otherwise defaults to 12*3 (12 train + 3 validation) + 3*3 (test) subjects/sessions.
        - fakeMask: If True, all masks will be False, effectively ignoring artifacts.
        '''
        self.data = []
        self.aug = aug

        # Default12 train + 3 validation = 15 test
        if train:
            videoIdx = [i for i in range(12)]
            self.nameList = [f'{i}_{j}' for i in range(1,13) for j in range(1,4)]
        elif val:
            videoIdx = [12,13,14]
            self.nameList = [f'{i}_{j}' for i in range(1,13) for j in range(1,4)]
        elif test:
            videoIdx = [i for i in range(15)]
            self.nameList = [f'{i}_{j}' for i in range(13,16) for j in range(1,4)]
        # Custom subject selection
        if nameList is not None:
            self.nameList = nameList
        ch_num = 62
        splitIdx = [0]
        datasetDE = np.empty((0,5,ch_num))
        datasetMask = np.empty((0,ch_num),dtype=bool)
        datasetTask = np.empty(0,dtype=np.int64)
        datasetName = np.empty(0,dtype=np.int64)
        datasetExp =  np.empty(0,dtype=np.int64)
        LABELS = [2,1,0,0,1,2,0,1,2,2,1,0,1,2,0] # Original labels: negative 0, neutral 1, positive 2
        for i,v in enumerate(LABELS):
            if v == 0: # Remap negative to 2
                LABELS[i] = 2
            elif v == 2: # Remap positive to 0 (happy)
                LABELS[i] = 0
        root_dir = str(Path(__file__).resolve().parent.parent.parent)

        for subjectName in self.nameList:
            name = int(subjectName.split('_')[0])-1
            experiment = int(subjectName.split('_')[1])-1
            # Load data
            dataFolder = f'{root_dir}/data/seed3/{subjectName}'
            # EEG data
            h5f = h5py.File(f'{dataFolder}/de.h5','r')
            de = h5f[subjectName]['de'][:] # (sample_num_all,feature_dim,ch_num)
            trialStart = h5f[subjectName].attrs['trialStart']
            trialEnd = h5f[subjectName].attrs['trialEnd']
            # Mask: bad channels are True
            mask = np.load(f'{dataFolder}/artifact/all.npy').T # (sample_num_all,ch_num)
            # Assume all channels are good channels
            if fakeMask:
                mask = np.full_like(mask,False,dtype=bool)

            # Construct dataset
            for i in videoIdx: # Only data within trials is needed
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
        
        # Calculate mean and standard deviation
        if train and means is None:
            datasetMask_expanded = np.expand_dims(datasetMask, axis=1) # (sample_num,1,ch_num)
            datasetMask_expanded = np.broadcast_to(datasetMask_expanded, datasetDE.shape)  # (sample_num,feature_dim,ch_num)
            datasetDE_masked = np.ma.masked_array(datasetDE, mask=datasetMask_expanded)
            self.means = datasetDE_masked.mean(axis=(0,2)) # (feature_dim,)
            self.stds = datasetDE_masked.std(axis=(0,2)) # (feature_dim,)
        else:
            self.means = means
            self.stds = stds
        # Normalize
        datasetDE = (datasetDE-self.means[:,np.newaxis])/self.stds[:,np.newaxis] # (sample_num,feature_dim,ch_num)

        # Convert to tensor
        datasetDE = torch.from_numpy(datasetDE).float() # (sample_num,feature_dim,ch_num)
        datasetDE = rearrange(datasetDE,'n f c -> n c f') # (sample_num,ch_num,feature_dim)
        sample_num,ch_num,feature_dim = datasetDE.size()
        datasetMask = torch.from_numpy(datasetMask) # (sample_num,ch_num)
        if floatTaskLabel:
            datasetTask = torch.from_numpy(datasetTask).float()
        else:
            datasetTask = torch.from_numpy(datasetTask) # (sample_num,)
        datasetName = torch.from_numpy(datasetName) # (sample_num,)
        datasetExp = torch.from_numpy(datasetExp) # (sample_num,)
        # Add 5 dummy nodes
        DEPad = torch.zeros((sample_num,5,feature_dim),dtype=torch.float32)
        datasetDE = torch.cat((datasetDE,DEPad),dim=1)
        maskPad = torch.full((sample_num,5),True,dtype=bool) # Set dummy nodes mask to True (bad)
        datasetMask = torch.cat((datasetMask,maskPad),dim=1)

        # Create final dataset: window_size DE features as a sequence
        if window_size == 1:
            for tupleData in zip(datasetDE,datasetMask,datasetName,datasetTask,datasetExp):
                self.data.append(tupleData)
        else:
            for i in range(len(splitIdx)-1):
                for j in range(splitIdx[i],splitIdx[i+1]-window_size+1):
                    self.data.append((datasetDE[j:j+window_size],datasetMask[j:j+window_size],datasetName[j],datasetTask[j],datasetExp[j]))

    def getMeanStd(self) -> Tuple[np.ndarray,np.ndarray]:
        '''
        Returns (means,stds) with shape (feature_dim,)
        '''
        return self.means, self.stds

    def __getitem__(self, index):
        '''
        Returns (de,mask,name,task,experiment)
        - de: (window_size, node_num, feature_dim) if window_size != 1, else (node_num, feature_dim)
        - mask: (window_size, node_num) if window_size != 1, else (node_num,)
        - name: scalar
        - task: scalar
        - experiment: scalar
        '''
        if self.aug:
            mask = self.data[index][1]
            window_size, node_num = mask.size()
            random_tensor = torch.rand((window_size,node_num))
            augmask = random_tensor < 0.1 # Each channel has a 10% chance of being considered bad
            return self.data[index][0], mask | augmask, self.data[index][2], self.data[index][3], self.data[index][4]
        else:
            return self.data[index]
    
    def __len__(self):
        return len(self.data)



class dataset_single(data.Dataset):
    '''
    Single-subject specified video SEED3 Dataset.
    By default, 12*3 (12 train + 3 validation) + 3*3 (test).
    Labels are originally negative 0, neutral 1, and positive 2.
    They are remapped to positive (happy) 0, neutral 1, and negative (sad) 2.
    '''
    def __init__(self, subjectName:str, videoIdx:List[int], means:np.ndarray, stds:np.ndarray, window_size=10, floatTaskLabel=False, aug=False):
        '''
        Parameters
        ----------
        - subjectName: e.g., '1_1' to '15_3', specifies the subject and session.
        - videoIdx: Specifies the video indices to use, e.g., [0,1,2,...].
        - means, stds: Parameters for normalizing DE features.
        - window_size: Sets the number of concatenated DE features for each output sample.
        '''
        self.aug = aug
        self.data = []

        ch_num = 62
        splitIdx = [0]
        datasetDE = np.empty((0,5,ch_num))
        datasetMask = np.empty((0,ch_num),dtype=bool)
        datasetTask = np.empty(0,dtype=np.int64)
        datasetName = np.empty(0,dtype=np.int64)
        datasetExp =  np.empty(0,dtype=np.int64)
        LABELS = [2,1,0,0,1,2,0,1,2,2,1,0,1,2,0] # Original labels
        for i,v in enumerate(LABELS):
            if v == 0: # Remap negative to 2
                LABELS[i] = 2
            elif v == 2: # Remap positive to 0
                LABELS[i] = 0
        root_dir = str(Path(__file__).resolve().parent.parent.parent)

        name = int(subjectName.split('_')[0])-1
        experiment = int(subjectName.split('_')[1])-1
        # Load data
        dataFolder = f'{root_dir}/data/seed3/{subjectName}'
        # EEG data
        h5f = h5py.File(f'{dataFolder}/de.h5','r')
        de = h5f[subjectName]['de'][:] # (sample_num_all,feature_dim,ch_num)
        trialStart = h5f[subjectName].attrs['trialStart']
        trialEnd = h5f[subjectName].attrs['trialEnd']
        # Mask: bad channels are True
        mask = np.load(f'{dataFolder}/artifact/all.npy').T # (sample_num_all,ch_num)

        # Construct dataset
        for i in videoIdx: # Only data within trials is needed
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
        
        # Normalize
        self.means = means
        self.stds = stds
        datasetDE = (datasetDE-self.means[:,np.newaxis])/self.stds[:,np.newaxis] # (sample_num,feature_dim,ch_num)

        # Convert to tensor
        datasetDE = torch.from_numpy(datasetDE).float() # (sample_num,feature_dim,ch_num)
        datasetDE = rearrange(datasetDE,'n f c -> n c f') # (sample_num,ch_num,feature_dim)
        sample_num,ch_num,feature_dim = datasetDE.size()
        datasetMask = torch.from_numpy(datasetMask) # (sample_num,ch_num)
        if floatTaskLabel:
            datasetTask = torch.from_numpy(datasetTask).float()
        else:
            datasetTask = torch.from_numpy(datasetTask) # (sample_num,)
        datasetName = torch.from_numpy(datasetName) # (sample_num,)
        datasetExp = torch.from_numpy(datasetExp) # (sample_num,)
        # Add 5 dummy nodes
        DEPad = torch.zeros((sample_num,5,feature_dim),dtype=torch.float32)
        datasetDE = torch.cat((datasetDE,DEPad),dim=1)
        maskPad = torch.full((sample_num,5),True,dtype=bool) # Set dummy nodes mask to True (bad)
        datasetMask = torch.cat((datasetMask,maskPad),dim=1)

        # Create final dataset: window_size DE features as a sequence
        if window_size == 1:
            for tupleData in zip(datasetDE,datasetMask,datasetName,datasetTask,datasetExp):
                self.data.append(tupleData)
        else:
            for i in range(len(splitIdx)-1):
                for j in range(splitIdx[i],splitIdx[i+1]-window_size+1):
                    self.data.append((datasetDE[j:j+window_size],datasetMask[j:j+window_size],datasetName[j],datasetTask[j],datasetExp[j]))

    def getMeanStd(self) -> Tuple[np.ndarray,np.ndarray]:
        '''
        Returns (means,stds) with shape (feature_dim,)
        '''
        return self.means, self.stds

    def __getitem__(self, index):
        '''
        Returns (de,mask,name,task,experiment)
        - de: (window_size, node_num, feature_dim) if window_size != 1, else (node_num, feature_dim)
        - mask: (window_size, node_num) if window_size != 1, else (node_num,)
        - name: scalar
        - task: scalar
        - experiment: scalar
        '''
        if self.aug:
            mask = self.data[index][1]
            window_size, node_num = mask.size()
            random_tensor = torch.rand((window_size,node_num))
            augmask = random_tensor < 0.1 # Each channel has a 10% chance of being considered bad
            return self.data[index][0], mask | augmask, self.data[index][2], self.data[index][3], self.data[index][4]
        else:
            return self.data[index]
    
    def __len__(self):
        return len(self.data)

class dataset_MAML:
    '''
    SEED3 MAML Dataset.
    Frankly, this could probably be implemented using several PyTorch DataLoaders, but since it's already custom-written, it remains as is.
    - The next() method returns de_spt, mask_spt, y_spt, de_qry, mask_qry, y_qry.
      - de_spt/de_qry shape: (subject_num,k_spt/k_qry,window_size,node_num,feature_dim)
      - mask_spt/mask_qry shape: (subject_num,k_spt/k_qry,window_size,node_num)
      - y_spt/y_qry shape: (subject_num,k_spt/k_qry)
    - The reset() method should be called at the beginning of each epoch to enable random sampling for the next() method.
    '''
    def __init__(
            self,
            means:np.ndarray,
            stds:np.ndarray,
            nameList:List[str] = [f'{i}_{j}' for i in range(1,15) for j in range(1,4)],
            k_spt = 64,
            k_qry = 16,
            sptVideoIdx:List[int] = [i for i in range(12)],
            qryVideoIdx:List[int] = [i for i in range(12,15)],
            window_size = 10
            ):
        '''
        Parameters
        ----------
        - means, stds: Parameters for normalizing DE features.
        - nameList: List of dataset files (subject_session names).
        - k_spt/k_qry: Size of support and query sets for each subject in a batch.
        - sptVideoIdx/qryVideoIdx: Video indices from which support and query samples are drawn, respectively.
        - window_size: Sets the number of concatenated DE features for each output sample; must be > 1.
        '''
        # For next() method
        self.k_spt = k_spt
        self.k_qry = k_qry

        data1 = {} # {'1_1':[(de,mask,label)*15]} helper dict; de(videolen,ch,dim), mask(videolen,ch), label scalar
        for subjectName in nameList:
            data1[subjectName] = []

        LABELS = [2,1,0,0,1,2,0,1,2,2,1,0,1,2,0] # Original labels
        for i,v in enumerate(LABELS):
            if v == 0: # Remap negative to 2
                LABELS[i] = 2
            elif v == 2: # Remap positive to 0
                LABELS[i] = 0
        root_dir = str(Path(__file__).resolve().parent.parent.parent)

        for subjectName in nameList:
            dataFolder = f'{root_dir}/data/seed3/{subjectName}'
            # EEG data
            h5f = h5py.File(f'{dataFolder}/de.h5','r')
            de = h5f[subjectName]['de'][:] # (sample_num,feature_dim,ch_num)
            de = (de-means[:,np.newaxis])/stds[:,np.newaxis] # Normalize
            de = torch.from_numpy(de).float()
            de = rearrange(de,'n f c -> n c f') # (sample_num,ch_num,feature_dim)
            sample_num,ch_num,feature_dim = de.size()
            DEPad = torch.zeros((sample_num,5,feature_dim),dtype=torch.float32)
            de = torch.cat((de,DEPad),dim=1) # (sample_num,ch_num+5,feature_dim)
            trialStart = h5f[subjectName].attrs['trialStart']
            trialEnd = h5f[subjectName].attrs['trialEnd']
            # Mask: bad channels are True
            mask = np.load(f'{dataFolder}/artifact/all.npy').T # (sample_num,ch_num)
            mask = torch.from_numpy(mask)
            maskPad = torch.full((sample_num,5),True,dtype=bool) # Set dummy nodes mask to True (bad)
            mask = torch.cat((mask,maskPad),dim=1) # (sample_num,ch_num+5)
            # Create data1
            for i in range(15):
                videoDE = de[trialStart[i]:trialEnd[i]]
                videoMask = mask[trialStart[i]:trialEnd[i]]
                data1[subjectName].append((videoDE,videoMask,LABELS[i]))
        
        # Support and query sets come from the same person, regardless of experiment session
        # self.data = {} # {'1':{'spt':[(de_sequence,mask_sequence,label),...],'qry':[(de_sequence,mask_sequence,label),...]},...}
        #              # de_sequence (winsz,ch,dim), mask_sequence (winsz,ch), label scalar
        # for subjectName in nameList:
        #     name = subjectName.split('_')[0]
        #     if name not in self.data:
        #         self.data[name] = {'spt':[],'qry':[]}
        #     for i in sptVideoIdx:
        #         de,mask,label = data1[subjectName][i]
        #         for j in range(de.size(0)-window_size+1):
        #             self.data[name]['spt'].append((de[j:j+window_size],mask[j:j+window_size],label))
        #     for i in qryVideoIdx:
        #         de,mask,label = data1[subjectName][i]
        #         for j in range(de.size(0)-window_size+1):
        #             self.data[name]['qry'].append((de[j:j+window_size],mask[j:j+window_size],label))
        # self.data = {} # {'1_1':{'spt':[(de_sequence,mask_sequence,label),...],'qry':[(de_sequence,mask_sequence,label),...]},...}
        #              # de_sequence (winsz,ch,dim), mask_sequence (winsz,ch), label scalar
        
        # Query uses data from other experimental sessions
        self.data = {} # {'1_1':{'spt':[(de_sequence,mask_sequence,label),...],'qry':[(de_sequence,mask_sequence,label),...]},...}
                     # de_sequence (winsz,ch,dim), mask_sequence (winsz,ch), label scalar
        name2otherExp = {} # {'1_1': ['1_2','1_3'], '1_2': ['1_1','1_3'],...}
        name2exp = {} # {'1': ['1_1','1_2','1_3'], '2': ['2_1','2_2','2_3'],...}
        for subjectName in nameList:
            name, _ = subjectName.split('_')
            if name not in name2exp:
                name2exp[name] = []
            name2exp[name].append(subjectName)
        for subjectName in nameList:
            name, _ = subjectName.split('_')
            name2otherExp[subjectName] = [other for other in name2exp[name] if other != subjectName]
        for subjectName in nameList:
            if subjectName not in self.data:
                self.data[subjectName] = {'spt':[],'qry':[]}
            for i in sptVideoIdx: # Support set uses its own session's data
                de,mask,label = data1[subjectName][i]
                for j in range(de.size(0)-window_size+1):
                    self.data[subjectName]['spt'].append((de[j:j+window_size],mask[j:j+window_size],label))
            for otherExp in name2otherExp[subjectName]: # Query set uses data from other sessions of the same subject
                for i in qryVideoIdx:
                    de,mask,label = data1[otherExp][i]
                    for j in range(de.size(0)-window_size+1):
                        self.data[subjectName]['qry'].append((de[j:j+window_size],mask[j:j+window_size],label))
        # Set indices
        self.reset()

    def maxIter(self):
        '''
        Returns the total number of batches in an epoch.
        '''
        name = list(self.data.keys())[0] # Pick an arbitrary subject
        qryNum = len(self.data[name]['qry'])//self.k_qry
        sptNum = len(self.data[name]['spt'])//self.k_spt
        return min(qryNum,sptNum)

    def reset(self):
        # Reset the indices for random sampling at the start of each epoch
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
        - de_spt/de_qry shape: (subject_num,k_spt/k_qry,window_size,node_num,feature_dim)
        - mask_spt/mask_qry shape: (subject_num,k_spt/k_qry,window_size,node_num)
        - y_spt/y_qry shape: (subject_num,k_spt/k_qry)
        '''
        """
        Retrieves a batch of data.
        :return: A tuple (spt_batch_data, qry_batch_data)
        """
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

            # Get random indices for the current subject
            spt_indices = self.indices[subject]['spt'][:self.k_spt]
            qry_indices = self.indices[subject]['qry'][:self.k_qry]

            # Extract corresponding samples from spt and qry sets
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

            # Update indices by removing the extracted samples
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
    means = np.array([-17.24, -19.72, -20.72, -22.07, -23.36])
    stds = np.array([2.090, 1.665, 1.581, 1.442, 1.799])
    nameList_train = [f"{i}_{j}" for i in range(1, 16) if i != 2 for j in range(1, 4)]
    dset = dataset_contrast_v2(val=True,means=means,stds=stds,nameList=nameList_train)
    cnt = np.zeros((15,3,3))
    for i in range(len(dset)):
        d,m,n,t,e = dset.__getitem__(i)
        cnt[n][e][t] += 1
    print(cnt)