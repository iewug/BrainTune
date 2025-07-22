import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import random
from collections import defaultdict
import numpy as np

class MixedSampler(Sampler):
    '''
    Stratified sampling of two datasets based on a given ratio.
    '''
    def __init__(self, dataset1, dataset2, batch_size, ratio=0.9):
        '''
        Parameters
        ----------
        - dataset1/dataset2: The two datasets.
        - batch_size: The batch size.
        - ratio: The ratio of samples to draw from dataset1 per batch relative to the batch size.
        '''
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.batch_size = batch_size
        self.num_samples1 = int(batch_size * ratio) # Number of samples to draw from dataset1 per batch
        self.num_samples2 = batch_size - self.num_samples1 # Number of samples to draw from dataset2 per batch

        # Initialize and shuffle indices
        self.dataset1_indices = random.sample(range(len(dataset1)), len(dataset1))
        self.dataset2_indices = random.sample(range(len(dataset2)), len(dataset2))

        self.current_idx1 = 0
        self.current_idx2 = 0
        self.total_batches = min(len(self.dataset1_indices) // self.num_samples1, # Total number of batches
                                 len(self.dataset2_indices) // self.num_samples2)
        
    def reset_indices(self):
        # Shuffle indices for a new epoch
        random.shuffle(self.dataset1_indices)
        random.shuffle(self.dataset2_indices)
        self.current_idx1 = 0
        self.current_idx2 = 0


    def __iter__(self):
        self.reset_indices()
        for _ in range(self.total_batches):
            # Get indices for the current batch
            batch_indices1 = self.dataset1_indices[self.current_idx1:self.current_idx1 + self.num_samples1]
            batch_indices2 = self.dataset2_indices[self.current_idx2:self.current_idx2 + self.num_samples2]

            # Update indices
            self.current_idx1 += self.num_samples1
            self.current_idx2 += self.num_samples2

            # Combine and shuffle indices from both datasets
            batch_indices = batch_indices1 + [idx + len(self.dataset1) for idx in batch_indices2]
            random.shuffle(batch_indices)

            yield batch_indices

    def __len__(self):
        # Return the expected number of batches
        return self.total_batches


class MixedDataset(Dataset):
    '''
    Combines two datasets.
    '''
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def __len__(self):
        return len(self.dataset1) + len(self.dataset2)

    def __getitem__(self, idx):
        if idx < len(self.dataset1):
            return self.dataset1[idx]
        else:
            return self.dataset2[idx - len(self.dataset1)]
        

class BalancedSampler(Sampler):
    '''
    Balanced sampling.
    '''
    def __init__(self, dataset, group_indices, samples_per_group):
        '''
        Parameters
        ----------
        - dataset: The dataset to sample from.
        - group_indices: The indices of the 0-dimensional tensors returned by the dataset's __getitem__ method
                         that will be used to form groups for balanced sampling.
        - samples_per_group: Number of samples to draw from each group per batch.
        '''
        self.dataset = dataset
        self.group_indices = group_indices
        self.samples_per_group = samples_per_group
        self.group_to_indices = self._group_indices()
        self.batch_num = min(len(v) for v in self.group_to_indices.values()) // samples_per_group

    def _group_indices(self):
        '''
        Returns a dictionary mapping each (name, task, exp) tuple to a list of dataset indices.
        Example: {('n','t','e'):[0,3,4,...],...} where n,t,e are 0-dimensional tensors.
        '''
        groups = defaultdict(list)
        for idx in range(len(self.dataset)):
            sample = self.dataset[idx]
            group_key = tuple(sample[i].item() for i in self.group_indices) # Note: requires 0-dimensional tensors here
            groups[group_key].append(idx)
        return groups
    
    def reset_indices(self):
        for value in self.group_to_indices.values():
            random.shuffle(value) 
        self.cur_idx = 0

    def __iter__(self):
        self.reset_indices()
        for _ in range(self.batch_num):
            batch_indices = []
            for value in self.group_to_indices.values():
                batch_indices.extend(value[self.cur_idx:self.cur_idx+self.samples_per_group])
            self.cur_idx += self.samples_per_group
            random.shuffle(batch_indices)
            yield batch_indices

    def __len__(self):
        return self.batch_num
    

# Test code
if __name__ == '__main__':
    class MyDataset(Dataset):
        def __init__(self, data):
            self.data = data  # data is a list containing (de, mask, name, task, experiment)

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    # Create dataset
    data = [
        (torch.randn(5), torch.tensor(1), torch.tensor(1), torch.tensor(1), torch.tensor(1)),
        (torch.randn(5), torch.tensor(2), torch.tensor(1), torch.tensor(2), torch.tensor(1)),
        (torch.randn(5), torch.tensor(3), torch.tensor(2), torch.tensor(1), torch.tensor(1)),
        (torch.randn(5), torch.tensor(4), torch.tensor(2), torch.tensor(2), torch.tensor(1)),
        (torch.randn(5), torch.tensor(5), torch.tensor(3), torch.tensor(1), torch.tensor(1)),
        (torch.randn(5), torch.tensor(6), torch.tensor(3), torch.tensor(2), torch.tensor(1)),
        (torch.randn(5), torch.tensor(11), torch.tensor(1), torch.tensor(1), torch.tensor(2)),
        (torch.randn(5), torch.tensor(22), torch.tensor(1), torch.tensor(2), torch.tensor(2)),
        (torch.randn(5), torch.tensor(33), torch.tensor(2), torch.tensor(1), torch.tensor(2)),
        (torch.randn(5), torch.tensor(44), torch.tensor(2), torch.tensor(2), torch.tensor(2)),
        (torch.randn(5), torch.tensor(55), torch.tensor(3), torch.tensor(1), torch.tensor(2)),
        (torch.randn(5), torch.tensor(66), torch.tensor(3), torch.tensor(2), torch.tensor(2)),
        (torch.randn(5), torch.tensor(101), torch.tensor(1), torch.tensor(1), torch.tensor(1)),
        (torch.randn(5), torch.tensor(102), torch.tensor(1), torch.tensor(2), torch.tensor(1)),
        (torch.randn(5), torch.tensor(103), torch.tensor(2), torch.tensor(1), torch.tensor(1)),
        (torch.randn(5), torch.tensor(104), torch.tensor(2), torch.tensor(2), torch.tensor(1)),
        (torch.randn(5), torch.tensor(105), torch.tensor(3), torch.tensor(1), torch.tensor(1)),
        (torch.randn(5), torch.tensor(106), torch.tensor(3), torch.tensor(2), torch.tensor(1)),
        (torch.randn(5), torch.tensor(111), torch.tensor(1), torch.tensor(1), torch.tensor(2)),
        (torch.randn(5), torch.tensor(122), torch.tensor(1), torch.tensor(2), torch.tensor(2)),
        (torch.randn(5), torch.tensor(133), torch.tensor(2), torch.tensor(1), torch.tensor(2)),
        (torch.randn(5), torch.tensor(144), torch.tensor(2), torch.tensor(2), torch.tensor(2)),
        (torch.randn(5), torch.tensor(155), torch.tensor(3), torch.tensor(1), torch.tensor(2)),
        (torch.randn(5), torch.tensor(166), torch.tensor(3), torch.tensor(2), torch.tensor(2)),
        # Add more samples...
    ]
    dataset = MyDataset(data)

    # Define balanced sampler, grouping by (name, task, experiment) with indices [2, 3, 4]
    sampler = BalancedSampler(
        dataset=dataset,
        group_indices=[2, 3, 4],  # Group by (name, task, experiment)
        samples_per_group=1
    )

    # Create DataLoader
    dataloader = DataLoader(dataset, batch_sampler=sampler) # batch_size = num_names * num_experiments * num_tasks * samples_per_group

    # Test DataLoader
    for epoch in range(3):
        for batch in dataloader:
            print(batch)  # batch is a list of tuples
        print()

    # class Dataset1(Dataset):
    #     def __init__(self):
    #         self.data = torch.tensor([[i,i,i] for i in range(50)])
    #     def __len__(self):
    #         return len(self.data)
    #     def __getitem__(self, idx):
    #         return self.data[idx]

    # class Dataset2(Dataset):
    #     def __init__(self):
    #         self.data = torch.tensor([[i,i,i] for i in range(50,100)])
    #     def __len__(self):
    #         return len(self.data)
    #     def __getitem__(self, idx):
    #         return self.data[idx]
    
    # # Instantiate datasets
    # dataset1 = Dataset1()
    # dataset2 = Dataset2()
    # batch_size = 10

    # # Create mixed dataset and sampler
    # mixed_dataset = MixedDataset(dataset1, dataset2)
    # mixed_sampler = MixedSampler(dataset1, dataset2, batch_size)

    # # Pass the sampler to DataLoader
    # dataloader = DataLoader(mixed_dataset, batch_sampler=mixed_sampler)

    # # Test reading a few batches
    # for epoch in range(3):
    #     for i, batch in enumerate(dataloader):
    #         print(f"Batch {i+1}")
    #         print(batch)