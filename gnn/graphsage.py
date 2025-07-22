'''
GraphSage model
'''
import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import List
from einops import repeat, rearrange
from copy import deepcopy

class GraphSageLayer(nn.Module):
    '''
    Aggregate, concatenate, and feedforward.
    '''
    def __init__(self, neigh_mat:torch.Tensor, in_dim:int, out_dim:int) -> None:
        '''
        Parameters
        ----------
        - neigh_mat: (node_num, node_num), neigh_mat[i][j]=1 indicates an edge from j to i.
        - in_dim: Input feature dimension.
        - out_dim: Output feature dimension.
        '''
        super().__init__()
        self.node_num = neigh_mat.size()[0]
        self.in_dim = in_dim
        self.neigh_mat = neigh_mat # (node_num, node_num)
        self.fc = nn.Linear(in_dim*2, out_dim)
    
    def forward(self, x:torch.Tensor, mask_bad:torch.Tensor) -> torch.Tensor:
        '''
        Parameters
        ----------
        - x: (batch_size, node_num, feature_dim)
        - mask_bad: (batch_size, node_num), dtype=bool, True for bad channels.

        Returns
        -------
        - x: (batch_size, node_num, out_dim)
        '''
        # 1. Aggregate (average over good neighbors)
        neigh_mat_batch = repeat(self.neigh_mat, 'n m -> b n m', b=x.size()[0]).clone() # (batch_size, node_num, node_num)
        neigh_mat_batch = neigh_mat_batch * ~mask_bad.unsqueeze(1) # Bad channels do not contribute values.
        # In-degree of each node
        num_neigh = neigh_mat_batch.sum(dim=-1) # (batch_size, node_num)
        no_neigh = (num_neigh == 0) # (batch_size, node_num)
        # Normalize
        num_neigh[no_neigh] = 1 # Avoid division by zero.
        neigh_mat_batch = neigh_mat_batch.div(num_neigh.unsqueeze(-1))
        # Average of good neighbors
        agg_x = torch.matmul(neigh_mat_batch, x) # (batch_size, node_num, feature_dim)

        # 2. Concatenate
        cat_x = torch.zeros(x.size()[0], self.node_num, self.in_dim * 2).to(x.device) # (batch_size, node_num, feature_dim * 2)
        # 1) All good
        cat_x[:,:,:self.in_dim] += x * ~mask_bad.unsqueeze(-1)
        cat_x[:,:,self.in_dim:] += agg_x * ~no_neigh.unsqueeze(-1)
        # 2) Self bad, neighbors good
        cat_x[:,:,:self.in_dim] += agg_x * (mask_bad & ~no_neigh).unsqueeze(-1)
        # 3) Self good, neighbors bad
        cat_x[:,:,self.in_dim:] += x * (~mask_bad & no_neigh).unsqueeze(-1)
        # Nodes that are bad themselves but have good neighbors are considered good.
        mask_bad[mask_bad & ~no_neigh] = False

        # 3. Feedforward
        return F.relu(self.fc(cat_x))
        

class GraphSage(nn.Module):
    def __init__(self, neigh_mat:torch.Tensor, out_dim=2, hid_dims:List[int]=[5,32,256]) -> None:
        '''
        Parameters
        ----------
        - neigh_mat: (node_num, node_num), neigh_mat[i][j]=True indicates an edge from j to i.
        - out_dim: Output dimension of the final fully connected layer.
        - hid_dims: List[int], hidden feature dimensions for each layer.
        '''
        super().__init__()
        self.node_num = neigh_mat.size()[0]
        self.layers = nn.ModuleList([GraphSageLayer(neigh_mat, hid_dims[i], hid_dims[i+1]) for i in range(len(hid_dims)-1)])
        self.fc = nn.Linear(hid_dims[-1]*5, out_dim)
    
    def forward(self, x:torch.Tensor, mask_bad:torch.Tensor) -> torch.Tensor:
        '''
        Parameters
        ----------
        - x: (batch_size, node_num, feature_dim)
        - mask_bad: (batch_size, node_num), dtype=bool, True for bad channels.

        Returns
        -------
        - x: (batch_size, out_dim)
        '''
        mask_bad = deepcopy(mask_bad) # Do not want to change the input value.
        for layer in self.layers:
            x = layer(x, mask_bad) # (batch_size, node_num, hid_dims[-1])
        out = rearrange(x[:,-5:,:], 'b n d -> b (n d)')
        return self.fc(out)


if __name__ == '__main__':
    neigh_mat = torch.tensor([[0,1,1,0,0,0,0],[1,0,0,1,0,0,0],[1,0,0,1,1,0,0],[0,1,1,0,0,1,0],[0,0,1,0,0,1,0],[0,0,0,1,1,0,0],[1,1,1,1,1,1,0]],dtype=torch.float32).to(0)
    model = GraphSage(neigh_mat,hid_dims=[5,5,6]).to(0)
    x = torch.tensor([[[1,2,3,4,5],[2,3,4,5,6],[3,4,5,6,7],[4,5,6,7,8],[5,6,7,8,9],[6,7,8,9,10],[0,0,0,0,0]],[[2,3,4,5,6],[1,2,3,4,5],[4,5,6,7,8],[3,4,5,6,7],[6,7,8,9,10],[5,6,7,8,9],[0,0,0,0,0]]],dtype=torch.float32).to(0)
    mask_bad = torch.tensor([[False,False,False,False,False,False,True],[False,False,False,False,False,False,True]]).to(0)
    print(model(x,mask_bad).size())
    # print(mask_bad)