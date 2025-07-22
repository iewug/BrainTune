import numpy as np
from typing import List, Union


def load_edge(file_path='edge.csv', node_num=38, retmat=False) -> Union[List[List[int]],np.ndarray]:
    '''
    Returns
    -------
    - neigh_lists: List[[List[int]]], neigh_lists[i]is which points has edges to nod i. 
    - or
    - neigh_matrix: np.ndarray, (node_num,node_num), M_{i,j}=True means there is edge from i to j.
    '''
    neigh_lists = [[] for _ in range(node_num)]
    neigh_matrix = np.zeros((node_num,node_num))
    mode = None
    with open(file_path) as f:
        for line in f:
            u,v = line.strip().split(',')
            if u == 'undirected':
                mode = 'undirected'
            elif u == 'directed':
                mode = 'directed'
            elif mode == 'undirected':
                u, v = int(u), int(v)
                neigh_lists[u].append(v)
                neigh_lists[v].append(u)
                neigh_matrix[u,v] = neigh_matrix[v,u] = 1
            elif mode == 'directed':
                u, v = int(u), int(v)
                neigh_lists[v].append(u)
                neigh_matrix[v,u] = 1
    if retmat:
        return neigh_matrix
    else:
        return neigh_lists



if __name__ == '__main__':
    # print(load_edge())
    print(load_edge(retmat=True))
