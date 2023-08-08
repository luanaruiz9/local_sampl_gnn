# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 14:57:45 2023

@author: Luana Ruiz
"""

import numpy as np
import torch

def return_node_idx(edge_list, m):
    
    N = torch.max(edge_list)
    edge_list = edge_list[:,torch.argsort(edge_list[1])]
    edge_list = edge_list[:,torch.argsort(edge_list[0])]
    
    idx = list()
    candidates = list()
    while len(idx) < m:
        if not len(candidates) == 0:
            i = candidates[0]
            candidates = candidates.remove(i)
        else:
            if len(idx) == 0:
                aux_list = list(np.arange(N+1))
            else:
                aux_list = list(np.arange(N+1))
                for j in idx:
                    if j in aux_list:
                        aux_list.remove(j)
            i = np.random.choice(np.array(aux_list))
            idx += [i]
            
        cur_edge_list = edge_list[1,torch.argwhere(edge_list[0]==i)]
        cur_edge_set = list(set(list(torch.flatten(cur_edge_list).cpu().numpy())))
        for j in idx:
            if j in cur_edge_set:
                cur_edge_set.remove(j)
                
        if len(cur_edge_set) + len(idx) <= m:
            idx = idx + cur_edge_set
            if candidates is None:
                candidates = list(set(cur_edge_set))
            elif len(candidates) == 0 :
                candidates = list(set(cur_edge_set))
            else:
                candidates = list(set(candidates.extend(cur_edge_set)))
        else:
            idx = idx + cur_edge_set[0:m-len(idx)]
    return idx