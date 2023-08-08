# No node or neighborhood sampling

import sys
import os
import datetime

import numpy as np
import torch
import pickle as pkl

from torch_geometric.datasets import Planetoid
from torch_geometric.loader import DataLoader

import matplotlib.pyplot as plt

import gnn
import train_test
from aux_functions import return_node_idx

dataset_name = sys.argv[1] # pubmed
n0 = int(sys.argv[2]) # subgraph size
n_epochs_per_n = int(sys.argv[3]) # interval between resampling, in nb. of epochs

# figure parameters
color = {}
color['SAGE'] = 'gray'
color['GCN'] = 'violet'
figSize = 5
plt.rcParams.update({'font.size': 16})

# file handling
thisFilename = dataset_name + '_' + str(n0) + '_' + str(n_epochs_per_n) #'800' 
# this is the general name of all related files
saveDirRoot = 'experiments' # In this case, relative location
saveDir = os.path.join(saveDirRoot, thisFilename) 
# create .txt to store the values of the setting parameters for easier
# reference when running multiple experiments
today = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
# append date and time of the run to the directory, to avoid several runs of
# overwritting each other.
saveDir = saveDir + '-' + today
# create directory 
if not os.path.exists(saveDir):
    os.makedirs(saveDir)

# aux class
class objectview(object):
    def __init__(self, d):
        self.__dict__ = d

# training hyperparamters
for args in [
        {'batch_size': 32, 'epochs': 300, 'opt': 'adam', 'opt_scheduler': 'none', 
         'opt_restart': 0, 'weight_decay': 5e-3, 'lr': 0.001},
    ]:
        args = objectview(args)

# training hyperparamters for large graph GNN
for args2 in [
        {'batch_size': 32, 'epochs': n_epochs_per_n, 'opt': 'adam', 
         'opt_scheduler': 'none', 'opt_restart': 0, 'weight_decay': 5e-3, 'lr': 0.001},
    ]:
        args2 = objectview(args2)
n_epochs = args.epochs
n_increases = int(n_epochs/n_epochs_per_n)

# lists for saving datasets and loaders over multiple realizations
dataset_vector = []
loader_vector = []
another_loader_vector = []

# Loss
loss = torch.nn.NLLLoss()

# data
dataset = Planetoid(root='/tmp/pubmed', name='PubMed', split='full')
F0 = dataset.num_node_features
C = dataset.num_classes
data = dataset.data 
m = data.num_nodes
data = data.subgraph(torch.randperm(data.num_nodes)[0:m])
edge_list = data.edge_index

# GNN models
modelList = dict()

F = [F0, 64, 32]
MLP = [32, C]
K = [2, 2]

SAGE = gnn.GNN('sage', F, MLP, True)
modelList['SAGE'] = SAGE

GCN = gnn.GNN('gcn', F, MLP, True)
modelList['GCN'] = GCN

SAGELarge = gnn.GNN('sage', F, MLP, True)
modelList['SAGE full'] = SAGELarge

GCNLarge = gnn.GNN('gcn', F, MLP, True)
modelList['GCN full'] = GCNLarge

# large graph data on which we will test all models  
dataset_transf = [data]
another_test_loader = DataLoader(dataset_transf, batch_size=args.batch_size, shuffle=False)

# generating datasets and loader for each sampled subgraph
m = n0
for i in range(n_increases+1):
    
    idx = return_node_idx(edge_list,m)
    sampledData = data.subgraph(torch.tensor(idx))
    dataset = [sampledData]
    dataset_vector.append(dataset)
    loader_vector.append(DataLoader(dataset, batch_size=args.batch_size, shuffle=False))
    another_loader_vector.append(another_test_loader)

loader_vector_dict = dict()
loader_vector_dict['SAGE'] = loader_vector
loader_vector_dict['GCN'] = loader_vector
loader_vector_dict['GNN'] = loader_vector
loader_vector_dict['SAGE full'] = another_loader_vector
loader_vector_dict['GCN full'] = another_loader_vector
loader_vector_dict['GNN full'] = another_loader_vector

# dictionaries to save test results
test_acc_dict = dict()
time_dict = dict()
best_accs = dict()

# initialize figures

fig1, fig_last = plt.subplots(figsize=(1.4*figSize, 1*figSize))
fig2, fig_best = plt.subplots(figsize=(1.4*figSize, 1*figSize))

# training and testing
for model_key, model in modelList.items():
    
    print('Training ' + model_key + '...')
    
    best_model = model
    best_acc = 0
    for count, loader in enumerate(loader_vector_dict[model_key]):

        best_acc_old = best_acc
        best_model_old = best_model
        test_accs, losses, best_model, last_model, best_acc, test_loader, training_time = train_test.train(loader, loader, model, loss, args2) 
        if count == 0:
            test_accs_full = test_accs
            total_time = training_time
        else:
            test_accs_full += test_accs
            total_time += training_time
        if best_acc < best_acc_old:
            best_model = best_model_old
            best_acc = best_acc_old
        
        print("Maximum validation set accuracy: {0}".format(max(test_accs)))
        print("Minimum loss: {0}".format(min(losses)))
        print()
        
    time_dict[model_key] = total_time/n_epochs
    test_acc_dict[model_key] = test_accs_full
    best_accs[model_key] = best_acc

    if 'SAGE' in model_key:
        col = color['SAGE']
    elif 'GCN' in model_key:
        col = color['GCN']
        
    if 'full' in model_key:
        fig_last.plot(test_accs_full[-1]*np.ones(len(test_accs_full)), '--', color=col, label=model_key)
        fig_best.plot(best_acc*np.ones(len(test_accs_full)), '--', color=col, label=model_key)
    else:
        fig_last.plot(test_accs_full, color=col, alpha=0.5, label=model_key)
        fig_best.plot(test_accs_full, color=col, alpha=0.5, label=model_key)

fig_last.set_ylabel('Accuracy')
fig_last.set_xlabel('Epochs')
fig_last.legend()
fig1.savefig(os.path.join(saveDir,'accuracies_last.pdf'))

fig_best.set_ylabel('Accuracy')
fig_best.set_xlabel('Epochs')
fig_best.legend()
fig2.savefig(os.path.join(saveDir,'accuracies_best.pdf'))

print()

pkl.dump(time_dict, open(os.path.join(saveDir,'time_per_epoch.p'), "wb"))
pkl.dump(test_acc_dict, open(os.path.join(saveDir,'test_accs_full.p'), "wb"))
pkl.dump(best_accs, open(os.path.join(saveDir,'best_accs.p'), "wb"))
