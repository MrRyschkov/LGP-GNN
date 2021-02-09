"""
    IMPORTING LIBS
"""
import dgl

import numpy as np
import os
import socket
import time
import random
import glob
import argparse, json
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter
from tqdm import tqdm

class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self

import sys


"""
    IMPORTING CUSTOM MODULES/METHODS
"""
from nets.SBMs_node_classification.load_net import gnn_model # import all GNNS
from data.data import LoadData # import dataset

"""
    GPU Setup
"""
def gpu_setup(use_gpu, gpu_id):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  

    if torch.cuda.is_available() and use_gpu:
        print('cuda available with GPU:',torch.cuda.get_device_name(0))
        device = torch.device("cuda")
    else:
        print('cuda not available')
        device = torch.device("cpu")
    return device

"""
    VIEWING MODEL CONFIG AND PARAMS
"""
def view_model_param(MODEL_NAME, net_params):
    model = gnn_model(MODEL_NAME, net_params)
    total_param = 0
    print("MODEL DETAILS:\n")
    #print(model)
    for param in model.parameters():
        # print(param.data.size())
        total_param += np.prod(list(param.data.size()))
    print('MODEL/Total parameters:', MODEL_NAME, total_param)
    return total_param


"""
    TRAINING CODE
"""

def train_val_pipeline(MODEL_NAME, dataset, params, net_params, dirs):
    start0 = time.time()
    per_epoch_time = []
    
    DATASET_NAME = dataset.name
    
    if MODEL_NAME in ['GCN', 'GAT']:
        if net_params['self_loop']:
            print("[!] Adding graph self-loops for GCN/GAT models (central node trick).")
            dataset._add_self_loops()
    
    if MODEL_NAME in ['GatedGCN']:
        if net_params['pos_enc']:
            print("[!] Adding graph positional encoding.")
            dataset._add_positional_encodings(net_params['pos_enc_dim'])
            print('Time PE:',time.time()-start0)
        
    trainset, valset, testset = dataset.train, dataset.val, dataset.test
        
    root_log_dir, root_ckpt_dir, write_file_name, write_config_file = dirs
    device = net_params['device']
    
    # Write network and optimization hyper-parameters in folder config/
    with open(write_config_file + '.txt', 'w') as f:
        f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n\nTotal Parameters: {}\n\n"""\
                .format(DATASET_NAME, MODEL_NAME, params, net_params, net_params['total_param']))
        
    log_dir = os.path.join(root_log_dir, "RUN_" + str(0))
    writer = SummaryWriter(log_dir=log_dir)

    # setting seeds
    random.seed(params['seed'])
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])
    if device.type == 'cuda':
        torch.cuda.manual_seed(params['seed'])
    
    print("Training Graphs: ", len(trainset))
    print("Validation Graphs: ", len(valset))
    print("Test Graphs: ", len(testset))
    print("Number of Classes: ", net_params['n_classes'])

    model = gnn_model(MODEL_NAME, net_params)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=params['lr_reduce_factor'],
                                                     patience=params['lr_schedule_patience'],
                                                     verbose=True)
    
    epoch_train_losses, epoch_val_losses = [], []
    epoch_train_accs, epoch_val_accs = [], [] 
    
    if MODEL_NAME in ['RingGNN', '3WLGNN']:
        # import train functions specific for WL-GNNs
        from train.train_SBMs_node_classification import train_epoch_dense as train_epoch, evaluate_network_dense as evaluate_network
        
        train_loader = DataLoader(trainset, shuffle=True, collate_fn=dataset.collate_dense_gnn)
        val_loader = DataLoader(valset, shuffle=False, collate_fn=dataset.collate_dense_gnn)
        test_loader = DataLoader(testset, shuffle=False, collate_fn=dataset.collate_dense_gnn)
        
    else:
        # import train functions for all other GCNs
        from train.train_SBMs_node_classification import train_epoch_sparse as train_epoch, evaluate_network_sparse as evaluate_network # import train functions
        
        train_loader = DataLoader(trainset, batch_size=params['batch_size'], shuffle=True, collate_fn=dataset.collate)
        val_loader = DataLoader(valset, batch_size=params['batch_size'], shuffle=False, collate_fn=dataset.collate)
        test_loader = DataLoader(testset, batch_size=params['batch_size'], shuffle=False, collate_fn=dataset.collate)
        
    # At any point you can hit Ctrl + C to break out of training early.
    try:
        with tqdm(range(params['epochs'])) as t:
            for epoch in t:

                t.set_description('Epoch %d' % epoch)

                start = time.time()

                if MODEL_NAME in ['RingGNN', '3WLGNN']: # since different batch training function for dense GNNs
                    epoch_train_loss, epoch_train_acc, optimizer = train_epoch(model, optimizer, device, train_loader, epoch, params['batch_size'])
                else:   # for all other models common train function
                    epoch_train_loss, epoch_train_acc, optimizer = train_epoch(model, optimizer, device, train_loader, epoch)
                    
                epoch_val_loss, epoch_val_acc = evaluate_network(model, device, val_loader, epoch)
                _, epoch_test_acc = evaluate_network(model, device, test_loader, epoch)        
                
                epoch_train_losses.append(epoch_train_loss)
                epoch_val_losses.append(epoch_val_loss)
                epoch_train_accs.append(epoch_train_acc)
                epoch_val_accs.append(epoch_val_acc)

                writer.add_scalar('train/_loss', epoch_train_loss, epoch)
                writer.add_scalar('val/_loss', epoch_val_loss, epoch)
                writer.add_scalar('train/_acc', epoch_train_acc, epoch)
                writer.add_scalar('val/_acc', epoch_val_acc, epoch)
                writer.add_scalar('test/_acc', epoch_test_acc, epoch)
                writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

                t.set_postfix(time=time.time()-start, lr=optimizer.param_groups[0]['lr'],
                              train_loss=epoch_train_loss, val_loss=epoch_val_loss,
                              train_acc=epoch_train_acc, val_acc=epoch_val_acc,
                              test_acc=epoch_test_acc)

                per_epoch_time.append(time.time()-start)

                # Saving checkpoint
                ckpt_dir = os.path.join(root_ckpt_dir, "RUN_")
                if not os.path.exists(ckpt_dir):
                    os.makedirs(ckpt_dir)
                torch.save(model.state_dict(), '{}.pkl'.format(ckpt_dir + "/epoch_" + str(epoch)))

                files = glob.glob(ckpt_dir + '/*.pkl')
                for file in files:
                    epoch_nb = file.split('_')[-1]
                    epoch_nb = int(epoch_nb.split('.')[0])
                    if epoch_nb < epoch-1:
                        os.remove(file)

                scheduler.step(epoch_val_loss)

                if optimizer.param_groups[0]['lr'] < params['min_lr']:
                    print("\n!! LR SMALLER OR EQUAL TO MIN LR THRESHOLD.")
                    break
                    
                # Stop training after params['max_time'] hours
                if time.time()-start0 > params['max_time']*3600:
                    print('-' * 89)
                    print("Max_time for training elapsed {:.2f} hours, so stopping".format(params['max_time']))
                    break
    
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early because of KeyboardInterrupt')
    
    
    _, test_acc = evaluate_network(model, device, test_loader, epoch)
    _, train_acc = evaluate_network(model, device, train_loader, epoch)
    print("Test Accuracy: {:.4f}".format(test_acc))
    print("Train Accuracy: {:.4f}".format(train_acc))
    print("Convergence Time (Epochs): {:.4f}".format(epoch))
    print("TOTAL TIME TAKEN: {:.4f}s".format(time.time()-start0))
    print("AVG TIME PER EPOCH: {:.4f}s".format(np.mean(per_epoch_time)))

    writer.close()

    """
        Write the results in out_dir/results folder
    """
    with open(write_file_name + '.txt', 'w') as f:
        f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n{}\n\nTotal Parameters: {}\n\n
    FINAL RESULTS\nTEST ACCURACY: {:.4f}\nTRAIN ACCURACY: {:.4f}\n\n
    Convergence Time (Epochs): {:.4f}\nTotal Time Taken: {:.4f} hrs\nAverage Time Per Epoch: {:.4f} s\n\n\n"""\
          .format(DATASET_NAME, MODEL_NAME, params, net_params, model, net_params['total_param'],
                  test_acc, train_acc, epoch, (time.time()-start0)/3600, np.mean(per_epoch_time)))
        

    
    
def set_all_params(dataset, MODEL_NAME, random_seed):
    
    n_heads = -1
    edge_feat = False
    pseudo_dim_MoNet = -1
    kernel = -1
    gnn_per_block = -1
    embedding_dim = -1
    pool_ratio = -1
    n_mlp_GIN = -1
    gated = False
    self_loop = False
    #self_loop = True
    max_time = 12
    pos_enc = True
    #pos_enc = False
    pos_enc_dim = 10
    
    if MODEL_NAME.startswith("GatedGCN"):
        if MODEL_NAME == "GatedGCN_E_PE":
            pos_enc = True
            pos_enc_dim = 10
        else:
            pos_enc = False
            pos_enc_dim = 10
        if MODEL_NAME == "GatedGCN_noE":
            edge_feat = False       
        MODEL_NAME="GatedGCN"
    

    if MODEL_NAME == 'GatedGCN':
        seed=41; epochs=1000; batch_size=5; init_lr=0.001; lr_reduce_factor=0.5; lr_schedule_patience=10; min_lr = 0.00001; weight_decay=0
        L=16; hidden_dim=70; out_dim=hidden_dim; dropout=0.0; readout='mean'

    if MODEL_NAME == 'GCN':
        seed=41; epochs=1000; batch_size=5; init_lr=0.001; lr_reduce_factor=0.5; lr_schedule_patience=10; min_lr = 0.00001; weight_decay=0
        L=16; hidden_dim=146; out_dim=hidden_dim; dropout=0.0; readout='mean'

    if MODEL_NAME == 'GAT':
        seed=41; epochs=1000; batch_size=50; init_lr=0.001; lr_reduce_factor=0.5; lr_schedule_patience=10; min_lr = 0.00001; weight_decay=0
        L=16; n_heads=8; hidden_dim=19; out_dim=n_heads*hidden_dim; dropout=0.0; readout='mean'
        print('True hidden dim:',out_dim)

    if MODEL_NAME == 'GraphSage':
        seed=41; epochs=1000; batch_size=50; init_lr=0.001; lr_reduce_factor=0.5; lr_schedule_patience=10; min_lr = 0.00001; weight_decay=0
        L=16; hidden_dim=108; out_dim=hidden_dim; dropout=0.0; readout='mean'


    if MODEL_NAME == 'MoNet':
        seed=41; epochs=1000; batch_size=50; init_lr=0.001; lr_reduce_factor=0.5; lr_schedule_patience=10; min_lr = 0.00001; weight_decay=0
        L=16; hidden_dim=90; out_dim=hidden_dim; dropout=0.0; readout='mean'
        pseudo_dim_MoNet=2; kernel=3;
        
      

        
    # generic new_params
    net_params = {}
    #net_params['in_dim'] = torch.unique(trainset[0][0].ndata['feat'],dim=0).size(0) # node_dim (feat is an integer)
    net_params['in_dim'] = dataset.train[0][0].ndata['feat'][0].shape[0]
    net_params['hidden_dim'] = hidden_dim
    net_params['out_dim'] = out_dim
    num_classes = torch.unique(dataset.train[0][1],dim=0).size(0)
    net_params['n_classes'] = num_classes
    net_params['L'] = L  # min L should be 2
    net_params['readout'] = "mean"
    net_params['layer_norm'] = True
    net_params['batch_norm'] = True
    net_params['in_feat_dropout'] = 0.0
    net_params['dropout'] = 0.0
    net_params['residual'] = True
    net_params['edge_feat'] = edge_feat
    net_params['self_loop'] = self_loop
    
    # for MLPNet 
    net_params['gated'] = gated  
    
    # for GAT
    net_params['n_heads'] = n_heads 
    
    # for graphsage
    net_params['sage_aggregator'] = 'meanpool' 

    # specific for GIN
    net_params['n_mlp_GIN'] = n_mlp_GIN
    net_params['learn_eps_GIN'] = True
    net_params['neighbor_aggr_GIN'] = 'sum'
    
    # specific for MoNet
    net_params['pseudo_dim_MoNet'] = pseudo_dim_MoNet
    net_params['kernel'] = kernel
    
    # specific for RingGNN
    net_params['radius'] = 2
    num_nodes = [dataset.train[i][0].number_of_nodes() for i in range(len(dataset.train))]
    net_params['avg_node_num'] = int(np.ceil(np.mean(num_nodes)))
    
    # specific for 3WLGNN
    net_params['depth_of_mlp'] = 2
    
    # specific for pos_enc_dim
    net_params['pos_enc'] = pos_enc
    net_params['pos_enc_dim'] = pos_enc_dim
    
    params = {}
    params['seed'] = random_seed
    params['epochs'] = epochs
    params['batch_size'] = batch_size
    params['init_lr'] = init_lr
    params['lr_reduce_factor'] = lr_reduce_factor 
    params['lr_schedule_patience'] = lr_schedule_patience
    params['min_lr'] = min_lr
    params['weight_decay'] = weight_decay
    params['print_epoch_interval'] = 5
    params['max_time'] = max_time
        
    return params, net_params


        

def perform_pattern_classification(DATASET_NAME, MODEL_NAME, random_seed):
    config = {}
    # gpu config: uncomment the second pair of lines to train on gpu
    use_gpu = False
    gpu_id = -1
    #use_gpu = True
    #gpu_id = 0
    gpu = {}
    gpu['use'] = use_gpu
    gpu['id'] = gpu_id
    config['gpu'] = gpu
    device = gpu_setup(config['gpu']['use'], config['gpu']['id'])
    
    out_dir = 'out/SBMs_node_classification/PATTERN/'
    
    root_log_dir = out_dir + 'logs/' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(config['gpu']['id']) + "_seed" + str(random_seed) + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    root_ckpt_dir = out_dir + 'checkpoints/' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" +str(random_seed)+ str(config['gpu']['id']) + "_seed" + str(random_seed) + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    write_file_name = out_dir + 'results/result_' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(config['gpu']['id']) + "_seed" + str(random_seed) + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    write_config_file = out_dir + 'configs/config_' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(config['gpu']['id']) + "_seed" + str(random_seed) + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    dirs = root_log_dir, root_ckpt_dir, write_file_name, write_config_file
    if not os.path.exists(out_dir + 'results'):
        os.makedirs(out_dir + 'results')
        
    if not os.path.exists(out_dir + 'configs'):
        os.makedirs(out_dir + 'configs')
    
    
    print("[I] Loading data (notebook) ...")
    dataset = LoadData(DATASET_NAME)    
    trainset, valset, testset = dataset.train, dataset.val, dataset.test
    print("[I] Finished loading.")
    
    params, net_params = set_all_params(dataset, MODEL_NAME, random_seed)
    
    if MODEL_NAME.startswith("GatedGCN"):      
        MODEL_NAME="GatedGCN"
    net_params['device'] = device
    net_params['gpu_id'] = config['gpu']['id']
    net_params['batch_size'] = params['batch_size']
    
    net_params['in_dim'] = dataset.train.graph_lists[0].ndata['feat'][0].shape[0]
    net_params['total_param'] = view_model_param(MODEL_NAME, net_params)
    train_val_pipeline(MODEL_NAME, dataset, params, net_params, dirs)
    
