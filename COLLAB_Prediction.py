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
from nets.COLLAB_edge_classification.load_net import gnn_model # import all GNNS
from data.data import LoadData # import dataset

"""
    GPU Setup
"""
def gpu_setup(use_gpu, gpu_id):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id) 
    print("Device Count", torch.cuda.device_count())
    if torch.cuda.is_available() and use_gpu:

        device=torch.device("cuda")
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
    t0 = time.time()
    per_epoch_time = []
        
    DATASET_NAME = dataset.name
    
    #assert net_params['self_loop'] == False, "No self-loop support for %s dataset" % DATASET_NAME
    
    if MODEL_NAME in ['GatedGCN']:
        if net_params['pos_enc']:
            print("[!] Adding graph positional encoding",net_params['pos_enc_dim'])
            dataset._add_positional_encodings(net_params['pos_enc_dim'])
            print('Time PE:',time.time()-t0)
        
    graph = dataset.graph
    
    evaluator = dataset.evaluator
    
    train_edges, val_edges, val_edges_neg, test_edges, test_edges_neg = dataset.train_edges, dataset.val_edges, dataset.val_edges_neg, dataset.test_edges, dataset.test_edges_neg
        
    root_log_dir, root_ckpt_dir, write_file_name, write_config_file = dirs
    device = net_params['device']
    
    # Write the network and optimization hyper-parameters in folder config/
    with open(write_config_file + '.txt', 'w') as f:
        f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n\nTotal Parameters: {}\n\n"""                .format(DATASET_NAME, MODEL_NAME, params, net_params, net_params['total_param']))
        
    log_dir = os.path.join(root_log_dir, "RUN_" + str(0))
    writer = SummaryWriter(log_dir=log_dir)

    # setting seeds
    random.seed(params['seed'])
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])
    if device.type == 'cuda':
        torch.cuda.manual_seed(params['seed'])
   
    print("Graph: ", graph)
    print("Training Edges: ", len(train_edges))
    print("Validation Edges: ", len(val_edges) + len(val_edges_neg))
    print("Test Edges: ", len(test_edges) + len(test_edges_neg))

    model = gnn_model(MODEL_NAME, net_params)
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                     factor=params['lr_reduce_factor'],
                                                     patience=params['lr_schedule_patience'],
                                                     verbose=True)
    
    epoch_train_losses = []
    epoch_train_hits, epoch_val_hits = [], [] 
    
    if MODEL_NAME in ['RingGNN', '3WLGNN']:
        raise NotImplementedError # gave OOM while preparing dense tensor
    else:
        # import train functions for all other GCNs
        from train.train_COLLAB_edge_classification import train_epoch_sparse as train_epoch, evaluate_network_sparse as evaluate_network
    
    # At any point you can hit Ctrl + C to break out of training early.
    try:
        monet_pseudo = None
        if MODEL_NAME == "MoNet":
            '''
            print("\nPre-computing MoNet pseudo-edges")
            # for MoNet: computing the 'pseudo' named tensor which depends on node degrees
            us, vs = graph.edges()
            # to avoid zero division in case in_degree is 0, we add constant '1' in all node degrees denoting self-loop
            monet_pseudo = [ 
                [1/np.sqrt(graph.in_degrees(us[i])+1), 1/np.sqrt(graph.in_degrees(vs[i])+1)] 
                    for i in range(graph.number_of_edges())
            ]
            monet_pseudo = torch.Tensor(monet_pseudo)
            torch.save(monet_pseudo, 'collab_monet_pseudo.pt')
            '''
            monet_pseudo = torch.load('collab_monet_pseudo.pt')
        with tqdm(range(params['epochs'])) as t:
            for epoch in t:

                t.set_description('Epoch %d' % epoch)    

                start = time.time()
                    
                epoch_train_loss, optimizer = train_epoch(model, optimizer, device, graph, train_edges, params['batch_size'], epoch, monet_pseudo)
                
                epoch_train_hits, epoch_val_hits, epoch_test_hits = evaluate_network(
                    model, device, graph, train_edges, val_edges, val_edges_neg, test_edges, test_edges_neg, evaluator, params['batch_size'], epoch, monet_pseudo)
                
                epoch_train_losses.append(epoch_train_loss)
                epoch_train_hits.append(epoch_train_hits)
                epoch_val_hits.append(epoch_val_hits)

                writer.add_scalar('train/_loss', epoch_train_loss, epoch)
                
                writer.add_scalar('train/_hits@10', epoch_train_hits[0]*100, epoch)
                writer.add_scalar('train/_hits@50', epoch_train_hits[1]*100, epoch)
                writer.add_scalar('train/_hits@100', epoch_train_hits[2]*100, epoch)
                
                writer.add_scalar('val/_hits@10', epoch_val_hits[0]*100, epoch)
                writer.add_scalar('val/_hits@50', epoch_val_hits[1]*100, epoch)
                writer.add_scalar('val/_hits@100', epoch_val_hits[2]*100, epoch)
                
                writer.add_scalar('test/_hits@10', epoch_test_hits[0]*100, epoch)
                writer.add_scalar('test/_hits@50', epoch_test_hits[1]*100, epoch)
                writer.add_scalar('test/_hits@100', epoch_test_hits[2]*100, epoch)
                
                writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)   

                t.set_postfix(time=time.time()-start, lr=optimizer.param_groups[0]['lr'],
                              train_loss=epoch_train_loss, train_hits=epoch_train_hits[1], 
                              val_hits=epoch_val_hits[1], test_hits=epoch_test_hits[1]) 

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

                scheduler.step(epoch_val_hits[1])

                if optimizer.param_groups[0]['lr'] < params['min_lr']:
                    print("\n!! LR EQUAL TO MIN LR SET.")
                    break
                    
                # Stop training after params['max_time'] hours
                if time.time()-t0 > params['max_time']*3600:
                    print('-' * 89)
                    print("Max_time for training elapsed {:.2f} hours, so stopping".format(params['max_time']))
                    break
    
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early because of KeyboardInterrupt')
    
    
    train_hits, val_hits, test_hits = evaluate_network(
        model, device, graph, train_edges, val_edges, val_edges_neg, test_edges, test_edges_neg, evaluator, params['batch_size'], epoch, monet_pseudo)

    print(f"Test:\nHits@10: {test_hits[0]*100:.4f}% \nHits@50: {test_hits[1]*100:.4f}% \nHits@100: {test_hits[2]*100:.4f}% \n")
    print(f"Train:\nHits@10: {train_hits[0]*100:.4f}% \nHits@50: {train_hits[1]*100:.4f}% \nHits@100: {train_hits[2]*100:.4f}% \n")
    print("Convergence Time (Epochs): {:.4f}".format(epoch))
    print("TOTAL TIME TAKEN: {:.4f}s".format(time.time()-t0))
    print("AVG TIME PER EPOCH: {:.4f}s".format(np.mean(per_epoch_time)))

    writer.close()

    """
        Write the results in out_dir/results folder
    """
    with open(write_file_name + '.txt', 'w') as f:
        f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n{}\n\nTotal Parameters: {}\n\n
    FINAL RESULTS\nTEST HITS@10: {:.4f}\nTEST HITS@50: {:.4f}\nTEST HITS@100: {:.4f}\nTRAIN HITS@10: {:.4f}\nTRAIN HITS@50: {:.4f}\nTRAIN HITS@100: {:.4f}\n\n
    Convergence Time (Epochs): {:.4f}\nTotal Time Taken: {:.4f}hrs\nAverage Time Per Epoch: {:.4f}s\n\n\n"""\
          .format(DATASET_NAME, MODEL_NAME, params, net_params, model, net_params['total_param'],
                  test_hits[0]*100, test_hits[1]*100, test_hits[2]*100, train_hits[0]*100, train_hits[1]*100, train_hits[2]*100,
                  epoch, (time.time()-t0)/3600, np.mean(per_epoch_time)))
    
    
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
    max_time = 12
    layer_type = 'dgl'
    num_embs = -1
    pos_enc = True
    #pos_enc = False
    pos_enc_dim = 10

    
    if MODEL_NAME.startswith("GatedGCN"):
        if MODEL_NAME == "GatedGCN_E_PE":
            pos_enc = True
            pos_enc_dim = 10
        else:
            pos_enc = False
        if MODEL_NAME == "GatedGCN_noE":
            edge_feat = False       
        MODEL_NAME="GatedGCN"
    
    if MODEL_NAME == 'MF':
        seed=random_seed; epochs=500; batch_size=32*1024; init_lr=0.01; lr_reduce_factor=0.5; lr_schedule_patience=10; min_lr = 1e-5; weight_decay=0
        L=0; hidden_dim=256; out_dim=hidden_dim; num_embs=235868;
    
    if MODEL_NAME == 'MLP':
        seed=random_seed; epochs=500; batch_size=32*1024; init_lr=0.001; lr_reduce_factor=0.5; lr_schedule_patience=10; min_lr = 1e-5; weight_decay=0
        L=3; hidden_dim=80; out_dim=hidden_dim; dropout=0.0; readout='mean'; gated = False  # Change gated = True for Gated MLP model
    
    if MODEL_NAME == 'GCN':
        seed=random_seed; epochs=500; batch_size=32*1024; init_lr=0.001; lr_reduce_factor=0.5; lr_schedule_patience=10; min_lr = 1e-5; weight_decay=0
        L=10; hidden_dim=74; out_dim=hidden_dim; dropout=0.0; readout='mean';
        
    if MODEL_NAME == 'GraphSage':
        seed=random_seed; epochs=500; batch_size=32*1024; init_lr=0.001; lr_reduce_factor=0.5; lr_schedule_patience=10; min_lr = 1e-5; weight_decay=0
        L=3; hidden_dim=38; out_dim=hidden_dim; dropout=0.0; readout='mean'; layer_type='edgefeat'

    if MODEL_NAME == 'GAT':
        seed=random_seed; epochs=500; batch_size=32*1024; init_lr=0.001; lr_reduce_factor=0.5; lr_schedule_patience=10; min_lr = 1e-5; weight_decay=0
        L=3; n_heads=3; hidden_dim=19; out_dim=n_heads*hidden_dim; dropout=0.0; readout='mean'; layer_type='dgl'
    
    if MODEL_NAME == 'GIN':
        seed=random_seed; epochs=500; batch_size=32*1024; init_lr=0.001; lr_reduce_factor=0.5; lr_schedule_patience=10; min_lr = 1e-5; weight_decay=0
        L=3; hidden_dim=60; out_dim=hidden_dim; dropout=0.0; readout='mean';
        
    if MODEL_NAME == 'MoNet':
        seed=random_seed; epochs=500; batch_size=32*1024; init_lr=0.001; lr_reduce_factor=0.5; lr_schedule_patience=10; min_lr = 1e-5; weight_decay=0
        L=3; hidden_dim=53; out_dim=hidden_dim; dropout=0.0; readout='mean';
        
    if MODEL_NAME == 'GatedGCN':
        seed=random_seed; epochs=500; batch_size=32*1024; init_lr=0.001; lr_reduce_factor=0.5; lr_schedule_patience=10; min_lr = 1e-5; weight_decay=0
        L=3; hidden_dim=35; out_dim=hidden_dim; dropout=0.0; readout='mean'; edge_feat = False; layer_type='edgereprfeat'
        
    # generic new_params
    net_params = {}
    net_params['in_dim'] = dataset.graph.ndata['feat'].shape[-1]
    net_params['in_dim_edge'] = dataset.graph.edata['feat'].shape[-1]
    net_params['residual'] = True
    net_params['hidden_dim'] = hidden_dim
    net_params['out_dim'] = out_dim
    num_classes = 1
    net_params['n_classes'] = num_classes
    net_params['n_heads'] = n_heads
    net_params['L'] = L  # min L should be 2
    net_params['readout'] = "mean"
    net_params['layer_norm'] = True
    net_params['batch_norm'] = True
    net_params['in_feat_dropout'] = 0.0
    net_params['dropout'] = 0.0
    net_params['edge_feat'] = edge_feat
    net_params['self_loop'] = self_loop
    net_params['layer_type'] = layer_type
    
    # for MF
    net_params['num_embs'] = num_embs
    
    # for MLPNet 
    net_params['gated'] = gated
    
    # specific for MoNet
    net_params['pseudo_dim_MoNet'] = 2
    net_params['kernel'] = 3
    
    # specific for GIN
    net_params['n_mlp_GIN'] = 2
    net_params['learn_eps_GIN'] = True
    net_params['neighbor_aggr_GIN'] = 'sum'
    
    # specific for graphsage
    net_params['sage_aggregator'] = 'maxpool'   
    
    # specific for pos_enc_dim
    net_params['pos_enc'] = pos_enc
    net_params['pos_enc_dim'] = pos_enc_dim

   
    
    params = {}
    params['seed'] = seed
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


        

def perform_collab_prediction(DATASET_NAME, MODEL_NAME, random_seed):
    config = {}
    # gpu config
    gpu_id = 0
    use_gpu = True
    device = None
    gpu = {}
    gpu['use'] = use_gpu
    gpu['id'] = gpu_id
    config['gpu'] = gpu
    device = gpu_setup(config['gpu']['use'], config['gpu']['id'])
    
    out_dir = 'out/COLLAB_edge_classification/debug/'
    
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
    print("[I] Finished loading.")
    
    params, net_params = set_all_params(dataset, MODEL_NAME, random_seed)
    
    if MODEL_NAME.startswith("GatedGCN"):      
        MODEL_NAME="GatedGCN"
    net_params['device'] = device
    net_params['gpu_id'] = config['gpu']['id']
    net_params['batch_size'] = params['batch_size']
    
    net_params['in_dim'] = dataset.graph.ndata['feat'][0].shape[0]
    net_params['total_param'] = view_model_param(MODEL_NAME, net_params)
    train_val_pipeline(MODEL_NAME, dataset, params, net_params, dirs)
    
