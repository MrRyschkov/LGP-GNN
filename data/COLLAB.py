import time
import dgl
import torch
from torch.utils.data import Dataset

from ogb.linkproppred import DglLinkPropPredDataset, Evaluator

from scipy import sparse as sp
import numpy as np


def positional_encoding(g, pos_enc_dim):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """
    
    # Laplacian
    A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # # Eigenvectors with numpy
    # EigVal, EigVec = np.linalg.eig(L.toarray())
    # idx = EigVal.argsort() # increasing order
    # EigVal, EigVec = EigVal[idx], np.real(EigVec[:,idx])
    # g.ndata['pos_enc'] = torch.from_numpy(np.abs(EigVec[:,1:pos_enc_dim+1])).float() 

    # Eigenvectors with scipy
    #EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR')
    EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR', tol=1e-2)
    EigVec = EigVec[:, EigVal.argsort()] # increasing order
    g.ndata['pos_enc'] = torch.from_numpy(np.real(EigVec[:,1:pos_enc_dim+1])).float() 

    return g


class COLLABDataset(Dataset):
    def __init__(self, name):
        start = time.time()
        print("[I] Loading dataset %s..." % (name))
        self.name = name
        self.dataset = DglLinkPropPredDataset(name='ogbl-collab')
        
        self.graph = self.dataset[0]  # single DGL graph
        
        # Create edge feat by concatenating weight and year
        self.graph.edata['feat'] = torch.cat( 
            [self.graph.edata['edge_weight'], self.graph.edata['edge_year']], 
            dim=1 
        )
        
        self.split_edge = self.dataset.get_edge_split()
        self.train_edges = self.split_edge['train']['edge']  # positive train edges
        self.val_edges = self.split_edge['valid']['edge']  # positive val edges
        self.val_edges_neg = self.split_edge['valid']['edge_neg']  # negative val edges
        self.test_edges = self.split_edge['test']['edge']  # positive test edges
        self.test_edges_neg = self.split_edge['test']['edge_neg']  # negative test edges
        
        self.evaluator = Evaluator(name='ogbl-collab')
        
        if self.name == "OGBL-COLLAB_3Cy":
            self.graph.ndata['feat'] = torch.load('data/COLLAB/collab_triangle_homs_norm.pt')
        elif self.name == "OGBL-COLLAB_4Cy":
            self.graph.ndata['feat'] = torch.load('data/COLLAB/collab_4cycle_homs_norm.pt')
        elif self.name == "OGBL-COLLAB_34Cy":
            self.graph.ndata['feat'] = torch.load('data/COLLAB/collab_34cycle_homs_norm.pt')
        elif self.name == "OGBL-COLLAB_4Cl":
            self.graph.ndata['feat'] = torch.load('data/COLLAB/collab_4clique_homs_norm.pt')
        elif self.name == "OGBL-COLLAB_34Cl":
            self.graph.ndata['feat'] = torch.load('data/COLLAB/collab_34clique_homs_norm.pt')
        elif self.name == "OGBL-COLLAB_5Cl":
            self.graph.ndata['feat'] = torch.load('data/COLLAB/collab_5clique_homs_norm.pt')
        elif self.name == "OGBL-COLLAB_345Cl":
            self.graph.ndata['feat'] = torch.load('data/COLLAB/collab_345clique_homs_norm.pt')
            
        print("[I] Finished loading.")
        print("[I] Data load time: {:.4f}s".format(time.time()-start))

    def _add_positional_encodings(self, pos_enc_dim):
        
        # Graph positional encoding v/ Laplacian eigenvectors
        self.graph = positional_encoding(self.graph, pos_enc_dim)
