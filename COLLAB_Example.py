import COLLAB_Prediction as cp

'''
The following function allows you to train the provided models with a feature-augmented version of the datasets. 

    zr.perform_collab_prediction(Dataset, Model, random_seed)

The available options for the dataset are: "OGBL-COLLAB", "OGBL-COLLAB_3Cl", "OGBL-COLLAB_4Cl",
"OGBL-COLLAB_5Cl", "OGBL-COLLAB_34Cl", "OGBL-COLLAB_345Cl"

The available models are "GAT", "GCN", "MoNet", "GraphSage" and "GatedGCN_E_PE".

A random seed of choise needs to be chosen as well (the results in the paper are obtained by averaging over 41, 42, 43 and 44).

An example (to reproduce the best results for the  model) is provided below:
'''

cp.perform_collab_prediction("OGBL-COLLAB_5Cl", "GCN", 43)
