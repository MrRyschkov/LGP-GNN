import CLUSTER_Node_Classification as cnc

'''
The following function allows you to train the provided models with a feature-augmented version of the datasets. 

    cnc.perform_cluster_classification(Dataset, Model, random_seed)

The available options for the dataset are: "SBM_CLUSTER", "SBM_CLUSTER_3Cy", "SBM_CLUSTER_4Cy", "SBM_CLUSTER_4Cl",
"SBM_CLUSTER_5Cl", "SBM_CLUSTER_34Cy", "SBM_CLUSTER_34Cl", "SBM_CLUSTER_345Cl"

The available models are "GAT", "GCN", "MoNet", "GraphSage" and "GatedGCN_E_PE".

A random seed of choise needs to be chosen as well (the results in the paper are obtained by averaging over 41, 42, 43 and 44).

An example (to reproduce the best results for the GraphSage model) is provided below:
'''

cnc.perform_cluster_classification("SBM_CLUSTER_4Cy", "GraphSage", 44)
