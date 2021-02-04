import PATTERN_Node_Classification as pnc

'''
The following function allows you to train the provided models with a feature-augmented version of the datasets. 

    pnc.perform_pattern_classification(Dataset, Model, random_seed)

The available options for the dataset are: "SBM_PATTERN", "SBM_PATTERN_3Cy", "SBM_PATTERN_4Cy", "SBM_PATTERN_4Cl",
"SBM_PATTERN_5Cl", "SBM_PATTERN_34Cy", "SBM_PATTERN_34Cl", "SBM_PATTERN_345Cl"

The available models are "GAT", "GCN", "MoNet", "GraphSage" and "GatedGCN_E_PE".

A random seed of choise needs to be chosen as well (the results in the paper are obtained by averaging over 41, 42, 43 and 44).

An example (to reproduce the best results for the MoNet model) is provided below:
'''

pnc.perform_pattern_classification("SBM_PATTERN_345Cl", "MoNet", 42)
