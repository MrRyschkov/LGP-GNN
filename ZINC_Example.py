import ZINC_Regression as zr

'''

The following function allows you to train the provided models with a feature-augmented version of the datasets. 

    zr.perform_zinc_regression(Dataset, Model, random_seed)

The available options for the dataset are: "ZINC_nofeat", "ZINC_3Cy", "ZINC_4Cy", "ZINC_5Cy", "ZINC_6Cy", "ZINC_34Cy", "ZINC_56Cy", "ZINC_456Cy", "ZINC_3456Cy"

The available models are "GAT", "GCN", "MoNet", "GraphSage" and "GatedGCN_E_PE".

A random seed of choise needs to be chosen as well (the results in the paper are obtained by using 41, 42, 43 and 44.
An example (to reproduce the best results for the GAT model) is provided below:

'''

zr.perform_zinc_regression("ZINC_3456", "GAT", 41)
