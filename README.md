# LGP-GNN
Downloadable code for the experiments in the paper "Graph Neural Networks with Local Graph Parameters". Our implementation is based on the implementation of the benchmarking experiments of the paper "Benchmarking Graph Neural Networks" by Dwivedi et al. (<https://github.com/graphdeeplearning/benchmarking-gnns>), the corresponding licence is included.

Steps to run code:
1. Install the necessary libraries. We advise creating a separate environment using the conda package and environment manager. <https://docs.conda.io/projects/conda/en/latest/user-guide/index.html>.

```
        conda create -n lgp_gnn python=3.7
        conda activate lgp_gnn
        conda install -c dglteam dgl-cuda10.2 #For Mac OS: conda install -c dglteam dgl
        conda install pytorch==1.6.0 torchvision==0.7.0 -c pytorch
        pip install -r requirements.txt
```

> :warning: **If you want to train the models on GPU: these installs of pytorch (1.6.0) and dgl (0.5.3) will work for CUDA 10.2**: If you want to work with another version of the cuda-toolkit on your system or device you should get the corresponding versions and change the commands accordingly. See <https://pytorch.org/get-started/previous-versions/> and <https://docs.dgl.ai/en/0.4.x/install/> for examples. 
   
2. Download the datasets from the ZIP archives in the following link. Unzip and place the .pkl and .pt files in the directories specified below:

<https://www.dropbox.com/sh/o2kyoyvh9qodi2r/AABx36JbiijPucYO2g4l61j8a?dl=0>

* The ZINC files should be placed in ./data/molecules
* The COLLAB files in ./data/COLLAB
* The CLUSTER and PATTERN files in ./data/SBMs
 
> :warning:  All the different versions of all datasets will take up a lot of space on your disk. Separate links are provided for every dataset.

3. Get the train-val-test pipeline running as in one of the .py scripts. Examples are provided to reproduce the best results for every dataset. Notebooks are also provided for every .py example script.

```
python ZINC_Example.py
```

> :warning: **If you want to train the models on GPU**: Uncomment the lines concerning use_gpu and gpu_id in ZINC_Regression.py, COLLAB_Prediction.py, CLUSTER_classification.py and PATTERN_classification.py.


Please reach out to <lgpgnn2021@gmail.com> for questions and remarks.

