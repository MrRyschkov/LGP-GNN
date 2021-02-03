# LGP-GNN
Downloadable code for the experiments in the paper "Graph Neural Networks with Local Graph Parameters"

Steps to run code:
1. Install dependencies from requirements.txt. If working with conda this can be done in the following way:

```
        conda create --name <env_name> --file requirements.txt
```
   Without conda this can be done as in the following guide: https://itsfoss.com/python-setup-linux/
2. Run download_datasets.sh (located in ./data) 
-- 
**_NOTE:_**  All the different versions of all datasets will take up a lot of space on your disk. Separate scripts are provided for every dataset.
--
3. Get the train-val-test pipeline running as in one of the notebooks. Examples are provided to reproduce the best results for every dataset. The notebooks are also provided as plain .py scripts.

