# CURBD with NeuroTorch

This code is used to reproduce the results of the CURBD repository (https://github.com/rajanlab/CURBD) associated
with the paper [Inferring brain-wide interactions using data-constrained recurrent neural network models](https://doi.org/10.1101/2020.12.18.423348)
from Matthew G. Perich and al. using the NeuroTorch library.


## Installation

To run the code, you need to install the requirements in requirements.txt.
```shell
pip install -r requirements.txt
```

## Usage

Then, you can run the code by running the following command:
```shell
python main.py
```


## Structure of the repository

The repository is structured as follows:
- `main.py`: main file to run the code
- `curbd_dataset.py`: dataset class to construct the curbd synthetic dataset
- `curbd_training.py`: training function
- `figures_script.py`: script with functions to generate the figures
- `utils.py`: utility functions
- `ts_dataset.py`: dataset class to construct a time series dataset from a npy file
- `main_other_dataset.py`: main file to run the code on other datasets



