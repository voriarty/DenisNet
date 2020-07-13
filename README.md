# DenisNet
Densely Connected and Inter-Sparse Convolutional Networks with aggregated Squeeze-and-Excitation transformations (DenisNet-SE)


The detault setting for this model is a DenisNet-SE, 100 layers, a growth rate of 12 and batch size 64,group 1.


Example usage with optional arguments for different hyperparameters (e.g., DenisNet-SE-40-g=2):
```sh
$ python train.py --layers 40   --group 2 --name DenisNet-SE-40-g=2
```


## Prerequisites

- Python 3.6
- GPU Memory 11g
- Numpy
- Pytorch 1.0+

