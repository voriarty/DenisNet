# DenisNet
Densely Connected and Inter-Sparse Convolutional Networks with aggregated Squeeze-and-Excitation transformations (DenisNet-SE)


The detault setting for this model is a DenisNet-SE, 100 layers, a growth rate of 12 and batch size 64,group 1.


Example usage with optional arguments for different hyperparameters (e.g., DenisNet-SE-40-g=2):
```sh
$ python train.py --layers 40   --group 2 --name DenisNet-SE-40-g=2
```
`--epochs` number of total epochs to run

`start-epoch` manual epoch number (useful on restarts)

`--batchsiz` batch size.

`--lr` initial learning rate

`--growth` number of new channels per layer (default: 12)

`--layers` total number of layers.(default: 100)

`--droprate` dropout probability (default: 0.0)

`--reduce` compression rate in transition stage (default: 0.5)

`--resume` path to latest checkpoint (default: none)

`--name` name of experiment.

`--no-se_model` To not use SE block.

`--se_re_model` store_true',help='use SE_residual block

`--reducation` Reduce channel in Se_model

`--group` num of group Conv

`--tensorboard` Log progress to TensorBoard', action='store_true

## Prerequisites

- Python 3.6
- GPU Memory 11g
- Numpy
- Pytorch 1.0+

