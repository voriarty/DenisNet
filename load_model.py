

import torch
import os
#加载断点
def load_checkpoint(model_path,resume=True,):
    if resume:
        assert os.path.isfile(model_path)
        checkpoint=torch.load(model_path)
        least_error=checkpoint['least_prec1']
        start_epoch=checkpoint['epoch']
        # optimizer.load_state_dict(checkpoint['optimizer'])
        print('Load checkpoint at epoch %d.'%start_epoch)
        print('least_error:%.4f'%least_error)
    else:
        print("Do not load checkpoint,Epoch start from 0")
        start_epoch=0
    return start_epoch

model_path='runs/DenseNet-BC-196-12-10-dr=0.2/model_best.pth.tar'
# model_path='runs/DenseNet-BC-SE-169-12-100/checkpoint.pth.tar'
load_checkpoint(model_path,resume=True)
