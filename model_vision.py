import torchvision
import torch
from tensorboardX import SummaryWriter
from DCRA.model.dcra import *


dummy_input=torch.rand(13,3,28,28)
# model=DCRA(depth=40,num_classes=10,SE_model=False,SE_Residual_model=True,bottleneck=False,num_group=2)
model=DCRA(depth=40,num_classes=10,SE_model=True,SE_Residual_model=False,bottleneck=False,num_group=6,Reducation=2)
with SummaryWriter(log_dir='../DCRA/runs/NET/Densenext-BC-SE-40-C-6',comment='Densenext-BC-SE-40-C-6') as w:
    w.add_graph(model,(dummy_input,))