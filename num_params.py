from torchsummary import summary
from model.densenet import *
from model.dcra import *
from model.Se import DenseSe
from model.SEnet import *
import torchvision

# ,num_group=6
# model=se_resnet164().cuda()
# model=torchvision.models.resnet50()
model=DCRA(depth=220,num_classes=10,SE_model=True,SE_Residual_model=False,bottleneck=False,num_group=6,Reducation=6)
# model=DenseSe(depth=142,num_classes=10,SE_model=True,SE_Residual_model=False,bottleneck=True,num_group=1,Reducation=6).cuda()
# model=DenseNet3(depth=100,num_classes=10).cuda()
# model=torchvision.models.resnet50().cuda()
# print('Number of model parameters: {}'.format(
#         sum([p.data.nelement() for p in model.parameters()])))

summary(model,(3,256,128))