import torchvision.datasets as dsets
from Test.Function import *

# print(numpy.__version__)
# print(tensorflow.__version__)

test_dataset = dsets.CIFAR10(root='../data/',
                             train=False,
                             transform=test_transform)

loss_fn=F.cross_entropy
device=get_default_device()
model=torchvision.models.DenseNet(num_classes=10).cuda()


#加载训练效果最好的模型
model,start_epoch=load_checkpoint(model,Train=False)
test_loader=DataLoader(test_dataset,batch_size=200)
test_loader=DeviceDataLoader(test_loader,device)
test_loss,total,test_err,test_time=test_model(model,loss_fn,test_loader,metric=error)
print('Test:Loss:{:.4f},Error:{:.3f},Total:{:d},Testtime:{:.3f}'.format(test_loss,test_err,total,test_time))

