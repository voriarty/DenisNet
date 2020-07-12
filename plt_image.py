import csv
import matplotlib.pyplot as plt
import numpy as np

# 读取文件
def Readtext(textfile):
    fp = open(textfile, 'r')
    data = []
    for line in fp.readlines():
        data.append(round(float(line),2))
    return data

#绘制图
def Image(label1,label2,train1,valid1,train2,valid2,image):
    x=range(len(train1))
    # train_label_1=label1+'_train'
    # train_label_2=label2+'_train'
    #
    # valid_label_1=label1+'_valid'
    # valid_label_2=label2+'_valid'

    train_label_1 ='DenseNet-BC-28(1x48d)train'
    train_label_2 ='DenisNet-40(6x8d)train'
    valid_label_1 ='DenseNet-BC-28(1x48d)val'
    valid_label_2 ='DenisNet-40(6x8d)val'
    y1 = train1
    y2=train2

    z1 =valid1
    z2 = valid2

    plt.plot(x, y1, label=train_label_1, color='b', linestyle='--')
    plt.plot(x, z1, label=valid_label_1, color='b', linestyle='-')
    plt.plot(x, y2, label=train_label_2, color='r', linestyle='--')
    plt.plot(x, z2, label=valid_label_2, color='r', linestyle='-')

    plt.xlabel('epochs')
    plt.ylabel('error(%)')
    plt.legend()
    plt.savefig(image)
    plt.show()

# textfile_1='DenseNet-BC-28-12-100'
textfile_1='DenseNet-BC-28-12-10'
data_1=Readtext('../DCRA/Text/DenseNext-C-Group/'+textfile_1+'_train.txt')
data_2=Readtext('../DCRA/Text/DenseNext-C-Group/'+textfile_1+'_test.txt')


# textfile_2='DenseNet-BC-40-12-100-C-6-n=3'
textfile_2='DenseNet-BC-40-12-10-C-6-n=3'
data_3=Readtext('../DCRA/Text/DenseNext-C-Group/'+textfile_2+'_train.txt')
data_4=Readtext('../DCRA/Text/DenseNext-C-Group/'+textfile_2+'_test.txt')

# Image_save='../DCRA/image/DenisNet-100.eps'
Image_save='../DCRA/image/DenisNet-10.eps'
Image(textfile_1,textfile_2,data_1,data_2,data_3,data_4,Image_save)