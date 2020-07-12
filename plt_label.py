import csv
import matplotlib.pyplot as plt
import numpy as np

# 读取文件
def Readtext(textfile):
    fp = open(textfile, 'r')
    list=[]
    for line in fp.readlines():
        # split，以空格为分隔符，分割字符串，的默认参数是空格，所以不传递任何参数时
        a = round(float(line.split()[0]), 2)
        list.append(a)

    return list

train1=Readtext('../DCRA/Text/DenseNet-BC-SERE-L=169/DenseNet-BC-169-12-10_train.txt')
test1=Readtext('../DCRA/Text/DenseNet-BC-SERE-L=169/DenseNet-BC-169-12-10_test.txt')

train2=Readtext('../DCRA/Text/DenseNet-BC-SERE-L=169/DenseNet-BC-SE-169-12-10_train.txt')
test2=Readtext('../DCRA/Text/DenseNet-BC-SERE-L=169/DenseNet-BC-SE-169-12-10_test.txt')

train3=Readtext('../DCRA/Text/DenseNet-BC-SERE-L=169/DenseNet-BC-SERE-169-12-10_train.txt')
test3=Readtext('../DCRA/Text/DenseNet-BC-SERE-L=169/DenseNet-BC-SERE-169-12-10_test.txt')

#绘制图
def Image(train1,valid1,train2,valid2,train3,valid3,image):
    x=range(300)
    y1 = train1
    y2=train2
    y3=train3

    z1 =valid1
    z2 = valid2
    z3=valid3



    plt.plot(x, y1, label='DenseNet-BC-100 train', color='b', linestyle='--')
    plt.plot(x, y2, label='DenseNet-BC-SE-100 train', color='r', linestyle='--')
    plt.plot(x, y3, label='DenseNet-BC-SERE-100 train', color='g', linestyle='--')

    plt.plot(x, z1, label='DenseNet-BC-100 test', color='b', linestyle='-')
    plt.plot(x, z2, label='DenseNet-BC-SE-100 test', color='r', linestyle='-')
    plt.plot(x, z3, label='DenseNet-BC-SERE-100 test', color='g', linestyle='-')

    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.savefig(image)
    plt.show()

image_file='../DCRA/image/BC-SE-RE-169-10.jpg'
Image(train1,test1,train2,test2,train3,test3,image_file)