import argparse
import os
import shutil
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from model import dcra as dn
from model import Se as dnse
# from DCRA.model import Se as dnse
# used for logging to TensorBoard
from tensorboard_logger import configure, log_value

# 创建一个解析对象
# description=None,    - help时显示的开始文字
parser = argparse.ArgumentParser(description='PyTorch DenseNet Training')

# 向该对象中添加你要关注的命令行参数和选项
# name or flags...    - 必选，指定参数的形式，一般写两个，一个短参数，一个长参数
# default ：默认值
# type ：参数类型，默认为str
# help		可以写帮助信息
parser.add_argument('--epochs', default=300, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    help='mini-batch size (default: 64)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--layers', default=100, type=int,
                    help='total number of layers (default: 100)')
parser.add_argument('--growth', default=12, type=int,
                    help='number of new channels per layer (default: 12)')
parser.add_argument('--droprate', default=0, type=float,
                    help='dropout probability (default: 0.0)')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='whether to use standard augmentation (default: True)')
parser.add_argument('--reduce', default=0.5, type=float,
                    help='compression rate in transition stage (default: 0.5)')
parser.add_argument('--no-bottleneck', dest='bottleneck', action='store_false',
                    help='To not use bottleneck block')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='DenseNet_BC_100_12', type=str,
                    help='name of experiment')
parser.add_argument('--no-se_model',dest='se_model',action='store_false',help='To not use SE block')
parser.add_argument('--se_re_model',dest='se_re_model',action='store_true',help='use SE_residual block')
parser.add_argument('--reducation',default=6,type=int,
                    help='Reduce channel in Se_model')
parser.add_argument('--group',default=1,type=int,
                    help='num of group Conv')
parser.add_argument('--tensorboard',
                    help='Log progress to TensorBoard', action='store_true')
parser.set_defaults(bottleneck=True)
parser.set_defaults(augment=True)
parser.set_defaults(se_model=True)


least_prec1 = 100


def main():
    #固定随机seed，保证两次运行相同的代码得到相同的输出，消除变化因子
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    # global：定义全局变量
    global args, least_prec1


    # 进行解析
    args = parser.parse_args()
    if args.tensorboard: configure("runs/%s" % (args.name))

    # Data loading code
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

    text_name1 = '../DCRA/Text/DenseNet-BC-Dr/' + args.name + '_train.txt'
    text_name2 = '../DCRA/Text/DenseNet-BC-Dr/' + args.name + '_test.txt'
    print(text_name1)
    print(text_name2)

    print("augment")
    print(args.augment)
    print("bottleneck")
    print(args.bottleneck)
    # print("reseme")
    # print(args.resume)
    print("se_model")
    print(args.se_model)
    print("se_re_model")
    print(args.se_re_model)
    print("tensorboard")
    print(args.tensorboard)
    print("num_group")
    print(args.group)
    print("reducation")
    print(args.reducation)

    if args.augment:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    kwargs = {'num_workers': 1, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=True, download=True,
                         transform=transform_train),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=False, transform=transform_test),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    # create model
    # model=dnse.DenseSe(args.layers, 10, args.growth, reduction=args.reduce,
    #                      bottleneck=args.bottleneck, dropRate=args.droprate,
    #                      SE_model=args.se_model,SE_Residual_model=args.se_re_model,Reducation=args.reducation,num_group=args.group)
    model = dn.DCRA(args.layers, 10, args.growth, reduction=args.reduce,
                         bottleneck=args.bottleneck, dropRate=args.droprate,
                         SE_model=args.se_model,SE_Residual_model=args.se_re_model,Reducation=args.reducation,num_group=args.group)

    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    # for training on multiple GPUs.
    # Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
    model = torch.nn.DataParallel(model).cuda()
    # model = model.cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            least_prec1 = checkpoint['least_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # 在程序刚开始加这条语句可以提升一点训练速度，没什么额外开销，一般都会加
    cudnn.benchmark = True

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                nesterov=True,
                                weight_decay=args.weight_decay)
    tra_out = []
    val_out = []

    for epoch in range(args.start_epoch, args.epochs):

        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        out=train(train_loader, model, criterion, optimizer, epoch)
        tra_out.append(out)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, epoch)
        val_out.append(prec1)

        # remember best prec@1 and save checkpoint
        is_best = prec1 < least_prec1
        least_prec1 = min(prec1, least_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'least_prec1': least_prec1,
        }, is_best)
    print('Least error: ', least_prec1)

    # 模型一训练，验证结果


    write_txt(tra_out, text_name1)
    write_txt(val_out, text_name2)


def train(train_loader, model, criterion, optimizer, epoch):
    """Train for one epoch on the training set"""
    """Computes and stores the average and current value"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()  # 返回当前时间的时间戳（1970纪元后经过的浮点秒数）

    # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中
    for i, (input, target) in enumerate(train_loader):
        target = target.cuda(async=True)
        input = input.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        # prec1 = accuracy(output.data, target, topk=(1,))[0]
        prec1=error(output,target)
        losses.update(loss.item(), input.size(0))
        top1.update(prec1, input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Error {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                loss=losses, top1=top1))
    # log to TensorBoard
    if args.tensorboard:
        log_value('train_loss', losses.avg, epoch)
        log_value('train_error', top1.avg, epoch)
    return top1.avg


def validate(val_loader, model, criterion, epoch):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(async=True)
            input = input.cuda()
            input_var = torch.autograd.Variable(input)
            # target_var = torch.autograd.Variable(target, volatile=True)
            target_var = torch.autograd.Variable(target)

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            # measure accuracy and record loss
            # prec1 = accuracy(output.data, target, topk=(1,))[0]
            prec1=error(output,target)
            losses.update(loss.item(), input.size(0))
            top1.update(prec1, input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Error {top1.val:.3f} ({top1.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1))

        print(' * Error {top1.avg:.3f}'.format(top1=top1))
        # log to TensorBoard
        if args.tensorboard:
            log_value('val_loss', losses.avg, epoch)
            log_value('val_error', top1.avg, epoch)
    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = "runs/%s/" % (args.name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'runs/%s/' % (args.name) + 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 after 150 and 225 epochs"""
    lr = args.lr * (0.1 ** (epoch // 150)) * (0.1 ** (epoch // 225))
    # log to TensorBoard
    if args.tensorboard:
        log_value('learning_rate', lr, epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# def accuracy(output, target, topk=(1,)):
#     """Computes the precision@k for the specified values of k"""
#     maxk = max(topk)
#     batch_size = target.size(0)
#
#     _, pred = output.topk(maxk, 1, True, True)
#     pred = pred.t()
#     correct = pred.eq(target.view(1, -1).expand_as(pred))
#
#     res = []
#     for k in topk:
#         correct_k = correct[:k].view(-1).float().sum(0)
#         res.append(correct_k.mul_(100.0 / batch_size))
#     return res

#定义错误率函数
def error(outputs,labels):
    _,preds=torch.max(outputs,dim=1)
    error=(1-torch.sum(preds==labels).item()/len(preds))*100
    return error


# 数据写入文本
def write_txt(data,text_name):
    f = open(text_name,'w')
    for i in range(len(data)):
        f.writelines(str(data[i]))
        i += 1
        f.write('\n')
    f.close()

if __name__ == '__main__':
    main()
