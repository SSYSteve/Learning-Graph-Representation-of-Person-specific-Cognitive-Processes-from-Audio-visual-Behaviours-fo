import os
import numpy as np
import torch
import torch.nn.functional as F
import random
from torch.utils.data.sampler import Sampler
import shutil
import torchvision.transforms as transforms
from torch.autograd import Variable
from evaluation_metric import PCC


class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_cifar10(args):
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    return train_transform, valid_transform


def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name) / 1e6


def save_checkpoint(state, is_best, save):
    filename = os.path.join(save, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)


def save(model, model_path, epoch, optimizer,optimizer_arch):
    saveDict = {'state_dict': model.state_dict(), 'arch_parameters': model.arch_parameters(), 'epoch': epoch,
                'optimizer_state_dict': optimizer.state_dict(),'optimizer_arch_state_dict': optimizer_arch.state_dict()}
    torch.save(saveDict, model_path)


def load(model, model_path):
    if 'state_dict' not in torch.load(model_path):
        model.load_state_dict(torch.load(model_path))
    else:
        saveDict = torch.load(model_path)
        model.load_state_dict(saveDict['state_dict'])
        model._arch_parameters = saveDict['arch_parameters']
        print('Has load arch_parameters')
def copy_state_dict(cur_state_dict, pre_state_dict, prefix = ''):
    def _get_params(key):
        key = prefix + key
        if key in pre_state_dict:
            return pre_state_dict[key]
        return None

    for k in cur_state_dict.keys():
        v = _get_params(k)
        try:
            if v is None:
                print('parameter {} not found'.format(k))
                continue
            cur_state_dict[k].copy_(v)
        except:
            print('copy param {} failed'.format(k))
            continue

def drop_path(x, drop_prob):
    if drop_prob > 0.:
        keep_prob = 1. - drop_prob
        mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
        x.div_(keep_prob)
        x.mul_(mask)
    return x


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.makedirs(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


class PersonalRandomSampler(Sampler):
    def __init__(self, data_source, batch_size):
        self.data_source = data_source
        self.batch_size = batch_size
        # 把data_source给按纯净分到各自的类中，不纯净（交界处）的就不要了
        split = [[[], []] for i in range(84)]
        for i in range(len(data_source)):
            clips_x_fl, clips_y_fl, clips_x_MFCC, clips_y_MFCC, XMean, YMean, CLASS, IDENTITY = data_source[i]
            isOneClass = (CLASS[0] == CLASS[-1] and IDENTITY[0] == IDENTITY[-1])
            if isOneClass:
                split[CLASS[0]][IDENTITY[0]].append(i)
        # 给各自的帧扩容到batch_size整除量
        for C in range(84):
            for I in range(2):
                if len(split[C][I]) == 0:
                    continue
                split[C][I] = self.expend(split[C][I])
        # 将所有的样本都按顺序放好，并且同一类内部打乱
        self.indexs = []
        for C in range(84):
            for I in range(2):
                if len(split[C][I]) == 0:
                    continue
                random.shuffle(split[C][I])
                self.indexs.extend(split[C][I])

    def expend(self, input_list):
        n = (len(input_list) // self.batch_size + 1) * self.batch_size
        mul = n // len(input_list)
        rem = n % len(input_list)
        out_list = []
        for i in range(mul):
            out_list.extend(input_list)
        rem_list = input_list[:rem]
        out_list.extend(rem_list)

        return out_list

    def __iter__(self):
        return iter(self.indexs)

    def __len__(self):
        return len(self.indexs)

class inBatchSequentialBatchShuffle(Sampler):
    def __init__(self, indices, batch_size):
        self.indices = indices
        self.batch_size = batch_size
        #以bs为步长取出指针
        self.pointers=[i for i in range(0,len(self.indices),self.batch_size)]

    def __iter__(self):
        random.shuffle(self.pointers)
        result=[]
        for i in self.pointers:
            if i+self.batch_size>len(self.indices):
                result = result + list(range(i, len(self.indices)))
            else:
                result = result+list(range(i,i+self.batch_size))
        return (self.indices[i] for i in result)

    def __len__(self):
        return len(self.indices)

class Adaptive_MSELoss_PCC(torch.nn.Module):
    def __init__(self):
        super(Adaptive_MSELoss_PCC, self).__init__()
    def forward(self,input, target):
        input_cpu = input.detach().cpu()
        target_cpu = target.detach().cpu()
        len_input = input.shape[-1]
        len_target = target.shape[-1]
        assert len_target>=len_input
        for index in range(len_target-len_input+1):
            # loss = F.mse_loss(input,target[:,:,index:index+len_input])
            pcc = PCC(input_cpu,target_cpu[:,:,index:index+len_input])
            if index==0:
                # min_index=0
                # min_loss = loss
                max_index = 0
                max_pcc = pcc
            else:
                if pcc>max_pcc:
                    # min_index = index
                    # min_loss = loss
                    max_index = index
                    max_pcc = pcc
        loss = F.mse_loss(input,target[:,:,max_index:max_index+len_input])
        return loss,max_index

class Adaptive_MSELoss_MSE(torch.nn.Module):
    def __init__(self):
        super(Adaptive_MSELoss_MSE, self).__init__()
    def forward(self,input, target):
        # input_cpu = input.detach().cpu()
        # target_cpu = target.detach().cpu()
        len_input = input.shape[-1]
        len_target = target.shape[-1]
        assert len_target>=len_input
        for index in range(len_target-len_input+1):
            loss = F.mse_loss(input,target[:,:,index:index+len_input])
            # pcc = PCC(input_cpu,target_cpu[:,:,index:index+len_input])
            if index==0:
                min_index=0
                min_loss = loss
                # max_index = 0
                # max_pcc = pcc
            else:
                if loss<min_loss:
                    min_index = index
                    min_loss = loss
                    # max_index = index
                    # max_pcc = pcc
        return min_loss,min_index