import os
import sys
import torch
import utils
import argparse
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from evaluation_metric import *
import pickle
import random

from Models.supernet_M_M import SandwichNetwork_Ind
from Data.Dataset_P import pickle_dataset
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser("Personality")
parser.add_argument('--data', type=str, default=None, help='location of the data')
parser.add_argument('--batch_size', type=int, default=50, help='batch size')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
parser.add_argument('--repo_freq', type=float, default=10, help='validation frequency, the result saved in tensorboard')
parser.add_argument('--save_arch_freq', type=float, default=10, help='save arch frequency')
parser.add_argument('--save_ckpt_freq', type=float, default=50, help='save ckpt frequency')
parser.add_argument('--valid_freq', type=float, default=50, help='validation frequency')
parser.add_argument('--gpu_id', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=500, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=32, help='num of init channels')
parser.add_argument('--layers', nargs='+', type=int, default=[1,1,1,2,1,1,1,2,1,1,1,2,1,1,1], help='shape of layers')
parser.add_argument('--model_path', type=str, default=None, help='reload ckpt path')
parser.add_argument('--save', type=str, default='temp', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--arch_learning_rate', type=float, default=0.05, help='learning rate for arch')
parser.add_argument('--arch_weight_decay', type=float, default=0, help='weight decay for arch')
parser.add_argument('--temp_len', type=int, default=80, help='seq_len of input')
parser.add_argument('--increase_L', type=int, default=0, help='Lengthen forward in time for the adaloss')
parser.add_argument('--increase_R', type=int, default=13, help='Lengthen backward in time for the adaloss')
parser.add_argument('--ID', type=int, default=0, help='ID of the talk')
parser.add_argument('--delay', type=int, default=0, help='delay for data slip window')
parser.add_argument('--over_lap', type=int, default=0, help='over_lap for data slip window')
parser.add_argument('--switch_epoch', type=int, default=300, help='the epoch to change loss form constant to adapt')
parser.add_argument('--Y_X', action="store_false", help='default is True,True: Expert 2 Novice; False: Novice 2 Expert')
args = parser.parse_args()
args.save = 'Log/supernet-delay{}-FT{}-{}/{}'.format(args.increase_R,args.switch_epoch,args.Y_X,args.ID)
utils.create_exp_dir(args.save)
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
writer = SummaryWriter(log_dir=args.save)


class VideoFrame():
    pass


class AudioFrame():
    pass


def main():
    if not torch.cuda.is_available():
        print('no gpu device available')
        sys.exit(1)
    np.random.seed(args.seed)
    random.seed(args.seed)
    cudnn.enabled = True
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = False
    print('gpu device = %d' % args.gpu_id)
    print("args = %s", args)

    criterion_ada = utils.Adaptive_MSELoss_MSE().cuda()
    criterion_constant = nn.MSELoss().cuda()
    model = SandwichNetwork_Ind(args.init_channels, args.layers, args.layers, None,
                   in_len=args.temp_len, out_len=args.temp_len,
                   down_times=args.layers.count(2)).cuda()

    print("param size = %fMB", utils.count_parameters_in_MB(model))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate,weight_decay=args.weight_decay)
    optimizer_arch = torch.optim.Adam(model.arch_parameters(),lr=args.arch_learning_rate, weight_decay=args.arch_weight_decay)

    dataset_train = pickle_dataset(args.data, ID=args.ID, over_lap=args.over_lap, layback=args.delay,
                                   seq_len=args.temp_len, increase_L=args.increase_L, increase_R=args.increase_R,Y_X=args.Y_X)

    num_train = len(dataset_train)
    indices = list(range(num_train))
    print(utils.count_parameters_in_MB(model))

    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size,shuffle=False,
        sampler=utils.inBatchSequentialBatchShuffle(indices,args.batch_size),
        pin_memory=True, num_workers=4)

    val_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size,shuffle=False,
        sampler=utils.inBatchSequentialBatchShuffle(indices, args.batch_size),
        pin_memory=True, num_workers=4)

    saved_epoch=0
    if args.model_path:
        saveDict = torch.load(args.model_path)
        model.load_state_dict(saveDict['state_dict'])
        model._arch_parameters = saveDict['arch_parameters']
        optimizer.load_state_dict(saveDict['optimizer_state_dict'])
        saved_epoch = saveDict['epoch']

    for epoch in range(saved_epoch,args.epochs):
        criterion = criterion_constant if epoch<args.switch_epoch else criterion_ada

        # training
        train_obj = train(train_loader, val_loader, model, None, criterion, optimizer, epoch, optimizer_arch)

        # validation
        if (epoch+1) % args.valid_freq == 0 or epoch == 0:
            with torch.no_grad():
                _, _, _ = infer(train_loader,val_loader, model, criterion, epoch)

        #save ckpt
        if (epoch+1) % args.save_ckpt_freq == 0 or epoch == args.epochs - 1:
            utils.save(model, os.path.join(args.save, 'model-'+str(epoch+1)), epoch, optimizer,optimizer_arch)

        #save arch parameter
        if (epoch+1) % args.save_arch_freq == 0 or epoch == args.epochs - 1:
            f1 = open(os.path.join(args.save, 'arch-'+str(epoch+1)+'.pickle'), 'wb')
            savepar = [parameter.detach().cpu().numpy() for parameter in model.arch_parameters()]
            pickle.dump(savepar, f1)
            f1.close()

def sample_data(loader):
    while True:
        for clips_x_fl, clips_y_fl, clips_x_MFCC, clips_y_MFCC, XMean, YMean in loader:
            yield clips_x_fl, clips_y_fl, clips_x_MFCC, clips_y_MFCC, XMean, YMean


def train(train_loader, val_loader, model, architect, criterion, optimizer, epoch, optimizer_arch):
    lossMtric = utils.AvgrageMeter()
    tqbar = tqdm(train_loader, desc='Train Epoch %d ' % epoch, miniters=1, dynamic_ncols=True)
    # val_infinite_iter = sample_data(val_loader)
    for clips_x_fl, clips_y_fl, clips_x_MFCC, clips_y_MFCC, XMean, YMean in tqbar:
        model.train()
        n = clips_x_fl.size(0)
        clips_x_fl = clips_x_fl.cuda()
        clips_y_fl = clips_y_fl.cuda()
        if args.Y_X:
            clips_x_MFCC = clips_x_MFCC.cuda()
        else:
            clips_y_MFCC = clips_y_MFCC.cuda()
        XMean = XMean.cuda()
        YMean = YMean.cuda()

        optimizer.zero_grad()
        optimizer_arch.zero_grad()
        if args.Y_X:
            logits = model(clips_x_fl - XMean, clips_x_MFCC)
        else:
            logits = model(clips_y_fl - YMean, clips_y_MFCC)

        if isinstance(criterion,nn.MSELoss):
            if args.Y_X:
                loss = criterion(logits, (clips_y_fl - YMean)[:,:,0:args.temp_len])
            else:
                loss = criterion(logits, (clips_x_fl - XMean)[:,:,0:args.temp_len])
        else:
            if args.Y_X:
                loss,_ = criterion(logits, clips_y_fl - YMean)
            else:
                loss,_ = criterion(logits, clips_x_fl - XMean)

        loss.backward()
        optimizer.step()
        optimizer_arch.step()

        lossMtric.update(loss.item(), n)
        tqbar.set_postfix({'loss': lossMtric.avg})
    if args.Y_X:
        writer.add_scalar('X-Y-train/loss', lossMtric.avg, epoch)
    else:
        writer.add_scalar('Y-X-train/loss', lossMtric.avg, epoch)
    if (epoch+1)%args.repo_freq == 0 or epoch==0:
        for index,alpha in enumerate(model.arch_parameters()):
            writer.add_histogram('sigmoid/%d'%index,torch.sigmoid(alpha),epoch)
            writer.add_histogram('value/%d'%index,alpha,epoch)
    return lossMtric.avg


def infer(train_loader,val_loader, model, criterion, epoch):
    lossMtric = utils.AvgrageMeter()
    PCCMtric = utils.AvgrageMeter()
    CCCMtric = utils.AvgrageMeter()
    model.eval()

    tqbar = tqdm(train_loader, desc='Val-train ', miniters=1, dynamic_ncols=True)
    for clips_x_fl, clips_y_fl, clips_x_MFCC, clips_y_MFCC, XMean, YMean in tqbar:
        clips_x_fl = clips_x_fl.cuda()
        clips_y_fl = clips_y_fl.cuda()
        if args.Y_X:
            clips_x_MFCC = clips_x_MFCC.cuda()
        else:
            clips_y_MFCC = clips_y_MFCC.cuda()
        XMean = XMean.cuda()
        YMean = YMean.cuda()

        if args.Y_X:
            logits = model(clips_x_fl - XMean, clips_x_MFCC)
        else:
            logits = model(clips_y_fl - YMean, clips_y_MFCC)
        if isinstance(criterion,nn.MSELoss):
            if args.Y_X:
                loss = criterion(logits, (clips_y_fl - YMean)[:,:,0:args.temp_len])
                YMean = YMean[:,:,0:args.temp_len]
                clips_y_fl =clips_y_fl[:,:,0:args.temp_len]
            else:
                loss = criterion(logits, (clips_x_fl - XMean)[:,:,0:args.temp_len])
                XMean = XMean[:,:,0:args.temp_len]
                clips_x_fl =clips_x_fl[:,:,0:args.temp_len]
        else:
            if args.Y_X:
                loss = criterion(logits, clips_y_fl - YMean)
                YMean = YMean[:,:,loss[1]:loss[1]+args.temp_len]
                clips_y_fl =clips_y_fl[:,:,loss[1]:loss[1]+args.temp_len]
                loss = loss[0]
            else:
                loss = criterion(logits, clips_x_fl - XMean)
                XMean = XMean[:,:,loss[1]:loss[1]+args.temp_len]
                clips_x_fl =clips_x_fl[:,:,loss[1]:loss[1]+args.temp_len]
                loss = loss[0]
        if args.Y_X:
            MtricP = (logits + YMean).detach().cpu()
            MtricT = clips_y_fl.detach().cpu()
        else:
            MtricP = (logits + XMean).detach().cpu()
            MtricT = clips_x_fl.detach().cpu()
        lossMtric.update(loss.detach().cpu().numpy())
        PCCMtric.update(PCC(MtricP, MtricT))
        CCCMtric.update(CCC(MtricP, MtricT))
        tqbar.set_postfix({'T-loss': lossMtric.avg, 'T-PCC': PCCMtric.avg, 'T-CCC': CCCMtric.avg})
    writer.add_scalar('val-T/loss', lossMtric.avg, epoch)
    writer.add_scalar('val-T/PCC', PCCMtric.avg, epoch)
    writer.add_scalar('val-T/CCC', CCCMtric.avg, epoch)

    return lossMtric.avg, PCCMtric.avg, CCCMtric.avg


if __name__ == '__main__':
    main()
