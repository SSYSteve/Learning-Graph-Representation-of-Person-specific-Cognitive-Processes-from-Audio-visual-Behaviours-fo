__all__ = ['SandwichNetwork_Ind']

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from Models.operations import *
from Models.genotypes import *

class MixedOp(nn.Module):

    def __init__(self, C, stride):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            op = OPS[primitive](C, stride, False)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm1d(C, affine=False))
            self._ops.append(op)

    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self._ops))


class MixedDown(nn.Module):
    def __init__(self, C):
        super(MixedDown, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES_DOWN:
            op = OPS_DOWN[primitive](C, 2, False)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm1d(C, affine=False))
            self._ops.append(op)

    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self._ops))


class MixedUP(nn.Module):
    def __init__(self, C):
        super(MixedUP, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES_UP:
            op = OPS_UP[primitive](C, 2, False)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm1d(C, affine=False))
            self._ops.append(op)

    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self._ops))

class Cell_Bread_Encoder(nn.Module):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell_Bread_Encoder, self).__init__()
        self.reduction = reduction

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self._steps = steps
        self._multiplier = multiplier

        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()
        for i in range(self._steps):
            for j in range(2 + i):
                if reduction and j < 2:
                    op = MixedDown(C)
                else:
                    op = MixedOp(C, 1)
                self._ops.append(op)

    def forward(self, s0, s1, weights_1, weights_2):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        if self.reduction:
            offset_1 = 0
            offset_2 = 0
            for i in range(self._steps):
                sum_pool = []
                for j in range(2 + i):
                    if j < 2:
                        sum_pool.append(self._ops[offset_1 + offset_2](states[j], weights_1[offset_1]))
                        offset_1 += 1
                    else:
                        sum_pool.append(self._ops[offset_1 + offset_2](states[j], weights_2[offset_2]))
                        offset_2 += 1
                s = sum(sum_pool)
                states.append(s)

        else:
            weights = weights_1
            offset = 0
            for i in range(self._steps):
                s = sum(self._ops[offset + j](h, weights[offset + j]) for j, h in enumerate(states))
                offset += len(states)
                states.append(s)
        return torch.cat(states[-self._multiplier:], dim=1)


class Cell_Bread_Decoder(nn.Module):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell_Bread_Decoder, self).__init__()
        self.reduction = reduction

        if reduction_prev:
            self.preprocess0 = TranConv(C_prev_prev, C, 3, 2, 1, 1, False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self._steps = steps
        self._multiplier = multiplier

        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()
        for i in range(self._steps):
            for j in range(2 + i):
                if reduction and j < 2:
                    op = MixedUP(C)
                else:
                    op = MixedOp(C, 1)
                self._ops.append(op)

    def forward(self, s0, s1, weights_1, weights_2):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        states = [s0, s1]
        if self.reduction:
            offset_1 = 0
            offset_2 = 0
            for i in range(self._steps):
                sum_pool = []
                for j in range(2 + i):
                    if j < 2:
                        sum_pool.append(self._ops[offset_1 + offset_2](states[j], weights_1[offset_1]))
                        offset_1 += 1
                    else:
                        sum_pool.append(self._ops[offset_1 + offset_2](states[j], weights_2[offset_2]))
                        offset_2 += 1
                s = sum(sum_pool)
                states.append(s)
        else:
            weights = weights_1
            offset = 0
            for i in range(self._steps):
                s = sum(self._ops[offset + j](h, weights[offset + j]) for j, h in enumerate(states))
                offset += len(states)
                states.append(s)
        return torch.cat(states[-self._multiplier:], dim=1)

class Cell_Stuff_Encoder(nn.Module):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell_Stuff_Encoder, self).__init__()
        self.reduction = reduction

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self.preprocess2 = ReLUConvBN(C_prev*2, C, 1, 1, 0, affine=False)
        self._steps = steps
        self._multiplier = multiplier

        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()
        for i in range(self._steps):
            for j in range(3 + i):
                if reduction and j < 3:
                    op = MixedDown(C)
                else:
                    op = MixedOp(C, 1)
                self._ops.append(op)

    def forward(self, s0, s1, s2, weights_1, weights_2):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        s2 = self.preprocess2(s2)

        states = [s0, s1, s2]
        if self.reduction:
            offset_1 = 0
            offset_2 = 0
            for i in range(self._steps):
                sum_pool = []
                for j in range(3 + i):
                    if j < 3:
                        sum_pool.append(self._ops[offset_1 + offset_2](states[j], weights_1[offset_1]))
                        offset_1 += 1
                    else:
                        sum_pool.append(self._ops[offset_1 + offset_2](states[j], weights_2[offset_2]))
                        offset_2 += 1
                s = sum(sum_pool)
                states.append(s)

        else:
            weights = weights_1
            offset = 0
            for i in range(self._steps):
                s = sum(self._ops[offset + j](h, weights[offset + j]) for j, h in enumerate(states))
                offset += len(states)
                states.append(s)
        return torch.cat(states[-self._multiplier:], dim=1)

class Cell_Stuff_Decoder(nn.Module):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell_Stuff_Decoder, self).__init__()
        self.reduction = reduction

        if reduction_prev:
            self.preprocess0 = TranConv(C_prev_prev, C, 3, 2, 1, 1, False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self.preprocess2 = ReLUConvBN(C_prev*2, C, 1, 1, 0, affine=False)
        self._steps = steps
        self._multiplier = multiplier

        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()
        for i in range(self._steps):
            for j in range(3 + i):
                if reduction and j < 3:
                    op = MixedUP(C)
                else:
                    op = MixedOp(C, 1)
                self._ops.append(op)

    def forward(self, s0, s1, s2, weights_1, weights_2):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        s2 = self.preprocess2(s2)
        states = [s0, s1, s2]
        if self.reduction:
            offset_1 = 0
            offset_2 = 0
            for i in range(self._steps):
                sum_pool = []
                for j in range(3 + i):
                    if j < 3:
                        sum_pool.append(self._ops[offset_1 + offset_2](states[j], weights_1[offset_1]))
                        offset_1 += 1
                    else:
                        sum_pool.append(self._ops[offset_1 + offset_2](states[j], weights_2[offset_2]))
                        offset_2 += 1
                s = sum(sum_pool)
                states.append(s)
        else:
            weights = weights_1
            offset = 0
            for i in range(self._steps):
                s = sum(self._ops[offset + j](h, weights[offset + j]) for j, h in enumerate(states))
                offset += len(states)
                states.append(s)
        return torch.cat(states[-self._multiplier:], dim=1)

class BreadNetwork(nn.Module):
    def __init__(self, C_in, C, layers_encoder, layers_decoder, criterion, steps=4, multiplier=4,
                 stem_multiplier=4,rnn_seq_len=3):
        super(BreadNetwork, self).__init__()
        self._C = C
        self._layers_encoder = layers_encoder
        self._layers_decoder = layers_decoder
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier

        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv1d(C_in, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm1d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        # Encoder part
        self.cells_encoder = nn.ModuleList()
        reduction_prev = False
        for i in layers_encoder:
            if i == 2:
                C_curr = int(C_curr * 1.5)
                reduction = True
            else:
                reduction = False
            cell = Cell_Bread_Encoder(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells_encoder += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        self.LSTM = nn.LSTM(input_size=rnn_seq_len,hidden_size=rnn_seq_len,num_layers=3)

        # Decoder part
        C_prev_prev = C_prev
        self.cells_decoder = nn.ModuleList()
        reduction_prev = False
        for i in layers_decoder:
            if i == 2:
                C_curr = int(C_curr // 1.5)
                reduction = True
            else:
                reduction = False
            cell = Cell_Bread_Decoder(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells_decoder += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

    def forward(self):
        print('BreadNetwork can not use alone ')
        exit(0)

class StuffingNetwork(nn.Module):
    def __init__(self, C_in, C, layers_encoder, layers_decoder, criterion, steps=4, multiplier=4,
                 stem_multiplier=4,rnn_seq_len=3):
        super(StuffingNetwork, self).__init__()
        self._C = C
        self._layers_encode = layers_encoder
        self._layers_decode = layers_decoder
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier

        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv1d(C_in, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm1d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        # Encoder part
        self.cells_encoder = nn.ModuleList()
        reduction_prev = False
        for i in layers_encoder:
            if i == 2:
                C_curr = int(1.5 * C_curr)
                reduction = True
            else:
                reduction = False
            cell = Cell_Stuff_Encoder(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells_encoder += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        self.LSTM = nn.LSTM(input_size=rnn_seq_len,hidden_size=rnn_seq_len,num_layers=3)

        # Decoder part
        C_prev_prev = C_prev
        self.cells_decoder = nn.ModuleList()
        reduction_prev = False
        for i in layers_decoder:
            if i == 2:
                C_curr = int(C_curr // 1.5)
                reduction = True
            else:
                reduction = False
            cell = Cell_Stuff_Decoder(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells_decoder += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

    def forward(self):
        print('StuffingNetwork can not use alone ')
        exit(0)

class SandwichNetwork_Ind(nn.Module):
    def __init__(self, C, layers_encoder, layers_decoder, criterion, steps=4, multiplier=4,
                 stem_multiplier=4,in_len=80,out_len=80,down_times=3):
        super(SandwichNetwork_Ind, self).__init__()
        self._layers_encoder = layers_encoder
        self._layers_decoder = layers_decoder
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier
        self.out_len = out_len
        self.in_len = in_len
        self.down_times = down_times
        self.register_buffer('mask_encoder',torch.ones(layers_encoder.count(2)+1,int(layers_encoder.count(1)/(layers_encoder.count(2)+1)))) #[num_stage,num_depth]
        self.register_buffer('mask_decoder',torch.ones(layers_decoder.count(2)+1,int(layers_decoder.count(1)/(layers_decoder.count(2)+1))))
        self.Module_Modal_1 = BreadNetwork(136, C, layers_encoder, layers_decoder, criterion, steps=steps,
                                           multiplier=multiplier,
                                           stem_multiplier=stem_multiplier,rnn_seq_len=int(self.in_len/(2**self.down_times)))
        self.Module_Modal_2 = BreadNetwork(128, C, layers_encoder, layers_decoder, criterion, steps=steps,
                                           multiplier=multiplier,
                                           stem_multiplier=stem_multiplier,rnn_seq_len=int(self.in_len/(2**self.down_times)))
        self.Module_Fusion = StuffingNetwork(136+128, C, layers_encoder, layers_decoder, criterion, steps=steps,
                                             multiplier=multiplier,
                                             stem_multiplier=stem_multiplier,rnn_seq_len=int(self.in_len/(2**self.down_times)))
        self.mixlayer = ReLUConvBN( C * 3 * multiplier, 136, 1, 1, 0, affine=False)
        self.LSTM = nn.LSTM(input_size=self.out_len,hidden_size=self.out_len,num_layers=3)
        self.outlater = ReLUConv( 136, 136, 1, 1, 0, affine=False)
        self._initialize_alphas()

    def forward(self, x_0, x_1):
        it_arch_parameters = iter(self._arch_parameters)
        s0_1 = s1_1 = self.Module_Modal_1.stem(x_0)
        s0_2 = s1_2 = self.Module_Modal_2.stem(x_1)
        s0_f = s1_f = self.Module_Fusion.stem(torch.cat([x_0,x_1],dim=1))
        curry_depth_features0_1=torch.tensor([],device=x_0.device)
        curry_depth_features0_2=torch.tensor([],device=x_0.device)
        curry_depth_features0_f=torch.tensor([],device=x_0.device)
        curry_depth_features1_1=torch.tensor([],device=x_0.device)
        curry_depth_features1_2=torch.tensor([],device=x_0.device)
        curry_depth_features1_f=torch.tensor([],device=x_0.device)
        stage = 0
        for i, (cell_1, cell_f, cell_2) in enumerate(
                zip(self.Module_Modal_1.cells_encoder, self.Module_Fusion.cells_encoder,
                    self.Module_Modal_2.cells_encoder)):
            if cell_f.reduction:
                if len(curry_depth_features1_1):
                    s0_1 = (curry_depth_features0_1*self.mask_encoder[stage].view(-1,1,1,1)).sum(dim=0)/self.mask_encoder[stage].sum()
                    s0_2 = (curry_depth_features0_2*self.mask_encoder[stage].view(-1,1,1,1)).sum(dim=0)/self.mask_encoder[stage].sum()
                    s0_f = (curry_depth_features0_f*self.mask_encoder[stage].view(-1,1,1,1)).sum(dim=0)/self.mask_encoder[stage].sum()
                    s1_1 = (curry_depth_features1_1*self.mask_encoder[stage].view(-1,1,1,1)).sum(dim=0)/self.mask_encoder[stage].sum()
                    s1_2 = (curry_depth_features1_2*self.mask_encoder[stage].view(-1,1,1,1)).sum(dim=0)/self.mask_encoder[stage].sum()
                    s1_f = (curry_depth_features1_f*self.mask_encoder[stage].view(-1,1,1,1)).sum(dim=0)/self.mask_encoder[stage].sum()

                weights_1_M1 = F.sigmoid(next(it_arch_parameters))
                weights_2_M1 = F.sigmoid(next(it_arch_parameters))
                weights_1_M2 = F.sigmoid(next(it_arch_parameters))
                weights_2_M2 = F.sigmoid(next(it_arch_parameters))
                weights_1_F = F.sigmoid(next(it_arch_parameters))
                weights_2_F = F.sigmoid(next(it_arch_parameters))
                s0_f, s1_f = s1_f, cell_f(s0_f, s1_f, torch.cat([s1_1,s1_2],dim=1), weights_1_F, weights_2_F)
                s0_1, s1_1 = s1_1, cell_1(s0_1, s1_1, weights_1_M1, weights_2_M1)
                s0_2, s1_2 = s1_2, cell_2(s0_2, s1_2, weights_1_M2, weights_2_M2)
                curry_depth_features0_1=torch.tensor([],device=x_0.device)
                curry_depth_features0_2=torch.tensor([],device=x_0.device)
                curry_depth_features0_f=torch.tensor([],device=x_0.device)
                curry_depth_features1_1=torch.tensor([],device=x_0.device)
                curry_depth_features1_2=torch.tensor([],device=x_0.device)
                curry_depth_features1_f=torch.tensor([],device=x_0.device)
                stage+=1
            else:
                weights_M1 = F.sigmoid(next(it_arch_parameters))
                weights_M2 = F.sigmoid(next(it_arch_parameters))
                weights_F = F.sigmoid(next(it_arch_parameters))
                s0_f, s1_f = s1_f, cell_f(s0_f, s1_f, torch.cat([s1_1,s1_2],dim=1), weights_F, None)
                s0_1, s1_1 = s1_1, cell_1(s0_1, s1_1, weights_M1, None)
                s0_2, s1_2 = s1_2, cell_2(s0_2, s1_2, weights_M2, None)
                curry_depth_features0_1 = torch.cat((curry_depth_features0_1,s0_1.unsqueeze(0)))
                curry_depth_features0_2 = torch.cat((curry_depth_features0_2,s0_2.unsqueeze(0)))
                curry_depth_features0_f = torch.cat((curry_depth_features0_f,s0_f.unsqueeze(0)))
                curry_depth_features1_1 = torch.cat((curry_depth_features1_1,s1_1.unsqueeze(0)))
                curry_depth_features1_2 = torch.cat((curry_depth_features1_2,s1_2.unsqueeze(0)))
                curry_depth_features1_f = torch.cat((curry_depth_features1_f,s1_f.unsqueeze(0)))


        if len(curry_depth_features1_1):
            s1_1 = (curry_depth_features1_1*self.mask_encoder[stage].view(-1,1,1,1)).sum(dim=0)/self.mask_encoder[stage].sum()
            s1_2 = (curry_depth_features1_2*self.mask_encoder[stage].view(-1,1,1,1)).sum(dim=0)/self.mask_encoder[stage].sum()
            s1_f = (curry_depth_features1_f*self.mask_encoder[stage].view(-1,1,1,1)).sum(dim=0)/self.mask_encoder[stage].sum()
            curry_depth_features0_1=torch.tensor([],device=x_0.device)
            curry_depth_features0_2=torch.tensor([],device=x_0.device)
            curry_depth_features0_f=torch.tensor([],device=x_0.device)
            curry_depth_features1_1=torch.tensor([],device=x_0.device)
            curry_depth_features1_2=torch.tensor([],device=x_0.device)
            curry_depth_features1_f=torch.tensor([],device=x_0.device)

        s1_1,_ = self.Module_Modal_1.LSTM(s1_1)
        s1_2,_ = self.Module_Modal_2.LSTM(s1_2)
        s1_f,_ = self.Module_Fusion.LSTM(s1_f)

        s0_1 = s1_1
        s0_2 = s1_2
        s0_f = s1_f
        stage=0
        for i, (cell_1, cell_f, cell_2) in enumerate(
                zip(self.Module_Modal_1.cells_decoder, self.Module_Fusion.cells_decoder,
                    self.Module_Modal_2.cells_decoder)):
            if cell_f.reduction:
                if len(curry_depth_features1_1):
                    s0_1 = (curry_depth_features0_1*self.mask_decoder[stage].view(-1,1,1,1)).sum(dim=0)/self.mask_decoder[stage].sum()
                    s0_2 = (curry_depth_features0_2*self.mask_decoder[stage].view(-1,1,1,1)).sum(dim=0)/self.mask_decoder[stage].sum()
                    s0_f = (curry_depth_features0_f*self.mask_decoder[stage].view(-1,1,1,1)).sum(dim=0)/self.mask_decoder[stage].sum()
                    s1_1 = (curry_depth_features1_1*self.mask_decoder[stage].view(-1,1,1,1)).sum(dim=0)/self.mask_decoder[stage].sum()
                    s1_2 = (curry_depth_features1_2*self.mask_decoder[stage].view(-1,1,1,1)).sum(dim=0)/self.mask_decoder[stage].sum()
                    s1_f = (curry_depth_features1_f*self.mask_decoder[stage].view(-1,1,1,1)).sum(dim=0)/self.mask_decoder[stage].sum()
                weights_1_M1 = F.sigmoid(next(it_arch_parameters))
                weights_2_M1 = F.sigmoid(next(it_arch_parameters))
                weights_1_M2 = F.sigmoid(next(it_arch_parameters))
                weights_2_M2 = F.sigmoid(next(it_arch_parameters))
                weights_1_F = F.sigmoid(next(it_arch_parameters))
                weights_2_F = F.sigmoid(next(it_arch_parameters))
                s0_f, s1_f = s1_f, cell_f(s0_f, s1_f, torch.cat([s1_1,s1_2],dim=1), weights_1_F, weights_2_F)
                s0_1, s1_1 = s1_1, cell_1(s0_1, s1_1, weights_1_M1, weights_2_M1)
                s0_2, s1_2 = s1_2, cell_2(s0_2, s1_2, weights_1_M2, weights_2_M2)
                curry_depth_features0_1=torch.tensor([],device=x_0.device)
                curry_depth_features0_2=torch.tensor([],device=x_0.device)
                curry_depth_features0_f=torch.tensor([],device=x_0.device)
                curry_depth_features1_1=torch.tensor([],device=x_0.device)
                curry_depth_features1_2=torch.tensor([],device=x_0.device)
                curry_depth_features1_f=torch.tensor([],device=x_0.device)
                stage+=1
            else:
                weights_M1 = F.sigmoid(next(it_arch_parameters))
                weights_M2 = F.sigmoid(next(it_arch_parameters))
                weights_F = F.sigmoid(next(it_arch_parameters))
                s0_f, s1_f = s1_f, cell_f(s0_f, s1_f, torch.cat([s1_1,s1_2],dim=1), weights_F, None)
                s0_1, s1_1 = s1_1, cell_1(s0_1, s1_1, weights_M1, None)
                s0_2, s1_2 = s1_2, cell_2(s0_2, s1_2, weights_M2, None)
                curry_depth_features0_1 = torch.cat((curry_depth_features0_1,s0_1.unsqueeze(0)))
                curry_depth_features0_2 = torch.cat((curry_depth_features0_2,s0_2.unsqueeze(0)))
                curry_depth_features0_f = torch.cat((curry_depth_features0_f,s0_f.unsqueeze(0)))
                curry_depth_features1_1 =torch.cat((curry_depth_features1_1,s1_1.unsqueeze(0)))
                curry_depth_features1_2 =torch.cat((curry_depth_features1_2,s1_2.unsqueeze(0)))
                curry_depth_features1_f =torch.cat((curry_depth_features1_f,s1_f.unsqueeze(0)))

        if len(curry_depth_features1_1):
            s1_1 = (curry_depth_features1_1*self.mask_decoder[stage].view(-1,1,1,1)).sum(dim=0)/self.mask_decoder[stage].sum()
            s1_2 = (curry_depth_features1_2*self.mask_decoder[stage].view(-1,1,1,1)).sum(dim=0)/self.mask_decoder[stage].sum()
            s1_f = (curry_depth_features1_f*self.mask_decoder[stage].view(-1,1,1,1)).sum(dim=0)/self.mask_decoder[stage].sum()
            curry_depth_features1_1=torch.tensor([],device=x_0.device)
            curry_depth_features1_2=torch.tensor([],device=x_0.device)
            curry_depth_features1_f=torch.tensor([],device=x_0.device)
        out = self.mixlayer(torch.cat([s1_1, s1_2, s1_f], dim=1))
        out,_ = self.LSTM(out)
        out = self.outlater(out)

        return out

    def forward_discretize(self, x_0, x_1):
        #self.disc_encoder
        #self.disc_decoder
        it_arch_parameters = iter(self._arch_parameters)
        s0_1 = s1_1 = self.Module_Modal_1.stem(x_0)
        s0_2 = s1_2 = self.Module_Modal_2.stem(x_1)
        s0_f = s1_f = self.Module_Fusion.stem(torch.cat([x_0,x_1],dim=1))
        stage = 0
        for i, (cell_1, cell_f, cell_2, disc_flag) in enumerate(
                zip(self.Module_Modal_1.cells_encoder, self.Module_Fusion.cells_encoder,
                    self.Module_Modal_2.cells_encoder,self.disc_encoder)):
            if cell_f.reduction:
                weights_1_M1 = F.sigmoid(next(it_arch_parameters))
                weights_2_M1 = F.sigmoid(next(it_arch_parameters))
                weights_1_M2 = F.sigmoid(next(it_arch_parameters))
                weights_2_M2 = F.sigmoid(next(it_arch_parameters))
                weights_1_F = F.sigmoid(next(it_arch_parameters))
                weights_2_F = F.sigmoid(next(it_arch_parameters))
                if not disc_flag: continue
                s0_f, s1_f = s1_f, cell_f(s0_f, s1_f, torch.cat([s1_1,s1_2],dim=1), weights_1_F, weights_2_F)
                s0_1, s1_1 = s1_1, cell_1(s0_1, s1_1, weights_1_M1, weights_2_M1)
                s0_2, s1_2 = s1_2, cell_2(s0_2, s1_2, weights_1_M2, weights_2_M2)
            else:
                weights_M1 = F.sigmoid(next(it_arch_parameters))
                weights_M2 = F.sigmoid(next(it_arch_parameters))
                weights_F = F.sigmoid(next(it_arch_parameters))
                if not disc_flag: continue
                s0_f, s1_f = s1_f, cell_f(s0_f, s1_f, torch.cat([s1_1,s1_2],dim=1), weights_F, None)
                s0_1, s1_1 = s1_1, cell_1(s0_1, s1_1, weights_M1, None)
                s0_2, s1_2 = s1_2, cell_2(s0_2, s1_2, weights_M2, None)

        s1_1,_ = self.Module_Modal_1.LSTM(s1_1)
        s1_2,_ = self.Module_Modal_2.LSTM(s1_2)
        s1_f,_ = self.Module_Fusion.LSTM(s1_f)

        s0_1 = s1_1
        s0_2 = s1_2
        s0_f = s1_f
        for i, (cell_1, cell_f, cell_2,disc_flag) in enumerate(
                zip(self.Module_Modal_1.cells_decoder, self.Module_Fusion.cells_decoder,
                    self.Module_Modal_2.cells_decoder,self.disc_decoder)):
            if cell_f.reduction:
                weights_1_M1 = F.sigmoid(next(it_arch_parameters))
                weights_2_M1 = F.sigmoid(next(it_arch_parameters))
                weights_1_M2 = F.sigmoid(next(it_arch_parameters))
                weights_2_M2 = F.sigmoid(next(it_arch_parameters))
                weights_1_F = F.sigmoid(next(it_arch_parameters))
                weights_2_F = F.sigmoid(next(it_arch_parameters))
                if not disc_flag: continue
                s0_f, s1_f = s1_f, cell_f(s0_f, s1_f, torch.cat([s1_1,s1_2],dim=1), weights_1_F, weights_2_F)
                s0_1, s1_1 = s1_1, cell_1(s0_1, s1_1, weights_1_M1, weights_2_M1)
                s0_2, s1_2 = s1_2, cell_2(s0_2, s1_2, weights_1_M2, weights_2_M2)
            else:
                weights_M1 = F.sigmoid(next(it_arch_parameters))
                weights_M2 = F.sigmoid(next(it_arch_parameters))
                weights_F = F.sigmoid(next(it_arch_parameters))
                if not disc_flag: continue
                s0_f, s1_f = s1_f, cell_f(s0_f, s1_f, torch.cat([s1_1,s1_2],dim=1), weights_F, None)
                s0_1, s1_1 = s1_1, cell_1(s0_1, s1_1, weights_M1, None)
                s0_2, s1_2 = s1_2, cell_2(s0_2, s1_2, weights_M2, None)

        out = self.mixlayer(torch.cat([s1_1, s1_2, s1_f], dim=1))
        out,_ = self.LSTM(out)
        out = self.outlater(out)

        return out

    def discretization(self,disc_encoder,disc_decoder):
        self.disc_encoder = disc_encoder
        self.disc_decoder = disc_decoder
        self.forward = self.forward_discretize

    def setmask(self,type,stage,depth):
        assert type=='encoder' or 'decoder', 'type should be encoder or decoder'
        if type == 'encoder':
            curry_mask = self.mask_encoder
        else :
            curry_mask = self.mask_decoder
        depth_mask = torch.zeros_like(curry_mask[stage])
        depth_mask[depth]=1
        curry_mask[stage] = depth_mask
        if type == 'encoder':
            self.mask_encoder = curry_mask
        else :
            self.mask_decoder = curry_mask

    def removeDepth(self,type,stage,depth):
        assert type=='encoder' or 'decoder', 'type should be encoder or decoder'
        if type == 'encoder':
            curry_mask = self.mask_encoder
        else :
            curry_mask = self.mask_decoder
        depth_mask = torch.ones_like(curry_mask[stage])
        depth_mask[depth]=0
        curry_mask[stage] = depth_mask
        if type == 'encoder':
            self.mask_encoder = curry_mask
        else :
            self.mask_decoder = curry_mask

    def _initialize_alphas(self):
        self._arch_parameters = []
        # k代表是stride为1的cell的连接数，k_sp是special的意思，代表上下采样的连接数
        # For modal-1
        k_M = sum(1 for i in range(self._steps) for n in range(2 + i))
        k_sp_M = 0
        k_sp_normal_M = 0
        for i in range(self._steps):
            for j in range(2 + i):
                if j < 2:
                    k_sp_M += 1
                else:
                    k_sp_normal_M += 1
        k_F = sum(1 for i in range(self._steps) for n in range(3 + i))
        k_sp_F = 0
        k_sp_normal_F = 0
        for i in range(self._steps):
            for j in range(3 + i):
                if j < 3:
                    k_sp_F += 1
                else:
                    k_sp_normal_F += 1

        num_ops = len(PRIMITIVES)
        num_ops_up = len(PRIMITIVES_UP)
        num_ops_down = len(PRIMITIVES_DOWN)

        for index, layer in enumerate(self._layers_encoder):
            if layer==1:
                self._arch_parameters.append(Variable(torch.zeros(k_M, num_ops).cuda(), requires_grad=True))#M1
                self._arch_parameters.append(Variable(torch.zeros(k_M, num_ops).cuda(), requires_grad=True))#M2
                self._arch_parameters.append(Variable(torch.zeros(k_F, num_ops).cuda(), requires_grad=True))#M3
            else:
                self._arch_parameters.append(Variable(torch.zeros(k_sp_M, num_ops_down).cuda(), requires_grad=True))
                self._arch_parameters.append(Variable(torch.zeros(k_sp_normal_M, num_ops).cuda(), requires_grad=True))#M1
                self._arch_parameters.append(Variable(torch.zeros(k_sp_M, num_ops_down).cuda(), requires_grad=True))
                self._arch_parameters.append(Variable(torch.zeros(k_sp_normal_M, num_ops).cuda(), requires_grad=True))#M2
                self._arch_parameters.append(Variable(torch.zeros(k_sp_F, num_ops_down).cuda(), requires_grad=True))
                self._arch_parameters.append(Variable(torch.zeros(k_sp_normal_F, num_ops).cuda(), requires_grad=True))#MF
        for index, layer in enumerate(self._layers_decoder):
            if layer==1:
                self._arch_parameters.append(Variable(torch.zeros(k_M, num_ops).cuda(), requires_grad=True))#M1
                self._arch_parameters.append(Variable(torch.zeros(k_M, num_ops).cuda(), requires_grad=True))#M2
                self._arch_parameters.append(Variable(torch.zeros(k_F, num_ops).cuda(), requires_grad=True))#M3
            else:
                self._arch_parameters.append(Variable(torch.zeros(k_sp_M, num_ops_up).cuda(), requires_grad=True))
                self._arch_parameters.append(Variable(torch.zeros(k_sp_normal_M, num_ops).cuda(), requires_grad=True))#M1
                self._arch_parameters.append(Variable(torch.zeros(k_sp_M, num_ops_up).cuda(), requires_grad=True))
                self._arch_parameters.append(Variable(torch.zeros(k_sp_normal_M, num_ops).cuda(), requires_grad=True))#M2
                self._arch_parameters.append(Variable(torch.zeros(k_sp_F, num_ops_up).cuda(), requires_grad=True))
                self._arch_parameters.append(Variable(torch.zeros(k_sp_normal_F, num_ops).cuda(), requires_grad=True))#MF

    def _loss(self, input, target):
        logits = self(input[0],input[1])
        return self._criterion(logits, target)

    def arch_parameters(self):
        return self._arch_parameters

if __name__ == '__main__':
    layer = [1,1,1,2, 1,1,1, 2, 1,1,1, 2,1,1,1]
    model = SandwichNetwork_Ind(48, layer, layer, torch.nn.MSELoss()).cuda()
    # print(model.mask_encoder)
    x_1 = torch.randn(3, 136, 80).cuda()
    x_2 = torch.randn(3, 128, 80).cuda()
    out = model(x_1, x_2)
    print(out.shape)