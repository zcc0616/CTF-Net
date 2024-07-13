import torch
import torch.nn as nn
import torch.nn.functional as F
from net.ResNet import resnet50
from math import log
from net.Res2Net import res2net50_v1b_26w_4s
from net.pvtv2 import pvt_v2_b2

class ConvBNR(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False):
        super(ConvBNR, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size, stride=stride, padding=dilation, dilation=dilation, bias=bias),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class Conv1x1(nn.Module):
    def __init__(self, inplanes, planes):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, 1)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)

class BAM(nn.Module):
    def __init__(self):
        super(BAM, self).__init__()
        self.reduce1 = Conv1x1(256, 64)
        self.reduce2 = Conv1x1(512, 256)
        self.conv = ConvBNR(256, 256, 3)
        self.conv_out = nn.Conv2d(256, 1, 1)
        self.block = nn.Sequential(
            ConvBNR(256 + 64, 256, 3),
            ConvBNR(256, 256, 3))

        self.sa = SpatialAttention()

    def forward(self, x1, t4):
        size = x1.size()[2:]
        x1 = self.reduce1(x1)
        t4 = self.reduce2(t4)
        t4 = F.interpolate(t4, size, mode='bilinear', align_corners=False)
        x_t = torch.cat((t4, x1), dim=1)
        xe = self.block(x_t)

        xe_sa = self.sa(xe) * xe
        xe_conv = self.conv(xe_sa)
        out = xe_conv + xe
        out = self.conv_out(out)

        return out

class CDFM(nn.Module):
    def __init__(self, channel1, channel2):
        super(CDFM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.sigmoid = nn.Sigmoid()
        self.conv1 = ConvBNR(channel2, channel2, 1)
        self.conv2 = ConvBNR(channel2, channel2, 1)
        self.conv3 = Conv1x1(channel1, channel2)
        self.conv4 = Conv1x1(channel2+channel2, channel2)  
        self.conv5 = Conv1x1(channel1, channel2)
        self.conv6 = Conv1x1(channel2, channel2)
        self.conv7 = Conv1x1(channel1+channel2, channel2)  
        t = int(abs((log(channel2, 2) + 1) / 2))
        k = t if t % 2 else t + 1
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)

        channel = (channel2)//4
        self.local_att = nn.Sequential(
            nn.Conv2d(channel2, channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channel2),
            nn.Sigmoid()
        )


    def forward(self, x, t):
        w_A = self.avg_pool(t)
        w_M = self.max_pool(t)
        wg = w_A + w_M
        wg = self.conv1d(wg.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        wg = self.sigmoid(wg)
        x_g = self.conv3(x)
        x_g = x_g * wg
        x_g = torch.cat((x_g, t), dim=1)
        x_g = self.conv4(x_g)

        wl = self.conv5(x)
        wl = self.local_att(wl)        
        t_l = self.conv6(t)
        t_l = wl*t_l
        t_l = torch.cat((t_l, x), dim=1)
        t_l = self.conv7(t_l)

        out = x_g + t_l

        return out

class FEM(nn.Module):
    def __init__(self, hchannel, channel, ochannel):
        super(FEM, self).__init__()
        self.conv1 = nn.Conv2d(hchannel+channel, 2, kernel_size=3, padding=1)
        self.conv2 = ConvBNR(hchannel+channel, ochannel, 3)
        self.sigmoid = nn.Sigmoid()
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, lf, hf, et, pr):
        if lf.size()[2:] != hf.size()[2:]:
            hf = F.interpolate(hf, size=lf.size()[2:], mode='bilinear', align_corners=False)
        if lf.size()[2:] != et.size()[2:]:
            et = F.interpolate(et, size=lf.size()[2:], mode='bilinear', align_corners=False)
        if lf.size()[2:] != pr.size()[2:]:
            pr = F.interpolate(pr, size=lf.size()[2:], mode='bilinear', align_corners=False)
        att = et + pr
        hf_a = hf * att
        lf_a = lf * att

        concat_fea = torch.cat((hf_a, lf_a),dim=1)
        conv_fea = self.conv1(concat_fea)
        Gi = self.avg_pool(self.sigmoid(conv_fea))
        Gi_l, Gi_h = torch.split(Gi, 1, dim=1)

        hf_g = hf_a * Gi_h
        lf_g = lf_a * Gi_l

        out = torch.cat((hf_g, lf_g),dim=1)
        out = self.conv2(out)

        return out

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.resnet = res2net50_v1b_26w_4s(pretrained=True)
        self.trans=pvt_v2_b2()  # [64, 128, 320, 512]
        path = './pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.trans.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.trans.load_state_dict(model_dict)
        # if self.training:
        # self.initialize_weights()

        self.bam = BAM()

        self.cdfm1 = CDFM(256, 64)
        self.cdfm2 = CDFM(512, 128)
        self.cdfm3 = CDFM(1024, 320)
        self.cdfm4 = CDFM(2048, 512)

        self.reduce1 = Conv1x1(64, 64)
        self.reduce2 = Conv1x1(128, 128)
        self.reduce3 = Conv1x1(320, 256)
        self.reduce4 = Conv1x1(512, 256)

        self.fem3 = FEM(256, 256, 256)
        self.fem2 = FEM(256, 128, 128)
        self.fem1 = FEM(128, 64, 64)
     
        self.predictor1 = nn.Conv2d(64, 1, 1)
        self.predictor2 = nn.Conv2d(128, 1, 1)
        self.predictor3 = nn.Conv2d(256, 1, 1)

        self.block = nn.Sequential(
            ConvBNR(512 + 320 + 128 + 64, 256, 3),
            ConvBNR(256, 256, 3),
            nn.Conv2d(256, 1, 1))

    # def initialize_weights(self):
    # model_state = torch.load('./models/resnet50-19c8e357.pth')
    # self.resnet.load_state_dict(model_state, strict=False)

    def forward(self, x):
        x1, x2, x3, x4 = self.resnet(x)
        pvt = self.trans(x)
        t1, t2 ,t3 , t4=pvt[0], pvt[1], pvt[2], pvt[3]

        edge = self.bam(x1, t4)
        edge_att = torch.sigmoid(edge)

        c1 = self.cdfm1(x1, t1)
        c2 = self.cdfm2(x2, t2)
        c3 = self.cdfm3(x3, t3)
        c4 = self.cdfm4(x4, t4)

        c1_4 = F.interpolate(c1, size=c4.size()[2:], mode='bilinear', align_corners=False)
        c2_4 = F.interpolate(c2, size=c4.size()[2:], mode='bilinear', align_corners=False)
        c3_4 = F.interpolate(c3, size=c4.size()[2:], mode='bilinear', align_corners=False)
        o4 = self.block(torch.cat((c4, c1_4, c2_4, c3_4), dim=1))
        o4_r = torch.sigmoid(o4)
        o4 = F.interpolate(o4, scale_factor=32, mode='bilinear', align_corners=False)

        c1f = self.reduce1(c1)
        c2f = self.reduce2(c2)
        c3f = self.reduce3(c3)
        c4f = self.reduce4(c4)

        f3 = self.fem3(c3f, c4f, edge_att, o4_r)
        o3 = self.predictor3(f3)
        o3_r = torch.sigmoid(o3)
        o3 = F.interpolate(o3, scale_factor=16, mode='bilinear', align_corners=False)

        f2 = self.fem2(c2f, f3, edge_att, o3_r)
        o2 = self.predictor2(f2)
        o2_r = torch.sigmoid(o2)
        o2 = F.interpolate(o2, scale_factor=8, mode='bilinear', align_corners=False)

        f1 = self.fem1(c1f, f2, edge_att, o2_r)
        o1 = self.predictor1(f1)
        o1 = F.interpolate(o1, scale_factor=4, mode='bilinear', align_corners=False)

        oe = F.interpolate(edge_att, scale_factor=4, mode='bilinear', align_corners=False)

        return o4, o3, o2, o1, oe