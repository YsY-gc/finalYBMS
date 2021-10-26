
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from dcn_v2 import DCN


'''
@description: 下采样路径
@param {*}
@return {*}
'''
class DownSample_blk(nn.Module):
    def __init__(self, in_channels, out_channels, bias = 0.1):
        super(DownSample_blk,self).__init__()
        self.conv1 = nn.Conv2d(in_channels,(in_channels+out_channels)//2, kernel_size=3,padding=1,stride=1,bias=bias)
        self.conv2 = nn.Conv2d((in_channels+out_channels)//2,out_channels,kernel_size=3,padding=1,stride=1,bias=bias)
        self.conv3 = nn.Conv2d(in_channels,out_channels,kernel_size=1,bias=bias)      #使通道数相同

    def forward(self, X):
        Y = torch.relu(self.conv1(X)) 
        Y = torch.relu(self.conv2(Y))    
        X = self.conv3(X)
        return X + Y

def DownSample_lay():
    return nn.MaxPool2d(2, stride=2, padding=0)

'''
@description: 上采样路径  normal nn.Conv2d
@param {*} in_channels
@param {*} out_channels
@param {*} bias
@return {*}
'''
def UpSample_blk(in_channels, out_channels, bias=0.1):
    blk = []
    blk.append(nn.Conv2d(in_channels,(in_channels+out_channels)//2,kernel_size=3,padding=1,stride=1,bias=bias))
    blk.append(nn.ReLU())
    blk.append(nn.Conv2d((in_channels+out_channels)//2,out_channels,kernel_size=3,padding=1,stride=1,bias=bias))
    blk.append(nn.ReLU())
    return nn.Sequential(*blk)
'''
@description:   UpSample_blk with dcn
@param {*} in_channels
@param {*} out_channels
@param {*} bias
@return {*}
'''
def UpSample_blk_dcn(in_channels, out_channels, bias=0.1):
    blk = []
    blk.append(nn.Conv2d(in_channels,(in_channels+out_channels)//2,kernel_size=3,padding=1,stride=1,bias=bias))
    blk.append(nn.ReLU()) 
    blk.append(nn.Conv2d((in_channels+out_channels)//2,out_channels,kernel_size=3,padding=1,stride=1,bias=bias))
    blk.append(nn.ReLU()) 
    return nn.Sequential(*blk)

def UpSample_lay(in_channesl, out_channels):
    return nn.ConvTranspose2d(in_channesl,out_channels,2,stride=2)

'''
@description: 普通卷积块
@param {*}
@return {*}
'''
def Conv_blk(in_channels, out_channels, bias=0.1):
    blk = []
    blk.append(nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1,bias=bias))
    return nn.Sequential(*blk)
'''
@description: DCN卷积块
@param {*}
@return {*}
'''
class Dcn_blk(nn.Module):
    def __init__(self, in_channels, out_channels,group=1):
        super(Dcn_blk, self).__init__()
        self.conv = DCN(in_channels, out_channels, kernel_size=(3, 3), stride=1, padding=1, dilation=1, deformable_groups=group)

    def forward(self, x):
        x = self.conv(x) 
        return x


'''
@description: 
@param {*}
@return {*}
'''
class UNET(nn.Module):
    def __init__(self,inChannels=1):
        super(UNET,self).__init__()

        self.dsb1 = DownSample_blk(inChannels,32)
        self.dsl1 = DownSample_lay()

        self.dsb2 = DownSample_blk(32,64)
        self.dsl2 = DownSample_lay()

        self.dsb3 = DownSample_blk(64,128)
        self.dsl3 = DownSample_lay()

        self.dsb4 = DownSample_blk(128,256)
        self.dsl4 = DownSample_lay()

        self.cb1 = Conv_blk(256,256)

        self.usl1 = UpSample_lay(256,256)
        self.usb1 = UpSample_blk(512,128)

        self.usl2 = UpSample_lay(128,128)
        self.usb2 = UpSample_blk(256,64)

        self.usl3 = UpSample_lay(64,64)
        self.usb3 = UpSample_blk(128,32)

        self.usl4 = UpSample_lay(32,32)
        self.usb4 = UpSample_blk(64,32) 

        self.cb2 = Conv_blk(32,2) 

    def forward(self, X):
        cat1 = self.dsb1(X); D1 = self.dsl1(cat1)
        
        cat2 = self.dsb2(D1);D2 = self.dsl2(cat2)
        
        cat3 = self.dsb3(D2);D3 = self.dsl3(cat3)
        
        cat4 = self.dsb4(D3);D4 = self.dsl4(cat4)
        
        T = self.cb1(D4);U1 = self.usl1(T)
        tac1 = torch.cat((U1,cat4),1)

        T1 = self.usb1(tac1);U2 = self.usl2(T1)
        tac2 = torch.cat((U2,cat3),1)

        T2 = self.usb2(tac2);U3 = self.usl3(T2)
        tac3 = torch.cat((U3,cat2),1)

        T3 = self.usb3(tac3);U4 = self.usl4(T3)
        tac4 = torch.cat((U4,cat1),1)

        T4 = self.usb4(tac4)

        O = self.cb2(T4)
        return O

# net = UNET()
# x = torch.rand(1,1,512,512)
# y = net(x)
# print(y.shape)
# print(net)


#Angular spectrum spread
class AS(nn.Module):
    def __init__(self, d, psize=1.34e-6, Lambda=532e-9, size=(768,768)):
        super(AS,self).__init__()
        pointv = np.arange(1-size[1]/2,size[1]/2+1)
        pointu = np.arange(1-size[0]/2,size[0]/2+1)
        Lx = psize*size[1]
        Ly = psize*size[0]
        u, v = np. meshgrid(pointv, pointu)
        u = u/Lx*Lambda
        v = v/Ly*Lambda
        H = np.exp(1j*2*np.pi*d/Lambda*np.sqrt(1-pow(u,2)-pow(v,2)))
        H = np.fft.fftshift(H)
        c = np.real(H)  
        d = np.imag(H)
        self.c = torch.from_numpy(c).float().to('cuda')
        self.d = torch.from_numpy(d).float().to('cuda')
    def forward(self, x):
        if x.shape[1] == 1:
            img = torch.zeros(x.shape).to('cuda')
            x = torch.cat((x,img),1)
        x = x.squeeze(0).permute(1,2,0).contiguous()
        F = torch.fft(x, 2)
        a = F[:,:,0]
        b = F[:,:,1]
        T_real = a*self.c-b*self.d
        T_real = T_real.unsqueeze(2)
        T_imag = a*self.d+b*self.c
        T_imag = T_imag.unsqueeze(2)
        y = torch.ifft(torch.cat((T_real, T_imag), 2), 2)
        return y.permute(2,0,1).contiguous().unsqueeze(0)

#SRCNN
class SRCNN(nn.Module):
    def __init__(self, num_channels=1):
        super(SRCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(num_channels, 128, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv2d(64, 2, kernel_size=5, padding=5 // 2)
        self.relu = nn.PReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x

#cmos 振幅采样函数
class S(nn.Module):
    def __init__(self):
        super(S,self).__init__()

    def forward(self,x):
        x_real = x[:,0,:,:]
        x_imag = x[:,1,:,:]
        return (pow(x_real,2)+ pow(x_imag,2)).unsqueeze(1)



class UNET_mini(nn.Module):#1/2
    def __init__(self,inChannels=1):
        super(UNET_mini,self).__init__()

        self.dsb1 = DownSample_blk(inChannels,16)
        self.dsl1 = DownSample_lay()

        self.dsb2 = DownSample_blk(16,32)
        self.dsl2 = DownSample_lay()

        self.dsb3 = DownSample_blk(32,64)
        self.dsl3 = DownSample_lay()

        self.dsb4 = DownSample_blk(64,128)
        self.dsl4 = DownSample_lay()

        self.cb1 = Conv_blk(128,128)

        self.usl1 = UpSample_lay(128,128)
        self.usb1 = UpSample_blk(256,64)

        self.usl2 = UpSample_lay(64,64)
        self.usb2 = UpSample_blk(128,32)

        self.usl3 = UpSample_lay(32,32)
        self.usb3 = UpSample_blk(64,16)

        self.usl4 = UpSample_lay(16,16)
        self.usb4 = UpSample_blk(32,16) 

        self.cb2 = Conv_blk(16,2) 

    def forward(self, X):
        cat1 = self.dsb1(X); D1 = self.dsl1(cat1)
        
        cat2 = self.dsb2(D1);D2 = self.dsl2(cat2)
        
        cat3 = self.dsb3(D2);D3 = self.dsl3(cat3)
        
        cat4 = self.dsb4(D3);D4 = self.dsl4(cat4)
        
        T = self.cb1(D4);U1 = self.usl1(T)
        tac1 = torch.cat((U1,cat4),1)

        T1 = self.usb1(tac1);U2 = self.usl2(T1)
        tac2 = torch.cat((U2,cat3),1)

        T2 = self.usb2(tac2);U3 = self.usl3(T2)
        tac3 = torch.cat((U3,cat2),1)

        T3 = self.usb3(tac3);U4 = self.usl4(T3)
        tac4 = torch.cat((U4,cat1),1)

        T4 = self.usb4(tac4)

        O = self.cb2(T4)
        return O


class UNET_mini2(nn.Module):#1/4
    def __init__(self,inChannels=1):
        super(UNET_mini2,self).__init__()

        self.dsb1 = DownSample_blk(inChannels,8)
        self.dsl1 = DownSample_lay()

        self.dsb2 = DownSample_blk(8,16)
        self.dsl2 = DownSample_lay()

        self.dsb3 = DownSample_blk(16,32)
        self.dsl3 = DownSample_lay()

        self.dsb4 = DownSample_blk(32,64)
        self.dsl4 = DownSample_lay()

        self.cb1 = Conv_blk(64,64)

        self.usl1 = UpSample_lay(64,64)
        self.usb1 = UpSample_blk(128,32)

        self.usl2 = UpSample_lay(32,32)
        self.usb2 = UpSample_blk(64,16)

        self.usl3 = UpSample_lay(16,16)
        self.usb3 = UpSample_blk(32,8)

        self.usl4 = UpSample_lay(8,8)
        self.usb4 = UpSample_blk(16,8) 

        self.cb2 = Conv_blk(8,2) 

    def forward(self, X):
        cat1 = self.dsb1(X); D1 = self.dsl1(cat1)
        
        cat2 = self.dsb2(D1);D2 = self.dsl2(cat2)
        
        cat3 = self.dsb3(D2);D3 = self.dsl3(cat3)
        
        cat4 = self.dsb4(D3);D4 = self.dsl4(cat4)
        
        T = self.cb1(D4);U1 = self.usl1(T)
        tac1 = torch.cat((U1,cat4),1)

        T1 = self.usb1(tac1);U2 = self.usl2(T1)
        tac2 = torch.cat((U2,cat3),1)

        T2 = self.usb2(tac2);U3 = self.usl3(T2)
        tac3 = torch.cat((U3,cat2),1)

        T3 = self.usb3(tac3);U4 = self.usl4(T3)
        tac4 = torch.cat((U4,cat1),1)

        T4 = self.usb4(tac4)

        O = self.cb2(T4)
        return O


class UNET_mini3(nn.Module):#3/4
    def __init__(self,inChannels=1):
        super(UNET_mini3,self).__init__()

        self.dsb1 = DownSample_blk(inChannels,24)
        self.dsl1 = DownSample_lay()

        self.dsb2 = DownSample_blk(24,48)
        self.dsl2 = DownSample_lay()

        self.dsb3 = DownSample_blk(48,96)
        self.dsl3 = DownSample_lay()

        self.dsb4 = DownSample_blk(96,192)
        self.dsl4 = DownSample_lay()

        self.cb1 = Conv_blk(192,192)

        self.usl1 = UpSample_lay(192,192)
        self.usb1 = UpSample_blk(384,96)

        self.usl2 = UpSample_lay(96,96)
        self.usb2 = UpSample_blk(192,48)

        self.usl3 = UpSample_lay(48,48)
        self.usb3 = UpSample_blk(96,24)

        self.usl4 = UpSample_lay(24,24)
        self.usb4 = UpSample_blk(48,24) 

        self.cb2 = Conv_blk(24,2) 

    def forward(self, X):
        cat1 = self.dsb1(X); D1 = self.dsl1(cat1)
        
        cat2 = self.dsb2(D1);D2 = self.dsl2(cat2)
        
        cat3 = self.dsb3(D2);D3 = self.dsl3(cat3)
        
        cat4 = self.dsb4(D3);D4 = self.dsl4(cat4)
        
        T = self.cb1(D4);U1 = self.usl1(T)
        tac1 = torch.cat((U1,cat4),1)

        T1 = self.usb1(tac1);U2 = self.usl2(T1)
        tac2 = torch.cat((U2,cat3),1)

        T2 = self.usb2(tac2);U3 = self.usl3(T2)
        tac3 = torch.cat((U3,cat2),1)

        T3 = self.usb3(tac3);U4 = self.usl4(T3)
        tac4 = torch.cat((U4,cat1),1)

        T4 = self.usb4(tac4)

        O = self.cb2(T4)
        return O

# net = UNET_mini3(2)
# x = torch.rand(1,2,512,512)
# y = net(x)
# print(y.shape)
# print(net)