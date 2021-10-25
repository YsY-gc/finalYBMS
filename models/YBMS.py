
from torch import nn
import torch
import numpy as np
from PIL import Image
from models.net_tools import UNET, AS, S, UNET_mini
from torch.autograd import Variable

# z=[0.710mm 1.185mm 1.685mm 2.178mm 2.682mm 3.180mm]
class YBMS(nn.Module):
    def __init__(self,size = (768,768)):
        super(YBMS, self).__init__()

        self.cnn1 = UNET_mini()
        self.cnn2 = UNET_mini()
        self.cnn3 = UNET_mini()
        self.cnn4 = UNET_mini()
        self.cnn5 = UNET_mini()
        # self.cnn6 = UNET()
        # self.cnn7 = UNET_mini(2)
        self.cnn8 = UNET(2)

        self.as0_3 = AS(1.685e-3,size=size)

        self.as1_0 = AS(-0.710e-3,size=size)
        self.as2_0 = AS(-1.185e-3,size=size)
        self.as2_1 = AS(-0.475e-3,size=size)
        self.as3_0 = AS(-1.685e-3,size=size)

        self.as0_5 = AS(2.682e-3,size=size)

        self.as0_1 = AS(0.710e-3,size=size)
        self.as3_1 = AS(-0.975e-3,size=size)        #***
        self.as3_2 = AS(-0.500e-3,size=size)
        self.as1_2 = AS(0.475e-3,size=size)         #***
        self.as2_3 = AS(0.500e-3,size=size)         #***
        self.as2_4 = AS(0.993e-3,size=size)     #**
        self.as1_3 = AS(0.975e-3,size=size) #*

        self.as3_4 = AS(0.493e-3,size=size)
        self.as4_3 = AS(-0.493e-3,size=size)

        self.as4_5 = AS(0.504e-3,size=size)
        self.as5_4 = AS(-0.504e-3,size=size)

        self.as4_6 = AS(1.002e-3,size=size)     #**
        self.as3_5 = AS(0.997e-3,size=size) #*

        self.as5_6 = AS(0.498e-3,size=size) #*
        self.as6_1 = AS(-2.47e-3,size=size) #*
        self.as6_2 = AS(-1.995e-3,size=size)    #** 
        
        self.s = S()

        self.alpha1 = torch.nn.Parameter((torch.ones(1)*0.05).to('cuda').requires_grad_(True))
        # self.alpha11 = torch.nn.Parameter((torch.ones(1)*0.05).to('cuda').requires_grad_(True))
        self.alpha2 = torch.nn.Parameter((torch.ones(1)*0.05).to('cuda').requires_grad_(True))
        # self.alpha22 = torch.nn.Parameter((torch.ones(1)*0.05).to('cuda').requires_grad_(True))
        self.alpha3 = torch.nn.Parameter((torch.ones(1)*0.05).to('cuda').requires_grad_(True))
        self.alpha4 = torch.nn.Parameter((torch.ones(1)*0.05).to('cuda').requires_grad_(True))  #4当11
        self.alpha5 = torch.nn.Parameter((torch.ones(1)*0.05).to('cuda').requires_grad_(True))  #5当22

        self.alpha6 = torch.nn.Parameter((torch.ones(1)*0.05).to('cuda').requires_grad_(True))
        self.alpha7 = torch.nn.Parameter((torch.ones(1)*0.05).to('cuda').requires_grad_(True))  
        self.alpha8 = torch.nn.Parameter((torch.ones(1)*0.05).to('cuda').requires_grad_(True))  
        self.alpha9 = torch.nn.Parameter((torch.ones(1)*0.05).to('cuda').requires_grad_(True))  
        # self.alpha6 = None
        # self.alpha7 = None
        # self.alpha8 = None

    def forward(self,x1,x2,x3,x4,x5,x6):


        U1_ob = self.cnn1(x1)
        U2_ob = self.cnn2(x2)
        U3_ob = self.cnn3(x3)
        U4_ob = self.cnn4(x4)
        U5_ob = self.cnn5(x5)


        # U4_ob = self.cnn4(x4)
        # U5_ob = self.cnn5(x5)
        # U6_ob = self.cnn6(x6)
        # U_init = (self.as1_0(x1) + self.as2_0(x2) + self.as3_0(x3)) / 3 

        # iter 1

        U4_pr = self.as5_4(U5_ob)
        U4 = self.alpha4 * 10 * U4_ob + (1 - self.alpha4 * 10) * U4_pr

        U3_pr = self.as4_3(U4)
        U3 = self.alpha3 * 10 * U3_ob + (1 - self.alpha3 * 10) * U3_pr

        U2_pr = self.as3_2(U3)
        U2 = self.alpha2 * 10 * U2_ob + (1 - self.alpha2 * 10) * U2_pr

        U1_pr = self.as2_1(U2)
        U1 = self.alpha1 * 10 * U1_ob + (1 - self.alpha1 * 10) * U1_pr

        U0 = self.as1_0(U1)#===========================

        #iter 2
        # U0 = self.cnn7(U0)

        U5_pr = self.as0_5(U0)
        U5 = self.alpha5 * 10 * U5_ob + (1 - self.alpha5 * 10) * U5_pr

        U4_pr = self.as5_4(U5)
        U4 = self.alpha6 * 10 * U4_ob + (1 - self.alpha6 * 10) * U4_pr

        U3_pr = self.as4_3(U4)
        U3 = self.alpha7 * 10 * U3_ob + (1 - self.alpha7 * 10) * U3_pr

        U2_pr = self.as3_2(U3)
        U2 = self.alpha8 * 10 * U2_ob + (1 - self.alpha8 * 10) * U2_pr

        U1_pr = self.as2_1(U2)
        U1 = self.alpha9 * 10 * U1_ob + (1 - self.alpha9 * 10) * U1_pr

        U0 = self.as1_0(U1)#===========================

        #iter 3
        # U0 = self.cnn7(U0)

        # U3_pr = self.as0_3(U0)
        # U3 = self.alpha6 * 10 * U3_ob + (1 - self.alpha6 * 10) * U3_pr

        # U2_pr = self.as3_2(U3)
        # U2 = self.alpha7 * 10 * U2_ob + (1 - self.alpha7 * 10) * U2_pr

        # U1_pr = self.as2_1(U2)
        # U1 = self.alpha8 * 10 * U1_ob + (1 - self.alpha8 * 10) * U1_pr

        # U0 = self.as1_0(U1)#===========================


        U0 = self.cnn8(U0)

        U1 = self.as0_1(U0)
        U2 = self.as1_2(U1)
        U3 = self.as2_3(U2)
        U4 = self.as3_4(U3)
        U5 = self.as4_5(U4)
        
        U_test = None

        a1 = self.alpha1
        a2 = self.alpha2
        a3 = self.alpha3
        a4 = self.alpha4
        a5 = self.alpha5


        x1p = self.s(U1)
        x2p = self.s(U2)
        x3p = self.s(U3)
        x4p = self.s(U4)
        x5p = self.s(U5)
        # x6p = self.s(U6)
       
        return x1p, x2p, x3p, x4p, x5p, U1, a1, a2, a3, a4, a5

def build_model(size=(768,768)):
    net = YBMS(size=size)
    net.load_state_dict(torch.load("I:\\finalYBMS\\params\\params1\\bestparams.pth"))
    net.cnn8 = UNET(2)

    for param in net.parameters():
        param.requires_grad = False
    for param in net.cnn8.parameters():
        param.requires_grad = True

    net.alpha1.requires_grad  = True
    net.alpha2.requires_grad  = True
    net.alpha3.requires_grad  = True
    net.alpha4.requires_grad  = True
    net.alpha5.requires_grad  = True
    net.alpha6 = torch.nn.Parameter((torch.ones(1)*0.05).to('cuda').requires_grad_(True))
    net.alpha7 = torch.nn.Parameter((torch.ones(1)*0.05).to('cuda').requires_grad_(True))  
    net.alpha8 = torch.nn.Parameter((torch.ones(1)*0.05).to('cuda').requires_grad_(True))

    return net  
  
# x= torch.rand(1,2,512,512)
# net = UNET(2)
# y = net(x)
# print(y.shape)


# x = Image.open("I:\\YNMT\\DATASETS\\traindata\\07\\2.bmp")


# x = np.array(x)
# x = torch.from_numpy(x).float()


# print(x.shape)

# x = x.unsqueeze(0).unsqueeze(0)


# x_img = torch.zeros(x.shape)

# x = torch.cat((x,x_img),1)
# print(x.shape)
# x = torch.rand(1,2,256,256)
# net = YBMS()
# # y = net(x)
# for i, j in net.named_parameters():
#     # print(1)
#     if i == 'alpha1' or i == 'alpha2' or i== 'alpha3':
#         print(i,":",j)

    # print(j)
# print(y.shape)

# y = y.squeeze(0)
# y_real = y[0,:,:]
# y_real = y_real.squeeze(0)
# y_imag = y[1,:,:]
# y_imag = y_imag.squeeze(0)
# y = torch.sqrt(pow(y_real,2)+pow(y_imag,2))
# y = y.numpy()
# print(y)
# y = Image.fromarray(y).convert('L')
# y.show()

# x = torch.rand((1,2,4,4))
# net = YBMS()
# print(net)
# para = list(net.parameters())
# print(para)
# for params in net.parameters():
#     print(params)
# y = net(x)
# print(x)
# print("====")
# print(y)