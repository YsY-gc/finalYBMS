
from torch import ne, nn
import torch
import numpy as np
from PIL import Image
from models.net_tools import *
from torch.autograd import Variable
from models.YBMS import *

# z=[0.710mm 1.185mm 1.685mm 2.178mm 2.682mm 3.180mm]
class YBMS5_2(nn.Module):
    def __init__(self,size = (768,768)):
        super(YBMS5_2, self).__init__()

        self.cnn1 = UNET_mini3()
        self.cnn3 = UNET_mini3()
        # self.cnn5 = UNET_mini3()

        self.cnn8 = UNET(2)



        self.as1_0 = AS(-0.710e-3,size=size)
        self.as0_1 = AS(0.710e-3,size=size)


        # self.as0_5 = AS(2.682e-3,size=size)

        # self.as5_3 = AS(-0.997e-3,size=size)
        # self.as3_5 = AS(0.997e-3,size=size)

        self.as0_3 = AS(1.685e-3,size=size)

        self.as3_1 = AS(-0.975e-3,size=size)
        self.as1_3 = AS(0.975e-3,size=size)



        
        self.s = S()

        self.alpha1 = torch.nn.Parameter((torch.ones(1)*0.05).to('cuda').requires_grad_(True))

        # self.alpha3 = torch.nn.Parameter((torch.ones(1)*0.05).to('cuda').requires_grad_(True))

        # self.alpha5 = torch.nn.Parameter((torch.ones(1)*0.05).to('cuda').requires_grad_(True))  #5当22


        self.alpha7 = torch.nn.Parameter((torch.ones(1)*0.05).to('cuda').requires_grad_(True))  

        self.alpha9 = torch.nn.Parameter((torch.ones(1)*0.05).to('cuda').requires_grad_(True))  


    def forward(self,x1,x3):


        U1_ob = self.cnn1(x1)

        U3_ob = self.cnn3(x3)

        # U5_ob = self.cnn5(x5)



        # U3_pr = self.as5_3(U5_ob)
        # U3 = self.alpha3 * 10 * U3_ob + (1 - self.alpha3 * 10) * U3_pr


        U1_pr = self.as3_1(U3_ob)
        U1 = self.alpha1 * 10 * U1_ob + (1 - self.alpha1 * 10) * U1_pr

        U0 = self.as1_0(U1)#===========================

        #iter 2
        # U0 = self.cnn7(U0)

        # U5_pr = self.as0_5(U0)
        # U5 = self.alpha5 * 10 * U5_ob + (1 - self.alpha5 * 10) * U5_pr


        U3_pr = self.as0_3(U0)
        U3 = self.alpha7 * 10 * U3_ob + (1 - self.alpha7 * 10) * U3_pr


        U1_pr = self.as3_1(U3)
        U1 = self.alpha9 * 10 * U1_ob + (1 - self.alpha9 * 10) * U1_pr

        U0 = self.as1_0(U1)#===========================



        U0 = self.cnn8(U0)

        U1 = self.as0_1(U0)

        U3 = self.as1_3(U1)

        # U5 = self.as3_5(U3)
        
        U_test = None

        a1 = self.alpha1

        # a3 = self.alpha3

        # a5 = self.alpha5


        x1p = self.s(U1)

        x3p = self.s(U3)

        # x5p = self.s(U5)

       
        return x1p, x3p, U1, a1

def build_model(size=(768,768)):
    net_skin = YBMS(size=size)
    net_skin.load_state_dict(torch.load("I:\\finalYBMS\\params\\params9\\bestparams.pth"))
    net = YBMS5_2(size=size)

    net.cnn1 = net_skin.cnn1
    net.cnn3 = net_skin.cnn3
    # net.cnn5 = net_skin.cnn5
    net.cnn8 = net_skin.cnn8
    net.alpha1 = net_skin.alpha1
    # net.alpha3 = net_skin.alpha3
    # net.alpha5 = net_skin.alpha5
    net.alpha7 = net_skin.alpha7
    net.alpha9 = net_skin.alpha9    
    # net.cnn8 = UNET(2)

    # for param in net.parameters():
    #     param.requires_grad = False
    # for param in net.cnn8.parameters():
    #     param.requires_grad = True

    # net.alpha1.requires_grad  = True
    # net.alpha2.requires_grad  = True
    # net.alpha3.requires_grad  = True
    # net.alpha4.requires_grad  = True
    # net.alpha5.requires_grad  = True
    # net.alpha6.requires_grad  = True
    # net.alpha7.requires_grad  = True
    # net.alpha8.requires_grad  = True
    # net.alpha9.requires_grad  = True
    # net.alpha6 = torch.nn.Parameter((torch.ones(1)*0.05).to('cuda').requires_grad_(True))
    # net.alpha7 = torch.nn.Parameter((torch.ones(1)*0.05).to('cuda').requires_grad_(True))  
    # net.alpha8 = torch.nn.Parameter((torch.ones(1)*0.05).to('cuda').requires_grad_(True))

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