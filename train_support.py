import torch
from matplotlib import pyplot as plt
import time
from tqdm import tqdm



def train_mu(net, train_iter, test_iter, optimizer, device, num_epochs):
    net = net.to(device)
    net.load_state_dict(torch.load("I:\\finalYBMS\\params\\params9\\lastparams.pth"))

    print("training on", device)
    loss1 = torch.nn.L1Loss()

    label_epoch = []
    label_loss = []
    best_check_point = 27.5351 #51
    for epoch in range(num_epochs):
        train_l_sum, n, batch_count, start = 0.0, 0, 0, time.time()
        tqdm_train = tqdm(train_iter, ncols=60)
        for x1, x2, x3, x4, x5, x6 in tqdm_train:
            x1 = x1.to(device); x2 = x2.to(device); x3 = x3.to(device)
            x4 = x4.to(device); x5 = x5.to(device); x6 = x6.to(device)
 
            optimizer.zero_grad()
            x1p, x2p, x3p, x4p, x5p, U1, a1, a2, a3, a4, a5 = net(x1, x2, x3, x4, x5, x6)

            A1_loss = loss1(x1p,x1)
            A2_loss = loss1(x2p,x2)
            A3_loss = loss1(x3p,x3)
            A4_loss = loss1(x4p,x4)
            A5_loss = loss1(x5p,x5)
            # A6_loss = loss1(x6p,x6)           

            l =  A1_loss +  A2_loss  + A3_loss + A4_loss + A5_loss
            l.backward()
            
            optimizer.step()
            train_l_sum += l.cpu().item()
            n+= x1.shape[0]
            batch_count += 1
        #测试集损失    
        tl, psnr = test_loss(test_iter, net, device=device)
        print('epoch %d,train loss %.4f, test loss %.4f, psnr %.4f,time %.1f sec, a1 %.4f, a2 %.4f, a3 %.4f, a4 %.4f, a5 %.4f\n'
              % (epoch + 1 + 3, train_l_sum / batch_count, tl, psnr,time.time() - start, a1, a2, a3, a4, a5))
        message = 'epoch %d,train loss %.4f, test loss %.4f, psnr %.4f,time %.1f sec, a1 %.4f, a2 %.4f, a3 %.4f, a4 %.4f, a5 %.4f\n' \
                 % (epoch + 1 + 3, train_l_sum / batch_count, tl, psnr,time.time() - start, a1, a2, a3, a4, a5)
        file = open("I:\\finalYBMS\\log\\log9.txt",'a+')
        file.write(message)
        file.close()
        #画损失图
        label_epoch.append(epoch)
        label_loss.append(tl)

        #每个epoch保存一次模型参数
        if psnr > best_check_point:
            best_check_point = psnr
            torch.save(net.state_dict(),'I:\\finalYBMS\\params\\params9\\bestparams.pth')
        
        torch.save(net.state_dict(),'I:\\finalYBMS\\params\\params9\\lastparams.pth')
    plt.plot(label_epoch,label_loss)
    plt.show()
    
def cal_psnr(y_hat,y):
    y_hat = y_hat.squeeze(0).squeeze(0)
    p1 = y_hat
    y = y.squeeze(0).squeeze(0)
    p2 = y

    diff = p1-p2
    diff *= diff
    mse = torch.mean(diff)
    psnr = 10*torch.log10(65535.0/mse)
    return psnr

def test_loss(test_iter, net, device):
    test_loss_sum, batch_count, psnr = 0.0, 0, 0.0
    with torch.no_grad():
        for x1, x2, x3, x4, x5, x6 in test_iter:
            x1 = x1.to(device); x2 = x2.to(device); x3 = x3.to(device)
            x4 = x4.to(device); x5 = x5.to(device); x6 = x6.to(device)

            x1p, x2p, x3p, x4p, x5p, U1, a1, a2, a3, a4, a5 = net(x1, x2, x3, x4, x5, x6)

            test_loss1 = torch.nn.L1Loss()

            A1_loss = test_loss1(x1p,x1)
            A2_loss = test_loss1(x2p,x2)
            A3_loss = test_loss1(x3p,x3)
            A4_loss = test_loss1(x4p,x4)
            A5_loss = test_loss1(x5p,x5)
            # A6_loss = test_loss1(x6p,x6)


            l_test = A1_loss +  A2_loss  + A3_loss + A4_loss + A5_loss

            x1_psnr = cal_psnr(x1,x1p)
            x2_psnr = cal_psnr(x2,x2p)
            x3_psnr = cal_psnr(x3,x3p)
            x4_psnr = cal_psnr(x4,x4p)
            x5_psnr = cal_psnr(x5,x5p)
            # x6_psnr = cal_psnr(x6,x6p)

            b_psnr = (x1_psnr+x2_psnr+x3_psnr+x4_psnr+x5_psnr) / 5

            test_loss_sum += float(l_test)
            psnr += b_psnr
            batch_count += 1
    return test_loss_sum / batch_count, psnr / batch_count


