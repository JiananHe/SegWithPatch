import os
from time import time
import numpy as np
import torch
# from models.vnet import get_net
from models.td_unet import get_net
# from data_loader.btcv_data_loader import MyDataset
from data_loader.btcv_blc_data_loader import MyDataset
from torch.utils.data import DataLoader
from loss.ce_loss import CELoss
# from loss.dice_loss import DiceLoss
from loss.dice_loss_bg import DiceLoss
from utils import setup_seed, organs_properties
from val import dataset_prediction
from tensorboardX import SummaryWriter

# 超参数
organs_name = organs_properties['organs_name']
organs_weight = organs_properties['organs_weight']

on_server = True
resume_training = True
module_dir = './module/td_unet80-0.668-0.601.pth'

os.environ['CUDA_VISIBLE_DEVICES'] = '0' if on_server is False else '1,2,3'
torch.backends.cudnn.benchmark = True
Epoch = 2000
leaing_rate = 1e-4

batch_size = 1 if on_server else 1
num_workers = 2 if on_server else 1
pin_memory = True if on_server else False

# 梯度累计三次，即每三次batch更新一次网络参数
samples_every_vol = 8
grad_accum_steps = 3

# 设置种子，使得结果可复现
setup_seed(2018)

# 模型
net_name = 'td_unet'
net = get_net(True)
net = torch.nn.DataParallel(net).cuda() if on_server else net.cuda()
if resume_training:
    print('----------resume training-----------')
    net.load_state_dict(torch.load(module_dir))
    net.train()

# 损失函数
loss_func = DiceLoss()

# 优化器
opt = torch.optim.Adam(net.parameters(), lr=leaing_rate, weight_decay=0.0005)

# 学习率衰减
lr_decay = torch.optim.lr_scheduler.MultiStepLR(opt, [1000], gamma=0.1)
# lr_decay = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'max', factor=0.8, patience=30, verbose=True, cooldown=5, min_lr=1e-6)

# 训练
writer = SummaryWriter()
for epoch in range(1, Epoch+1):
    mean_loss = []
    epoch_start = time()

    # switch models to training mode, clear gradient accumulators
    net.train()
    opt.zero_grad()

    # 数据 (注意：虽然batch_size为1，但是实际包含samples_every_vol个patch)，每个epoch重新计算sampling count
    train_ds = MyDataset('csv_files/btcv_train_info.csv', organs_weight, samples_every_vol, grad_accum_steps)
    train_dl = DataLoader(train_ds, batch_size, True, num_workers=num_workers, pin_memory=pin_memory)

    for step, (ct, seg) in enumerate(train_dl): # 24 steps in BTCV
        ct = ct.squeeze(0).cuda()
        seg = seg.squeeze(0).cuda()

        # forward + backward + (after grad_accum_steps)optimize and clear grad
        output = net(ct)
        loss = loss_func(output, seg)
        mean_loss.append(loss)

        # loss regularization
        loss = loss / grad_accum_steps
        # back propagation (calculate grad)
        loss.backward()

        # update parameters of net
        if ((step + 1) % grad_accum_steps) == 0:
            # optimizer the net
            opt.step()  # update parameters of net
            opt.zero_grad()  # reset gradient

        s = 'epoch:{}, step:{}, loss:{:.3f}'.format(epoch, step, loss.item())
        os.system('echo %s' % s)

    mean_loss = sum(mean_loss) / len(mean_loss)

    # lr_decay.step(mean_acc)  # 如果10个epoch内train acc不上升，则lr = lr * 0.5
    lr_decay.step()

    writer.add_scalar('train/loss', mean_loss, epoch)
    # writer.add_scalar('lr', lr_decay.get_lr(), epoch)  # ReduceLROnPlateau没有get_lr()方法
    writer.add_scalar('lr', opt.param_groups[0]['lr'], epoch)

    s = '--- epoch:%d, mean loss:%.3f, epoch time:%.3f min\n' % (epoch, mean_loss, (time() - epoch_start) / 60)
    os.system('echo %s' % s)

    # valset accuracy
    if epoch % 20 is 0:
        os.system('echo %s' % "--------evaluation on validation set----------")
        val_eval_start = time()
        val_org_mean_dice = dataset_prediction(net, 'csv_files/btcv_val_info.csv')
        writer.add_scalars('valset orgs dice',
                           {name: val_org_mean_dice[i] for i, name in enumerate(organs_name)}, epoch)

        val_mean_dice = np.mean(val_org_mean_dice)
        s = "mean dice: %.3f, eval time: %.3f min" % (val_mean_dice, (time() - val_eval_start) / 60)
        os.system('echo %s' % s)
        writer.add_scalar("valset mean dice", val_mean_dice, epoch)

        # update organs weight according to dices
        organs_weight = 1.0 - np.array(val_org_mean_dice)
        os.system('echo %s' % "-----------------------------------------\n")

    # trainset accuracy
    if epoch % 40 is 0:
        os.system('echo %s' % "----------evaluation on training set-----------")
        train_eval_start = time()
        train_org_mean_dice = dataset_prediction(net, 'csv_files/btcv_train_info.csv')
        writer.add_scalars('trainset orgs dice',
                           {name: train_org_mean_dice[i] for i, name in enumerate(organs_name)}, epoch)

        train_mean_dice = np.mean(train_org_mean_dice)
        s = "mean dice: %.3f, eval time: %.3f min" % (train_mean_dice, (time() - train_eval_start) / 60)
        os.system('echo %s' % s)
        writer.add_scalar("trainset mean dice", train_mean_dice, epoch)
        torch.save(net.state_dict(), "./module/%s%d-%.3f-%.3f.pth"
                   % (net_name, epoch, np.mean(train_org_mean_dice), np.mean(val_org_mean_dice)))
        os.system('echo %s' % "-----------------------------------------\n")

    # 每十个个epoch保存一次模型参数
    if epoch % 10 is 0:
        torch.save(net.state_dict(), "./module/%s%d-%.3f.pth" % (net_name, epoch, mean_loss))

writer.close()
