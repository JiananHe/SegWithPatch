import os
from time import time
import numpy as np
import torch
from models.td_unet import get_net
from data_loader.batch_generator import get_data_loader
from torch.utils.data import DataLoader
from loss.dice_ce_loss import DC_and_CE_loss
from utils import *
from val import dataset_prediction
from tensorboardX import SummaryWriter


# 设置种子，使得结果可复现
setup_seed(2018)


def calc_batch_weights(image_names):
    batch_weights = []
    for img_name in image_names:
        if int(img_name[5:7]) > 60:
            batch_weights.append(batch_low_confidence_weight)
        else:
            batch_weights.append(1.0)
    return batch_weights


if __name__ == "__main__":
    # 模型
    net_name = 'td_unet'
    net = get_net(1)
    net = torch.nn.DataParallel(net).cuda()
    module_dir = './module/td_unet80-0.668-0.601.pth'
    if resume_training:
        print('----------resume training-----------')
        net.load_state_dict(torch.load(module_dir))
        net.train()

    # 损失函数
    loss_func = DC_and_CE_loss()

    # 优化器
    opt = torch.optim.Adam(net.parameters(), lr=leaing_rate, weight_decay=0.0005)

    # 训练
    writer = SummaryWriter()
    for epoch in range(1, Epoch+1):
        mean_loss = []
        epoch_start = time()

        # switch models to training mode, clear gradient accumulators
        net.train()
        opt.zero_grad()

        # 数据
        batch_loader = get_data_loader(class_weight, data_loader_processes)

        for step, batch in enumerate(batch_loader):
            data = batch['data']
            data = data.cuda().float()
            target = batch['seg']
            batch_weights = calc_batch_weights(batch['image_names'])

            # forward + backward + (after grad_accum_steps)optimize and clear grad
            outputs = net(data)
            loss = loss_func(outputs, target)

            # loss regularization
            loss = loss / grad_accum_steps
            # back propagation (calculate grad)
            loss.backward()
            mean_loss.append(loss.item())

            # update parameters of net
            if ((step + 1) % grad_accum_steps) == 0:
                opt.step()  # update parameters of net
                opt.zero_grad()  # reset gradient

            s = 'epoch:{}, step:{}, loss:{:.3f}'.format(epoch, step, loss.item())
            os.system('echo %s' % s)

        mean_loss = sum(mean_loss) / len(mean_loss)

        # 学习率递减
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
            val_org_mean_dice = dataset_prediction(net, 'info_files/btcv_val_info.csv')
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
            train_org_mean_dice = dataset_prediction(net, 'info_files/btcv_train_info.csv')
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
