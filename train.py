import os
from time import time
import numpy as np
import torch
from models.td_unet import get_net
from data_loader.batch_generator import get_data_loader
from torch.utils.data import DataLoader
from loss.dice_ce_loss import DC_and_CE_loss
from utils import *
from val import dataset_validation
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


def lr_delay(initial_lr, epoch):
    return initial_lr * pow((1 - epoch / Epoch), 0.9)


if __name__ == "__main__":
    # 模型
    net_name = 'td_unet'
    net = get_net(1)
    net = torch.nn.DataParallel(net).cuda()
    if resume_training:
        print('----------resume training-----------')
        net.load_state_dict(torch.load(module_dir))
        net.train()

    # 优化器
    opt = torch.optim.Adam(net.parameters(), lr=inital_learning_rate, weight_decay=0.0005)

    # 验证集数据
    _, _, val_samples_info, val_samples_name = split_train_val()
    print("samples for validation: ", val_samples_name)

    # 训练
    writer = SummaryWriter()
    for epoch in range(1, Epoch+1):
        mean_loss = []
        epoch_start = time()

        # switch models to training mode, clear gradient accumulators
        net.train()
        opt.zero_grad()

        # 损失函数
        loss_func = DC_and_CE_loss(class_weight)

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
        writer.add_scalar('train/loss', mean_loss, epoch)

        # 学习率递减
        for p in opt.param_groups:
            p['lr'] = lr_delay(inital_learning_rate, epoch)
        writer.add_scalar('lr', opt.param_groups[0]['lr'], epoch)

        s = '--- epoch:%d, mean loss:%.3f, epoch time:%.3f min\n' % (epoch, mean_loss, (time() - epoch_start) / 60)
        os.system('echo %s' % s)

        # valset accuracy
        os.system('echo %s' % "--------evaluation on validation set----------")
        val_eval_start = time()
        val_cls_mean_dice = dataset_validation(net, val_samples_info, show_sample_dice=True)
        writer.add_scalars('valset orgs dice',
                           {name: val_cls_mean_dice[i] for i, name in enumerate(classes_name)}, epoch)

        val_mean_dice = np.mean(val_cls_mean_dice)
        s = "mean dice: %.3f, eval time: %.3f min" % (val_mean_dice, (time() - val_eval_start) / 60)
        os.system('echo %s' % s)
        writer.add_scalar("valset mean dice", val_mean_dice, epoch)

        # update organs weight according to dices
        class_weight = 1.0 - np.array(val_cls_mean_dice)
        os.system('echo %s' % "---------------------------------------------\n")

        # 保存模型参数
        if epoch % 5 is 1:
            model_save_name = "%s%d-%.3f-%.3f.pth" % (net_name, epoch, mean_loss, val_mean_dice)
            torch.save(net.state_dict(), "./module/" + model_save_name)
            print("model saved as:  %s" % model_save_name)

    writer.close()
