import os
import torch

# 네트워크를 저장하거나 불러오는 함수 작성
def save(ckpt_dir, net, optim, epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    torch.save({'net': net.state_dict(), 'optim': optim.state_dict()},
               './%s/model_epoch%d.pth' % (ckpt_dir, epoch))


def load(ckpt_dir, net, optim):
    ckpt_list = os.listdir(ckpt_dir)
    ckpt_list.sort()

    dict_model = torch.load('./%s/%s' % (ckpt_dir, ckpt_list[-1]))

    net.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])

    return net, optim