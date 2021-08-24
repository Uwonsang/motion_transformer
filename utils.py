import os
import numpy as np
import torch

def adjust_learning_rate(ori_lr, args, optimizer, global_step, lr_steps):

    decay = 0.1

    for i in range(len(lr_steps)):
        if global_step >= int(lr_steps[i]):
            decay = decay * 0.1

    lr = ori_lr * decay
    decay = args.weight_decay
    for param_group in optimizer.param_groups:
        if 'lr_mult' in param_group:
            param_group['lr'] = lr * param_group['lr_mult']
        else:
            param_group['lr'] = lr

        if 'decay_mult' in param_group:
            param_group['weight_decay'] = decay * param_group['decay_mult']
    return lr


def save_checkpoint(state, epoch, folder, filename='min_loss_checkpoint.pth.tar'):
    if not os.path.exists(folder):
        os.makedirs(folder)

    filename = os.path.join(folder, filename)
    torch.save(state, filename)


def load_pretrained_model(model, pretrained_model_path):
    # Load trained model.
    trained_model = torch.load(pretrained_model_path)
    pretrained_dict = trained_model['state_dict']
    removed_prefix_pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
    model_dict = model.state_dict()

    for k in pretrained_dict.keys():
        print("{0} exists in ori pretrained model".format(k))
    for k in model_dict.keys():
        print("{0} exists in ori model".format(k))

    # Judge pretrained model and current model are multi-gpu or not
    multi_gpu_pretrained = False
    multi_gpu_current_model = False
    for k, v in pretrained_dict.items():
        if "module" in k:
            multi_gpu_pretrained = True
            break;

    for k, v in model_dict.items():
        if "module" in k:
            multi_gpu_current_model = True
            break;

    # Different ways to deal with diff cases
    if multi_gpu_pretrained and multi_gpu_current_model:
        updated_pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    elif multi_gpu_pretrained and not multi_gpu_current_model:
        updated_pretrained_dict = {k: v for k, v in removed_prefix_pretrained_dict.items() if k in model_dict}
    elif not multi_gpu_pretrained and multi_gpu_current_model:
        updated_pretrained_dict = {}
        for current_k in model_dict:
            removed_prefix_k = current_k.replace("module.", "")
            updated_pretrained_dict[current_k] = pretrained_dict[removed_prefix_k]
    else:
        updated_pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    for k in updated_pretrained_dict.keys():
        print("{0} is loaded successfully".format(k))
    # 2. overwrite entries in the existing state dict
    model_dict.update(updated_pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if n > 0:
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count