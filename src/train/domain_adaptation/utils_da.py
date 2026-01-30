'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import logging
import random


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            # init.normal(m.weight, std=1e-3)
            init.xavier_normal_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)


try:
    _, term_width = os.popen('stty size', 'r').read().split()
except ValueError:
    term_width = 80
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def slice_patches(imgs, hight_slice=2, width_slice=4):
    b, c, h, w = imgs.size()
    h_patch, w_patch = int(h / hight_slice), int(w / width_slice)
    patches = imgs.unfold(2, h_patch, h_patch).unfold(3, w_patch, w_patch)
    patches = patches.contiguous().view(b, c, -1, h_patch, w_patch)
    patches = patches.transpose(1,2)
    patches = patches.reshape(-1, c, h_patch, w_patch)
    return patches


import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Function
from sklearn.manifold import TSNE
import torch
import itertools
import os


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


def optimizer_scheduler(optimizer, p):
    """
    Adjust the learning rate of optimizer
    :param optimizer: optimizer for updating parameters
    :param p: a variable for adjusting learning rate
    :return: optimizer
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = 0.01 / (1. + 10 * p) ** 0.75

    return optimizer


def one_hot_embedding(labels, num_classes=10):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes)
    return y[labels]


def save_model(encoder, classifier, discriminator, training_mode, directory_log, epoch):
    print('Saving models ...')
    save_folder = os.path.join(directory_log, "trained_models")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    torch.save(encoder.state_dict(), os.path.join(save_folder, f'encoder_{training_mode}_epoch{epoch}.torch'))
    torch.save(classifier.state_dict(), os.path.join(save_folder, f'classifier_{training_mode}_epoch{epoch}.torch'))

    if training_mode == 'DANN':
        torch.save(discriminator.state_dict(), os.path.join(save_folder, f'discriminator_{training_mode}_epoch{epoch}.torch'))

    print('The model has been successfully saved!')


def plot_embedding(X, y, d, training_mode):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    y = list(itertools.chain.from_iterable(y))
    y = np.asarray(y)

    plt.figure(figsize=(10, 10))
    for i in range(len(d)):  # X.shape[0] : 1024
        # plot colored number
        if d[i] == 0:
            colors = (0.0, 0.0, 1.0, 1.0)
        else:
            colors = (1.0, 0.0, 0.0, 1.0)
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=colors,
                 fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([]), plt.yticks([])

    save_folder = 'saved_plot'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    fig_name = 'saved_plot/' + str(training_mode) + '.png'
    plt.savefig(fig_name)
    print('{} has been successfully saved!'.format(fig_name))


# def visualize(encoder, training_mode, device):
#     # Draw 512 samples in test_data
#     source_test_loader = mnist.mnist_test_loader
#     target_test_loader = mnistm.mnistm_test_loader
#
#     # Get source_test samples
#     source_label_list = []
#     source_img_list = []
#     for i, test_data in enumerate(source_test_loader):
#         if i >= 16:  # to get only 512 samples
#             break
#         img, label = test_data
#         label = label.numpy()
#         img = img.to(device)
#         img = torch.cat((img, img, img), 1)  # MNIST channel 1 -> 3
#         source_label_list.append(label)
#         source_img_list.append(img)
#
#     source_img_list = torch.stack(source_img_list)
#     source_img_list = source_img_list.view(-1, 3, 28, 28)
#
#     # Get target_test samples
#     target_label_list = []
#     target_img_list = []
#     for i, test_data in enumerate(target_test_loader):
#         if i >= 16:
#             break
#         img, label = test_data
#         label = label.numpy()
#         img = img.to(device)
#         target_label_list.append(label)
#         target_img_list.append(img)
#
#     target_img_list = torch.stack(target_img_list)
#     target_img_list = target_img_list.view(-1, 3, 28, 28)
#
#     # Stack source_list + target_list
#     combined_label_list = source_label_list
#     combined_label_list.extend(target_label_list)
#     combined_img_list = torch.cat((source_img_list, target_img_list), 0)
#
#     source_domain_list = torch.zeros(512).type(torch.LongTensor)
#     target_domain_list = torch.ones(512).type(torch.LongTensor)
#     combined_domain_list = torch.cat((source_domain_list, target_domain_list), 0).to(device)
#
#     print("Extracting features to draw t-SNE plot...")
#     combined_feature = encoder(combined_img_list)  # combined_feature : 1024,2352
#
#     tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000)
#     dann_tsne = tsne.fit_transform(combined_feature.detach().cpu().numpy())
#
#     print('Drawing t-SNE plot ...')
#     plot_embedding(dann_tsne, combined_label_list, combined_domain_list, training_mode)
#
#
# def visualize_input(device):
#     source_test_loader = mnist.mnist_test_loader
#     target_test_loader = mnistm.mnistm_test_loader
#
#     # Get source_test samples
#     source_label_list = []
#     source_img_list = []
#     for i, test_data in enumerate(source_test_loader):
#         if i >= 16:  # to get only 512 samples
#             break
#         img, label = test_data
#         label = label.numpy()
#         img = img.to(device)
#         img = torch.cat((img, img, img), 1)  # MNIST channel 1 -> 3
#         source_label_list.append(label)
#         source_img_list.append(img)
#
#     source_img_list = torch.stack(source_img_list)
#     source_img_list = source_img_list.view(-1, 3, 28, 28)
#
#     # Get target_test samples
#     target_label_list = []
#     target_img_list = []
#     for i, test_data in enumerate(target_test_loader):
#         if i >= 16:
#             break
#         img, label = test_data
#         label = label.numpy()
#         img = img.to(device)
#         target_label_list.append(label)
#         target_img_list.append(img)
#
#     target_img_list = torch.stack(target_img_list)
#     target_img_list = target_img_list.view(-1, 3, 28, 28)
#
#     # Stack source_list + target_list
#     combined_label_list = source_label_list
#     combined_label_list.extend(target_label_list)
#     combined_img_list = torch.cat((source_img_list, target_img_list), 0)
#
#     source_domain_list = torch.zeros(512).type(torch.LongTensor)
#     target_domain_list = torch.ones(512).type(torch.LongTensor)
#     combined_domain_list = torch.cat((source_domain_list, target_domain_list), 0).to(device)
#
#     print("Extracting features to draw t-SNE plot...")
#     combined_feature = combined_img_list  # combined_feature : 1024,3,28,28
#     combined_feature = combined_feature.view(1024, -1)  # flatten
#     # print(type(combined_feature), combined_feature.shape)
#
#     tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000)
#     dann_tsne = tsne.fit_transform(combined_feature.detach().cpu().numpy())
#     print('Drawing t-SNE plot ...')
#     plot_embedding(dann_tsne, combined_label_list, combined_domain_list, 'input')


def set_model_mode(mode='train', models=None):
    for model in models:
        if mode == 'train':
            model.train()
        else:
            model.eval()
