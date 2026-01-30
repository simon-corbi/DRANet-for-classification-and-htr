import torch
import torch.nn as nn
import torch.nn.functional as F

def loss_weights(task, dsets):
    '''
        You must set these hyperparameters to apply our method to other datasets.
        These hyperparameters may not be the optimal value for your machine.
    '''

    alpha = dict()
    alpha['style'], alpha['dis'], alpha['gen'] = dict(), dict(), dict()
    if task == 'clf':
        alpha['recon'], alpha['consis'], alpha['content'] = 5, 1, 1
        # READ2018 datasets
        if 'read_2018_general' in dsets and 'read_2018_30866' in dsets and 'U' not in dsets:
            alpha['style']['read_2018_general2read_2018_30866'], alpha['style']['read_2018_308662read_2018_general'] = 5e4, 1e4
            alpha['dis']['read_2018_general'], alpha['dis']['read_2018_30866'] = 0.5, 0.5
            alpha['gen']['read_2018_general'], alpha['gen']['read_2018_30866'] = 0.5, 1.0

        elif 'read_2018_general' in dsets and 'read_2018_30882' in dsets and 'U' not in dsets:
            alpha['style']['read_2018_general2read_2018_30882'], alpha['style']['read_2018_308822read_2018_general'] = 5e4, 1e4
            alpha['dis']['read_2018_general'], alpha['dis']['read_2018_30882'] = 0.5, 0.5
            alpha['gen']['read_2018_general'], alpha['gen']['read_2018_30882'] = 0.5, 1.0

        elif 'read_2018_general' in dsets and 'read_2018_30893' in dsets and 'U' not in dsets:
            alpha['style']['read_2018_general2read_2018_30893'], alpha['style']['read_2018_308932read_2018_general'] = 5e4, 1e4
            alpha['dis']['read_2018_general'], alpha['dis']['read_2018_30893'] = 0.5, 0.5
            alpha['gen']['read_2018_general'], alpha['gen']['read_2018_30893'] = 0.5, 1.0

        elif 'read_2018_general' in dsets and 'read_2018_35013' in dsets and 'U' not in dsets:
            alpha['style']['read_2018_general2read_2018_35013'], alpha['style']['read_2018_350132read_2018_general'] = 5e4, 1e4
            alpha['dis']['read_2018_general'], alpha['dis']['read_2018_35013'] = 0.5, 0.5
            alpha['gen']['read_2018_general'], alpha['gen']['read_2018_35013'] = 0.5, 1.0

        elif 'read_2018_general' in dsets and 'read_2018_35015' in dsets and 'U' not in dsets:
            alpha['style']['read_2018_general2read_2018_35015'], alpha['style']['read_2018_350152read_2018_general'] = 5e4, 1e4
            alpha['dis']['read_2018_general'], alpha['dis']['read_2018_35015'] = 0.5, 0.5
            alpha['gen']['read_2018_general'], alpha['gen']['read_2018_35015'] = 0.5, 1.0

    return alpha


class Loss_Functions:
    def __init__(self, args):
        self.args = args
        self.alpha = dict()
        self.alpha['style'], self.alpha['dis'], self.alpha['gen'] = dict(), dict(), dict()
        self.alpha['recon'], self.alpha['consis'], self.alpha['content'] = 5, 1, 1
        self.alpha['style'][args.dataset_source + '2' + args.dataset_target] = 5e4
        self.alpha['style'][args.dataset_target + '2' + args.dataset_source] = 1e4
        self.alpha['dis'][args.dataset_source] = 0.5
        self.alpha['dis'][args.dataset_target] = 0.5
        self.alpha['gen'][args.dataset_source] = 0.5
        self.alpha['gen'][args.dataset_target] =  1.0

    def recon(self, imgs, recon_imgs):
        recon_loss = 0
        for dset in imgs.keys():
            recon_loss += F.l1_loss(imgs[dset], recon_imgs[dset])
        return self.alpha['recon'] * recon_loss

    def dis(self, real, fake):
        dis_loss = 0
        if self.args.task == 'clf':  # DCGAN loss
            for dset in real.keys():
                dis_loss += self.alpha['dis'][dset] * F.binary_cross_entropy(real[dset], torch.ones_like(real[dset]))
            for cv in fake.keys():
                source, target = cv.split("2read_", 1)
                target = "read_" + target
                dis_loss += self.alpha['dis'][target] * F.binary_cross_entropy(fake[cv], torch.zeros_like(fake[cv]))
        elif self.args.task == 'seg':  # Hinge loss
            for dset in real.keys():
                dis_loss += self.alpha['dis'][dset] * F.relu(1. - real[dset]).mean()
            for cv in fake.keys():
                source, target = cv.split("2read_", 1)
                target = "read_" + target
                dis_loss += self.alpha['dis'][target] * F.relu(1. + fake[cv]).mean()
        return dis_loss

    def gen(self, fake):
        gen_loss = 0
        for cv in fake.keys():
            source, target = cv.split("2read_", 1)
            target = "read_" + target
            if self.args.task == 'clf':
                gen_loss += self.alpha['gen'][target] * F.binary_cross_entropy(fake[cv], torch.ones_like(fake[cv]))
            elif self.args.task == 'seg':
                gen_loss += -self.alpha['gen'][target] * fake[cv].mean()
        return gen_loss

    def content_perceptual(self, perceptual, perceptual_converted):
        content_perceptual_loss = 0
        for cv in perceptual_converted.keys():
            source, target = cv.split("2read_", 1)
            target = "read_" + target
            content_perceptual_loss += F.mse_loss(perceptual[source][-1], perceptual_converted[cv][-1])
        return self.alpha['content'] * content_perceptual_loss

    def style_perceptual(self, style_gram, style_gram_converted):
        style_percptual_loss = 0
        for cv in style_gram_converted.keys():
            source, target = cv.split("2read_", 1)
            target = "read_" + target
            for gr in range(len(style_gram[target])):
                style_percptual_loss += self.alpha['style'][cv] * F.mse_loss(style_gram[target][gr],
                                                                             style_gram_converted[cv][gr])
        return style_percptual_loss

    def consistency(self, contents, styles, contents_converted, styles_converted, converts):
        consistency_loss = 0
        for cv in converts:
            source, target = cv.split("2read_", 1)
            target = "read_" + target
            consistency_loss += F.l1_loss(contents[cv], contents_converted[cv])
            consistency_loss += F.l1_loss(styles[target], styles_converted[cv])
        return self.alpha['consis'] * consistency_loss

    def task(self, pred, gt):
        task_loss = 0
        for key in pred.keys():
            if '2' in key:
                source, target = key.split('2')
            else:
                source = key
            task_loss += F.cross_entropy(pred[key], gt[source], ignore_index=-1)
        return task_loss

