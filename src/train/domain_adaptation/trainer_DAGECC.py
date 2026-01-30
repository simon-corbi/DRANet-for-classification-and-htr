from __future__ import print_function

import os
from logging import Formatter, StreamHandler, getLogger, FileHandler
from random import seed

import numpy as np
import torch.backends.cudnn
import wandb
from sklearn.metrics import classification_report, f1_score
from torch import optim
from torch.utils.data import DataLoader

from src.data.collate.classif_batch_collate import CollateImageLabelClassification
from src.data.config_reader.reader_conf_char_classif import read_json_config_char_classif_da
from src.data.datasets.multiple_dataset_classif import MultipleImageLabelDataset
from src.data.datasets.one_dataset_classif import OneImageLabelDataset
from src.data.image.augmentations.config_aug import get_config_aug_classif
from src.data.text.charset_token_DAGECC import CharsetToken
from src.models.domain_adaptation import model_DAGECC
from src.models.domain_adaptation.DRANet_DAGECC import *
from src.models.domain_adaptation.DRANet_DAGECC import Generator, Separator
from src.models.domain_adaptation.cadt import *
from src.models.domain_adaptation.cnn.resnet_torchvision import ResNetTorch, BasicBlock
from src.models.domain_adaptation.drn import drn26
from src.models.domain_adaptation.loss_functions_DAGECC import *
from src.models.domain_adaptation.model_DAGECC import *
from src.models.expert_models.resnet_torchvision import BasicBlock, ResNetTorch
from src.models.init_weights import init_params
from src.models.model_utils import load_pretrained_model
from utils_da import *


def set_converts(datasets):
    training_converts, test_converts = [], []
    center_dset = datasets[0]
    for source in datasets:  # source
        if not center_dset == source:
            training_converts.append(center_dset + '2' + source)
            training_converts.append(source + '2' + center_dset)
        for target in datasets:  # target
            if not source == target:
                test_converts.append(source + '2' + target)

    return training_converts, test_converts


class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = args.device

        self.datasets = ["source", "target"]
        self.training_converts, self.test_converts = set_converts(self.datasets)

        # #  Initialization
        self.train_loader, self.test_loader = dict(), dict()

        self.nets, self.optims, self.losses = dict(), dict(), dict()

        # self.checkpoint = './checkpoint/%s/%s' % (args.task, args.ex)
        self.step = 0

        if args.task == 'seg':
            self.imsize = (2 * args.imsize, args.imsize)  # (width, height)
            self.best_miou = 0.
        elif args.task == 'clf':
            self.imsize = (args.imsize, args.imsize)
            self.acc = dict()
            self.best_acc = dict()
            self.best_f1 = {cv: 0. for cv in self.test_converts}
            for cv in self.test_converts:
                self.best_acc[cv] = 0.

        self.loss_fns = Loss_Functions(args)

        self.logger = getLogger()
        self.checkpoint = os.path.join(
            self.args.path_log,
            'checkpoint',
            'source',
            'target',
        )
        self.step = 0

        # Paths data
        source_infos_train, source_infos_test, target_info_train, target_info_test, charset_path = read_json_config_char_classif_da(
            args.config_file)

        # Alphabet
        charset = CharsetToken(charset_path)
        self.charset = charset
        self.nb_char_all = charset.get_nb_char()

        # For class not present in charset
        ignore_index = self.nb_char_all  # + 1
        charset.add_label("IGNORE")
        self.nb_char_all = charset.get_nb_char()

        self.loss_classif = nn.CrossEntropyLoss(ignore_index=ignore_index)

        self.char_list = charset.get_charset_list()
        char_dict = charset.get_charset_dictionary()

        # Data
        fixed_size_img = (args.height_max, args.width_max)

        config_augmentation = get_config_aug_classif()

        # Source
        train_db = MultipleImageLabelDataset(source_infos_train,
                                             fixed_size_img,
                                             char_dict,
                                             ignore_index,
                                             apply_augmentation=True,
                                             config_augmentation=config_augmentation)
        print('Nb samples train source {}:'.format(len(train_db)))

        # Pad img with black = 0
        c_collate_fn = CollateImageLabelClassification(imgs_pad_value=[0])
        collate_fn = c_collate_fn.collate_fn

        self.train_loader["source"] = DataLoader(train_db, num_workers=args.workers, batch_size=args.batch,
                                                 pin_memory=True,
                                                 collate_fn=collate_fn, shuffle=True)

        test_db = OneImageLabelDataset(source_infos_test[0][1],
                                       fixed_size_img,
                                       char_dict,
                                       ignore_index)

        print('Nb samples test source{}:'.format(len(test_db)))

        self.test_loader["source"] = DataLoader(test_db, num_workers=args.workers, batch_size=args.batch,
                                                pin_memory=True,
                                                collate_fn=collate_fn, shuffle=False)
        # Target
        train_db = MultipleImageLabelDataset(target_info_train,
                                             fixed_size_img,
                                             char_dict,
                                             ignore_index,
                                             apply_augmentation=True,
                                             config_augmentation=config_augmentation)
        print('Nb samples train target {}:'.format(len(train_db)))

        # Pad img with black = 0
        c_collate_fn = CollateImageLabelClassification(imgs_pad_value=[0])
        collate_fn = c_collate_fn.collate_fn

        self.train_loader["target"] = DataLoader(train_db, num_workers=args.workers, batch_size=args.batch,
                                                 pin_memory=True,
                                                 collate_fn=collate_fn, shuffle=True)

        test_db = OneImageLabelDataset(target_info_test[0][1],
                                       fixed_size_img,
                                       char_dict,
                                       ignore_index)

        print('Nb samples test target{}:'.format(len(test_db)))

        self.test_loader["target"] = DataLoader(test_db, num_workers=args.workers, batch_size=args.batch,
                                                pin_memory=True,
                                                collate_fn=collate_fn, shuffle=False)

    def set_default(self):
        # Enable cuDNN benchmark only if CUDA is available
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

        ## Random Seed ##
        print("Random Seed: ", self.args.manualSeed)
        seed(self.args.manualSeed)
        torch.manual_seed(self.args.manualSeed)

        # Only set CUDA seeds if CUDA is available
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.args.manualSeed)

        ## Logger ##
        file_log_handler = FileHandler(self.args.logfile)
        self.logger.addHandler(file_log_handler)
        stderr_log_handler = StreamHandler(sys.stdout)
        self.logger.addHandler(stderr_log_handler)
        self.logger.setLevel('INFO')
        formatter = Formatter()
        file_log_handler.setFormatter(formatter)
        stderr_log_handler.setFormatter(formatter)

    def save_networks(self):
        step_dir = os.path.join(self.checkpoint, '%d' % self.step)
        os.makedirs(step_dir, exist_ok=True)
        for key in self.nets.keys():
            if key == 'D':
                for dset in self.args.datasets:
                    torch.save(self.nets[key][dset].state_dict(),
                               self.checkpoint + '/%d/net%s_%s.pth' % (self.step, key, dset))
            elif key == 'T':
                for cv in self.test_converts:
                    torch.save(self.nets[key][cv].state_dict(),
                               self.checkpoint + '/%d/net%s_%s.pth' % (self.step, key, cv))
            else:
                torch.save(self.nets[key].state_dict(), self.checkpoint + '/%d/net%s.pth' % (self.step, key))

    def save_networks_last(self):
        if not os.path.exists(self.checkpoint):
            os.mkdir(self.checkpoint)
        for key in self.nets.keys():
            if key == 'D':
                for dset in self.args.datasets:
                    torch.save(self.nets[key][dset].state_dict(),
                               self.checkpoint + '/net_last_%s_%s.pth' % (key, dset))
            elif key == 'T':
                for cv in self.test_converts:
                    torch.save(self.nets[key][cv].state_dict(),
                               self.checkpoint + '/net_last_%s_%s.pth' % (key, cv))
            else:
                torch.save(self.nets[key].state_dict(), self.checkpoint + '/net%s.pth' % (key))

    def load_networks(self, step):
        self.step = step
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        for key in self.nets.keys():
            if key == 'D':
                for dset in self.args.datasets:
                    self.nets[key][dset].load_state_dict(
                        torch.load(
                            self.checkpoint + '/%d/net%s_%s.pth' % (step, key, dset),
                            map_location=device
                        )
                    )
            elif key == 'T':
                if self.args.task == 'clf':
                    for cv in self.test_converts:
                        self.nets[key][cv].load_state_dict(
                            torch.load(
                                self.checkpoint + '/%d/net%s_%s.pth' % (step, key, cv),
                                map_location=device
                            )
                        )
            else:
                self.nets[key].load_state_dict(
                    torch.load(
                        self.checkpoint + '/%d/net%s.pth' % (step, key),
                        map_location=device
                    )
                )

    def set_networks(self):
        # Model
        charset = self.charset
        nb_char_all = charset.get_nb_char()

        # For class not present in charset
        ignore_index = nb_char_all  # + 1
        charset.add_label("IGNORE")
        char_list = charset.get_charset_list()
        char_dict = charset.get_charset_dictionary()
        nb_char_all = charset.get_nb_char()

        complete_model = ResNetTorch(BasicBlock, [3, 4, 6, 3], nb_channel_input=1,
                                     num_classes=nb_char_all)

        if os.path.isfile(self.args.path_model):
            load_pretrained_model(self.args.path_model, complete_model, self.device)
            print(f"✅ Loaded pretrained weights from {self.args.path_model}")
        else:
            print(f"❌ Model file not found at {self.args.path_model}")

        # Replace DRANet Encoder with your custom Extractor
        self.nets['E'] = model_DAGECC.Extractor(complete_model).to(self.device)

        self.nets['G'] = Generator()
        self.nets['S'] = Separator(self.imsize, self.training_converts)
        self.nets['D'] = dict()
        for dset in self.args.datasets:
            if self.args.task == 'clf':
                if dset == 'U':
                    self.nets['D'][dset] = Discriminator_USPS()
                else:
                    self.nets['D'][dset] = Discriminator_MNIST()
            else:
                self.nets['D'][dset] = PatchGAN_Discriminator()
        self.nets['T'] = dict()
        for cv in self.test_converts:
            if self.args.task == 'clf':
                self.nets['T'][cv] = complete_model  # Classifier() change to custom classifier
            elif self.args.task == 'seg':
                self.nets['T'][cv] = drn26()

        for net in self.nets.keys():
            if net == 'D':
                for dset in self.args.datasets:
                    init_params(self.nets[net][dset])
            elif net == 'T':
                if self.args.task == 'clf':
                    for cv in self.test_converts:
                        continue  # Ne pas initialiser T[cv] si clf
                elif self.args.task == 'seg':
                    for cv in self.test_converts:
                        init_params(self.nets[net][cv])  # Initialiser uniquement pour segmentation
            elif net == 'E':
                continue  # Ne pas initialiser E
            else:
                init_params(self.nets[net])
        self.nets['P'] = VGG19()

        # Move nets to device
        for net in self.nets.keys():
            if net == 'D':
                for dset in self.args.datasets:
                    self.nets[net][dset] = self.nets[net][dset].to(self.device)
            elif net == 'T':
                for cv in self.test_converts:
                    self.nets[net][cv] = self.nets[net][cv].to(self.device)
            else:
                self.nets[net] = self.nets[net].to(self.device)

    def set_optimizers(self):
        self.optims['E'] = optim.Adam(self.nets['E'].parameters(), lr=self.args.lr_dra,
                                      betas=(self.args.beta1, 0.999),
                                      weight_decay=self.args.weight_decay_dra)

        self.optims['D'] = dict()
        for dset in self.args.datasets:
            print(f"dataset {dset}")
            self.optims['D'][dset] = optim.Adam(self.nets['D'][dset].parameters(), lr=self.args.lr_dra,
                                                betas=(self.args.beta1, 0.999),
                                                weight_decay=self.args.weight_decay_dra)

        self.optims['G'] = optim.Adam(self.nets['G'].parameters(), lr=self.args.lr_dra,
                                      betas=(self.args.beta1, 0.999),
                                      weight_decay=self.args.weight_decay_dra)

        self.optims['S'] = optim.Adam(self.nets['S'].parameters(), lr=self.args.lr_dra,
                                      betas=(self.args.beta1, 0.999),
                                      weight_decay=self.args.weight_decay_dra)

        self.optims['T'] = dict()
        for convert in self.test_converts:
            if self.args.task == 'clf':
                self.optims['T'][convert] = optim.SGD(self.nets['T'][convert].parameters(), lr=self.args.lr_clf,
                                                      momentum=0.9,
                                                      weight_decay=self.args.weight_decay_task)
            elif self.args.task == 'seg':
                self.optims['T'][convert] = optim.SGD(self.nets['T'][convert].parameters(), lr=self.args.lr_seg,
                                                      momentum=0.9,
                                                      weight_decay=self.args.weight_decay_task)

    def set_zero_grad(self):
        for net in self.nets.keys():
            if net == 'D':
                for dset in self.args.datasets:
                    self.nets[net][dset].zero_grad()
            elif net == 'T':
                for convert in self.test_converts:
                    self.nets[net][convert].zero_grad()
            else:
                self.nets[net].zero_grad()

    def set_train(self):
        for net in self.nets.keys():
            if net == 'D':
                for dset in self.args.datasets:
                    self.nets[net][dset].train()
            elif net == 'T':
                for convert in self.test_converts:
                    self.nets[net][convert].train()
            else:
                self.nets[net].train()

    def set_eval(self):
        for convert in self.test_converts:
            self.nets['T'][convert].eval()

    def get_batch(self, batch_data_iter):
        batch_data = dict()
        for dset in self.args.datasets:
            try:
                batch_data[dset] = next(batch_data_iter[dset])
            except StopIteration:
                batch_data_iter[dset] = iter(self.train_loader[dset])
                batch_data[dset] = next(batch_data_iter[dset])
        return batch_data

    def train_dis(self, imgs):  # Train Discriminators (D)
        self.set_zero_grad()
        features, converted_imgs, D_outputs_fake, D_outputs_real = dict(), dict(), dict(), dict()

        # Real
        for dset in self.args.datasets:
            features[dset] = self.nets['E'](imgs[dset])
            if self.args.task == 'clf':
                D_outputs_real[dset] = self.nets['D'][dset](imgs[dset])
            else:
                D_outputs_real[dset] = self.nets['D'][dset](slice_patches(imgs[dset]))

        contents, styles = self.nets['S'](features, self.training_converts)

        # CADT
        if self.args.CADT:
            for convert in self.training_converts:
                source, target = convert.split('2')
                _, styles[target] = cadt(contents[source], contents[target], styles[target])

        # Fake
        for convert in self.training_converts:
            source, target = convert.split('2')
            converted_imgs[convert] = self.nets['G'](contents[convert], styles[target])
            if self.args.task == 'clf':
                D_outputs_fake[convert] = self.nets['D'][target](converted_imgs[convert])
            else:
                D_outputs_fake[convert] = self.nets['D'][target](slice_patches(converted_imgs[convert]))

        errD = self.loss_fns.dis(D_outputs_real, D_outputs_fake)
        errD.backward()
        for optimizer in self.optims['D'].values():
            optimizer.step()
        self.losses['D'] = errD.data.item()

    def train_task(self, imgs, labels):  # Train Task Networks (T)
        self.set_zero_grad()
        features = dict()
        converted_imgs = dict()
        pred = dict()
        converts = self.training_converts if self.args.task == 'clf' else self.test_converts
        with torch.no_grad():
            for dset in self.args.datasets:
                features[dset] = self.nets['E'](imgs[dset])
            contents, styles = self.nets['S'](features, converts)
            for convert in converts:
                source, target = convert.split('2')
                if self.args.CADT:
                    _, styles[target] = cadt(contents[source], contents[target], styles[target])
                converted_imgs[convert] = self.nets['G'](contents[convert], styles[target])

            # 3 datasets (MNIST, MNIST-M, USPS)
            # DRANet can convert USPS <-> MNIST-M without training the conversion directly.
            for convert in list(set(self.test_converts) - set(self.training_converts)):
                features_mid = dict()
                source, target = convert.split('2')
                mid = list(set(self.args.datasets) - {source, target})[0]
                convert1 = source + '2' + mid
                convert2 = mid + '2' + target
                features_mid[convert1] = self.nets['E'](converted_imgs[convert1])
                contents_mid, _ = self.nets['S'](features_mid, [convert2])
                converted_imgs[convert] = self.nets['G'](contents_mid[convert2], styles[target])

        for convert in self.test_converts:
            pred[convert] = self.nets['T'][convert](converted_imgs[convert])
            source, target = convert.split('2')
            pred[source] = self.nets['T'][convert](imgs[source])

        errT = self.loss_fns.task(pred, labels)
        errT.backward()
        for optimizer in self.optims['T'].values():
            optimizer.step()
        self.losses['T'] = errT.data.item()

    def train_esg(self, imgs):  # Train Encoder(E), Separator(S), Generator(G)
        self.set_zero_grad()
        features, converted_imgs, recon_imgs, D_outputs_fake = dict(), dict(), dict(), dict()
        features_converted = dict()
        perceptual, style_gram = dict(), dict()
        perceptual_converted, style_gram_converted = dict(), dict()
        con_sim = dict()
        for dset in self.args.datasets:
            features[dset] = self.nets['E'](imgs[dset])
            recon_imgs[dset] = self.nets['G'](features[dset], 0)
            perceptual[dset] = self.nets['P'](imgs[dset])
            style_gram[dset] = [gram(fmap) for fmap in perceptual[dset][:-1]]
        contents, styles = self.nets['S'](features, self.training_converts)

        for convert in self.training_converts:
            source, target = convert.split('2')
            if self.args.CADT:
                con_sim[convert], styles[target] = cadt(contents[source], contents[target], styles[target])
                style_gram[target] = cadt_gram(style_gram[target], con_sim[convert])
            converted_imgs[convert] = self.nets['G'](contents[convert], styles[target])
            if self.args.task == 'clf':
                D_outputs_fake[convert] = self.nets['D'][target](converted_imgs[convert])
            else:
                D_outputs_fake[convert] = self.nets['D'][target](slice_patches(converted_imgs[convert]))
            features_converted[convert] = self.nets['E'](converted_imgs[convert])
            perceptual_converted[convert] = self.nets['P'](converted_imgs[convert])
            style_gram_converted[convert] = [gram(fmap) for fmap in perceptual_converted[convert][:-1]]
        contents_converted, styles_converted = self.nets['S'](features_converted)

        Content_loss = self.loss_fns.content_perceptual(perceptual, perceptual_converted)
        Style_loss = self.loss_fns.style_perceptual(style_gram, style_gram_converted)
        Consistency_loss = self.loss_fns.consistency(contents, styles, contents_converted, styles_converted,
                                                     self.training_converts)
        G_loss = self.loss_fns.gen(D_outputs_fake)
        Recon_loss = self.loss_fns.recon(imgs, recon_imgs)

        # Pondération des pertes avant log
        self.losses['G'] = self.args.lambda_g * G_loss.data.item()
        self.losses['Recon'] = self.args.lambda_recon * Recon_loss.data.item()
        self.losses['Consis'] = self.args.lambda_consis * Consistency_loss.data.item()
        self.losses['Content'] = self.args.lambda_content * Content_loss.data.item()
        self.losses['Style'] = self.args.lambda_style * Style_loss.data.item()

        errESG = G_loss + Content_loss + Style_loss + Consistency_loss + Recon_loss

        errESG.backward()
        for net in ['E', 'S', 'G']:
            self.optims[net].step()

        self.losses['G'] = G_loss.data.item()
        self.losses['Recon'] = Recon_loss.data.item()
        self.losses['Consis'] = Consistency_loss.data.item()
        self.losses['Content'] = Content_loss.data.item()
        self.losses['Style'] = Style_loss.data.item()

    def eval(self, cv):
        source, target = cv.split('2')
        self.set_eval()

        if self.args.task == 'clf':
            correct = 0
            total = 0
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for batch_idx, (imgs, labels) in enumerate(self.test_loader[target]):
                    imgs, labels = imgs.to(self.device), labels.to(self.device)
                    pred = self.nets['T'][cv](imgs)
                    _, predicted = torch.max(pred.data, 1)

                    total += labels.size(0)
                    correct += predicted.eq(labels.data).cpu().sum()

                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

                    progress_bar(batch_idx, len(self.test_loader[target]),
                                 'Acc: %.3f%% (%d/%d)' % (100. * correct / total, correct, total))

            # Accuracy
            acc = 100. * correct / total
            self.logger.info('======================================================')
            self.logger.info('Step: %d | Acc: %.3f%% (%d/%d)' %
                             (self.step / len(self.test_loader[target]), acc, correct, total))
            self.logger.info('======================================================')

            # -----------------------------
            # Handle IGNORE class

            charset = self.charset
            nb_char_all = charset.get_nb_char()
            IGNORE_IDX = nb_char_all - 1  # assuming IGNORE was added last
            valid_idx = [i for i, l in enumerate(all_labels) if l != IGNORE_IDX]
            filtered_labels = [all_labels[i] for i in valid_idx]
            filtered_preds = [all_preds[i] for i in valid_idx]

            # F1 Macro
            if len(set(filtered_labels)) > 1:
                f1_macro = f1_score(filtered_labels, filtered_preds, average='macro')
            else:
                f1_macro = 0.0
                self.logger.warning("⚠️ Moins de 2 classes valides après exclusion de IGNORE — F1 macro mis à 0.")

            self.logger.info('F1 Macro (sans IGNORE): %.4f' % f1_macro)

            # Classification Report
            report = classification_report(filtered_labels, filtered_preds, digits=4)
            self.logger.info('Classification Report (sans IGNORE):\n%s' % report)

            # Debug unique classes
            self.logger.info('Labels uniques : %s', np.unique(filtered_labels))
            self.logger.info('Prédictions uniques : %s', np.unique(filtered_preds))

            # Sauvegarde des meilleurs résultats
            if acc > self.best_acc[cv]:
                self.best_acc[cv] = acc

                self.save_networks()

            if f1_macro > self.best_f1[cv]:
                self.best_f1[cv] = f1_macro

        self.set_train()

    def print_loss(self):
        best = ''
        wandb_log_data = {}

        if self.args.task == 'clf':
            for cv in self.test_converts:
                acc = self.best_acc[cv]
                f1 = self.best_f1[cv]
                best += f'{cv}: acc={acc:.2f}, f1={f1:.2f}|'
                wandb_log_data[f'best_acc/{cv}'] = acc
                wandb_log_data[f'best_f1_macro/{cv}'] = f1
        elif self.args.task == 'seg':
            miou = self.best_miou
            best += f'{miou:.2f}'
            wandb_log_data['best_mIoU'] = miou

        # Log current losses
        losses = ''
        for key, value in self.losses.items():
            losses += f'{key}: {value:.2f}|'
            wandb_log_data[f'loss/{key}'] = value

        # Terminal log
        self.logger.info('[%d/%d] %s| %s %s' % (self.step, self.args.iter, losses, best, self.args.ex))

        return wandb_log_data

    def train(self):
        self.set_default()
        self.set_networks()
        self.set_optimizers()
        self.set_train()
        self.logger.info(self.loss_fns.alpha)

        batch_data_iter = dict()
        for dset in self.args.datasets:
            batch_data_iter[dset] = iter(self.train_loader[dset])

        # eval epoch 0
        for cv in self.test_converts:
            eval_metrics = self.eval(cv)

            # Optionally log evaluation metrics to wandb if eval() returns any
            if isinstance(eval_metrics, dict):
                wandb.log({f"eval/{k}": v for k, v in eval_metrics.items()}, step=self.step)

        for i in range(self.args.iter):
            self.step += 1

            # get batch data
            batch_data = self.get_batch(batch_data_iter)
            imgs, labels = dict(), dict()
            min_batch = self.args.batch

            for dset in self.args.datasets:
                imgs[dset], labels[dset] = batch_data[dset]
                imgs[dset], labels[dset] = imgs[dset].to(self.device), labels[dset].to(self.device)
                if self.args.task == 'seg':
                    labels[dset] = labels[dset].long()
                if imgs[dset].size(0) < min_batch:
                    min_batch = imgs[dset].size(0)

            if min_batch < self.args.batch:
                for dset in self.args.datasets:
                    imgs[dset], labels[dset] = imgs[dset][:min_batch], labels[dset][:min_batch]

            # training steps
            self.train_dis(imgs)
            for t in range(2):
                self.train_esg(imgs)
            self.train_task(imgs, labels)

            # evaluation
            if self.step % self.args.eval_freq == 0:
                for cv in self.test_converts:
                    eval_metrics = self.eval(cv)

                    # Optionally log evaluation metrics to wandb if eval() returns any
                    if isinstance(eval_metrics, dict):
                        wandb.log({f"eval/{k}": v for k, v in eval_metrics.items()}, step=self.step)

            # print and log losses
            losses = self.print_loss()  # Assuming this returns a dict like {'loss_total': x, 'loss_task': y}
            if isinstance(losses, dict):
                wandb.log(losses, step=self.step)

    def test(self):
        self.set_default()
        self.set_networks()
        self.load_networks(self.args.load_step)
        for cv in self.test_converts:
            self.eval(cv)
