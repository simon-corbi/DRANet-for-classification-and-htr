from __future__ import print_function

from logging import Formatter, StreamHandler, getLogger, FileHandler
from random import seed

import editdistance
import torch.backends.cudnn
import wandb
from prettytable import PrettyTable

from src.models.domain_adaptation.loss_functions_READ import *
from src.models.domain_adaptation.model_READ import *
from src.data.collate.collate_batch import CollateImageLabelHTR
from src.data.datasets.htr_dataset import HTRDataset
from src.data.global_values.text_global_values import BLANK_STR_TOKEN, CTC_PAD
from src.data.image.augmentations.img_augmentation import get_img_data_augmentation
from src.data.readers.reader_config import read_json_config, read_labels_files
from src.data.text.best_path_ctc import ctc_best_path_one
from src.data.text.charset_token import CharsetToken
from src.evaluate.evaluate_recognition import nb_chars_from_list
from src.evaluate.metrics_counter import MetricLossCERWER
from src.models.crnn import CRNN
from src.models.model_utils import load_pretrained_model
from utils_da import *
from src.models.domain_adaptation.DRANet_READ import *

from src.models.domain_adaptation.cadt import *
from src.models.domain_adaptation.drn import drn26


import os


def set_converts(datasets, task):
    training_converts, test_converts = [], []
    center_dset = datasets[0]
    for source in datasets:  # source
        if not center_dset == source:
            training_converts.append(center_dset + '2' + source)
            training_converts.append(source + '2' + center_dset)
        if task == 'clf':
            for target in datasets:  # target
                if not source == target:
                    test_converts.append(source + '2' + target)
    if task == 'clf':
        tensorboard_converts = test_converts
    elif task == 'seg':
        test_converts.append('G2C')
        tensorboard_converts = training_converts
    else:
        raise Exeception("Does not support the task")

    return training_converts, test_converts, tensorboard_converts


class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = args.device

        self.training_converts, self.test_converts, self.tensorboard_converts = set_converts(args.datasets, args.task)
        if args.task == 'seg':
            self.imsize = (2 * args.imsize, args.imsize)  # (width, height)
            self.best_miou = 0.
        elif args.task == 'clf':
            self.imsize = args.imsize
            self.best_cer = {cv: float('inf') for cv in self.test_converts}

            self.best_cer_shortcut = {cv: float('inf') for cv in self.test_converts}

        # Paths
        dbs_info, charsets_path, all_labels_files, _ = read_json_config(args.config_file)
        all_labels = read_labels_files(all_labels_files)

        # Alphabet
        self.charset = CharsetToken(charsets_path, use_blank=True)
        self.char_list = self.charset.get_charset_list()
        self.char_dict = self.charset.get_charset_dictionary()

        # Data
        fixed_size_img = (args.height_max, args.width_max)
        aug_transforms = get_img_data_augmentation()

        # Pad img with black = 0
        c_collate_fn = CollateImageLabelHTR(imgs_pad_value=[0], pad_txt=CTC_PAD)
        collate_fn = c_collate_fn.collate_fn

        width_divisor = 8  # Relative to the model

        # data loader
        self.train_loader, self.test_loader, self.data_iter = dict(), dict(), dict()
        for dset in self.args.datasets:
            print('Dataset: ' + dset)

            info_one_db = dbs_info[dset]
            train_info = [[dset, info_one_db["train"]["img_dir"], info_one_db["train"]["extension_img"]]]
            # Merge train db
            train_dataset = HTRDataset(train_info,
                                       fixed_size_img,
                                       width_divisor,
                                       args.pad_left,
                                       args.pad_right,
                                       all_labels,
                                       self.char_dict,
                                       aug_transforms,
                                       apply_noise=1,
                                       is_trainset=True)

            test_info = [[dset + "_test", info_one_db["train"]["img_dir"], info_one_db["train"]["extension_img"]]]
            test_dataset = HTRDataset(test_info,
                                      fixed_size_img,
                                      width_divisor,
                                      args.pad_left,
                                      args.pad_right,
                                      all_labels,
                                      self.char_dict,
                                      aug_transforms,
                                      apply_noise=0,        # No noise on test
                                      is_trainset=False)

            print("Train dataset size:", len(train_dataset))
            print("Test dataset size:", len(test_dataset))

            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=self.args.batch,
                shuffle=True,
                num_workers=int(self.args.workers),
                pin_memory=True,
                collate_fn=collate_fn)

            test_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=self.args.batch * 4,
                shuffle=False,
                num_workers=int(self.args.workers),
                collate_fn=collate_fn)

            self.train_loader[dset] = train_loader
            self.test_loader[dset] = test_loader

        self.nets, self.optims, self.losses = dict(), dict(), dict()
        self.loss_fns = Loss_Functions(args)

        self.logger = getLogger()
        # self.checkpoint = './checkpoint/%s/%s' % (args.task, args.ex)
        self.checkpoint = os.path.join(args.path_log, "checkpoint")
        self.step = 0

        self.name_classes_19 = [
            "road", "sidewalk", "building", "wall", "fence", "pole", "trafflight", "traffsign", "vegetation",
            "terrain", "sky", "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle",
        ]



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
        if not os.path.exists(self.checkpoint + '/%d' % self.step):
            os.mkdir(self.checkpoint + '/%d' % self.step)
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
        cnn_cfg = [(2, 64), 'M', (4, 128), 'M', (4, 256)]
        head_cfg = (256, 3)  # (hidden dimension, num_layers blstm)
        # width_divisor = 8

        model_reco = CRNN(cnn_cfg, head_cfg, self.charset.get_nb_char())
        before_blstm = model_reco.features
        # after_blstm = model_reco.top

        # Init model
        print("Initializing model weights kaiming")
        for p in model_reco.parameters():
            if p.dim() > 1:
                torch.nn.init.kaiming_normal_(p, nonlinearity="relu")

        if os.path.isfile(self.args.path_model):
            load_pretrained_model(self.args.path_model, model_reco, self.device)
        else:
            print("No pretrained model load")

        print(f"Transferring model to {str(self.device)}...")

        model_reco = model_reco.to(self.device)

        # Replace DRANet Encoder with your custom Extractor
        self.nets['E'] = before_blstm.to(self.device)
        with torch.no_grad():
            # dummy = torch.zeros(1, 1, self.imsize[0], self.imsize[1]).to(self.device)  # 1 channel input
            dummy = torch.zeros(1, 1, self.args.height_max,
                                self.args.width_max + self.args.pad_left + self.args.pad_right).to(self.device)  # 1 channel input
            feat = self.nets['E'](dummy)
            encoder_out_channels = feat.size(1)
        print(f"Encoder output channels: {encoder_out_channels}")

        self.nets['G'] = Generator(out_height=self.args.height_max,
                                   out_width_with_pad=self.args.width_max + self.args.pad_left + self.args.pad_right)
        # self.nets['S'] = Separator(self.imsize, self.training_converts)
        self.nets['S'] = Separator((self.args.height_max,
                                    self.args.width_max + self.args.pad_left + self.args.pad_right),
                                    self.training_converts)

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
                self.nets['T'][cv] = model_reco  # Classifier() change to Simon classifier
            elif self.args.task == 'seg':
                self.nets['T'][cv] = drn26()

        # Initialisation des paramètres (sauf E et T[cv] si clf)
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
        # self.set_zero_grad()
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
                source, target = convert.split("2read_", 1)
                target = "read_" + target
                _, styles[target] = cadt(contents[source], contents[target], styles[target])

        # Fake
        for convert in self.training_converts:
            source, target = convert.split("2read_", 1)
            target = "read_" + target
            converted_imgs[convert] = self.nets['G'](contents[convert], styles[target])
            if self.args.task == 'clf':
                D_outputs_fake[convert] = self.nets['D'][target](converted_imgs[convert])
            else:
                D_outputs_fake[convert] = self.nets['D'][target](slice_patches(converted_imgs[convert]))

        errD = self.loss_fns.dis(D_outputs_real, D_outputs_fake)
        errD.backward()
        if self.step % self.args.accum_steps == 0:
            for optimizer in self.optims['D'].values():
                optimizer.step()
                optimizer.zero_grad()
        self.losses['D'] = errD.data.item() / self.args.accum_steps

    def train_task(self, imgs, labels, x, x_reduced_len, y_enc, y_len_enc, y_gt_txt, print_target_source):  # Train Task Networks (T)
        # self.set_zero_grad()
        features = dict()
        converted_imgs = dict()
        pred = dict()
        converts = self.training_converts if self.args.task == 'clf' else self.test_converts
        with torch.no_grad():
            for dset in self.args.datasets:
                features[dset] = self.nets['E'](imgs[dset])
            contents, styles = self.nets['S'](features, converts)
            for convert in converts:
                source, target = convert.split("2read_", 1)
                target = "read_" + target
                if self.args.CADT:
                    _, styles[target] = cadt(contents[source], contents[target], styles[target])
                converted_imgs[convert] = self.nets['G'](contents[convert], styles[target])
                converted_imgs[convert] = torch.clamp(converted_imgs[convert], 0.0, 1.0)

            for convert in list(set(self.test_converts) - set(self.training_converts)):
                features_mid = dict()
                source, target = convert.split("2read_", 1)
                target = "read_" + target
                if print_target_source:
                    print("source: " + source)
                    print("target: " + target)
                mid = list(set(self.args.datasets) - {source, target})[0]
                convert1 = source + '2' + mid
                convert2 = mid + '2' + target
                features_mid[convert1] = self.nets['E'](converted_imgs[convert1])
                contents_mid, _ = self.nets['S'](features_mid, [convert2])
                converted_imgs[convert] = self.nets['G'](contents_mid[convert2], styles[target])
                converted_imgs[convert] = torch.clamp(converted_imgs[convert], 0.0, 1.0)

        for convert in self.test_converts:
            pred[convert] = self.nets['T'][convert](converted_imgs[convert])
            source, target = convert.split("2read_", 1)
            target = "read_" + target
            if print_target_source:
                print("source: " + source)
                print("target: " + target)
            pred[source] = self.nets['T'][convert](imgs[source])

        task_loss = None
        loss_enc_main_epoch = 0
        loss_enc_shortcut_epoch = 0
        weight_loss_shortcut = 0.1
        ctc_loss = torch.nn.CTCLoss(zero_infinity=True, reduction="mean")
        for key in pred.keys():
            if '2read_' in key:
                source, target = convert.split("2read_", 1)
                target = "read_" + target
            else:
                source = key

            # Simon training part for ctc loss
            y, _, after_blstm = pred[key]
            # Recognition loss
            output, aux_output = y

            output = torch.nn.functional.log_softmax(output, dim=-1)

            # Recognition losses
            try:
                B = output.shape[1]  # batch size
                if len(x_reduced_len) != B:
                    print(f"[CTC] Skipping batch: input_lengths={len(x_reduced_len)} batch_size={B}")
                    continue
                if len(y_len_enc) != y_enc.size(0):
                    print(f"[CTC] Skipping batch: target_lengths={len(y_len_enc)} != targets={y_enc.size(0)}")
                    continue

                loss = ctc_loss(output.cpu(), y_enc, x_reduced_len, y_len_enc)
                loss_enc_main_epoch += loss.detach().item()

                aux_output = torch.nn.functional.log_softmax(aux_output, dim=-1)
                loss_shortcut = weight_loss_shortcut * ctc_loss(aux_output.cpu(), y_enc, x_reduced_len, y_len_enc)
                loss_enc_shortcut_epoch += loss_shortcut.detach().item()

                loss += loss_shortcut
                task_loss = loss

            except RuntimeError as e:
                print(f"[CTC] RuntimeError caught, skipping batch: {e}")
                continue

        if task_loss is not None:
            errT = task_loss / self.args.accum_steps
            errT.backward()
            if self.step % self.args.accum_steps == 0:
                for optimizer in self.optims['T'].values():
                    optimizer.step()
                    optimizer.zero_grad()

            self.losses['T'] = errT.item()
        else:
            print("[CTC] No valid batch in this step, skipping optimizer step.")

    def train_esg(self, imgs):  # Train Encoder(E), Separator(S), Generator(G)
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
            source, target = convert.split("2read_", 1)
            target = "read_" + target
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
        if self.step % self.args.accum_steps == 0:
            for net in ['E', 'S', 'G']:
                self.optims[net].step()
                self.optims[net].zero_grad()

        self.losses['G'] = G_loss.data.item() / self.args.accum_steps
        self.losses['Recon'] = Recon_loss.data.item() / self.args.accum_steps
        self.losses['Consis'] = Consistency_loss.data.item() / self.args.accum_steps
        self.losses['Content'] = Content_loss.data.item() / self.args.accum_steps
        self.losses['Style'] = Style_loss.data.item() / self.args.accum_steps


    def eval(self, cv):
        source, target = cv.split("2read_", 1)
        target = "read_" + target
        print("source: " + source)
        print("target: " + target)
        self.set_eval()
        token_blank = self.char_dict[BLANK_STR_TOKEN]

        if self.args.task == 'clf':
            metrics_main = MetricLossCERWER("Main")
            metrics_shortcut = MetricLossCERWER("Shortcut")
            ctc_loss_fn = torch.nn.CTCLoss(zero_infinity=True, reduction="mean")

            with torch.no_grad():
                for batch_idx, batch_data in enumerate(self.test_loader[target]):
                    x = batch_data["imgs"]
                    x_reduced_len = batch_data["w_reduce"]

                    y_enc = batch_data["label_ind"]
                    y_len_enc = batch_data["label_ind_length"]

                    y_gt_txt = batch_data["label_str"]

                    x, y_enc = x.to(self.device), y_enc.to(self.device)

                    # Remove text padding
                    y_gt_txt = [t.strip() for t in y_gt_txt]
                    # Remove double spaces
                    y_gt_txt = [t.replace("  ", " ") for t in y_gt_txt]

                    nb_item_batch = x.shape[0]

                    y_pred, _, _ = self.nets['T'][cv](x)
                    output, aux_output = y_pred

                    # Main head
                    output_log = torch.nn.functional.log_softmax(output, dim=-1)

                    ctc_loss = ctc_loss_fn(output_log, y_enc, x_reduced_len, y_len_enc)
                    metrics_main.add_loss(ctc_loss.item(), nb_item_batch)

                    # (Nb frames, Batch size, Nb characters) -> (Batch size, Nb frames, Nb characters)
                    output_log = output_log.transpose(0, 1)

                    top = [torch.argmax(lp, dim=1).detach().cpu().numpy()[:x_reduced_len[j]] for j, lp in
                           enumerate(output_log)]
                    predictions_text = [ctc_best_path_one(p, self.char_list, token_blank) for p in top]

                    predictions_text = [t.strip() for t in predictions_text]  # Remove text padding
                    predictions_text = [t.replace("  ", " ") for t in predictions_text]

                    cers = [editdistance.eval(u, v) for u, v in zip(y_gt_txt, predictions_text)]
                    metrics_main.add_cer(sum(cers), nb_chars_from_list(y_gt_txt))

                    # Shortcut head
                    output_log = torch.nn.functional.log_softmax(aux_output, dim=-1)

                    ctc_loss = ctc_loss_fn(output_log, y_enc, x_reduced_len, y_len_enc)
                    metrics_shortcut.add_loss(ctc_loss.item(), nb_item_batch)

                    # (Nb frames, Batch size, Nb characters) -> (Batch size, Nb frames, Nb characters)
                    output_log = output_log.transpose(0, 1)

                    top_aux = [torch.argmax(lp, dim=1).detach().cpu().numpy()[:x_reduced_len[j]] for j, lp in
                               enumerate(output_log)]

                    predictions_text_aux = [ctc_best_path_one(p, self.char_list, token_blank) for p in top_aux]
                    predictions_text_aux = [t.strip() for t in predictions_text_aux]  # Remove text padding
                    predictions_text_aux = [t.replace("  ", " ") for t in predictions_text_aux]

                    cers = [editdistance.eval(u, v) for u, v in zip(y_gt_txt, predictions_text_aux)]
                    metrics_shortcut.add_cer(sum(cers), nb_chars_from_list(y_gt_txt))

            # Main head
            if metrics_main.nb_letters != 0:
                cer_main = metrics_main.cer / metrics_main.nb_letters
            else:
                cer_main = 1.0  # fallback if no letters

            # Shortcut head
            if metrics_shortcut.nb_letters != 0:
                cer_shortcut = metrics_shortcut.cer / metrics_shortcut.nb_letters
            else:
                cer_shortcut = 1.0

            if self.args.debug_pc == 0:
                # Log current metrics to wandb
                wandb.log({
                    f"Main/CER_{cv}": cer_main,
                    f"Shortcut/CER_{cv}": cer_shortcut,
                }, step=self.step)

            # Sauvegarde des meilleurs résultats
            if cer_main < self.best_cer[cv]:
                self.best_cer[cv] = cer_main

                if self.args.debug_pc == 0:
                    # self.writer.add_scalar('Best_Main_CER/%s' % cv, cer_main, self.step)
                    wandb.log({f'Best_Main_CER/{cv}': cer_main}, step=self.step)

            self.save_networks_last()

        self.set_train()

    def print_loss(self):
        best = ''
        wandb_log_data = {}

        if self.args.task == 'clf':
            for cv in self.test_converts:
                cer_main = self.best_cer[cv]
                cer_shortcut = self.best_cer_shortcut[cv]
                best += (f'{cv}: '
                         f'Main_CER={cer_main:.4f}, '
                         f'Shortcut_CER={cer_shortcut:.4f} |')
                # Log to wandb
                wandb_log_data[f'best_main_cer/{cv}'] = cer_main
                wandb_log_data[f'best_shortcut_cer/{cv}'] = cer_shortcut

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
        print("Eval start:")
        for cv in self.test_converts:
            self.eval(cv)

        for i in range(self.args.iter):
            self.step += 1

            # get batch data
            batch_data = self.get_batch(batch_data_iter)
            imgs, labels = dict(), dict()
            min_batch = self.args.batch

            for dset in self.args.datasets:
                x = batch_data[dset]["imgs"]

                x_reduced_len = batch_data[dset]["w_reduce"]

                y_enc = batch_data[dset]["label_ind"]
                y_len_enc = batch_data[dset]["label_ind_length"]

                y_gt_txt = batch_data[dset]["label_str"]

                x, y_enc = x.to(self.device), y_enc.to(self.device)

                # Remove text padding
                y_gt_txt = [t.strip() for t in y_gt_txt]
                # Remove double spaces
                y_gt_txt = [t.replace("  ", " ") for t in y_gt_txt]
                imgs[dset], labels[dset] = x.to(self.device), y_gt_txt

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

            print_target_source = False
            if i < 2:
                print_target_source = True

            self.train_task(imgs, labels, x, x_reduced_len, y_enc, y_len_enc, y_gt_txt, print_target_source)

            # evaluation
            if self.step % self.args.eval_freq == 0:
                for cv in self.test_converts:
                    self.eval(cv)

            # print and log losses
            losses = self.print_loss()  # Assuming this returns a dict like {'loss_total': x, 'loss_task': y}

            if self.args.debug_pc == 0:
                if isinstance(losses, dict):
                    wandb.log(losses, step=self.step)

    def test(self):
        self.set_default()
        self.set_networks()
        self.load_networks(self.args.load_step)
        for cv in self.test_converts:
            self.eval(cv)
