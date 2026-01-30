import argparse
import faulthandler
import os
import time

import torch
from torch.utils.data import DataLoader

from src.data.collate.collate_batch import CollateImageLabelHTR
from src.data.datasets.htr_dataset import HTRDataset
from src.data.global_values.text_global_values import CTC_PAD, BLANK_STR_TOKEN
from src.data.image.augmentations.aug_htr import get_img_data_augmentation_htr
from src.data.readers.reader_config import read_json_config, read_labels_files, read_json_config_crnn_supervised
from src.data.text.charset_token import CharsetToken
from src.evaluate.metrics_counter import MetricLossCERWER
from src.evaluate.supervised.eval_one_epoch_crnn import evaluate_one_epoch_crnn
from src.models.crnn import CRNN
from src.models.losses.compute_center_cluster import compute_center_coordinates_crnn
from src.models.model_utils import load_pretrained_model
from src.train.supervised.train_crnn_one_epoch import train_crnn_one_epoch

parser = argparse.ArgumentParser()

parser.add_argument("config_file")
parser.add_argument("log_dir")

parser.add_argument('--learning_rate', default=5e-4, type=float)
parser.add_argument('--batch_size', default=4, type=int)  # To update
parser.add_argument('--num_workers', default=0, type=int)
parser.add_argument('--nb_epochs_max', default=2, type=int)  # To update
parser.add_argument("--path_model", default="", type=str)
parser.add_argument("--path_optimizer", default="", help="", type=str)
parser.add_argument('--height_max', default=160, type=int)
parser.add_argument('--width_max', default=1570, type=int)
parser.add_argument('--pad_left', default=64, type=int)
parser.add_argument('--pad_right', default=64, type=int)

parser.add_argument('--milestones_lr_1', default=700, type=int)
parser.add_argument('--lr_decay_1', default=10, type=float)

# Regularization
parser.add_argument('--use_regularization', default=1, type=int)
parser.add_argument('--epoch_start_regularization', default=1, type=int) # To update
parser.add_argument('--weight_loss_regularization_ok', default=0.55, type=float)
print("===============================================================================")

begin = time.time()
args = parser.parse_args()
print(args)

faulthandler.enable()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("device :")
print(device)
print("torch.cuda.is_available(): " + str(torch.cuda.is_available()))
print("torch.cuda.device_count(): " + str(torch.cuda.device_count()))

# Paths
train_info, val_info, test_info, charsets_path, all_labels_files, _ = read_json_config_crnn_supervised(args.config_file)
all_labels = read_labels_files(all_labels_files)

# Alphabet
charset = CharsetToken(charsets_path, use_blank=True)
char_list = charset.get_charset_list()
char_dict = charset.get_charset_dictionary()

# Model
cnn_cfg = [(2, 64), 'M', (4, 128), 'M', (4, 256)]
head_cfg = (256, 3)  # (hidden dimension, num_layers blstm)
width_divisor = 8

model_reco = CRNN(cnn_cfg, head_cfg, charset.get_nb_char())

# Data
fixed_size_img = (args.height_max, args.width_max)
aug_transforms = get_img_data_augmentation_htr()

# Pad img with black = 0
c_collate_fn = CollateImageLabelHTR(imgs_pad_value=[0], pad_txt=CTC_PAD)
collate_fn = c_collate_fn.collate_fn

# Merge train db
train_db = HTRDataset(train_info,
                      fixed_size_img,
                      width_divisor,
                      args.pad_left,
                      args.pad_right,
                      all_labels,
                      char_dict,
                      aug_transforms,
                      apply_noise=1,
                      is_trainset=True)
print('Nb samples train {}:'.format(len(train_db)))
train_dataloader = DataLoader(train_db, num_workers=args.num_workers, batch_size=args.batch_size, pin_memory=True,
                              collate_fn=collate_fn, shuffle=True)

# Separe validation db
all_val_dataloader = []
all_val_dataset = []

for one_db in val_info:
    val_db = HTRDataset([one_db],
                        fixed_size_img,
                        width_divisor,
                        args.pad_left,
                        args.pad_right,
                        all_labels,
                        char_dict)
    print('Nb samples val {}:'.format(len(val_db)))

    val_dataloader = DataLoader(val_db, num_workers=args.num_workers, batch_size=args.batch_size, pin_memory=True,
                                collate_fn=collate_fn, shuffle=False)

    all_val_dataloader.append(val_dataloader)
    all_val_dataset.append(val_db)  # To get name DB

all_test_dataloader = []
all_test_dataset = []

# Separate test db
for one_db in test_info:
    test_db = HTRDataset([one_db],
                         fixed_size_img,
                         width_divisor,
                         args.pad_left,
                         args.pad_right,
                         all_labels,
                         char_dict)

    print('Nb samples test {}:'.format(len(test_db)))

    test_dataloader = DataLoader(test_db, num_workers=args.num_workers, batch_size=args.batch_size, pin_memory=True,
                                 collate_fn=collate_fn, shuffle=False)

    all_test_dataloader.append(test_dataloader)
    all_test_dataset.append(test_db)

# Init model
print("Initializing model weights kaiming")
for p in model_reco.parameters():
    if p.dim() > 1:
        torch.nn.init.kaiming_normal_(p, nonlinearity="relu")

if os.path.isfile(args.path_model):
    load_pretrained_model(args.path_model, model_reco, device)

print(f"Transferring model to {str(device)}...")
model_reco = model_reco.to(device)

number_parameters = sum(p.numel() for p in model_reco.parameters() if p.requires_grad)
print(f"Model has {number_parameters:,} trainable parameters.")

print_summary = ""

ctc_loss_fn = torch.nn.CTCLoss(zero_infinity=True, reduction="mean")

optimizer = torch.optim.Adam(model_reco.parameters(), lr=args.learning_rate)

if os.path.isfile(args.path_optimizer):
    try:
        checkpoint = torch.load(args.path_optimizer, map_location=device)
        optimizer.load_state_dict(checkpoint)
        print("Load optimizer")
    except:
        print("Error load optimizer")
        optimizer = torch.optim.Adam(model_reco.parameters(), lr=args.learning_rate)

best_cer = 1.0
best_epoch = 0

path_save_model_best = os.path.join(args.log_dir, "crnn_best.torch")
path_save_model_last = os.path.join(args.log_dir, "crnn_last.torch")

path_save_optimizer_best = os.path.join(args.log_dir, "optimizer_best.torch")
path_save_optimizer_last = os.path.join(args.log_dir, "optimizer_last.torch")

lr = args.learning_rate
# For center loss
index_class_to_filter = [char_dict[BLANK_STR_TOKEN], char_dict[" "]]
loss_reg = torch.nn.MSELoss(reduction="mean")

conf_reg = {
    "index_class_to_filter": index_class_to_filter,
    "loss_reg": loss_reg,
    "weight_loss_regularization_ok": args.weight_loss_regularization_ok
}

ceneters_value = []
compute_loss_reg = False

begin_train = time.time()
# Training
for epoch in range(0, args.nb_epochs_max):
    begin_time_epoch = time.time()
    print('EPOCH {}:'.format(epoch))

    # Learning rate values
    if epoch < args.milestones_lr_1:
        lr = args.learning_rate
    # decay
    else:
        lr = args.learning_rate / args.lr_decay_2

    for g in optimizer.param_groups:
        g['lr'] = lr
    print("lr:" + str(lr))

    # Training
    dict_losses = train_crnn_one_epoch(train_dataloader,
                                       optimizer,
                                       model_reco,
                                       device,
                                       ctc_loss_fn,
                                       compute_loss_reg,
                                       conf_reg,
                                       ceneters_value,
                                       char_list,
                                       char_dict[BLANK_STR_TOKEN])

    print('train_loss_main {}'.format(dict_losses["loss_main"]))

    if compute_loss_reg:
        print('train_loss_reg_epoch {}'.format(dict_losses["loss_reg_epoch"]))

    for i_db in range(len(all_val_dataloader)):
        print("--------------Evaluate-------------------------------")
        current_db_name = all_val_dataset[i_db].name_db
        print(current_db_name)

        dict_result = evaluate_one_epoch_crnn(all_val_dataloader[i_db],
                                              model_reco,
                                              device,
                                              char_list,
                                              char_dict[BLANK_STR_TOKEN],
                                              ctc_loss_fn)

        dict_result["metrics_main"].print_cer_wer()

        # Save best model
        # -> to update if multiple validation datasets
        if dict_result["metrics_main"].get_cer() < best_cer:
            best_cer = dict_result["metrics_main"].get_cer()
            best_epoch = epoch
            print("Best cer, save model.")

            torch.save(model_reco.state_dict(), path_save_model_best)
            torch.save(optimizer.state_dict(), path_save_optimizer_best)

    # Compute clusters coordinates if activate
    if args.use_regularization == 1:
        if epoch >= args.epoch_start_regularization:
            print("Compute prototype")
            compute_loss_reg = True

            ceneters_value = compute_center_coordinates_crnn(train_dataloader,
                                                        model_reco,
                                                        device,
                                                        char_list,
                                                        char_dict["<BLANK>"],
                                                        index_class_to_filter)

    end_time_epoch = time.time()
    print("Time one epoch (s): " + str((end_time_epoch - begin_time_epoch)))
    print("")

    torch.save(model_reco.state_dict(), path_save_model_last)
    torch.save(optimizer.state_dict(), path_save_optimizer_last)

end_train = time.time()
print("best_epoch: " + str(best_epoch))
print("best_cer val: " + str(best_cer))
print("Time all (s): " + str((end_train - begin_train)))
print("End training")

print_summary += "best_epoch: " + str(best_epoch) + "\n"
print_summary += "best_cer val: "
print_summary += f"{100 * best_cer:.2f}% \n"
print_summary += "\n"

metrics_test_last = MetricLossCERWER("All Tests last")
metrics_test_best = MetricLossCERWER("All Tests best")

print("--------------Testing-------------------------------")
for i_db in range(len(all_test_dataloader)):
    # Load last model -> case multiple test datasets
    if os.path.isfile(path_save_model_last):
        load_pretrained_model(path_save_model_last, model_reco, device, print_load_ok=False)

    print()
    current_db_name = all_test_dataset[i_db].name_db
    print(current_db_name)
    print_summary += current_db_name
    print_summary += " \n"

    print()
    print("--------Begin Testing last-----------")
    dict_result = evaluate_one_epoch_crnn(all_test_dataloader[i_db],
                                          model_reco,
                                          device,
                                          char_list,
                                          char_dict[BLANK_STR_TOKEN],
                                          ctc_loss_fn)

    dict_result["metrics_main"].print_cer_wer()

    metrics_test_last.add_metric(dict_result["metrics_main"])

    print_summary += "Testing last \n"
    str_cer_wer = dict_result["metrics_main"].str_cer_wer()
    print_summary += str_cer_wer + "\n"

    if len(all_val_dataloader) == 1:
        print("--------Begin Testing best cer val-----------")
        # Load best model
        if os.path.isfile(path_save_model_best):
            load_pretrained_model(path_save_model_best, model_reco, device, print_load_ok=False)

        dict_result = evaluate_one_epoch_crnn(all_test_dataloader[i_db],
                                              model_reco,
                                              device,
                                              char_list,
                                              char_dict[BLANK_STR_TOKEN],
                                              ctc_loss_fn)

        dict_result["metrics_main"].print_cer_wer()
        metrics_test_best.add_metric(dict_result["metrics_main"])

        print_summary += "\n"
        print_summary += "Testing best val cer \n"
        str_cer_wer = dict_result["metrics_main"].str_cer_wer()
        print_summary += str_cer_wer + "\n"


print()
# for all tests -> norm by all characters
metrics_test_last.print_cer()
if len(all_val_dataloader) == 1:
    metrics_test_best.print_cer()

print()
print("Summary:")
print(print_summary)

