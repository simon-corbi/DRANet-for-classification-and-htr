import argparse
import faulthandler
import os
import time

import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader

from src.data.collate.classif_batch_collate import CollateImageLabelClassification
from src.data.config_reader.reader_conf_char_classif import read_json_config_char_classif
from src.data.datasets.multiple_dataset_classif import MultipleImageLabelDataset
from src.data.datasets.one_dataset_classif import OneImageLabelDataset
from src.data.image.augmentations.config_aug import get_config_aug_classif
from src.data.text.charset_token import CharsetToken
from src.evaluate.supervised.eval_one_epoch_cnn import evaluate_cnn_one_epoch
from src.models.expert_models.resnet_torchvision import BasicBlock, ResNetTorch
from src.models.load_pretrained_model import load_pretrained_model
from src.models.losses.center_loss_fct import compute_center_loss_k1
from src.models.losses.compute_center_cluster import compute_cluster_center_k_1_cnn

parser = argparse.ArgumentParser()

parser.add_argument("config_file")
parser.add_argument("log_dir")

parser.add_argument('--learning_rate', default=1e-4, type=float)
parser.add_argument('--batch_size', default=1024, type=int)
parser.add_argument('--num_workers', default=0, type=int)
parser.add_argument('--nb_epochs_max', default=150, type=int)
parser.add_argument("--path_model", default="", help="path of pretrained model", type=str)
parser.add_argument('--height_max', default=48, type=int)
parser.add_argument('--width_max', default=48, type=int)

parser.add_argument('--milestones_lr', default=100, type=int)
parser.add_argument('--lr_decay', default=10, type=float)

parser.add_argument('--use_center_loss', default=1, type=int)
parser.add_argument('--epoch_start_center_loss', default=100, type=int)
parser.add_argument('--weight_center_loss_ok', default=1.0, type=float)
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

# Paths data
train_info, val_info, test_info, charset_path = read_json_config_char_classif(args.config_file)
print(train_info)

# Alphabet
charset = CharsetToken(charset_path)
nb_char_all = charset.get_nb_char()

# For class not present in charset
ignore_index = nb_char_all  # + 1
charset.add_label("IGNORE")

char_list = charset.get_charset_list()
char_dict = charset.get_charset_dictionary()
nb_char_all = charset.get_nb_char()

# Data
fixed_size_img = (args.height_max, args.width_max)

image_size = args.height_max

config_augmentation = get_config_aug_classif()

begin_init_train = time.time()
train_db = MultipleImageLabelDataset(train_info,
                                     fixed_size_img,
                                     char_dict,
                                     ignore_index,
                                     apply_augmentation=True,
                                     config_augmentation=config_augmentation)

end_init_train = time.time()
print("Time init train (s): " + str((end_init_train - begin_init_train)))
print('Nb samples train {}:'.format(len(train_db)))

# Pad img with black = 0
c_collate_fn = CollateImageLabelClassification(imgs_pad_value=[0])
collate_fn = c_collate_fn.collate_fn

train_dataloader = DataLoader(train_db, num_workers=args.num_workers, batch_size=args.batch_size, pin_memory=True,
                              collate_fn=collate_fn, shuffle=True)

all_val_dataloader = []
dbs_val = []
for db in val_info:
    val_db = OneImageLabelDataset(db[1],
                                  fixed_size_img,
                                  char_dict,
                                  ignore_index)

    print('Nb samples val {}:'.format(len(val_db)))

    val_dataloader = DataLoader(val_db, num_workers=args.num_workers, batch_size=args.batch_size, pin_memory=True,
                                collate_fn=collate_fn, shuffle=False)

    all_val_dataloader.append(val_dataloader)
    dbs_val.append(db[0])

all_test_dataloader = []
dbs_test = []
for db in test_info:
    test_db = OneImageLabelDataset(db[1],
                                   fixed_size_img,
                                   char_dict,
                                   ignore_index)

    print('Nb samples test {}:'.format(len(test_db)))

    test_dataloader = DataLoader(test_db, num_workers=args.num_workers, batch_size=args.batch_size, pin_memory=True,
                                 collate_fn=collate_fn, shuffle=False)

    all_test_dataloader.append(test_dataloader)
    dbs_test.append(db[0])

# Model ResNet34
nb_channel_input = 1
dim_features_center_loss = 512

model = ResNetTorch(BasicBlock, [3, 4, 6, 3], nb_channel_input=nb_channel_input, num_classes=nb_char_all)

if os.path.isfile(args.path_model):
    load_pretrained_model(args.path_model, model, device)

print(f"Transferring model to {str(device)}...")
model = model.to(device)

number_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model has {number_parameters:,} trainable parameters.")

optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

loss_classif = nn.CrossEntropyLoss(ignore_index=ignore_index)
loss_center_fct = torch.nn.MSELoss(reduction="mean")

path_save_model_last = os.path.join(args.log_dir, "cnn_last.torch")
path_save_model_best = os.path.join(args.log_dir, "cnn_best.torch")

best_f1_val = 0
best_f1_val_epoch = -1

begin_train = time.time()

centers_coordinates = []
compute_center_loss = False

for epoch in range(0, args.nb_epochs_max):
    begin_time_epoch = time.time()
    print('EPOCH {}:'.format(epoch))

    # Learning rate values  -> refactor with step scheduler
    if epoch < args.milestones_lr:
        lr = args.learning_rate
    # First decay
    else:
        lr = args.learning_rate / args.lr_decay

    for g in optimizer.param_groups:
        g['lr'] = lr
    print("lr:" + str(lr))

    # Training
    model.train()
    id_batch = 0
    total_loss_train = 0
    center_loss_epoch = 0
    nb_item = 0
    for i, (images, labels) in enumerate(train_dataloader):
        # Move tensors to the configured device
        images = images.to(device)
        labels = labels.to(device)

        batch_size = images.shape[0]

        # Error with batch size
        if batch_size == 1:
            continue

        # Forward pass
        outputs, features = model.forward_return_intermediate_values(images)

        loss = loss_classif(outputs, labels)

        if compute_center_loss:
            features = torch.sigmoid(features)

            _, predicted = torch.max(outputs.data, 1)

            # Group features by character
            dict_feature_per_class = {}
            for i_item in range(batch_size):
                if predicted[i_item] == labels[i_item]:
                    if predicted[i_item].item() in dict_feature_per_class:
                        dict_feature_per_class[predicted[i_item].item()].append(features[i_item])
                    else:
                        dict_feature_per_class[predicted[i_item].item()] = [features[i_item]]

            center_loss_batch = compute_center_loss_k1(dict_feature_per_class, centers_coordinates, loss_center_fct)

            center_loss_batch *= args.weight_center_loss_ok

            # Cas all predictions are classes filtered or cer != 0
            if not isinstance(center_loss_batch, float) and not isinstance(center_loss_batch, int):
                center_loss_batch = center_loss_batch.to(loss.device)
                loss += center_loss_batch
                center_loss_epoch += center_loss_batch.detach().item()

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        total_loss_train += loss.item()
        nb_item += batch_size

    if nb_item != 0:
        total_loss_train /= nb_item
        center_loss_epoch /= nb_item

    print('Train Loss: {:.4f}'.format(total_loss_train))

    if compute_center_loss:
        print('Train center Loss: {:.4f}'.format(center_loss_epoch))

    print("Validation")
    mean_f1 = 0
    for i_db in range(len(val_info)):
        print("--------------------------------------")
        print(dbs_val[i_db])
        f1_macro_total, f1_per_class, loss_val_total = evaluate_cnn_one_epoch(model, all_val_dataloader[i_db], device,
                                                                              loss_classif,
                                                                              f1_score)

        print('F1-Macro : {:.2f} %'.format(100 * f1_macro_total))
        print('Val Loss : ' + str(loss_val_total))

        nb_char = len(char_list)
        nb_class_val = len(f1_per_class)

        for i_c in range(nb_char):
            if i_c < nb_class_val:
                print(char_list[i_c] + f": {f1_per_class[i_c]:.2f}")

        mean_f1 += f1_macro_total

    # Best if mean of all f1
    if len(val_info) > 0:
        mean_f1 /= len(val_info)

    if mean_f1 > best_f1_val:
        print("Find better model")
        best_f1_val = mean_f1
        best_f1_val_epoch = epoch
        torch.save(model.state_dict(), path_save_model_best)

    # Compute cluster if activate
    if args.use_center_loss == 1:
        if epoch >= args.epoch_start_center_loss:
            print("Compute prototype:")
            compute_center_loss = True

            centers_coordinates = compute_cluster_center_k_1_cnn(train_dataloader,
                                                             model,
                                                             device,
                                                             nb_classes=nb_char_all - 1,  # without ignore class
                                                             dim_features=dim_features_center_loss)

    end_time_epoch = time.time()
    print("Time one epoch (s): " + str((end_time_epoch - begin_time_epoch)))
    print("")

    torch.save(model.state_dict(), path_save_model_last)

end_train = time.time()

print("best_f1_val_epoch:" + str(best_f1_val_epoch))
print(f"best_f1_val: {100 * best_f1_val:.2f}%")

print("Time all (s): " + str((end_train - begin_train)))
print("End training")

print("--------Testing-----------")
for i_db in range(len(test_info)):
    print("--------------------------------------")
    print(dbs_test[i_db])

    if os.path.isfile(path_save_model_last):
        load_pretrained_model(path_save_model_last, model, device, print_load_ok=False)

    f1_macro_total, f1_per_class, loss_val_total = evaluate_cnn_one_epoch(model, all_test_dataloader[i_db], device,
                                                                          loss_classif, f1_score)

    print("Last model ")
    print('F1-Macro : {:.2f} %'.format(100 * f1_macro_total))
    print('Val Loss : ' + str(loss_val_total))

    nb_char = len(char_list)
    nb_class_val = len(f1_per_class)

    for i_c in range(nb_char):
        if i_c < nb_class_val:
            print(char_list[i_c] + f": {f1_per_class[i_c]:.2f}")

    print("DB " + str(dbs_test[i_db]))
    print("Last model F1-Macro: ")
    print(f"{100 * f1_macro_total:.2f}% \n")

    if os.path.isfile(path_save_model_best):
        load_pretrained_model(path_save_model_best, model, device, print_load_ok=False)

    f1_macro_total, f1_per_class, loss_val_total = evaluate_cnn_one_epoch(model,
                                                                          all_test_dataloader[i_db],
                                                                          device,
                                                                          loss_classif,
                                                                          f1_score,
                                                                          print_first_batch=True,
                                                                          char_list=char_list)

    print("Best model ")
    print('F1-Macro : {:.2f} %'.format(100 * f1_macro_total))
    print('Val Loss : ' + str(loss_val_total))

    nb_char = len(char_list)
    nb_class_val = len(f1_per_class)

    for i_c in range(nb_char):
        if i_c < nb_class_val:
            print(char_list[i_c] + f": {f1_per_class[i_c]:.2f}")

    print("Best model F1-Macro: ")
    print(f"{100 * f1_macro_total:.2f}% \n")

print("End")

