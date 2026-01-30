import os
import pickle

import idx2numpy
import numpy as np
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.utils import save_image


def extract_and_save_emnist(dir_save, split_version):
    dir_save_split = os.path.join(dir_save, split_version)
    os.makedirs(dir_save_split, exist_ok=True)

    training_data = datasets.EMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
        split=split_version
    )

    path_train = os.path.join(dir_save_split, "train")

    data = training_data.data
    data = torch.permute(data, (0, 2, 1))

    db = {
        "data": data,
        "label": training_data.targets,
        "class_to_idx": training_data.class_to_idx
    }

    dbfile = open(path_train, 'ab')

    # source, destination
    pickle.dump(db, dbfile)
    dbfile.close()

    test_data = datasets.EMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
        split=split_version
    )

    path_test = os.path.join(dir_save_split, "test")

    data = test_data.data
    data = torch.permute(data, (0, 2, 1))

    db = {
        "data": data,
        "label": test_data.targets,
        "class_to_idx": test_data.class_to_idx
    }

    dbfile = open(path_test, 'ab')

    # source, destination
    pickle.dump(db, dbfile)
    dbfile.close()


def extract_and_save_emnist_uppercase(dir_save):
    dir_save_split = os.path.join(dir_save, "upper_letters")
    os.makedirs(dir_save_split, exist_ok=True)

    training_data = datasets.EMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
        split="byclass"
    )

    path_train = os.path.join(dir_save_split, "train")

    data = training_data.data
    data = torch.permute(data, (0, 2, 1))

    # id: 10 digits then 26 lower case then 26 upper case
    filter_upper = (9 < training_data.targets) & (training_data.targets < 36)

    data = data[filter_upper]
    labels = training_data.targets[filter_upper]

    db = {
        "data": data,
        "label": labels,
        "class_to_idx": training_data.class_to_idx
    }

    dbfile = open(path_train, 'ab')

    # source, destination
    pickle.dump(db, dbfile)
    dbfile.close()

    test_data = datasets.EMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
        split="byclass"
    )

    path_test = os.path.join(dir_save_split, "test")

    data = test_data.data
    data = torch.permute(data, (0, 2, 1))

    # id: 10 digits then 26 lower case then 26 upper case
    filter_upper = (9 < test_data.targets) & (test_data.targets < 36)

    data = data[filter_upper]
    labels = test_data.targets[filter_upper]

    db = {
        "data": data,
        "label": labels,
        "class_to_idx": test_data.class_to_idx
    }

    dbfile = open(path_test, 'ab')

    # source, destination
    pickle.dump(db, dbfile)
    dbfile.close()


# Create subset to debug on CPU
def create_debug_dataset(path_origin, path_new, number_to_keep):
    dbfile = open(path_origin, 'rb')
    db = pickle.load(dbfile)

    data_d = db["data"].clone()
    label_d = db["label"].clone()

    all_index = torch.randperm(data_d.shape[0])
    index_filter = all_index[:number_to_keep]

    data_d = data_d[index_filter]
    label_d = label_d[index_filter]

    dbfile.close()

    db_debug = {
        "data": data_d,
        "label": label_d,
        "class_to_idx": db["class_to_idx"]
    }

    dbfile_debug = open(path_new, 'ab')

    # source, destination
    pickle.dump(db_debug, dbfile_debug)
    dbfile_debug.close()


def split_train_val(path_origin, path_save_train, path_save_val, ratio_train):
    dbfile = open(path_origin, 'rb')
    db = pickle.load(dbfile)

    data = db["data"].clone()
    label = db["label"].clone()
    class_to_idx = db["class_to_idx"]
    dbfile.close()

    nb_item = data.shape[0]
    number_train = int(ratio_train * nb_item)

    all_index = torch.randperm(nb_item)
    #Train
    index_train = all_index[:number_train]

    data_train = data[index_train]
    label_train = label[index_train]

    db_train = {
        "data": data_train,
        "label": label_train,
        "class_to_idx": class_to_idx
    }

    dbfile_train = open(path_save_train, 'ab')

    pickle.dump(db_train, dbfile_train)
    dbfile_train.close()

    # Validation
    index_val = all_index[number_train:]

    data_val = data[index_val]
    label_val = label[index_val]

    db_val = {
        "data": data_val,
        "label": label_val,
        "class_to_idx": class_to_idx
    }

    dbfile_val = open(path_save_val, 'ab')

    pickle.dump(db_val, dbfile_val)
    dbfile_val.close()


def save_tensor_as_img(path_data, dir_save):

    dbfile = open(path_data, 'rb')
    db = pickle.load(dbfile)

    index = 0
    for one_img in db["data"]:
        one_img_n = one_img / 255
        path_save = os.path.join(dir_save, str(index) + ".png")
        save_image(one_img_n, path_save)

        index += 1


if __name__ == '__main__':

    # To do: define the path
    working_dir = ""

    # # EMNIST
    ratio_train = 0.8

    # Digits
    extract_and_save_emnist(working_dir, "digits")

    # Split train into training and validation
    path_train = os.path.join(working_dir, "digits", "train")
    path_save_train = os.path.join(working_dir, "digits", "train_80")
    path_save_val = os.path.join(working_dir, "digits", "val_20")
    split_train_val(path_train, path_save_train, path_save_val, ratio_train)

    # # Upper Letter
    extract_and_save_emnist_uppercase(working_dir)

    # Split train into training and validation
    path_train = os.path.join(working_dir, "upper_letters", "train")
    path_save_train = os.path.join(working_dir, "upper_letters", "train_80")
    path_save_val = os.path.join(working_dir, "upper_letters", "val_20")
    split_train_val(path_train, path_save_train, path_save_val, ratio_train)


