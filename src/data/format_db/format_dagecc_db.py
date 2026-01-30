import os.path
import csv
import glob
import pickle

import torch
from skimage import io

from src.data.image.transform.rescal_transform import rescale_fix_size_img


def format_safran_mnist(dir_img, path_csv_label, height_resize, width_resize, path_save, ext_img="png"):
    set_label = set()

    dict_id_label = {}

    # Read label gt
    with open(path_csv_label, newline='', encoding='utf-8') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')

        i_line = 0
        for row in spamreader:
            if i_line > 0:
                dict_id_label[row[0]] = row[1]

                set_label.add(row[1])

            i_line += 1

    # Order label
    list_label = list(set_label)
    list_label = sorted(list_label)

    # Create class_to_idx
    class_to_idx = {}

    for i_s in range(len(list_label)):
        class_to_idx[list_label[i_s]] = i_s

    files_img = glob.glob(dir_img + '/*.' + ext_img)

    labels_all = []
    imgs_all = []
    # Foreach image
    for one_path_img in files_img:
        img = io.imread(one_path_img, as_gray=True)
        # Resize
        img = rescale_fix_size_img(img, height_resize, width_resize, pad_value=0)

        imgs_all.append(img)

        split_name = os.path.split(one_path_img)
        split_name = split_name[1].split(sep=".")  # Filename and extension
        id_file = split_name[0]
        label = dict_id_label[id_file]
        label_index = class_to_idx[label]

        labels_all.append(label_index)

    labels_all = torch.tensor(labels_all)
    imgs_all = torch.tensor(imgs_all)

    db_data = {
        "data": imgs_all,
        "label": labels_all,
        "class_to_idx": class_to_idx
    }

    dbfile = open(path_save, 'ab')

    # Save
    pickle.dump(db_data, dbfile)
    dbfile.close()


if __name__ == '__main__':
    # To do: define the path where the data are
    working_dir = ""
    height_resize = 48
    width_resize = 48

    dir_save = os.path.join(working_dir, "preprocess_48_48")
    os.makedirs(dir_save, exist_ok=True)

    # Note: unzip images tar.gz

    path_csv_label = os.path.join(working_dir, "Safran-MNIST-DLS_gt_training.csv")

    path_save = os.path.join(dir_save, "train")
    dir_img = os.path.join(working_dir, "training")

    format_safran_mnist(dir_img, path_csv_label, height_resize, width_resize, path_save)

    path_csv_label = os.path.join(working_dir, "Safran-MNIST-DLS_gt_validation.csv")

    path_save = os.path.join(dir_save, "validation")
    dir_img = os.path.join(working_dir, "validation")

    format_safran_mnist(dir_img, path_csv_label, height_resize, width_resize, path_save)

    path_csv_label = os.path.join(working_dir, "Safran-MNIST-DLS_gt_testing.csv")

    path_save = os.path.join(dir_save, "test")
    dir_img = os.path.join(working_dir, "testing")

    format_safran_mnist(dir_img, path_csv_label, height_resize, width_resize, path_save)

