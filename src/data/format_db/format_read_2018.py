import json
import os
import glob
import random
import shutil
from pathlib import Path

from src.data.text.charset_util import create_charset, merge_charset


def format_dir_data(origin_dir, save_dir, ext_img=".jpg"):
    files_img = Path(origin_dir).rglob('*' + ext_img)

    save_img = os.path.join(save_dir, "img")
    save_label = os.path.join(save_dir, "label")

    os.makedirs(save_img, exist_ok=True)
    os.makedirs(save_label, exist_ok=True)

    for one_img_file_path in files_img:
        one_img_file_path_str = str(one_img_file_path)
        one_label_file_path = one_img_file_path_str + ".txt"

        id_sample = one_img_file_path.stem

        if os.path.isfile(one_label_file_path):
            path_to = os.path.join(save_img, id_sample + ext_img)
            shutil.copyfile(one_img_file_path_str, path_to)

            path_to = os.path.join(save_label, id_sample + ".txt")
            shutil.copyfile(one_label_file_path, path_to)

        else:
            print("File does not exist:" + one_label_file_path)


def split_train_val(origin_dir, save_dir, ratio_train, ext_img):

    files_img = glob.glob(origin_dir + '/**/*.' + ext_img, recursive=True)
    random.shuffle(files_img)

    nb_train = int(ratio_train * len(files_img))

    # Train
    save_dir_train = os.path.join(save_dir, "train")

    save_img = os.path.join(save_dir_train, "img")
    save_label = os.path.join(save_dir_train, "label")

    os.makedirs(save_img, exist_ok=True)
    os.makedirs(save_label, exist_ok=True)

    for one_img_path in files_img[:nb_train]:

        id_sample = Path(one_img_path).stem

        path_to = os.path.join(save_img, id_sample + "." + ext_img)
        shutil.copyfile(one_img_path, path_to)

        path_label_origin = os.path.join(origin_dir, "label", id_sample + ".txt")

        path_to = os.path.join(save_label, id_sample + ".txt")
        shutil.copyfile(path_label_origin, path_to)

    # Validation
    save_dir_val = os.path.join(save_dir, "validation")

    save_img = os.path.join(save_dir_val, "img")
    save_label = os.path.join(save_dir_val, "label")

    os.makedirs(save_img, exist_ok=True)
    os.makedirs(save_label, exist_ok=True)

    for one_img_path in files_img[nb_train:]:

        id_sample = Path(one_img_path).stem

        path_to = os.path.join(save_img, id_sample + "." + ext_img)
        shutil.copyfile(one_img_path, path_to)

        path_label_origin = os.path.join(origin_dir, "label", id_sample + ".txt")

        path_to = os.path.join(save_label, id_sample + ".txt")
        shutil.copyfile(path_label_origin, path_to)


def format_specific_data_all_list(origin_dir_data, origin_dir_data_split, dir_save, ext_img):

    listes_split = glob.glob(origin_dir_data_split + '/**/*', recursive=True)

    for one_list_path in listes_split:
        format_specific_data_one_list(origin_dir_data, one_list_path, dir_save, ext_img)


def format_specific_data_one_list(origin_dir_data, path_file_list, dir_save, ext_img):

    name_list = Path(path_file_list).stem
    dir_save_one_list = os.path.join(dir_save, name_list)

    save_img = os.path.join(dir_save_one_list, "img")
    save_label = os.path.join(dir_save_one_list, "label")

    os.makedirs(save_img, exist_ok=True)
    os.makedirs(save_label, exist_ok=True)

    # Read name + extension image
    with open(path_file_list) as fp:
        for id_item in fp:
            if len(id_item) > 1:
                if id_item[-1] == "\n":
                    id_item = id_item[:-1]  # Remove \n

            # Search img path
            img_paths = glob.glob(origin_dir_data + '/**/' + id_item, recursive=True)

            if len(img_paths) == 1:
                # remove extension
                id_sample = Path(img_paths[0]).stem

                path_to = os.path.join(save_img, id_sample + "." + ext_img)
                shutil.copyfile(img_paths[0], path_to)

                path_label_origin = img_paths[0] + ".txt"

                path_to = os.path.join(save_label, id_sample + ".txt")
                shutil.copyfile(path_label_origin, path_to)

            else:
                print("File not found:" + id_item)


def create_charset_list_dir(dir_root):
    list_dir = glob.glob(dir_root + '/*/')

    for one_d in list_dir:

        name_dir = Path(one_d).stem

        path_charset = os.path.join(dir_root, "charset_" + name_dir + ".txt")
        path_label = os.path.join(one_d, "label")

        create_charset(path_label, path_charset)


def split_list_dir(root_dir_origin, root_dir_save, ratio_train, ext_img):
    list_dir = glob.glob(root_dir_origin + '/*/')

    for one_d in list_dir:
        if os.path.isdir(one_d):

            name_dir = Path(one_d).stem
            save_dir = os.path.join(root_dir_save, name_dir)
            os.makedirs(save_dir, exist_ok=True)

            split_train_val(one_d, save_dir, ratio_train, ext_img)


# def read_label_file(label_file):
#     dict_id_label = {}
#
#     with open(label_file) as fp:
#         for text_line in fp:
#             if len(text_line) > 1:
#                 if text_line[-1] == "\n":
#                     text_line = text_line[:-1]  # Remove \n
#
#                 # Example line
#                 # test_data/30866/Konzilsprotokolle_C/30866_0027_1063780_r1_r1l19.jpg derliche mitzutheilen und allenfalls mit seinem
#                 elts = text_line.split(sep=" ", maxsplit=1)
#
#                 if len(elts) == 2:
#                     id_sample = elts[0].split(sep="/") # test_data/30866/Konzilsprotokolle_C/30866_0027_1063780_r1_r1l19.jpg
#                     id_sample = id_sample[-1]  # 30866_0027_1063780_r1_r1l19.jpg
#                     id_sample = id_sample[:-4]  # 30866_0027_1063780_r1_r1l19
#
#                     label = elts[1]
#
#                     dict_id_label[id_sample] = label
#
#     return dict_id_label


def format_one_test_dir(dir_origin, dir_save, all_label_dict, ext_img):
    save_img = os.path.join(dir_save, "img")
    save_label = os.path.join(dir_save, "label")

    os.makedirs(save_img, exist_ok=True)
    os.makedirs(save_label, exist_ok=True)

    files_img = glob.glob(dir_origin + '/**/*.' + ext_img, recursive=True)

    for one_img_path in files_img:

        id_sample = Path(one_img_path).stem

        if id_sample in all_label_dict:

            path_to = os.path.join(save_img, id_sample + "." + ext_img)
            shutil.copyfile(one_img_path, path_to)

            path_to = os.path.join(save_label, id_sample + ".txt")

            with open(path_to, 'w', encoding="utf-8") as file:
                file.write(all_label_dict[id_sample])
        else:
            print("label not present in dict")


if __name__ == '__main__':
    # First unzip data in working dir

    # To do: define the path
    working_dir = "C:/dev/data/READ/2018_to_delete/"

    dir_format = os.path.join(working_dir, "format")
    os.makedirs(dir_format, exist_ok=True)

    # Format general data
    dir_general = os.path.join(dir_format, "general")
    os.makedirs(dir_general, exist_ok=True)

    dir_general_all = os.path.join(dir_general, "all")
    os.makedirs(dir_general_all, exist_ok=True)

    origin_dir_general = os.path.join(working_dir, "general_data")

    format_dir_data(origin_dir_general, dir_general_all, ext_img=".jpg")

    # To gather labels into one json file: src\data\text\gather_labels_one_file.py

    dir_label_general_all = os.path.join(dir_general_all, "label")
    dir_img_general_all = os.path.join(dir_general_all, "img")

    # Create charset general
    path_charset_read_general = os.path.join(dir_general, "charset_general.txt")
    create_charset(dir_label_general_all, path_charset_read_general)

    dir_general_split = os.path.join(dir_general, "split")
    split_train_val(dir_general_all, dir_general_split, ratio_train=0.8, ext_img="jpg")

    dir_specific = os.path.join(dir_format, "specific")
    os.makedirs(dir_specific, exist_ok=True)

    dir_specific_all = os.path.join(dir_specific, "all")
    os.makedirs(dir_specific_all, exist_ok=True)

    origin_specific_data = os.path.join(working_dir, "specific_data")
    origin_specific_data_split = os.path.join(working_dir, "specific_data_train_list")

    format_specific_data_all_list(origin_specific_data, origin_specific_data_split, dir_specific_all, ext_img="jpg")

    create_charset_list_dir(dir_specific_all)
    path_charset_specific = os.path.join(dir_specific, "charset_specific.txt")
    create_charset(origin_specific_data, path_charset_specific)

    dir_specific_split = os.path.join(dir_format, "specific/split")
    os.makedirs(dir_specific_split, exist_ok=True)

    split_list_dir(dir_specific_all, dir_specific_split, ratio_train=0.8, ext_img="jpg")

    dir_img_test_all = os.path.join(working_dir, "test_data")

    dir_save_all = os.path.join(dir_format, "test")
    os.makedirs(dir_save_all, exist_ok=True)

    labels_test = "../../../data/read2018/all_labels_30866_test.json"

    dict_id_label = {}
    with open(labels_test, "r") as fp:
        dict_id_label = json.load(fp)

    dir_data_test = os.path.join(working_dir, "test_data", "data", "30866")
    dir_save_test = os.path.join(dir_save_all, "30866")
    format_one_test_dir(dir_data_test, dir_save_test, dict_id_label, ext_img="jpg")

    path_charset_test_1 = os.path.join(dir_save_all, "charset_test_30866.txt")
    create_charset(os.path.join(dir_save_all, "30866", "label"), path_charset_test_1)

    # # Example to merge charsets
    # path_charset_test_2 = os.path.join(dir_save_all, "charset_test_30882.txt")
    # create_charset(os.path.join(dir_save_all, "30882", "label"), path_charset_test_2)
    #
    # path_charset_test_3 = os.path.join(dir_save_all, "charset_test_30893.txt")
    # create_charset(os.path.join(dir_save_all, "30893", "label"), path_charset_test_3)
    #
    # path_charset_test_4 = os.path.join(dir_save_all, "charset_test_35013.txt")
    # create_charset(os.path.join(dir_save_all, "35013", "label"), path_charset_test_4)
    #
    # path_charset_test_5 = os.path.join(dir_save_all, "charset_test_35015.txt")
    # create_charset(os.path.join(dir_save_all, "35015", "label"), path_charset_test_5)
    #
    # all_charsets = [path_charset_read_general, path_charset_specific, path_charset_test_1, path_charset_test_2,
    #                path_charset_test_3, path_charset_test_4, path_charset_test_5]
    #
    # path_charset_read = "C:/Users/simcor/dev/data/READ/2018/charset_read_2018.txt"
    # merge_charset(all_charsets, path_charset_read)


