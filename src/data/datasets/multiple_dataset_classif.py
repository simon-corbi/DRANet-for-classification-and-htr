import pickle

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import adjust_contrast, adjust_brightness, gaussian_blur, rgb_to_grayscale

from src.data.image.augmentations.erosion_dilatation_batch import Erosion2d, Dilation2d
from src.data.image.transform.rescal_transform import rescale_fix_size_batch


class MultipleImageLabelDataset(Dataset):
    """
    """

    def __init__(self,
                 list_db: list,
                 fixed_size,
                 charset_dict_all,
                 ignore_index,
                 apply_augmentation,
                 config_augmentation,
                 pad_left=0,
                 pad_right=0):
        """
        """

        self.image_paths = []

        self.data = []
        self.labels_ind = []

        self.id_item = []

        self.apply_augmentation = apply_augmentation
        self.config_augmentation = config_augmentation

        self.fixed_size = fixed_size
        # self.transforms = transforms
        self.pad_left = pad_left
        self.pad_right = pad_right

        dim_input_channel = 1

        # # Initialize the Transforms for image NET -> to refactor clean
        # # First test RestNet 34
        # weights = ResNet34_Weights.DEFAULT
        # self.preprocess_image_net = weights.transforms()

        self.erode_t = Erosion2d(dim_input_channel, dim_input_channel, 3, soft_max=True)
        self.dilate_t = Dilation2d(dim_input_channel, dim_input_channel, 3, soft_max=True)

        for one_db_path in list_db:
            dbfile = open(one_db_path[1], 'rb')
            db = pickle.load(dbfile)

            data_one_db = db["data"]
            label_one_db = db["label"]
            class_to_idx = db["class_to_idx"]

            list_index_item = []
            list_index_target = []

            # # Get list index source
            for key, index_v in class_to_idx.items():
                index_item = label_one_db == index_v
                list_index_item.append(index_item)

                if key not in charset_dict_all:
                    # EMNIST upper and lower are the same, label in lower case
                    key_upper = key.upper()

                    if key_upper in charset_dict_all:
                        list_index_target.append(charset_dict_all[key_upper])
                    else:
                        list_index_target.append(ignore_index)
                else:
                    list_index_target.append( charset_dict_all[key])

            # Update target index
            for index_item, i_t in zip(list_index_item, list_index_target):
                label_one_db[index_item] = i_t

            print(one_db_path)
            print("Size origin: " + str(data_one_db.shape))
            print()

            shape_db = data_one_db.shape

            max_value = torch.max(data_one_db)
            # Some db are already normalize
            if max_value > 1:
                data_one_db = data_one_db / 255.0

            # N, Channel, Height, Width
            if len(shape_db) == 4:
                #if grayscale_activate == 1:
                if shape_db[1] == 3:
                    data_one_db = rgb_to_grayscale(data_one_db)
                # else:
                #     if shape_db[1] == 1:
                #         data_one_db = data_one_db.repeat(1, 3, 1, 1)

            # N, Height, Width
            elif len(shape_db) == 3:
                data_one_db = data_one_db.unsqueeze(1)
                #
                # if grayscale_activate == 0:
                #     data_one_db = data_one_db.repeat(1, 3, 1, 1)

            shape_db = data_one_db.shape
            # save_image(data_one_db, 'C:/Users/simcor/dev/logs/img_digit_img_net.png')

            # Pad height, width
            data_one_db = rescale_fix_size_batch(data_one_db, self.fixed_size[0], self.fixed_size[1], pad_value=0)

            self.data.append(data_one_db)
            self.labels_ind.append(label_one_db)

        self.data_origin = self.data
        self.labels_ind_origin = self.labels_ind

        self.data = torch.cat(self.data)
        self.labels_ind = torch.cat(self.labels_ind)

        self.ratio_data_origin = 1.0

    def add_unlabel_data(self, new_data, new_label):
        new_data = new_data.to(self.data.device)
        new_label = new_label.to(self.labels_ind.device)

        data_origin_filter = []
        label_origin_filter = []

        for one_db, one_label in zip(self.data_origin, self.labels_ind_origin):
            nb_item = one_db.shape[0]
            nb_item_filter = int(self.ratio_data_origin * nb_item)

            all_index = torch.randperm(nb_item)

            index_filter = all_index[:nb_item_filter]

            data_one_db = one_db[index_filter]
            label_one_db = one_label[index_filter]

            data_origin_filter.append(data_one_db)
            label_origin_filter.append(label_one_db)

        data_origin_filter = torch.cat(data_origin_filter)
        label_origin_filter = torch.cat(label_origin_filter)

        self.data = torch.cat((data_origin_filter, new_data))
        self.labels_ind = torch.cat((label_origin_filter, new_label))

        print("New nb item:" + str(self.data.shape[0]))

    def __len__(self):
        """
        """

        return self.data.shape[0]

    def __getitem__(self, idx):
        """
        """

        img = self.data[idx]  # tensor

        if self.apply_augmentation:
            # https://pytorch.org/vision/0.15/transforms.html
            # apply random combination of transformation
            aug = self.config_augmentation

            # Apply transform random composition
            # Erosion
            if "erosion" in aug.keys() and np.random.rand() < aug["erosion"]["proba"]:
                img_erode = img.unsqueeze(0)  # add batch dim
                img_erode = self.erode_t(img_erode)
                img_erode = img_erode.squeeze(0)  # remove batch dim
                img = img_erode.detach()  # remove grad link to erode transform
            # Dilatation
            if "dilatation" in aug.keys() and np.random.rand() < aug["dilatation"]["proba"]:
                img_dilate = img.unsqueeze(0)  # add batch dim
                img_dilate = self.dilate_t(img_dilate)
                img_dilate = img_dilate.squeeze(0)  # remove batch dim
                img = img_dilate.detach()  # remove grad link to erode transform
            # Contrast
            if "contrast" in aug.keys() and np.random.rand() < aug["contrast"]["proba"]:
                factor = np.random.uniform(aug["contrast"]["min_factor"], aug["contrast"]["max_factor"])
                img = adjust_contrast(img, factor)
            # Bright
            if "brightness" in aug.keys() and np.random.rand() < aug["brightness"]["proba"]:
                factor = np.random.uniform(aug["brightness"]["min_factor"], aug["brightness"]["max_factor"])
                img = adjust_brightness(img, factor)
            # Gaussian Blur
            if "gaussian_blur" in aug.keys() and np.random.rand() < aug["gaussian_blur"]["proba"]:
                img = gaussian_blur(img, kernel_size=(3, 7), sigma=(0.3, 4.))

            if "sign_flipping" in aug.keys() and np.random.rand() < aug["sign_flipping"]["proba"]:
                img = 1 + (-1 * img)
            if "gaussian_noise" in aug.keys() and np.random.rand() < aug["gaussian_noise"]["proba"]:
                img += torch.rand(img.size())
            if "mobius" in aug.keys():
                # To refactor
                img = img.cpu().detach().numpy()
                img = np.squeeze(img, axis=0)  # Remove channel gray
                img = aug["mobius"]["transform"](img)
                img = np.expand_dims(img, axis=0)  # add channel gray
                img = torch.as_tensor(img, dtype=torch.float32)

        return img, self.labels_ind[idx]
