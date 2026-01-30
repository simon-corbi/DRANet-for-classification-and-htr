import pickle

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms.functional import rgb_to_grayscale

from src.data.image.transform.rescal_transform import rescale_fix_size_batch


class OneImageLabelDataset(Dataset):
    """
    """

    def __init__(self,
                 db_path: list,
                 fixed_size,
                 charset_dict_all,
                 ignore_index,
                 transforms: list = None,):
        """
        """

        self.image_paths = []

        self.data = []
        self.labels_ind = []

        self.id_item = []

        self.fixed_size = fixed_size
        self.transforms = transforms

        dbfile = open(db_path, 'rb')
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

        shape_db = data_one_db.shape

        max_value = torch.max(data_one_db)
        # Some db are already normalize
        if max_value > 1:
            data_one_db = data_one_db / 255.0

        # N, Channel, Height, Width
        if len(shape_db) == 4:
            if shape_db[1] == 3:
                data_one_db = rgb_to_grayscale(data_one_db)
        # N, Height, Width
        elif len(shape_db) == 3:
            data_one_db = data_one_db.unsqueeze(1)

        label_one_db = label_one_db.to(torch.int64)

        data_one_db = rescale_fix_size_batch(data_one_db, self.fixed_size[0], self.fixed_size[1], pad_value=0)

        self.data = data_one_db
        self.labels_ind = label_one_db

    def __len__(self):
        """
        """

        return self.data.shape[0]

    def __getitem__(self, idx):
        """
        """
        img = self.data[idx]  # tensor

        return img, self.labels_ind[idx]
