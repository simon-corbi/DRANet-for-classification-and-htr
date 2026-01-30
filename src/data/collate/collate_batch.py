import torch
import numpy as np


def pad_sequences_1D(data, padding_value):
    """
    Pad data with padding_value to get same length
    """
    x_lengths = [len(x) for x in data]
    longest_x = max(x_lengths)
    padded_data = np.ones((len(data), longest_x)).astype(np.int32) * padding_value
    for i, x_len in enumerate(x_lengths):
        padded_data[i, :x_len] = data[i][:x_len]
    return padded_data


def pad_images_2D(data, padding_value):
    """
    data: list of numpy array
    """
    x_lengths = [x.shape[2] for x in data]
    y_lengths = [x.shape[1] for x in data]
    longest_x = max(x_lengths)
    longest_y = max(y_lengths)

    nb_channel = data[0].shape[0]

    padded_data = np.ones((len(data), nb_channel, longest_y, longest_x)) * padding_value
    for i, xy_len in enumerate(zip(x_lengths, y_lengths)):
        x_len, y_len = xy_len
        padded_data[i, :, :y_len, :x_len] = data[i][:, :y_len, :x_len]
    return padded_data


class CollateImageLabelHTR(object):
    """"""

    def __init__(self, imgs_pad_value, pad_txt):
        self.imgs_pad_value = imgs_pad_value

        self.pad_txt = pad_txt

    def collate_fn(self, batch_data):
        """
        """

        ids = [batch_data[i]["ids"] for i in range(len(batch_data))]

        label_str = [batch_data[i]["label_str"] for i in range(len(batch_data))]

        label_ind = [batch_data[i]["label_ind"] for i in range(len(batch_data))]
        label_ind_length = [len(l) for l in label_ind]
        label_ind = pad_sequences_1D(label_ind, padding_value=self.pad_txt)
        label_ind = torch.tensor(label_ind).long()

        imgs = [batch_data[i]["img"] for i in range(len(batch_data))]
        w_reduce = [batch_data[i]["w_reduce"] for i in range(len(batch_data))]

        imgs = pad_images_2D(imgs, padding_value=self.imgs_pad_value)

        imgs = torch.tensor(imgs).float()

        # Collate all
        formatted_batch_data = {
            "ids": ids,

            "imgs": imgs,
            "w_reduce": w_reduce,

            "label_str": label_str,

            "label_ind": label_ind,
            "label_ind_length": label_ind_length,
        }

        return formatted_batch_data
