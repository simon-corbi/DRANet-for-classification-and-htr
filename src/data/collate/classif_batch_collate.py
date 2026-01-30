import torch


class CollateImage(object):
    def collate_fn(self, batch_data):
        """
        """

        imgs = [batch_data[i]["img"] for i in range(len(batch_data))]
        imgs = torch.stack(imgs)

        ids = [batch_data[i]["ids"] for i in range(len(batch_data))]

        return imgs, ids


class CollateImageLabelClassification(object):
    def __init__(self, imgs_pad_value):
        self.imgs_pad_value = imgs_pad_value

    def collate_fn(self, batch_data):
        """
        """

        imgs = [batch_data[i][0] for i in range(len(batch_data))]
        imgs = torch.stack(imgs)

        labels = [batch_data[i][1] for i in range(len(batch_data))]
        labels = torch.stack(labels)

        return imgs, labels