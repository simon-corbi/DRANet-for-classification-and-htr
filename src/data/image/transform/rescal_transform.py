from skimage.transform import resize
import numpy as np
from torchvision.transforms import Resize
import torch.nn.functional as F


def rescale_bigger_pad_smaller_image(img, height_min, width_min, pad_left, pad_right, pad_bottom, inverse_color=True, pad_value=0):
    temp_img = img
    temp_img = temp_img.astype(np.float32)
    temp_img /= 255.0       # Normalize image

    if inverse_color:
        temp_img = 1 - temp_img

    h, w = temp_img.shape

    # Down scale large image
    if h > height_min or w > width_min:
        scale_y = float(height_min) / float(h)
        scale_x = float(width_min) / float(w)

        scale = min(scale_x, scale_y)

        width_new = int(scale * w)
        height_new = int(scale * h)

        temp_img = resize(image=temp_img, output_shape=(height_new, width_new)).astype(np.float32)

    total_pad_right = pad_right
    total_pad_bottom = pad_bottom

    h, w = temp_img.shape

    if h < height_min:
        total_pad_bottom += height_min - h

    if w < width_min:
        total_pad_right += width_min - w

    temp_img = np.pad(temp_img, ((0, total_pad_bottom), (pad_left, total_pad_right)), 'constant', constant_values=pad_value)

    return temp_img


def rescale_fix_size_batch(imgs, height_target, width_target, pad_value=0):

    temp_img = imgs

    b, c, h, w = temp_img.shape

    scale_y = height_target / float(h)
    scale_x = width_target / float(w)

    # Upscale small image
    if h < height_target or w < width_target:
        # fix: before -> max
        scale = min(scale_x, scale_y)

        width_new = int(scale * w)
        height_new = int(scale * h)

        resize = Resize((height_new, width_new))

        temp_img = resize(temp_img)

    b, c, h, w = temp_img.shape

    scale_y = height_target / float(h)
    scale_x = width_target / float(w)

    # Down scale large image
    if h > height_target or w > width_target:
        scale = min(scale_x, scale_y)

        width_new = int(scale * w)
        height_new = int(scale * h)

        resize = Resize((height_new, width_new))

        temp_img = resize(temp_img)

    b, c, h, w = temp_img.shape

    pad_height = 0
    pad_width = 0

    if h < height_target:
        pad_height += height_target - h

    if w < width_target:
        pad_width += width_target - w

    temp_img = F.pad(input=temp_img, pad=(0, pad_width, 0, pad_height, 0, 0, 0, 0), mode='constant', value=pad_value)

    return temp_img


def rescale_fix_size_img(img, height_target, width_target, pad_value=0):

    # ndarray
    temp_img = img

    h, w = temp_img.shape

    scale_y = height_target / float(h)
    scale_x = width_target / float(w)

    # Upscale small image
    if h < height_target or w < width_target:
        scale = min(scale_x, scale_y)

        width_new = int(scale * w)
        height_new = int(scale * h)

        # normalize value / 255
        temp_img = resize(image=temp_img, output_shape=(height_new, width_new)).astype(np.float32)

    h, w = temp_img.shape

    scale_y = height_target / float(h)
    scale_x = width_target / float(w)

    # Down scale large image
    if h > height_target or w > width_target:
        scale = min(scale_x, scale_y)

        width_new = int(scale * w)
        height_new = int(scale * h)

        # normalize value / 255
        temp_img = resize(image=temp_img, output_shape=(height_new, width_new)).astype(np.float32)

    h, w = temp_img.shape

    pad_height = 0
    pad_width = 0

    if h < height_target:
        pad_height += height_target - h

    if w < width_target:
        pad_width += width_target - w

    temp_img = np.pad(temp_img, ((0, pad_height), (0, pad_width)), 'constant', constant_values=pad_value)

    return temp_img
