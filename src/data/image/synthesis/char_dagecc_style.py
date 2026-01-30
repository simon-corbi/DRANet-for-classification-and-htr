import os.path
import pickle
import random

import numpy as np
import torch
from PIL import Image, ImageFont, ImageDraw
from torchvision.transforms.functional import gaussian_blur, pil_to_tensor, adjust_brightness
from torchvision.utils import save_image

from src.data.image.transform.rescal_transform import rescale_bigger_pad_smaller_image
from src.data.text.charset_token import CharsetToken


def generate_safran_char(label, font, height, width, path):
    """
    Generate character image in Safran-MNIST-DLS style

    Save image for debug and visualization purpose, return image

    :param label: character to print
    :param path: where to save image
    """
    min_grayscale = 50
    max_grayscale = 180

    # Get char label size
    # Image dimension width, height
    img_dumb = Image.new("L", (width, height))  # RGB
    draw = ImageDraw.Draw(img_dumb)

    font_size = random.randint(int(height * 0.8), 2 * height)

    font = ImageFont.truetype(font, font_size)

    _, _, w, h = draw.textbbox((0, 0), label, font=font)

    offset = 0

    if label == ".":
        offset = 4

    real_width = w + 4 + offset
    real_height = h + 6  # 4

    # Generate char label
    img = Image.new("L", (real_width, real_height))
    draw = ImageDraw.Draw(img)

    use_shadow_img = True

    if label == ".":
        _, _, w, h = draw.textbbox((0, 0), label, font=font)

        ink_color_bright = random.randint(200, 255)
        # coordinate (x, y)
        draw.text((3 + offset, 0), label, font=font, fill=ink_color_bright)

        img_shadow = Image.new("L", (real_width, real_height))
        draw_shadow = ImageDraw.Draw(img_shadow)

        ink_color_dark = random.randint(min_grayscale + 12, max_grayscale - 24)

        # draw_shadow.text((4 + offset, -3), label, font=font, fill=ink_color_dark)
        draw_shadow.text((4 + offset, 1), label, font=font, fill=ink_color_dark)
    else:
        _, _, w, h = draw.textbbox((0, 0), label, font=font)

        ink_color_bright = random.randint(200, 255)
        # coordinate (x, y)
        draw.text((3 + offset, 0), label, font=font, fill=ink_color_bright)

        img_shadow = Image.new("L", (real_width, real_height))
        draw_shadow = ImageDraw.Draw(img_shadow)

        ink_color_dark = random.randint(min_grayscale + 12, max_grayscale - 24)
        offset_shadow = random.randint(-2, 2)
        draw_shadow.text((4 + offset_shadow + offset, offset_shadow), label, font=font, fill=ink_color_dark)

    # Apply shear/perspective transform
    if np.random.rand() < 0.4:
        m = random.uniform(0.0, 0.3)
        xshift = abs(m) * real_width
        new_width = real_width + int(round(xshift))

        img = img.transform((new_width, real_height), Image.AFFINE, (1, m, -xshift if m > 0 else 0, 0, 1, 0),
                            Image.BICUBIC)

        if use_shadow_img:
            img_shadow = img_shadow.transform((new_width, real_height), Image.AFFINE,
                                              (1, m, -xshift if m > 0 else 0, 0, 1, 0), Image.BICUBIC)

    char_tensor = pil_to_tensor(img)
    char_tensor = char_tensor.float()
    char_tensor /= 255.0

    if use_shadow_img:
        shadow_tensor = pil_to_tensor(img_shadow)
        shadow_tensor = shadow_tensor.float()
        shadow_tensor /= 255.0

    # Chnage with transform
    char_width, char_height = img.size

    # Geneate background
    # Tensor dimension, channel, height, width  -> different than PIL
    img_background = torch.zeros((1, char_height, char_width))
    img_background += min_grayscale + (torch.rand(img_background.size()) * (max_grayscale - min_grayscale))

    img_background /= 255  # space 0 ; 1

    odd_number = [1 + i * 2 for i in range(1, 4)]
    kx = odd_number[random.randint(0, 2)]
    ky = odd_number[random.randint(0, 2)]
    img_background = gaussian_blur(img_background, kernel_size=[kx, ky],
                                   sigma=(0.1, 5.))  # V5 kernel_size=(3, 7), sigma=(0.3, 4.))

    if np.random.rand() < 0.4:
        kx = odd_number[random.randint(0, 2)]
        ky = odd_number[random.randint(0, 2)]
        img_background = gaussian_blur(img_background, kernel_size=(kx, ky), sigma=(0.3, 4.))

    if np.random.rand() < 0.2:
        kx = odd_number[random.randint(0, 2)]
        ky = odd_number[random.randint(0, 2)]
        img_background = gaussian_blur(img_background, kernel_size=(kx, ky), sigma=(0.3, 4.))

    merge = img_background
    merge = merge.squeeze(0)  # Remove dimension channel

    if use_shadow_img:
        # Set shadow value
        if np.random.rand() < 0.5:
            shadow_tensor = shadow_tensor.squeeze(0)
            index_shadow = shadow_tensor != 0
            merge[index_shadow] = shadow_tensor[index_shadow]

    # Set light value
    char_tensor = char_tensor.squeeze(0)
    index_light = char_tensor > 0.2  # filter dark artifact  v5 0,33
    merge[index_light] = char_tensor[index_light]

    if np.random.rand() > 0.5:
        # Bright
        factor = np.random.uniform(0.5, 2)
        merge = adjust_brightness(merge, factor)

    # Set in grayscale space [0 ; 255}
    merge *= 255

    merge = merge.cpu().detach().numpy()

    # To refactor -> this function / 255
    merge = rescale_bigger_pad_smaller_image(merge, height, width, 0, 0, pad_bottom=0, inverse_color=False)

    merge = torch.from_numpy(merge)
    # value between 0 - 1
    save_image(merge, path)

    # value between 0 - 255
    merge *= 255

    return merge


def generate_n_sample(dir_save, char_label, fonts, nb_generate_per_class, height, width, path_save_db):
    # Make dir for extracted characters
    for one_c in char_label:
        dir_c = os.path.join(dir_save, one_c)

        if one_c == ".":
            dir_c = os.path.join(dir_save, "char_dot")
        if one_c == "/":
            dir_c = os.path.join(dir_save, "char_slash")

        os.makedirs(dir_c, exist_ok=True)

    id_item = 0

    all_img = []
    all_label = []

    # Generate and save for each class
    for one_font in fonts:
        for one_class in char_label:
            nb_to_generate = nb_generate_per_class

            # # Create more data for non digit class
            for i in range(nb_to_generate):
                # Class not present in this font
                if one_font == font_leddotmatrix and one_class == "/":
                    continue
                else:
                    name_img = "id_" + str(id_item) + ".png"
                    path_img = os.path.join(dir_save, one_class, name_img)

                    if one_class == ".":
                        path_img = os.path.join(dir_save, "char_dot", name_img)
                    if one_class == "/":
                        path_img = os.path.join(dir_save, "char_slash", name_img)

                    img = generate_safran_char(one_class, one_font, height, width, path_img)

                    all_img.append(img)
                    all_label.append(char_dict[one_class])

                    id_item += 1

    all_img = torch.stack(all_img)

    all_label = np.array(all_label)
    all_label = torch.from_numpy(all_label)
    all_label = all_label.to(torch.int64)

    db = {
        "data": all_img,
        "label": all_label,
        "class_to_idx": char_dict
    }

    dbfile = open(path_save_db, 'ab')

    # source, destination
    pickle.dump(db, dbfile)
    dbfile.close()


if __name__ == '__main__':
    dir_font = "../../../../data/fonts/"

    font_leddotmatrix = os.path.join(dir_font, "LED Dot-Matrix.ttf")  # No slash
    font_enhanceddotdigit = os.path.join(dir_font, "enhanced_dot_digital-7.ttf")
    font_bpdots = os.path.join(dir_font, "bpdots.regular.otf")

    fonts = [font_leddotmatrix, font_enhanceddotdigit, font_bpdots]

    charset_file = "../../../../data/charset_digits_competition_task2.txt"
    charset = CharsetToken(charset_file)
    char_list = charset.get_charset_list()
    char_dict = charset.get_charset_dictionary()

    dir_save = "../../../../data/dagecc/"

    height = 48
    width = 48

    # # # Debug
    dir_save_debug = os.path.join(dir_save, "debug")
    # Train
    dir_train = os.path.join(dir_save_debug, "train_img")
    os.makedirs(dir_train, exist_ok=True)

    path_save_db_train = os.path.join(dir_save_debug, "train")
    generate_n_sample(dir_train, char_list, fonts, 2, height, width, path_save_db_train)
    # Val
    dir_val = os.path.join(dir_save_debug, "val_img")
    os.makedirs(dir_val, exist_ok=True)

    path_save_db_val = os.path.join(dir_save_debug, "val")
    generate_n_sample(dir_val, char_list, fonts, 2, height, width, path_save_db_val)
    # Train
    dir_test = os.path.join(dir_save_debug, "test_img")
    os.makedirs(dir_test, exist_ok=True)

    path_save_db_test = os.path.join(dir_save_debug, "test")
    generate_n_sample(dir_test, char_list, fonts, 2, height, width, path_save_db_test)

    # # # # # # # All
    # dir_save = "C:/Users/simcor/dev/data/Safran-MNIST_DLS/synthetic_v8_brightness_n2000/"
    # n_train = 2000
    # n_val = 500
    # n_test = 500
    # dir_save_all = os.path.join(dir_save, "all")
    # # # # Train
    # dir_train = os.path.join(dir_save_all, "train_img")
    # os.makedirs(dir_train, exist_ok=True)
    #
    # path_save_db_train = os.path.join(dir_save_all, "train")
    #
    # generate_n_sample(dir_train, char_list, fonts, n_train, height, width, path_save_db_train)
    # # # Validation
    # dir_val = os.path.join(dir_save_all, "val_img")
    # os.makedirs(dir_val, exist_ok=True)
    #
    # path_save_db_val = os.path.join(dir_save_all, "val")
    #
    # generate_n_sample(dir_val, char_list, fonts, n_val, height, width, path_save_db_val)
    # # Test
    # dir_test = os.path.join(dir_save_all, "test_img")
    # os.makedirs(dir_test, exist_ok=True)
    #
    # path_save_db_test = os.path.join(dir_save_all, "test")
    #
    # generate_n_sample(dir_test, char_list, fonts, n_test, height, width, path_save_db_test)

    print("End")
