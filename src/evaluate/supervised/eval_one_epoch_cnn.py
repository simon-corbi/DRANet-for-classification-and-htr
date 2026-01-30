import time

import torch

from src.data.text.index_to_text import convert_int_to_chars


def evaluate_cnn_one_epoch(model, data_loader, device, loss_classif, f1_score, print_first_batch=False, char_list=None):
    model.eval()

    all_gt = []
    all_pred = []

    with torch.no_grad():
        loss_val_total = 0

        begin_time = time.time()
        nb_items = 0
        nb_batch = 0
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            nb_items += images.shape[0]
            nb_batch += 1
            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)

            loss = loss_classif(outputs, labels)

            if not torch.isnan(loss):
                loss_val_total += loss.item()

            predicted = predicted.cpu()
            labels = labels.cpu()

            all_gt.append(labels)
            all_pred.append(predicted)

            if nb_batch < 3 and print_first_batch:
                if char_list is not None:
                    print("GT:")
                    print(convert_int_to_chars(labels, char_list))
                    print("Predicton:")
                    print(convert_int_to_chars(predicted, char_list))

        end_time = time.time()
        total_time = end_time - begin_time
        time_one = total_time / nb_items
        print("Time one (s): " + str(time_one))
        print("nb_items: " + str(nb_items))
        all_gt = torch.cat(all_gt)
        all_pred = torch.cat(all_pred)

        f1_macro_total = f1_score(all_pred, all_gt, average='macro')
        f1_per_class = f1_score(all_pred, all_gt, average=None)

        loss_val_total /= nb_items

        return f1_macro_total, f1_per_class, loss_val_total
