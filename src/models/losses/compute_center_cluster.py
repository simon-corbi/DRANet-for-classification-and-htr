import editdistance
import torch

from src.data.text.best_path_ctc import ctc_best_path_one


# for CNN
def compute_cluster_center_k_1_cnn(data_loader,
                               model,
                               device,
                               nb_classes,
                               dim_features):
    model.eval()

    nb_ok = 0
    nb_ko = 0

    # Size of features from CRNN
    center_coordinates = torch.zeros([nb_classes, dim_features]).to(device)

    dict_feature_per_class = {}

    # Get prediction features
    with torch.no_grad():
        for i_batch, (images, labels) in enumerate(data_loader):
            images = images.to(device)
            labels = labels.to(device)

            batch_size = images.shape[0]

            # Error with batch size
            if batch_size == 1:
                continue

            # Forward pass
            outputs, features = model.forward_return_intermediate_values(images)
            features = torch.sigmoid(features)

            _, predicted = torch.max(outputs.data, 1)

            for i_item in range(batch_size):
                if predicted[i_item] != labels[i_item]:
                    nb_ko += 1
                else:
                    nb_ok += 1
                    # Group features by character
                    if predicted[i_item].item() in dict_feature_per_class:
                        dict_feature_per_class[predicted[i_item].item()].append(features[i_item])
                    else:
                        dict_feature_per_class[predicted[i_item].item()] = [features[i_item]]

    print("nb_ko: " + str(nb_ko))
    print("nb_ok: " + str(nb_ok))

    # Compute center coordinate
    for key in dict_feature_per_class:
        if len(dict_feature_per_class[key]) > 0:
            # mean_value = compute_means_features(dict_feature_per_class[key])
            features_tensor = torch.stack(dict_feature_per_class[key])

            mean_tensor = torch.mean(features_tensor, 0)
            mean_tensor = mean_tensor.detach()

            center_coordinates[key] = mean_tensor

    return center_coordinates


# For CRNN
def compute_center_coordinates_crnn(data_loader,
                               model,
                               device,
                               char_list,
                               token_blank,
                               index_class_to_filter):
    model.eval()

    prototypes_after = torch.zeros([len(char_list), 512]).to(device)

    dict_feature_per_class_after = {}

    # Get prediction features
    with torch.no_grad():
        for index_batch, batch_data in enumerate(data_loader):
            x = batch_data["imgs"].to(device)
            x_reduced_len = batch_data["w_reduce"]

            y_gt_txt = batch_data["label_str"]

            nb_item_batch = x.shape[0]

            y_pred, _, after_blstm = model(x)

            after_blstm = torch.permute(after_blstm, (1, 0, 2))
            after_blstm = torch.sigmoid(after_blstm)

            # Encoder
            encoder_outputs_main, encoder_outputs_shortcut = y_pred
            encoder_outputs_main = torch.nn.functional.log_softmax(encoder_outputs_main, dim=-1)

            # (Nb frames, Batch size, Nb characters) -> (Batch size, Nb frames, Nb characters)
            encoder_outputs_main = encoder_outputs_main.transpose(0, 1)

            top_main_enc = [torch.argmax(lp, dim=1).detach().cpu().numpy()[:x_reduced_len[j]] for j, lp in
                            enumerate(encoder_outputs_main)]
            predictions_text_main_enc = [ctc_best_path_one(p, char_list, token_blank) for p in top_main_enc]

            cers_enc = [editdistance.eval(u, v) for u, v in zip(y_gt_txt, predictions_text_main_enc)]

            for i in range(nb_item_batch):
                if cers_enc[i] == 0:

                    # Group features by character
                    for f, y in zip(after_blstm[i], top_main_enc[i]):
                        if y in index_class_to_filter:
                            continue
                        if y in dict_feature_per_class_after:
                            dict_feature_per_class_after[y].append(f)
                        else:
                            dict_feature_per_class_after[y] = [f]

    # Compute means
    for key in dict_feature_per_class_after:
        if len(dict_feature_per_class_after[key]) > 0:
            # N, nb features
            features_tensor = torch.stack(dict_feature_per_class_after[key])

            mean_value = torch.mean(features_tensor, 0)
            mean_value = mean_value.detach()

            prototypes_after[key] = mean_value

    return prototypes_after
