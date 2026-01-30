from src.data.global_values.text_global_values import CTC_PAD


def compute_center_loss_k1(dict_features_per_class, clusters, loss_fct):
    loss_center_all_class = 0

    nb_class = 0
    for index_class, features in dict_features_per_class.items():
        nb_frames_used_class = 0
        loss_one_class = 0
        for one_feature in features:
            index_class_loss = index_class

            loss_reg = loss_fct(one_feature, clusters[index_class_loss])

            loss_one_class += loss_reg
            nb_frames_used_class += 1

        # Norm per class, not all item because classes are unbalanced
        if nb_frames_used_class != 0:
            loss_one_class /= nb_frames_used_class
            nb_class += 1

        loss_center_all_class += loss_one_class

    if nb_class != 0:
        loss_center_all_class /= nb_class

    return loss_center_all_class


def groupe_features_per_class(features, gt_seq_frames, index_class_to_filter):
    dict_feature_per_class = {}

    for features_one_item, y_one_item in zip(features, gt_seq_frames):

        if y_one_item is None:
            continue
        # y_one_item: tensor
        for f, y in zip(features_one_item, y_one_item):
            if y.item() in index_class_to_filter:
                continue
            else:
                if y.item() == CTC_PAD:
                    print("Error Pad class is used")
                else:
                    if y.item() in dict_feature_per_class:
                        dict_feature_per_class[y.item()].append(f)
                    else:
                        dict_feature_per_class[y.item()] = [f]

    return dict_feature_per_class
