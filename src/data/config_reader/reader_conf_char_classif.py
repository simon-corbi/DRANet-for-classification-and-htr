import json


def read_json_config_char_classif(path_file):
    train_info = []
    val_info = []
    test_info = []
    charset_path = ""

    with open(path_file, "r") as fp:
        config_values = json.load(fp)

        print(config_values)

        if "train_data" in config_values:
            for key, value in config_values["train_data"].items():
                train_info.append([key, value])
        if "val_data" in config_values:
            for key, value in config_values["val_data"].items():
                val_info.append([key, value])
        if "test_data" in config_values:
            for key, value in config_values["test_data"].items():
                test_info.append([key, value])

        if "charset_file" in config_values:
            charset_path = config_values["charset_file"]

    return train_info, val_info, test_info, charset_path


def read_json_config_char_classif_da(path_file):
    source_infos_train = []
    source_infos_test = []
    target_info_train = []
    target_info_test = []
    charset_path = ""

    with open(path_file, "r") as fp:
        config_values = json.load(fp)

        print(config_values)

        for key, value in config_values["dbs"]["source"].items():
            source_infos_train.append([key, value["train"]])
            source_infos_test.append([key, value["test"]])

        target_info_train.append(["target_train", config_values["dbs"]["target"]["train"]])
        target_info_test.append(["target_test", config_values["dbs"]["target"]["test"]])

        if "charset_file" in config_values:
            charset_path = config_values["charset_file"]

    return source_infos_train, source_infos_test, target_info_train, target_info_test, charset_path