import os
import torch


def load_pretrained_model(pretrained_model_file, model, device, print_load_ok=True):
    print("Loading pretrained model (from provided location: " + pretrained_model_file + ")...")
    if os.path.isfile(pretrained_model_file):
        checkpoint = torch.load(pretrained_model_file, map_location=device)

        pretrained_dict = checkpoint
        model_dict = model.state_dict()

        pretrained_keys = []
        skipped_keys = []
        scratch_keys = []
        for k in model_dict.keys():
            key = k

            if key in pretrained_dict:
                if model_dict[k].shape == pretrained_dict[key].shape:
                    pretrained_keys.append(k)
                else:
                    skipped_keys.append(k)
            else:
                scratch_keys.append(k)

        if print_load_ok:
            print('-' * 80)
            print("Loading following pretrained weights:")
        for k in pretrained_keys:
            key = k
            if print_load_ok:
                print(k)
            model_dict[k] = pretrained_dict[key]

        print('-' * 80)
        print("Training following weights from scratch:")
        for k in scratch_keys:
            print(k)

        print('-' * 80)
        print("Skipping following pretrained weights, because shapes mismatch:")
        for k in skipped_keys:
            key = k
            print(k)
            print(f"Model shape: '{model_dict[k].shape}'")
            print(f"Pretrained model shape: '{pretrained_dict[key].shape}'")

        model.load_state_dict(model_dict)

        print('-' * 80)
        print("Pretrained weights loaded.")

    else:
        print("Cannot load pretrained model from provided location: " + pretrained_model_file + " ...")

    model.to(device)
