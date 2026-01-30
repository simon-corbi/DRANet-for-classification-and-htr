from src.data.image.augmentations.mobius_transform import MobiusTransform_Improved


def get_config_aug_classif():
    config_augmentation = {
        "brightness": {
            "proba": 0.2,
            "min_factor": 0.5,
            "max_factor": 1.5,
        },
        "contrast": {
            "proba": 0.2,
            "min_factor": 0.01,
            "max_factor": 1.0,
        },
        "sign_flipping": {
            "proba": 0.1,
        },
        "gaussian_blur": {
            "proba": 0.2,
        },
        "erosion": {
            "proba": 0.09,
        },
        "dilatation": {
            "proba": 0.02,
        },
        "gaussian_noise": {
            "proba": 0.07,
        },
        "mobius": {
            "transform": MobiusTransform_Improved(p=0.01)
        }
    }

    return config_augmentation