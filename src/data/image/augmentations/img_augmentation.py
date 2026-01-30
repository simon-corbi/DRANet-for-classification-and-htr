import albumentations as A


# https://github.com/georgeretsi/HTR-best-practices/blob/main/utils/transforms.py
def get_img_data_augmentation(proba=0.5):
    # albumentations transforms for text augmentation
    aug_transforms = A.Compose([

        # geometric augmentation
        A.Affine(rotate=(-1, 1), shear={'x': (-30, 30), 'y': (-5, 5)}, scale=(0.6, 1.2), translate_percent=0.02, mode=1,
                 p=proba),

        # perspective transform
        # A.Perspective(scale=(0.05, 0.1), p=0.5),

        # distortions
        A.OneOf([
            A.GridDistortion(distort_limit=(-.1, .1), p=proba),
            # A.ElasticTransform(alpha=60, sigma=20, alpha_affine=0.5, p=0.5),  # Repo old version albumentations # defaut alpha_affine: float = 50,
            A.ElasticTransform(alpha=60, sigma=20, p=proba),  # alpha_affine param was remove
        ], p=proba),

        # erosion & dilation
        A.OneOf([
            A.Morphological(p=proba, scale=3, operation='dilation'),
            A.Morphological(p=proba, scale=3, operation='erosion'),
        ], p=proba),

        # color invertion - negative
        # A.InvertImg(p=0.5),

        # color augmentation - only grayscale images
        A.RandomBrightnessContrast(p=proba, brightness_limit=0.2, contrast_limit=0.2),

        # color contrast
        A.RandomGamma(p=proba, gamma_limit=(80, 120)),
    ])

    return aug_transforms
