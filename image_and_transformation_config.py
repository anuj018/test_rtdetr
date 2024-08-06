import albumentations as A #image augmentation library
from albumentations.pytorch import ToTensorV2

image_mean = [0.485, 0.456, 0.406 ]
image_std = [0.229, 0.224, 0.225]
IMAGE_SIZE = (640,640) #WIDTH,HEIGHT

# Define the augmentation pipeline for training
train_augmentation_and_transform_yolo = A.Compose(
    [
        A.Perspective(p=0.1),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(p=0.1),
    ],
    bbox_params=A.BboxParams(
        format="yolo",
        label_fields=["category"],
    ),
)

# Define the augmentation pipeline for validation (no augmentation, just conversion to tensor)
valid_transform_yolo = A.Compose(
    [A.NoOp()],
    bbox_params=A.BboxParams(
        format="yolo",
        label_fields=["category"],
    ),
)

train_augmentation_and_transform = A.Compose(
    [
        A.Perspective(p=0.1),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(p=0.1),
    ],
    bbox_params=A.BboxParams(
        format="pascal_voc",
        label_fields=["category"],
        clip=True,
        min_area=25
    ),
)
valid_transform = A.Compose(
    [A.NoOp()],
    bbox_params=A.BboxParams(
        format="pascal_voc",
        label_fields=["category"],
        clip=True,
        min_area=1
    ),
)
