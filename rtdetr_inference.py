from pathlib import Path
import torch
import requests
import cv2

import numpy as np

import supervision as sv

from PIL import Image
from dataclasses import dataclass, replace
from torch.utils.data import Dataset
from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
    TrainingArguments,
    Trainer
)

from utils.evaluation_utils import (annotate,
                                    display_dataset_sample,
                                    MAPEvaluator)

from utils.dataset_utils import (load_detection_datasets,
                                PyTorchDetectionDataset,
                                collate_fn)

from image_and_transformation_config import (IMAGE_SIZE, 
                                            image_mean,image_std,
                                            train_augmentation_and_transform,
                                            valid_transform)

torch.cuda.empty_cache()
CHECKPOINT = "./saved_model_path" #add to drive and give path 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)
dataset_path = './dataset'
ds_train, ds_valid, ds_test = load_detection_datasets(dataset_path=dataset_path,format='yolo')
# Preparing function to compute mAP
id2label = {id: label for id, label in enumerate(ds_train.classes)}
label2id = {label: id for id, label in enumerate(ds_train.classes)}
print(id2label)
print(label2id)

processor = AutoImageProcessor.from_pretrained(
    CHECKPOINT,
    do_resize=True,
    size={"width": IMAGE_SIZE[0] ,"height": IMAGE_SIZE[1]},
)

# Now you can combine the image and annotation transformations to use on a batch of examples:
pytorch_dataset_train = PyTorchDetectionDataset(
    ds_train, processor, transform=train_augmentation_and_transform,annotation_format = "yolo")
pytorch_dataset_valid = PyTorchDetectionDataset(
    ds_valid, processor, transform=valid_transform,annotation_format = "yolo")
pytorch_dataset_test = PyTorchDetectionDataset(
    ds_test, processor, transform=valid_transform,annotation_format = "yolo")

eval_compute_metrics_fn = MAPEvaluator(image_processor=processor, threshold=0.01, id2label=id2label,label2id=label2id)

model = AutoModelForObjectDetection.from_pretrained(
    CHECKPOINT,
    id2label=id2label,
    label2id=label2id,
    anchor_image_size=None,
    ignore_mismatched_sizes=True,
)

model = model.to(DEVICE)

targets = []
predictions = []

for i in range(len(ds_test)):
    path, sourece_image, annotations = ds_test[i]

    image = Image.open(path)
    inputs = processor(image, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)

    w, h = image.size
    results = processor.post_process_object_detection(
        outputs, target_sizes=[(h, w)], threshold=0.3)

    detections = sv.Detections.from_transformers(results[0])

    targets.append(annotations)
    predictions.append(detections)


# @title Calculate mAP
mean_average_precision = sv.MeanAveragePrecision.from_detections(
    predictions=predictions,
    targets=targets,
)

print(f"map50_95: {mean_average_precision.map50_95:.2f}")
print(f"map50: {mean_average_precision.map50:.2f}")
print(f"map75: {mean_average_precision.map75:.2f}")

confusion_matrix = sv.ConfusionMatrix.from_detections(
    predictions=predictions,
    targets=targets,
    classes=ds_test.classes
)

conf_mat = confusion_matrix.plot()
conf_mat.savefig("figure.png")
# print(type(conf_mat))
# cv2.imwrite('confmat.png', conf_mat)


