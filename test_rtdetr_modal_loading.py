# -*- coding: utf-8 -*-
"""rtdetr_final.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1aXhCKokXOq-Jq6S7m7dn0yme3j43QDHv
"""
from pathlib import Path
import torch
import requests

import numpy as np
import supervision as sv #for loading the dataset
import albumentations as A #image augmentation library

from PIL import Image
from dataclasses import dataclass, replace
from torch.utils.data import Dataset
from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
    TrainingArguments,
    Trainer
)
# from torchmetrics.detection.mean_ap import MeanAveragePrecision

# @title Load model

CHECKPOINT = "./rtdetr_r50vd" #add to drive and give path
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForObjectDetection.from_pretrained(CHECKPOINT).to(DEVICE)
processor = AutoImageProcessor.from_pretrained(CHECKPOINT)

# @title Run inference

IMAGE_PATH = Path('./images/test_images/bus.jpeg')
RESULT_IMAGE_PATH = Path('./images/result.jpeg')

image = Image.open(IMAGE_PATH)
inputs = processor(image, return_tensors="pt").to(DEVICE)

with torch.no_grad():
    outputs = model(**inputs)

w, h = image.size
results = processor.post_process_object_detection(
    outputs, target_sizes=[(h, w)], threshold=0.3)

# @title Display result with NMS

detections = sv.Detections.from_transformers(results[0])
labels = [
    model.config.id2label[class_id]
    for class_id
    in detections.class_id
]

annotated_image = image.copy()
annotated_image = sv.BoundingBoxAnnotator().annotate(annotated_image, detections)
annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels=labels)
annotated_image.thumbnail((600, 600))
annotated_image

# @title Display result with NMS

detections = sv.Detections.from_transformers(results[0]).with_nms(threshold=0.1)
labels = [
    model.config.id2label[class_id]
    for class_id
    in detections.class_id
]

annotated_image = image.copy()
annotated_image = sv.BoundingBoxAnnotator().annotate(annotated_image, detections)
annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels=labels)
annotated_image.thumbnail((600, 600))
annotated_image.save(RESULT_IMAGE_PATH)
