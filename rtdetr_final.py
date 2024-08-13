# Training involves the following steps:

# Load the model with AutoModelForObjectDetection using the same checkpoint as in the preprocessing.
# Define your training hyperparameters in TrainingArguments.
# Pass the training arguments to Trainer along with the model, dataset, image processor, and data collator.
# Call train() to finetune your model.
# When loading the model from the same checkpoint that you used for the preprocessing, remember to pass the label2id and id2label maps that you created earlier from the dataset's metadata. Additionally, we specify ignore_mismatched_sizes=True to replace the existing classification head with a new one.

from pathlib import Path
import torch
import requests
import cv2

import numpy as np

from PIL import Image
from dataclasses import dataclass, replace
from torch.utils.data import Dataset
from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
    TrainingArguments,
    Trainer
)

import supervision as sv


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
CHECKPOINT = "./rtdetr_r50vd" #add to drive and give path 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_path = './17cls_plastic/split/'
ds_train, ds_valid, ds_test = load_detection_datasets(dataset_path=dataset_path,format='yolo')
# Preparing function to compute mAP
id2label = {id: label for id, label in enumerate(ds_train.classes)}
label2id = {label: id for id, label in enumerate(ds_train.classes)}
print(id2label)
print(label2id)

processor = AutoImageProcessor.from_pretrained(
    CHECKPOINT,
    do_resize=True,
    size={"height": IMAGE_SIZE[1] ,"width": IMAGE_SIZE[0]},
    pad_size = {"height": 640, "width": 640}
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
# In the TrainingArguments use output_dir to specify where to save your model, then configure hyperparameters as you see fit. 
# Do not remove unused columns because this will drop the image column. Without the image column, you can't create pixel_values. For this reason, set remove_unused_columns to False.
# Set eval_do_concat_batches=False to get proper evaluation results. Images have different number of target boxes, if batches are concatenated we will not be able to determine which boxes belongs to particular image.

training_args = TrainingArguments(
    output_dir='./fine_tuned_models',
    num_train_epochs=300,
    max_grad_norm=0.1,
    learning_rate=5e-5,
    warmup_steps=300,
    per_device_train_batch_size=64,
    dataloader_num_workers=2,
    metric_for_best_model="eval_map",
    greater_is_better=True,
    load_best_model_at_end=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    remove_unused_columns=False,
    eval_do_concat_batches=False,
)

print("successfully loaded training arguments")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=pytorch_dataset_train,
    eval_dataset=pytorch_dataset_valid,
    tokenizer=processor,
    data_collator=collate_fn,
    compute_metrics=eval_compute_metrics_fn,
)

trainer.train()

# Save fine-tuned model on hard drive
model.save_pretrained("./saved_model_path")
processor.save_pretrained("./saved_model_path")
# Inference with fine-tuned RT-DETR model
