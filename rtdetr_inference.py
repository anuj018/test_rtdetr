from pathlib import Path
import torch
import cv2
import numpy as np
import supervision as sv
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset
from transformers import AutoImageProcessor, AutoModelForObjectDetection
import os
from utils.evaluation_utils import MAPEvaluator
from utils.dataset_utils import load_detection_datasets, PyTorchDetectionDataset
from image_and_transformation_config import (IMAGE_SIZE, 
                                            image_mean, image_std,
                                            train_augmentation_and_transform,
                                            valid_transform)
import time

# Folder to save images with bounding boxes
output_folder = "./predicted_images/"
os.makedirs(output_folder, exist_ok=True)

torch.cuda.empty_cache()
CHECKPOINT = "./saved_model_path"  # Add your model path
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)
dataset_path = './17cls_plastic/split/'
ds_train, ds_valid, ds_test = load_detection_datasets(dataset_path=dataset_path, format='yolo')

# Preparing function to compute mAP
id2label = {id: label for id, label in enumerate(ds_train.classes)}
label2id = {label: id for id, label in enumerate(ds_train.classes)}
print(id2label)
print(label2id)

processor = AutoImageProcessor.from_pretrained(
    CHECKPOINT,
    do_resize=True,
    size={"height": IMAGE_SIZE[1], "width": IMAGE_SIZE[0]},
    pad_size={"height": 640, "width": 640}
)

pytorch_dataset_train = PyTorchDetectionDataset(
    ds_train, processor, transform=train_augmentation_and_transform, annotation_format="yolo")
pytorch_dataset_valid = PyTorchDetectionDataset(
    ds_valid, processor, transform=valid_transform, annotation_format="yolo")
pytorch_dataset_test = PyTorchDetectionDataset(
    ds_test, processor, transform=valid_transform, annotation_format="yolo")

eval_compute_metrics_fn = MAPEvaluator(image_processor=processor, threshold=0.01, id2label=id2label, label2id=label2id)

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




# Iterate through the test dataset
# Iterate through the test dataset
for i in range(len(ds_test)):
    path, source_image, annotations = ds_test[i]

    image = Image.open(path)
    original_w, original_h = image.size  # Original image dimensions
    print(original_w, original_h)
    inputs = processor(image, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)

    # Processed image dimensions
    processed_h, processed_w = inputs['pixel_values'].shape[-2:]

    # Padding added during preprocessing
    pad_y = 280  # Since padding was only added at the bottom
    scale_y = 2  # Scaling factor

    results = processor.post_process_object_detection(
        outputs, target_sizes=[(original_h, original_w)], threshold=0.3)[0]

    # Adjust bounding boxes according to the scaling factors
    draw = ImageDraw.Draw(image)
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = box.tolist()
        # y1_pred = box[1]  # y1 predicted
        # y2_pred = box[3]  # y2 predicted

        # # Correct the y1 and y2 coordinates
        # y1_corrected = (y1_pred * scale_y) - pad_y
        # y2_corrected = (y2_pred * scale_y) - pad_y

        # # Clamp the values to ensure they are within the bounds of the original image
        # y1_corrected = max(0, min(y1_corrected, original_h))
        # y2_corrected = max(0, min(y2_corrected, original_h))

        # # Apply scaling to y coordinates only
        # # ymin = round((box[1] + pad_y) * scale_y, 2)  # ymin
        # # ymax = round((box[3] + pad_y) * scale_y, 2)  # ymax

        # # # Limit ymin and ymax to valid image bounds
        # # ymin = max(0, ymin)
        # # ymax = min(original_h, ymax)

        # # Update the bounding box
        # box[1] = y1_corrected
        # box[3] = y2_corrected



        print(
            f"Detected {model.config.id2label[label.item()]} with confidence "
            f"{round(score.item(), 3)} at location {box}"
        )

        # Draw the bounding box
        draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="red", width=3)
        
        # Draw the label and confidence score
        label_text = f"{model.config.id2label[label.item()]}: {round(score.item(), 2)}"
        draw.text((box[0], box[1]), label_text, fill="white")

    # Save the image with bounding boxes
    output_path = os.path.join(output_folder, f"predicted_{i}.png")
    image.save(output_path)

    targets.append(annotations)
    predictions.append(results)

# (Optional) Calculate mAP and confusion matrix if needed
# Calculate mAP
mean_average_precision = sv.MeanAveragePrecision.from_detections(
    predictions=predictions,
    targets=targets,
)

print(f"map50_95: {mean_average_precision.map50_95:.2f}")
print(f"map50: {mean_average_precision.map50:.2f}")
print(f"map75: {mean_average_precision.map75:.2f}")

# Plot and save the confusion matrix
confusion_matrix = sv.ConfusionMatrix.from_detections(
    predictions=predictions,
    targets=targets,
    classes=ds_test.classes
)

conf_mat = confusion_matrix.plot()
conf_mat.savefig("figure.png")

# Export the model to ONNX (Optional)
dummy_input = torch.randn(1, 3, 224, 224).to(DEVICE)

torch.onnx.export(
    model,                      # Model to export
    dummy_input,                # Example input data
    "rtdetr_r50vd.onnx",        # Output ONNX file path
    input_names=["input"],      # Input name
    output_names=["output"],    # Output name
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}, # Dynamic batch size
    opset_version=16            # ONNX opset version
)

print("Model successfully exported to ONNX format!")
