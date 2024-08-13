from pathlib import Path
import supervision as sv
from torch.utils.data import Dataset
import albumentations as A
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time 


def load_detection_datasets(dataset_path: str, format: str = 'yolo'):
    print(dataset_path)
    if format == 'yolo':
        ds_train = sv.DetectionDataset.from_yolo(
            images_directory_path=Path(f"{dataset_path}/train/images"),
            annotations_directory_path=Path(f"{dataset_path}/train/labels"),
            data_yaml_path=Path(f"{dataset_path}/data.yaml")
        )
        ds_valid = sv.DetectionDataset.from_yolo(
            images_directory_path=Path(f"{dataset_path}/val/images"),
            annotations_directory_path=Path(f"{dataset_path}/val/labels"),
            data_yaml_path=Path(f"{dataset_path}/data.yaml")
        )
        ds_test = sv.DetectionDataset.from_yolo(
            images_directory_path=Path(f"{dataset_path}/test/images"),
            annotations_directory_path=Path(f"{dataset_path}/test/labels"),
            data_yaml_path=Path(f"{dataset_path}/data.yaml")
        )
    elif format == 'coco':
        ds_train = sv.DetectionDataset.from_coco(
            images_directory_path=Path(f"{dataset_path}/train"),
            annotations_path=Path(f"{dataset_path}/train/_annotations.coco.json")
        )
        ds_valid = sv.DetectionDataset.from_coco(
            images_directory_path=Path(f"{dataset_path}/valid"),
            annotations_path=Path(f"{dataset_path}/valid/_annotations.coco.json")
        )
        ds_test = sv.DetectionDataset.from_coco(
            images_directory_path=Path(f"{dataset_path}/test"),
            annotations_path=Path(f"{dataset_path}/test/_annotations.coco.json")
        )
    else:
        raise ValueError("Unsupported format. Please use 'yolo' or 'coco'.")

    return ds_train, ds_valid, ds_test

def collate_fn(batch):
    data = {}
    data["pixel_values"] = torch.stack([x["pixel_values"] for x in batch])
    data["labels"] = [x["labels"] for x in batch]
    return data

def count_annotations(dataset):
    num_annotations = 0
    for index, image_path in enumerate(dataset.images):
        #print(f"Image index: {index}")
        #print(f"Classes: {dataset.classes}")

        annotations = dataset.annotations[image_path]
        #print(f"Annotations: {annotations}")

    #     # Assuming 'bboxes' is an attribute of the annotations
    #     num_annotations += len(annotations['bboxes'])
    # return num_annotations

# Function to draw bounding boxes on the image
def draw_bounding_boxes(image, boxes, categories):
    for box, category in zip(boxes, categories):
        # Extract box coordinates
        x_min, y_min, x_max, y_max = map(int, box)
        
        # Draw the bounding box
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        # Optionally, add category label
        cv2.putText(image, str(category), (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return image

def save_plot_with_original_and_transformed(original_image, transformed_image, idx, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.5))
    axes[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image with Boxes')
    axes[0].axis('off')
    
    axes[1].imshow(cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Transformed Image with Boxes')
    axes[1].axis('off')
    
    plt.tight_layout()
    plot_save_path = save_path / f"original_and_transformed_{idx}.jpg"
    plt.savefig(plot_save_path)
    plt.close()
    #print(f"Plot saved at {plot_save_path}")

def find_empty_annotations(dataset):
    empty_indexes = []
    for index, (image_path, image, annotations) in enumerate(dataset):
        if annotations is None or len(annotations.xyxy) == 0:
            empty_indexes.append(index)
            #print(f"Empty annotations found for image at index {index} with path {image_path}")
    return empty_indexes




class PyTorchDetectionDataset(Dataset):
    def __init__(self, dataset, processor, transform: A.Compose = None, annotation_format="original"):
        self.dataset = dataset
        self.processor = processor
        self.transform = transform
        self.annotation_format = annotation_format

    @staticmethod
    def yolo_to_pascal_voc(bbox, img_width, img_height):
        """
        Convert bounding boxes from YOLO to Pascal VOC format.
        """
        x_center, y_center, width, height = bbox
        xmin = int((x_center - width / 2) * img_width)
        xmax = int((x_center + width / 2) * img_width)
        ymin = int((y_center - height / 2) * img_height)
        ymax = int((y_center + height / 2) * img_height)
        return [xmin, ymin, xmax, ymax]

    @staticmethod
    def annotations_as_coco(image_id, categories, boxes):
        annotations = []
        for category, bbox in zip(categories, boxes):
            x1, y1, x2, y2 = bbox
            formatted_annotation = {
                "image_id": image_id,
                "category_id": category,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "iscrowd": 0,
                "area": (x2 - x1) * (y2 - y1),
            }
            annotations.append(formatted_annotation)

        return {
            "image_id": image_id,
            "annotations": annotations,
        }

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        _, image, annotations = self.dataset[idx]
        # Convert BGR to RGB if necessary
        if image.shape[-1] == 3:  # Ensure the image has 3 channels
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # print(f'orig image shape is {image}')
        # orig_image_PIL = Image.fromarray(image)
        # orig_image_PIL.save(f"orig_image_PIL_{idx}.png")
        boxes = annotations.xyxy
        # print(f'boxes before is {boxes}')
        categories = annotations.class_id
        if self.transform:
            transformed = self.transform(
                image=image,
                bboxes=boxes,
                category=categories
            )
            transformed_image = transformed["image"]
            transformed_boxes = transformed["bboxes"]
            # print(f'transformed_boxes are {transformed_boxes}')
            #print(f'transformed boxes are {transformed_boxes}')
            transformed_categories = transformed["category"]
        formatted_annotations = self.annotations_as_coco(
            image_id=idx, categories=transformed_categories, boxes=transformed_boxes)
        # print(f'formatted_annotations is {formatted_annotations}')
        # ann = formatted_annotations
        #print(f'formatted annotations are {ann}')
        result = self.processor(
            images=transformed_image, annotations=formatted_annotations, return_tensors="pt")

        # Image processor expands batch dimension, let's squeeze it
        
        result = {k: v[0] for k, v in result.items()}
        # print(f'result is {result}')
        
        # # Get the image tensor
        # pixel_values = result['pixel_values']

        # # Convert the tensor to a numpy array
        # pixel_values = pixel_values.squeeze().detach().cpu().numpy()

        # # Normalize if necessary (e.g., if in range [-1, 1])
        # pixel_values = (pixel_values - pixel_values.min()) / (pixel_values.max() - pixel_values.min()) * 255

        # # Convert to uint8
        # pixel_values = pixel_values.astype(np.uint8)

        # # Convert numpy array to PIL Image
        # print(f'shape is {pixel_values.shape}')
        # processed_image_pil = Image.fromarray(pixel_values.transpose(1, 2, 0))  # Convert from CxHxW to HxWxC

        # # Save the image
        # processed_image_pil.save(f"square_processed_image_{idx}.png")
        # # time.sleep(35.0)
        return result