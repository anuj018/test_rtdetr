import supervision as sv
from pathlib import Path
from PIL import Image
import torch
from dataclasses import dataclass
import numpy as np
from torchmetrics.detection.mean_ap import MeanAveragePrecision


def annotate(image, annotations, classes):
    labels = [
        classes[class_id]
        for class_id
        in annotations.class_id
    ]

    bounding_box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator(text_scale=1, text_thickness=2)

    annotated_image = image.copy()
    annotated_image = bounding_box_annotator.annotate(annotated_image, annotations)
    annotated_image = label_annotator.annotate(annotated_image, annotations, labels=labels)
    return annotated_image

def get_image_dimensions(image_path):
   """ Load an image and return its width and height. Parameters: image_path (str or Path): The path to the image file. Returns: tuple: A tuple containing the width and height of the image. """ # Ensure the image path is
   image_path = Path(image_path) # Open the image file
   with Image.open(image_path) as img:
    width, height = img.size
    return width, height
   
def display_dataset_sample(ds_train, grid_size=5, single_tile_size=(400, 400), plot_size=(10, 10)):
    """
    Displays a sample of the dataset in a grid format with annotations.

    Parameters:
    ds_train (Dataset): The dataset containing images and annotations.
    grid_size (int): The size of the grid (number of images along one dimension).
    single_tile_size (tuple): The size of each individual tile (width, height).
    plot_size (tuple): The size of the plot (width, height).
    """
    annotated_images = []
    for i in range(grid_size * grid_size):
        _, image, annotations = ds_train[i]
        annotated_image = annotate(image, annotations, ds_train.classes)
        annotated_images.append(annotated_image)

    grid = sv.create_tiles(
        annotated_images,
        grid_size=(grid_size, grid_size),
        single_tile_size=single_tile_size,
        tile_padding_color=sv.Color.WHITE,
        tile_margin_color=sv.Color.WHITE
    )
    sv.plot_image(grid, size=plot_size)

@dataclass
class ModelOutput:
    logits: torch.Tensor
    pred_boxes: torch.Tensor


class MAPEvaluator:

    def __init__(self, image_processor, threshold=0.00, id2label=None,label2id = None):
        self.image_processor = image_processor
        self.threshold = threshold
        self.id2label = id2label
        self.label2id = label2id

    def collect_image_sizes(self, targets):
        """Collect image sizes across the dataset as list of tensors with shape [batch_size, 2]."""
        image_sizes = []
        for batch in targets:
            batch_image_sizes = torch.tensor(np.array([x["size"] for x in batch]))
            image_sizes.append(batch_image_sizes)
        return image_sizes

    def collect_targets(self, targets, image_sizes):
        post_processed_targets = []
        for target_batch, image_size_batch in zip(targets, image_sizes):
            for target, (height, width) in zip(target_batch, image_size_batch):
                boxes = target["boxes"]
                boxes = sv.xcycwh_to_xyxy(boxes)
                boxes = boxes * np.array([width, height, width, height])
                boxes = torch.tensor(boxes)
                labels = torch.tensor(target["class_labels"])
                post_processed_targets.append({"boxes": boxes, "labels": labels})
        return post_processed_targets

    def collect_predictions(self, predictions, image_sizes):
        post_processed_predictions = []
        for batch, target_sizes in zip(predictions, image_sizes):
            batch_logits, batch_boxes = batch[1], batch[2]
            output = ModelOutput(logits=torch.tensor(batch_logits), pred_boxes=torch.tensor(batch_boxes))
            post_processed_output = self.image_processor.post_process_object_detection(
                output, threshold=self.threshold, target_sizes=target_sizes
            )
            post_processed_predictions.extend(post_processed_output)
        return post_processed_predictions

    @torch.no_grad()
    def __call__(self, evaluation_results):

        predictions, targets = evaluation_results.predictions, evaluation_results.label_ids

        image_sizes = self.collect_image_sizes(targets)
        post_processed_targets = self.collect_targets(targets, image_sizes)
        post_processed_predictions = self.collect_predictions(predictions, image_sizes)

        evaluator = MeanAveragePrecision(box_format="xyxy", class_metrics=True)
        evaluator.warn_on_many_detections = False
        evaluator.update(post_processed_predictions, post_processed_targets)

        metrics = evaluator.compute()

        # Replace list of per class metrics with separate metric for each class
        classes = metrics.pop("classes")
        map_per_class = metrics.pop("map_per_class")
        mar_100_per_class = metrics.pop("mar_100_per_class")
        for class_id, class_map, class_mar in zip(classes, map_per_class, mar_100_per_class):
            class_name = self.id2label[class_id.item()] if self.id2label is not None else class_id.item()
            metrics[f"map_{class_name}"] = class_map
            metrics[f"mar_100_{class_name}"] = class_mar

        metrics = {k: round(v.item(), 4) for k, v in metrics.items()}

        return metrics
