from pathlib import Path
import torch
import yaml

from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
    TrainingArguments,
    Trainer
)
from utils.evaluation_utils import (MAPEvaluator)
from utils.dataset_utils import (load_detection_datasets, PyTorchDetectionDataset, collate_fn)
from image_and_transformation_config import (train_augmentation_and_transform, valid_transform)

# Load configuration from the config.yaml file
config_file_path = "./runtime_config.yaml"
with open(config_file_path, 'r') as file:
    config = yaml.safe_load(file)

torch.cuda.empty_cache()

CHECKPOINT = config['model']['checkpoint']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_path = config['dataset']['path']
ds_train, ds_valid, ds_test = load_detection_datasets(dataset_path=dataset_path, format=config['dataset']['format'])

# Preparing function to compute mAP
id2label = {id: label for id, label in enumerate(ds_train.classes)}
label2id = {label: id for id, label in enumerate(ds_train.classes)}
print(id2label)
print(label2id)

config['model']['id2label'] = id2label
config['model']['label2id'] = label2id

processor = AutoImageProcessor.from_pretrained(
    CHECKPOINT,
    do_resize=config['image_processing']['do_resize'],
    size=config['image_processing']['size'],
)

# Now you can combine the image and annotation transformations to use on a batch of examples:
pytorch_dataset_train = PyTorchDetectionDataset(
    ds_train, processor, transform=train_augmentation_and_transform, annotation_format=config['dataset']['format'])
pytorch_dataset_valid = PyTorchDetectionDataset(
    ds_valid, processor, transform=valid_transform, annotation_format=config['dataset']['format'])
pytorch_dataset_test = PyTorchDetectionDataset(
    ds_test, processor, transform=valid_transform, annotation_format=config['dataset']['format'])

eval_compute_metrics_fn = MAPEvaluator(image_processor=processor, threshold=0.01, id2label=id2label, label2id=label2id)

model = AutoModelForObjectDetection.from_pretrained(
    CHECKPOINT,
    id2label=id2label,
    label2id=label2id,
    anchor_image_size=config['model']['anchor_image_size'],
    ignore_mismatched_sizes=config['model']['ignore_mismatched_sizes'],
)

training_args = TrainingArguments(
    output_dir=config['training_args']['output_dir'],
    num_train_epochs=config['training_args']['num_train_epochs'],
    max_grad_norm=config['training_args']['max_grad_norm'],
    learning_rate=config['training_args']['learning_rate'],
    warmup_steps=config['training_args']['warmup_steps'],
    per_device_train_batch_size=config['training_args']['per_device_train_batch_size'],
    dataloader_num_workers=config['training_args']['dataloader_num_workers'],
    metric_for_best_model=config['training_args']['metric_for_best_model'],
    greater_is_better=config['training_args']['greater_is_better'],
    load_best_model_at_end=config['training_args']['load_best_model_at_end'],
    eval_strategy=config['training_args']['eval_strategy'],
    save_strategy=config['training_args']['save_strategy'],
    save_total_limit=config['training_args']['save_total_limit'],
    remove_unused_columns=config['training_args']['remove_unused_columns'],
    eval_do_concat_batches=config['training_args']['eval_do_concat_batches'],
)

print("Successfully loaded training arguments")

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
model.save_pretrained(config['output']['saved_model_path'])
processor.save_pretrained(config['output']['saved_model_path'])
