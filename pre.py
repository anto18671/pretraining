from pytorch_lightning import LightningModule, Trainer
from timm.models.resnet import ResNet, BasicBlock
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import Accuracy
from albumentations.pytorch import ToTensorV2
from datasets import load_dataset
from torch.nn import Module
import numpy as np
import mlflow
import torch
import math
import os

# Albumentations
from albumentations import (
    Compose, HorizontalFlip, Normalize, ColorJitter,
    PadIfNeeded, VerticalFlip, ToGray, Rotate,
)

# Define the line model
class CustomModel(Module):
    def __init__(self, num_classes, dropout=0.0):
        super(CustomModel, self).__init__()
        # Define the backbone model without the classifier
        self.backbone = ResNet(
            block=BasicBlock,
            layers=[2, 2, 2, 2],
            drop_rate=dropout,
            num_classes=num_classes,
            channels=(32, 64, 128, 256),
        )

    # Forward pass
    def forward(self, x):
        # Return final output
        return self.backbone(x)
    
# This class defines a warmup + exponential scheduler
class WarmupExponentialLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, decay_factor, warmup_steps=0, last_epoch=-1):
        # Set warmup steps
        self.warmup_steps = warmup_steps
        
        # Compute decay factor
        self.decay_factor = decay_factor
        
        # Initialize step counter
        self.current_step = 0
        
        # Set initial learning rate
        super().__init__(optimizer, last_epoch)

    # Step function
    def step(self, epoch=None):
        # Update step count
        self.current_step += 1

        # Get learning rate
        lr = self.get_lr()[0]

        # Update learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    # Compute learning rate
    def get_lr(self):
        # Get initial learning rate
        initial_lr = self.base_lrs[0]

        # If warmup steps are set, apply warmup
        if self.current_step <= self.warmup_steps:
            # Compute progress
            progress = self.current_step / self.warmup_steps

            # Linear warmup
            return [initial_lr * 0.5 * (1 + math.cos(math.pi * (1 - progress)))]
        
        # Compute decay steps
        decay_steps = self.current_step - self.warmup_steps

        # Exponential decay
        return [initial_lr * (self.decay_factor ** decay_steps)]

    # Save scheduler state
    def state_dict(self):
        return {"current_step": self.current_step}

    # Load scheduler state
    def load_state_dict(self, state_dict):
        self.current_step = state_dict["current_step"]

# Custom Dataset to apply transformations
class PretrainingDataset(Dataset):
    def __init__(self, dataset, transform, num_classes):
        # Store dataset
        self.dataset = dataset
        
        # Store transform
        self.transform = transform
        
        # Store number of classes
        self.num_classes = num_classes

    # Get dataset size
    def __len__(self):
        # Return dataset size
        return len(self.dataset)

    # Get item from dataset
    def __getitem__(self, idx):
        # Get image
        img = self.dataset[idx]['image']
        
        # Get fine label
        fine_label = self.dataset[idx]['label']
        
        # Convert image to numpy array
        img = np.array(img)
        
        # Apply transformations if defined
        img = self.transform(image=img)['image']
        
        # Convert label to one-hot encoding
        fine_label = torch.tensor(fine_label, dtype=torch.long)
        
        # Return image and label
        return img, fine_label

# Define the pretraining model with MLflow logging
class PretrainModel(LightningModule):
    def __init__(self, model, num_classes, num_batches):
        super().__init__()
        # Load the model
        self.model = model
        
        # Define the loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # Define the optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-4, weight_decay=1e-2)

        # Log the model
        self.scheduler = WarmupExponentialLR(self.optimizer, decay_factor=0.85 ** (1 / num_batches), warmup_steps=1024)

        # Define accuracy metric
        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        
    # Training step
    def training_step(self, batch, batch_idx):
        # Get the images and targets from the batch
        images, targets = batch
        
        # Perform the forward pass
        outputs = self.model(images)
        
        # Calculate the loss
        loss = self.criterion(outputs, targets)

        # Calculate accuracy
        preds = torch.argmax(outputs, dim=1)
        acc = self.train_accuracy(preds, targets)

        # Get current learning rate from optimizer
        lr = self.optimizers().param_groups[0]['lr']
        
        # Log the training loss, accuracy and learning rate in progress bar
        self.log("train_loss", loss, prog_bar=True, logger=False)
        self.log("train_acc", acc, prog_bar=True, logger=False)
        self.log("lr", lr, prog_bar=True, logger=False)

        # Log the training loss and accuracy to MLflow
        mlflow.log_metric("train_loss", loss.item(), step=batch_idx)
        mlflow.log_metric("train_acc", acc.item(), step=batch_idx)

        # Update LR per step
        self.lr_schedulers().step()
        
        # Return the loss
        return loss

    # Validation epoch start
    def on_validation_epoch_start(self):
        # Save the model
        torch.save(self.model.state_dict(), "models/pretrained_model.pth")

    # Validation step
    def validation_step(self, batch, batch_idx):
        # Get the images and targets from the batch
        images, targets = batch
        
        # Perform the forward pass
        outputs = self.model(images)
        
        # Calculate the loss
        loss = self.criterion(outputs, targets)

        # Calculate accuracy
        preds = torch.argmax(outputs, dim=1)
        acc = self.val_accuracy(preds, targets)
        
        # Log the validation loss and accuracy in progress bar
        self.log("val_loss", loss, prog_bar=True, logger=False)
        self.log("val_acc", acc, prog_bar=True, logger=False)

        # Log the validation loss and accuracy to MLflow
        mlflow.log_metric("val_loss", loss.item(), step=batch_idx)
        mlflow.log_metric("val_acc", acc.item(), step=batch_idx)
        
        # Return the loss
        return loss

    # Configure optimizer and scheduler
    def configure_optimizers(self):
        # Return the optimizer and scheduler
        return [self.optimizer], [{"scheduler": self.scheduler, "interval": "step"}]

# Load the pretraining data
def load_pretraining_data():
    # Define dataset paths
    data_path = os.path.join(os.getenv("HOME_DATASETS"), "imagenet21k-p-arrow")

    # Make paths for train and validation
    train_path = os.path.join(data_path, "train")
    val_path = os.path.join(data_path, "validation")
    
    # Load dataset from disk
    train_dataset = load_from_disk(train_path)
    val_dataset = load_from_disk(val_path)

    # Dynamically determine the number of classes
    num_classes = train_dataset.features['label'].num_classes

    # Print the number of classes
    print(f"Number of classes in dataset: {num_classes}")
    
    # Define training transformations
    train_transform = Compose([
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.25),
        ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, p=1.0),
        ToGray(p=0.1),
        PadIfNeeded(224, 224, border_mode=0),
        Rotate(limit=10, p=0.25),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    
    # Define validation transformations
    val_transform = Compose([
        PadIfNeeded(224, 224, border_mode=0),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    
    # Create training dataset
    train_dataset = PretrainingDataset(
        train_dataset,
        train_transform,
        num_classes,
    )
    
    # Create validation dataset
    val_dataset = PretrainingDataset(
        val_dataset,
        val_transform,
        num_classes,
    )
    
    # Create training loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=192,
        num_workers=12,
        shuffle=True,
        drop_last=True,
        persistent_workers=True
    )
    
    # Create validation loaders
    val_loader = DataLoader(
        val_dataset,
        batch_size=192,
        num_workers=4,
        shuffle=False,
        drop_last=True,
        persistent_workers=True
    )
    
    # Return the loaders
    return train_loader, val_loader, num_classes

# Main function
def main():
    # Backend configuration
    torch.backends.cudnn.benchmark = True

    # Set fp32 precision to 'high' for TF32 usage
    torch.set_float32_matmul_precision("high")

    # Set the experiment name
    experiment_name = "Custom model pretraining"
    mlflow.set_experiment(experiment_name)

    # Create the models directory
    os.makedirs("models", exist_ok=True)

    # Load the pretraining data
    train_loader, val_loader, num_classes = load_pretraining_data()
    print("Data loaded...")

    # Create the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device set to {device}")

    # Load the model
    model = CustomModel(num_classes).to(device)
    print("Model loaded...")

    # Compile the model
    torch.compile(model, backend='inductor', dynamic=False, fullgraph=True)
    print("Model compiled...")

    # Initialize the model training
    trainer = Trainer(
        max_epochs=5,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        log_every_n_steps=512,
        enable_checkpointing=False,
        enable_progress_bar=True,
        logger=False,
    )

    # Find the number of batches
    num_batches = len(train_loader)

    # Train the model with validation
    trainer.fit(PretrainModel(model, num_classes, num_batches), train_loader, val_loader)

    # Print the training completion
    print("Training completed")

# Run the main function
if __name__ == "__main__":
    main()
