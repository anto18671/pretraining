from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import Accuracy
from albumentations.pytorch import ToTensorV2
from timm import create_model
from torch import nn
import mlflow
import torch
import math
import cv2
import os

# Import albumentations
from albumentations import (
    Resize, VerticalFlip, ToGray, Affine, RandomGamma, PlanckianJitter,
    Compose, HorizontalFlip, Normalize, ColorJitter,
    Resize, GaussNoise, MotionBlur,
)

# Define the CustomModel
class CustomModel(nn.Module):
    # Initializes the model
    def __init__(self, num_classes):
        super().__init__()
        # Initialize EfficientViT model
        self.efficientvit = create_model(
            'efficientvit_b2.r288_in1k',
            num_classes=num_classes,
            pretrained=True,
        )
        
        # Ablate specific blocks
        self.ablate_stage(0, [])
        self.ablate_stage(1, [])
        self.ablate_stage(2, [3, 4])
        self.ablate_stage(3, [0, 1, 2, 3, 4, 5, 6])

        # Replace the head with a custom head
        self.efficientvit.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(192, 576, bias=False),
            nn.LayerNorm((576,), eps=1e-5, elementwise_affine=True),
            nn.Hardswish(),
            nn.Dropout(p=0.1, inplace=False),
            nn.Linear(576, 2400, bias=True),
        )

    # Remove the and head
    def remove_head(self):
        # Delete the head from the model
        del self.efficientvit.head
    
    # Ablate specific blocks in a stage
    def ablate_stage(self, stage_idx, block_indices):
        # Get the stage
        stage = self.efficientvit.stages[stage_idx]

        # Convert the blocks to a list
        blocks = list(stage.blocks)

        # Print the number of blocks before ablation
        print(f"Stage {stage_idx} - Blocks before ablation: {len(blocks)}")

        # Remove blocks in reverse order
        for index in sorted(block_indices, reverse=True):
            del blocks[index]

        # Print the number of blocks after ablation
        print(f"Stage {stage_idx} - Blocks after ablation: {len(blocks)}")

        # Check if no blocks are left
        if len(blocks) == 0:
            # Remove the stage if no blocks are left
            del self.efficientvit.stages[stage_idx]

        # Update the blocks
        stage.blocks = nn.Sequential(*blocks)

    # Forward method
    def forward(self, x):
        # Return the features from the backbone
        return self.efficientvit(x)
    
# Define a warmup and exponential scheduler with an optional cycling element
class WarmupExponentialLR:
    # Initialize scheduler with optimizer, decay factor, warmup steps, cycle frequency, and cycle amplitude
    def __init__(self, optimizer, decay_factor, warmup_steps=0, cycle_frequency=1, cycle_amplitude=0.0):
        # Store the optimizer
        self.optimizer = optimizer

        # Store the warmup steps
        self.warmup_steps = warmup_steps

        # Store the decay factor
        self.decay_factor = decay_factor

        # Store the cycle frequency
        self.cycle_frequency = cycle_frequency

        # Store the cycle amplitude
        self.cycle_amplitude = cycle_amplitude

        # Initialize the current step counter
        self.current_step = 0

        # Loop over parameter groups in the optimizer
        for param_group in self.optimizer.param_groups:
            # Set the initial learning rate if not already present
            param_group.setdefault('initial_lr', param_group['lr'])
    
    # Update the learning rate at each step
    def step(self):
        # Increment the current step counter
        self.current_step += 1

        # Compute the new learning rate based on the current step
        lr = self.get_lr(self.current_step)

        # Loop over parameter groups in the optimizer
        for param_group in self.optimizer.param_groups:
            # Update the learning rate in the parameter group
            param_group['lr'] = lr
    
    # Compute the learning rate based on warmup, exponential decay, and cycling
    def get_lr(self, step):
        # Retrieve the initial learning rate from the first parameter group
        initial_lr = self.optimizer.param_groups[0]['initial_lr']

        # Check if the current step is within the warmup period
        if step <= self.warmup_steps:
            # Calculate the progress fraction during warmup
            progress = step / self.warmup_steps

            # Calculate the cosine-based warmup learning rate
            base_lr = initial_lr * 0.5 * (1 + math.cos(math.pi * (1 - progress)))
        else:
            # Calculate the number of steps after warmup
            decay_steps = step - self.warmup_steps

            # Apply exponential decay after warmup
            base_lr = initial_lr * (self.decay_factor ** decay_steps)

        # Compute the sine-based cycling factor
        cycle_factor = 1 + self.cycle_amplitude * math.sin(2 * math.pi * step / self.cycle_frequency)

        # Return the final learning rate including the cycling component
        return base_lr * cycle_factor
    
    # Return the current state of the scheduler as a dictionary
    def state_dict(self):
        # Return the state dictionary containing the current step
        return {'current_step': self.current_step}
    
    # Load the scheduler state from a state dictionary
    def load_state_dict(self, state):
        # Set the current step from the provided state dictionary
        self.current_step = state.get('current_step', 0)

# Define a class for structured progressive unfreezing of the model block-by-block
class FeatureExtractorTrainer(LightningModule):
    # Initialize with a model, number of classes, and total batches
    def __init__(self, model, num_classes, num_batches, unfreeze_steps=2048):
        # Call the parent class initializer
        super().__init__()
        # Store the model reference
        self.model = model

        # Store unfreeze steps
        self.unfreeze_steps = unfreeze_steps

        # Freeze all parameters of the feature extractor
        for param in self.model.efficientvit.parameters():
            param.requires_grad = False

        # Unfreeze parameters of the head only
        for param in self.model.efficientvit.head.parameters():
            param.requires_grad = True

        # Define the loss function as CrossEntropyLoss
        self.criterion = torch.nn.CrossEntropyLoss()

        # Define the optimizer as AdamW with model parameters
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            weight_decay=2.5e-4,
            lr=1e-4,
        )

        # Calculate decay factor for exponential decay
        decay_factor = 0.88 ** (1 / num_batches)

        # Define the learning rate scheduler with warmup and exponential decay
        self.scheduler = WarmupExponentialLR(
            self.optimizer,
            decay_factor=decay_factor,
            warmup_steps=4096,
            cycle_frequency=512,
            cycle_amplitude=1/3,
        )

        # Initialize the training accuracy metric for multiclass classification
        self.train_accuracy = Accuracy(
            task="multiclass",
            num_classes=num_classes,
        )

        # Initialize the validation accuracy metric for multiclass classification
        self.val_accuracy = Accuracy(
            task="multiclass",
            num_classes=num_classes,
        )

    # Define a hook to update parameter freezing based on training steps
    def on_train_batch_start(self, batch, batch_idx, dataloader_idx=0):
        # Check if the current global step has reached the target number of steps
        if self.global_step == self.unfreeze_steps:
            # Unfreeze all parameters of the feature extractor
            for param in self.model.efficientvit.parameters():
                param.requires_grad = True

    # Define the training step executed during each training iteration
    def training_step(self, batch, batch_idx):
        # Unpack the batch into images and targets
        images, targets = batch

        # Compute model outputs from the images
        outputs = self.model(images)

        # Calculate the loss between outputs and targets using CrossEntropyLoss
        loss = self.criterion(outputs, targets)

        # Determine the predicted classes by taking the argmax of outputs
        preds = torch.argmax(outputs, dim=1)

        # Calculate the accuracy for the training batch
        acc = self.train_accuracy(preds, targets)

        # Retrieve the current learning rate from the optimizer
        lr = self.optimizers().param_groups[0]['lr']

        # Log the training loss in the progress bar without using a logger
        self.log("train_loss", loss, prog_bar=True, logger=False)

        # Log the training accuracy in the progress bar without using a logger
        self.log("train_acc", acc, prog_bar=True, logger=False)

        # Log the current learning rate in the progress bar without using a logger
        self.log("lr", lr, prog_bar=True, logger=False)

        # Log the training loss to MLflow with the current batch index as step
        mlflow.log_metric("train_loss", loss.item(), step=batch_idx)

        # Log the training accuracy to MLflow with the current batch index as step
        mlflow.log_metric("train_acc", acc.item(), step=batch_idx)
        
        # Return the computed loss for backpropagation
        return loss

    # Override the LightningModule lr_scheduler_step hook for custom scheduler stepping
    def lr_scheduler_step(self, scheduler, optimizer_idx):
        # Call the custom scheduler's step function to update the learning rate
        scheduler.step()

    # Save the model at the start of each validation epoch
    def on_validation_epoch_start(self):
        # Save the current model state to a file
        torch.save(self.model.state_dict(), "models/pretrained_model.pth")

    # Define the validation step executed during each validation iteration
    def validation_step(self, batch, batch_idx):
        # Unpack the batch into images and targets
        images, targets = batch

        # Compute model outputs from the images
        outputs = self.model(images)

        # Calculate the loss between outputs and targets using CrossEntropyLoss
        loss = self.criterion(outputs, targets)

        # Determine the predicted classes by taking the argmax of outputs
        preds = torch.argmax(outputs, dim=1)

        # Calculate the accuracy for the validation batch
        acc = self.val_accuracy(preds, targets)

        # Log the validation loss in the progress bar without using a logger
        self.log("val_loss", loss, prog_bar=True, logger=False)

        # Log the validation accuracy in the progress bar without using a logger
        self.log("val_acc", acc, prog_bar=True, logger=False)

        # Log the validation loss to MLflow with the current batch index as step
        mlflow.log_metric("val_loss", loss.item(), step=batch_idx)

        # Log the validation accuracy to MLflow with the current batch index as step
        mlflow.log_metric("val_acc", acc.item(), step=batch_idx)
        
        # Return the computed loss for evaluation
        return loss

    # Configure the optimizer and learning rate scheduler
    def configure_optimizers(self):
        # Return the optimizer and a dictionary specifying the scheduler and its step interval
        return [self.optimizer], [{"scheduler": self.scheduler, "interval": "step"}]

# Define a custom dataset class
class CustomImageDataset(Dataset):
    # Initialize the custom image dataset with base path, split, class index mapping, and transformation pipeline
    def __init__(self, base_path, split, class_to_idx, transform):
        # Combine base_path and split to form the dataset directory
        self.based = os.path.join(base_path, split)
        
        # Store the transformation pipeline
        self.transform = transform
        
        # Store the class-to-index mapping
        self.class_to_idx = class_to_idx
        
        # List dataset samples with consistent class indices
        self.samples = [
            (os.path.join(class_name, fname), self.class_to_idx[class_name])
            for class_name in self.class_to_idx.keys()
            for fname in os.listdir(os.path.join(self.based, class_name))
        ]

    # Return the total number of samples in the dataset
    def __len__(self):
        return len(self.samples)

    # Retrieve a sample from the dataset given its index
    def __getitem__(self, idx):
        # Retrieve the relative image path and label
        img_rel_path, label = self.samples[idx]
        
        # Construct the full image path
        full_img_path = os.path.join(self.based, img_rel_path)
        
        # Read and convert the image from BGR to RGB
        image = cv2.imread(full_img_path)
        
        # Convert the image to RGB format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply the transformation pipeline
        image = self.transform(image=image)["image"]
        
        # Return the transformed image and label
        return image, label
        

# Load the pretraining data using torchvision's ImageFolder
def load_pretraining_data():
    # Define the transformation pipeline for training data using albumentations
    train_transform = Compose([
        # Flip transformations
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),

        # Apply one of the following transformations randomly
        ColorJitter(
            brightness=0.25,
            contrast=0.25,
            saturation=0.25,
            hue=0.025,
            p=1.0,
        ),

        # Random gamma correction
        RandomGamma(
            gamma_limit=(92.5, 107.5),
            p=0.5,
        ),

        # Visual transform with special effects
        PlanckianJitter(
            mode="cied",
            temperature_limit=(5000,8000),
            sampling_method="gaussian",
            p=0.5,
        ),

        # Apply Gaussian noise
        GaussNoise(
            std_range=(0.0, 0.02),
            p=0.25,
        ),

        # Apply motion blur
        MotionBlur(
            blur_limit=(3, 9),
            allow_shifted=True,
            p=0.25,
        ),

        # Convert the image to grayscale
        ToGray(p=0.1),

        # Spatial transformations
        Affine(
            translate_percent=(0.0, 0.125),
            scale=(0.825, 1.1),
            rotate=(-25.0, 25.0),
            shear=(-12.5, 12.2),
            p=1.0,
        ),

        # Normalize the image with mean and standard deviation
        Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),

        # Resize the image to a fixed size
        Resize(448, 448),

        # Convert the image to PyTorch tensor format
        ToTensorV2(),
    ])

    # Define the transformation pipeline for validation data using albumentations
    val_transform = Compose([
        # Normalize the image with mean and standard deviation
        Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),

        # Resize the image to a fixed size
        Resize(448, 448),

        # Convert the image to PyTorch tensor format
        ToTensorV2(),
    ])

    # Construct the base path for datasets using the HOME_DATASETS environment variable
    data_path = os.path.join(os.getenv("HOME_DATASETS"), "imagenet2_4k")

    # Get the sorted class names and create a consistent class-to-index mapping
    class_names = sorted(os.listdir(os.path.join(data_path, "train")))
    class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}

    # Get the number of classes
    num_classes = len(class_names)
    print(f"Number of classes in dataset: {num_classes}")

    # Wrap the raw training dataset with PretrainingDataset to apply transformations
    train_dataset = CustomImageDataset(
        data_path,
        "train",
        class_to_idx,
        train_transform,
    )

    # Wrap the raw validation dataset with PretrainingDataset to apply transformations
    val_dataset = CustomImageDataset(
        data_path,
        "validation",
        class_to_idx,
        val_transform,
    )

    # Create a DataLoader for the training dataset with specified batch size and workers
    train_loader = DataLoader(
        train_dataset,
        batch_size=60,
        num_workers=6,
        shuffle=True,
        persistent_workers=True,
    )

    # Create a DataLoader for the validation dataset with specified batch size and workers
    val_loader = DataLoader(
        val_dataset,
        batch_size=60,
        num_workers=3,
        shuffle=False,
        persistent_workers=True,
    )

    # Return the training DataLoader, validation DataLoader, and number of classes
    return train_loader, val_loader, num_classes

# Define the main function to set up training and execute the training loop
def main():
    # Enable benchmark mode for cudnn to improve performance
    torch.backends.cudnn.benchmark = True

    # Set the experiment name for MLflow logging
    experiment_name = "Custom model pretraining"
    mlflow.set_experiment(experiment_name)

    # Create a directory named "models" if it does not exist
    os.makedirs("models", exist_ok=True)

    # Load the pretraining data and obtain the DataLoaders and number of classes
    train_loader, val_loader, num_classes = load_pretraining_data()

    # Print a message indicating that the data has been loaded
    print("Data loaded...")

    # Set the device to CUDA if available; otherwise use CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Print the device being used for training
    print(f"Device set to {device}...")

    # Initialize the CustomModel with the determined number of classes and move it to the selected device
    model = CustomModel(num_classes).to(device)

    # Print the model architecture
    print(model)

    # Print a message indicating that the model has been loaded
    print("Model loaded...")

    # Initialize the Trainer with specified parameters for epochs, accelerator, and logging
    trainer = Trainer(
        max_epochs=16,
        precision="bf16-mixed",
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        enable_checkpointing=False,
        enable_progress_bar=True,
        log_every_n_steps=512,
        logger=False,
    )

    # Determine the number of batches in the training DataLoader
    num_batches = len(train_loader)

    # Start training using the FeatureExtractorTrainer wrapper with the training and validation DataLoaders
    trainer.fit(FeatureExtractorTrainer(model, num_classes, num_batches), train_loader, val_loader)

    # Print a message indicating that training has been completed
    print("Training completed")

# Run the main function if the script is executed
if __name__ == "__main__":
    main()
