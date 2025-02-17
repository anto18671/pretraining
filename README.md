# Overview

This repository contains a training script for pretraining a custom computer vision model. The solution leverages several state-of-the-art libraries such as PyTorch Lightning for model training, timm for model backbones, albumentations for data augmentation, MLflow for experiment tracking, and Hugging Face Datasets for data streaming. The script integrates these components to provide an end-to-end pipeline for model development, training, logging, and validation.

The core features include:

- A custom model built with a ResNet backbone.
- A custom learning rate scheduler combining a warmup phase with exponential decay.
- Integration of MLflow to log training and validation metrics.
- Extensive data augmentation using albumentations.
- Streaming of data from an image folder using Hugging Face Datasets.
- Support for GPU training with performance optimizations.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Training Details](#training-details)
- [Logging and Experiment Tracking](#logging-and-experiment-tracking)
- [Code Structure](#code-structure)
- [Customization](#customization)
- [Contribution](#contribution)
- [License](#license)

## Features

- **Custom Model Architecture:**  
  Uses a ResNet backbone from timm with configurable dropout and class settings.

- **Advanced Learning Rate Scheduler:**  
  Implements a warmup phase followed by exponential decay to adjust learning rates throughout training.

- **Data Augmentation:**  
  Applies a suite of image transformations including horizontal/vertical flips, color jitter, grayscale conversion, padding, rotation, and normalization.

- **Streaming Data Loader:**  
  Utilizes the Hugging Face Datasets library to stream data from a filesystem directory in an efficient manner.

- **MLflow Integration:**  
  Logs training and validation metrics (loss, accuracy, learning rate) to MLflow for comprehensive experiment tracking.

- **Performance Optimizations:**  
  Uses CUDA if available, enables cuDNN benchmark mode, and sets TF32 precision for improved training performance.

## Prerequisites

Before running the script, ensure that you have the following:

- Python 3.8 or later.
- A working CUDA installation (if training on GPU).
- Access to the dataset in an ImageFolder format.
- Environment variable `HOME_DATASETS` pointing to the dataset directory.

Required libraries include:

- PyTorch and torchvision
- PyTorch Lightning
- timm
- albumentations
- MLflow
- torchmetrics
- Hugging Face Datasets
- NumPy

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/anto18671/pretraining.git
   cd pretraining
   ```

2. **Set up a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install the required packages:**

   ```bash
   pip install -r requirements.txt
   ```

## Data Preparation

The script uses the Hugging Face Datasets library to load images from a filesystem directory. To prepare your dataset:

- Organize your images in an ImageFolder format.
- Set the `HOME_DATASETS` environment variable to the root directory containing your dataset. For example:

  ```bash
  export HOME_DATASETS=/path/to/your/dataset
  ```

- Ensure that the directory structure follows the expected pattern for both training and validation splits (e.g., `train/**/*` and `val/**/*`).

## Configuration

The script includes several configurable components:

- **Model Parameters:**
  You can modify the number of classes and dropout rates in the custom model by editing the initialization parameters.

- **Learning Rate Scheduler:**
  The scheduler is set up with a warmup phase (1024 steps) followed by an exponential decay determined by a decay factor. Adjust these parameters as needed.

- **Data Augmentation:**
  The training and validation transformations are defined using albumentations. Feel free to customize the list of transformations in the respective sections.

- **Batch Sizes and Workers:**
  The DataLoader for training uses a batch size of 192 with 12 workers, while the validation DataLoader uses a batch size of 192 with 4 workers. These values can be adjusted based on hardware capabilities.

## Usage

1. **Ensure that the required environment variables are set:**
   For example, set `HOME_DATASETS` to point to your dataset directory.

2. **Run the training script:**

   ```bash
   python pre.py
   ```

3. **Monitor training progress:**
   The script prints key information to the console including data loading confirmation, device selection, model compilation status, and training progress.

## Training Details

- **Model Compilation:**
  The script compiles the model using the Inductor backend with full graph compilation and disables dynamic shapes for potential performance benefits.

- **Optimizer and Scheduler:**
  Uses the AdamW optimizer with a learning rate of 2e-4 and weight decay of 1e-2. The custom learning rate scheduler updates the learning rate at every step.

- **Training Loop:**
  The training loop is managed by PyTorch Lightning. Training metrics such as loss and accuracy are computed at each step and logged accordingly.

- **Validation:**
  The validation step is executed at the start of each validation epoch where the model is saved to disk. Metrics are similarly logged for monitoring performance.

## Logging and Experiment Tracking

- **MLflow Integration:**
  The script is integrated with MLflow. It logs training and validation metrics including:

  - Training loss and accuracy
  - Validation loss and accuracy
  - Learning rate updates

- **Experiment Setup:**
  The MLflow experiment is configured at the start of the script. Metrics are logged with step information, which can be later visualized using MLflow’s UI.

## Code Structure

- **Custom Model Definition:**
  The `CustomModel` class wraps a ResNet backbone from timm and provides a simple forward pass.

- **Custom Learning Rate Scheduler:**
  The `WarmupExponentialLR` class implements a learning rate schedule with a warmup period followed by exponential decay.

- **Dataset and Data Augmentation:**
  The `PretrainingDataset` class applies image transformations using albumentations, converting images and labels appropriately.

- **Lightning Module for Training:**
  The `PretrainModel` class extends PyTorch Lightning’s `LightningModule` and contains the training and validation steps along with MLflow logging.

- **Main Function:**
  The `main` function sets backend configurations, loads data, compiles the model, and initializes the training process via PyTorch Lightning’s Trainer.

## Customization

You can tailor various parts of the script to meet your needs:

- **Modify Model Architecture:**
  Change the parameters of the ResNet backbone or swap it out for a different architecture from timm.

- **Adjust Data Augmentation:**
  Edit the transformation pipelines in the training and validation sections to experiment with different augmentation strategies.

- **Learning Rate Scheduler Tuning:**
  Adjust the warmup steps and decay factor in the custom learning rate scheduler based on your training dynamics.

- **Experiment Tracking:**
  Customize the MLflow logging by adding more metrics or altering the logging frequency.

## License

This repository is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

```

```
