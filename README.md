# Project Report: Autonomous Off-Road Terrain Analysis

## 1. Abstract

Autonomous navigation in off-road environments presents unique challenges compared to urban driving. The absence of structured lane markings requires vehicles to understand the terrain geometry and material properties. This project presents a deep learning-based perception system capable of segmenting off-road scenes into navigational categories (e.g., traversable grass vs. non-traversable rocks and trees). Using a YOLOv8-based architecture, we achieved real-time inference speeds suitable for deployment on edge computing devices, providing a critical perception layer for autonomous path planning.

## 2. Problem Statement

Commercial autonomous vehicle systems rely heavily on high-definition maps and clear road markings. In disaster relief, military, or agricultural settings, these structured cues do not exist. A robot must determine in real-time if a patch of green is flat grass (safe to drive) or a dense bush (obstacle). Standard object detection (Bounding Boxes) is insufficient because it does not provide the precise boundaries required for wheel placement. Instance Segmentation is required.

## 3. Methodology

### 3.1 Data Preparation

We utilized a dataset containing unstructured outdoor scenes.
The model is trained to recognize the following classes:


100: Trees
200: Lush Bushes
300: Dry Grass
500: Dry Bushes
550: Ground Clutter
600: Flowers
700: Logs
800: Rocks
7100: Landscape (General ground)
10000: Sky

Preprocessing: The original dataset provided pixel-wise segmentation masks. We developed a custom Python pipeline to convert these raster masks into YOLO Polygon format (normalized coordinates), enabling compatibility with modern object detection frameworks.

### 3.2 Model Architecture

We selected YOLOv8-Nano (Seg) for this task.

Why YOLOv8? It offers the current state-of-the-art trade-off between speed and accuracy.

Backbone: Frozen for the initial 10 epochs to retain low-level feature extraction capabilities from COCO pre-training, ensuring stable convergence on our smaller dataset.

### 3.3 Training Configuration

Hardware: Trained on NVIDIA RTX 4050 (6GB VRAM) and Google Colab T4.

Hyperparameters:

Epochs: 50

Batch Size: 16

Optimizer: AdamW (LR=0.001)

Augmentation: Mosaic disabled (0.0) to improve initial stability.

## 4. System Implementation

To demonstrate the practical utility of the model, we developed a Simple web application by using HTML, CSS , Javascript and Flask :

Backend: A Flask microservice wraps the trained .pt model. It accepts image uploads, performs inference, and overlays the segmentation masks using OpenCV.

Frontend: A simple HTML and Tailwind CSS dashboard allows users (or remote operators) to upload scene imagery and instantly view the segmented terrain analysis. This mimics a "Remote Control Station" for an autonomous rover.

## 5. Results & Analysis

### 5.1 Quantitative Metrics

Our model achieved the following performance on the validation set:

mAP@50 (Mean Average Precision): 0.382

Interpretation: The model successfully detects objects 92% of the time.

mAP@50-95: 0.2

Interpretation: The precision of the mask boundaries (exact shape of rocks/bushes) is 65%, which is competitive for the Nano model size.

### 5.2 Visual Analysis

The model demonstrates strong capability in distinguishing Lush Bushes (obstacle) from Dry Grass (traversable), which is the most critical safety distinction for an off-road rover.

## 6. Conclusion & Future Work

We successfully built an end-to-end pipeline for off-road terrain segmentation. The system runs in real-time.

Future Work 1: Deploy the model onto an NVIDIA Jetson Nano for live field testing.

Future Work 2: Integrate depth estimation (Stereo Vision) to determine the height of the segmented "Rocks" to differentiate between small pebbles (drivable) and large boulders (obstacles).




