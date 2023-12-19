## Data Augmentation using Stable Diffusions for improving Object Detection performance

## 1. Introduction

Object detection is a crucial task in computer vision, with applications in various domains such as autonomous driving, surveillance, and medical imaging. Deep learning models have shown remarkable performance in this task, but their effectiveness relies on large and diverse datasets for training. Collecting such datasets is expensive, time-consuming, and prone to bias and class imbalance issues. To address this challenge, this project proposes a novel approach to synthetic image data augmentation using Large Language Models (LLMs). The project aims to fill the gap in existing research by exploring advanced techniques to generate high-quality synthetic images for improving object detection.

## 2. Proposed Method
<!-- ![Architecture](https://github.com/ruthvikauwm/ObjectDetection/assets/54182107/bb62cf47-f627-4b66-9653-e1f095ca39d2) -->
![proposed_architecture](https://github.com/ruthvikauwm/ObjectDetection/assets/34669956/4ab63cf1-17ba-4f89-a09c-ae6bd4193781)

## 3. Project Implementation

To set up the project environment and install the required packages, follow the steps below. 
Note: Executing this project demands substantial memory resources and GPU capacity. The image generation and training processes are anticipated to take several hours. We recommend implementing it in a GPU environment instead of a Conda environment for optimal performance and efficiency.

## Step 1: Create a Conda Environment
```
conda create --name cs762project python=3.8
```
## Activate the conda environment (Linux/Mac/Windows)
```
conda activate cs762project
```
## Step 2: Install required packages
```
pip install numpy torch torchvision imageio pillow scipy nltk cma scikit-learn
```
```
pip install transformers scipy ftfy
```
```
pip install diffusers==0.4.0
```
```
pip install "ipywidgets>=7,<8"
```

## Step 3: Training Data Creation

Our dataset is curated from three parts, the original set, the synthetic set, and the test set. For the original set, we utilized the COCO 2017 dataset (https://cocodataset.org/#download), which is a widely used benchmark dataset for object detection, segmentation, and captioning. This dataset contains over 330,000 images with more than 2.5 million object instances labeled across 80 categories. We specifically used the COCO train 2017 dataset, which is a subset of the larger COCO dataset containing 118,287 images. To augment our dataset, we explored two methods - Generative Adversarial Networks (GANs) and Stable Diffusions step.

The COCODataSubsetCreation.ipynb file contains the code to extract only the images related to the specified labels from the entire COCO dataset.

The TrainDataCreation.ipynb file manages unzipping these folders from the drive after mounting it. It generates various combinations of data such as 
1. natural_only
2. natural_aug
3. natural_gan_sd
4. gan_sd
5. natural_sd_aug
6. natural_gan_sd_aug.

These combinations are organized into images and labels folders for both training and validation purposes.

Note: We've conducted experiments involving textual inversion with our Stable Diffusion model. Further details can be found in the documentation at https://huggingface.co/docs/diffusers/training/text_inversion

## Step 4: Testing Data Creation

This step is optional. Ideally, testing can be conducted on random images (of the 10 specified labels) downloaded from the internet. We've scripted a process to collect a few images for each label to efficiently test the model. The TestDataCreation.ipynb file handles this segment.

## Step 5: Model Training

Utilizing the YOLOV5 model (accessible at https://github.com/ultralytics/yolov5), we've trained the model using the 6 datasets created earlier. 

The Object Detection.ipynb file contains the code for the training process. This code allows you to experiment with various training parameters and save the best model weights. The weights file will be stored at '/content/yolov5/runs/train/exp/weights/<.pt file>' after the training completes.

## Step 6: Model Testing

The detect.py file within the yolo folder serves to test the object detection model. For instance, you can use the following command:
```
!python detect.py --weight /content/yolov5/runs/train/exp2/weights/best.pt --source /content/yolov5/YOLO_TEST
```
This command prompts the model to detect objects in the images within the YOLO_TEST folder and save the results at runs/detect/exp.

The Object Detection Test.ipynb file includes code to execute the model against the 6 datasets and store the results.

## Step 7: Metrics Calculation

To evaluate the effectiveness of synthetic data augmentation in improving object detection performance and the model’s generalization ability, we utilized several standard evaluation metrics.

1. TP,FP,FN:- True Positives, False Positive, False Negative
2. Recall:- TP / (TP + FN)
3. Precision:- TP / (TP + FP)
4. F1 Score:- 2 * (Precision * Recall) / (Precision + Recall)
5. mAP:- (AP1 + AP2 + ... + APₙ) / n

Executing metrics.ipynb generates the necessary graphs to evaluate the model's performance.


