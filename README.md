## Data Augmentation using Stable Diffusions for improving Object Detection performance

## 1. Introduction

Object detection is a crucial task in computer vision, with applications in various domains such as autonomous driving, surveillance, and medical imaging. Deep learning models have shown remarkable performance in this task, but their effectiveness relies on large and diverse datasets for training. Collecting such datasets is expensive, time-consuming, and prone to bias and class imbalance issues. To address this challenge, this project proposes a novel approach to synthetic image data augmentation using Large Language Models (LLMs). The project aims to fill the gap in existing research by exploring advanced techniques to generate high-quality synthetic images for improving object detection.

## 2. Proposed Method
![Architecture](https://github.com/ruthvikauwm/ObjectDetection/assets/54182107/bb62cf47-f627-4b66-9653-e1f095ca39d2)

## 3. Installations

To set up the project environment and install the required packages, follow the steps below. 
Note: This project execution requires a lot of memory resources and GPU. It would take a few hours to generate the images and complete the training process.

# Step 1: Create a Conda Environment
# Create a conda environment
```
conda create --name cs762project python=3.8
```
# Activate the conda environment (Linux/Mac/Windows)
```
conda activate cs762project
```
# Step 2: Install required packages
# Install required packages
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
**Step 1: Training Data Creation:**

We've selected a specific number of images from the COCO dataset for 10 labels and annotated them using the tool available at https://www.makesense.ai/. These images have been generated using basic augmentation techniques like rotation, flip, resize, and reshape, along with SD and GAN techniques. The GAN code is available in the file Icgan_colab.ipynb. The resulting images are stored in zip files named YOLO_NATURAL, YOLO_AUG, YOLO_GAN, and YOLO_SD on our drive.

The COCODataSubsetCreation.ipynb file contains the code to extract only the images related to the specified labels from the entire COCO dataset.

The TrainDataCreation.ipynb file manages unzipping these folders from the drive after mounting it. It generates various combinations of data such as natural_only, natural_aug, natural_gan_sd, gan_sd, natural_sd_aug, and natural_gan_sd_aug. These combinations are organized into images and labels folders for both training and validation purposes.

**Step 2: Testing Data Creation:**

This step is optional. Ideally, testing can be conducted on random images (of the 10 specified labels) downloaded from the internet. We've scripted a process to collect a few images for each label to efficiently test the model. The TestDataCreation.ipynb file handles this segment.

**Step 3: Model Training:**

Utilizing the YOLOV5 model (accessible at https://github.com/ultralytics/yolov5), we've trained the model using the 6 datasets created earlier. 

The Object Detection.ipynb file contains the code for the training process. This code allows you to experiment with various training parameters and save the best model weights. The weights file will be stored at '/content/yolov5/runs/train/exp/weights/<.pt file>' after the training completes.

**Step 4: Model Testing:**

The detect.py file within the yolo folder serves to test the object detection model. For instance, you can use the following command:
```
!python detect.py --weight /content/yolov5/runs/train/exp2/weights/best.pt --source /content/yolov5/YOLO_TEST
```
This command prompts the model to detect objects in the images within the YOLO_TEST folder and save the results at runs/detect/exp.

The Object Detection Test.ipynb file includes code to execute the model against the 6 datasets and store the results.

**Step 5: Metrics Calculation:**

Executing metrics.ipynb generates the necessary graphs to evaluate the model's performance.


