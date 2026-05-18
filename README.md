# WMA - PJATK

Machine Vision (Wizja Maszynowa) projects developed in Python using OpenCV, TensorFlow/Keras and YOLO (Ultralytics).

## Topics
- Image processing
- Object tracking
- Hough transform
- Feature descriptors (SIFT, ORB)
- Convolutional Neural Networks (CNN)
- Object detection and tracking with YOLO

---

# Projects

## Project 1 — Red Ball Tracking
**Topic:** Morphological operations, color detection, object tracking

### Description
Application for tracking a red ball in a video stream using OpenCV.

### Features
- Video frame acquisition
- Frame resizing
- Detection of the ball center and radius
- Morphological operations for noise reduction
- Real-time object tracking

### Technologies
- Python
- OpenCV
- NumPy

---

## Project 2 — Coin Detection and Counting
**Topic:** Hough Transform, image preprocessing

### Description
Program for detecting and counting coins on images.

### Features
- Coin detection using Hough Transform
- Edge detection and thresholding
- Counting different coin types
- Tray / non-tray object analysis
- Image preprocessing optimization

### Technologies
- Python
- OpenCV
- NumPy

---

## Project 3 — Image Descriptors and Object Tracking
**Topic:** Feature descriptors (SIFT, ORB)

### Description
Application for feature extraction, image matching and object tracking.

### Features
- Keypoint detection
- Descriptor extraction using:
  - SIFT
  - ORB
- Feature matching between images
- Object localization and tracking
- Real-time video processing

### Technologies
- Python
- OpenCV

---

## Project 4 — Convolutional Neural Networks
**Topic:** Deep Learning, image classification

### Description
Implementation of a Convolutional Neural Network (CNN) for image classification using TensorFlow and Keras.

### Requirements
- Dataset preparation with image augmentation
- At least 450 labeled training images
- CNN architecture with max-pooling layers
- Activation function optimization
- Model optimization and evaluation
- Test image classification

### Classification Classes
- Banana
- Orange
- Lemon

### Technologies
- Python
- OpenCV
- TensorFlow
- Keras

---

## Project 5 — YOLO Object Detection and Tracking
**Topic:** Ultralytics YOLO, object detection

### Description
Object detection and tracking using Ultralytics YOLO framework.

### Features
- Custom dataset preparation
- Object labeling and augmentation
- Training YOLO models
- Object detection in videos
- Object tracking
- Comparison with descriptor-based tracking methods from previous projects

### Example Classes
- Chainsaw
- Airplane
- Custom objects from Roboflow datasets

### Technologies
- Python
- OpenCV
- Ultralytics YOLO
- Roboflow

### Additional Information
The project includes:
- Installation and configuration of Ultralytics package
- Dataset preparation and augmentation
- Model training and evaluation
- Detection and tracking tests on videos

---

# Repository Structure

```text
WMA_projects/
│
├── WMA_1/
├── WMA_2/
├── WMA_3/
├── WMA_4/
└── WMA_5/
