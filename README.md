![Python](https://img.shields.io/badge/Python-3.11-blue)

![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)

![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO-purple)

![RT-DETR](https://img.shields.io/badge/RT--DETR-Transformer-green)

![License](https://img.shields.io/badge/License-MIT-yellow)

# PPE Detection Comparative Analysis

> Comparative Performance Analysis of CNN-based and Transformer-based Architectures for Real-Time PPE Detection in IoT Systems.


## Overview

This project presents a comprehensive comparative study of modern object detection architectures for Personal Protective Equipment (PPE) detection.

Instead of focusing on a single model, this work evaluates multiple state-of-the-art CNN-based and Transformer-based detectors under identical experimental conditions to analyze their effectiveness for real-time IoT deployment.

The comparison includes:

* YOLOv8
* YOLO26
* RT-DETR

The study evaluates detection accuracy, computational efficiency, inference speed, qualitative performance, and deployment suitability.


## Key Features

* Comparative analysis of multiple object detection architectures
* Benchmarking under identical datasets
* Quantitative evaluation
* Qualitative evaluation
* Failure case analysis
* Computational performance analysis
* COCO dataset validation utilities
* Ready for real-time IoT deployment experiments


## Project Architecture

```mermaid
flowchart TD

Dataset

-->

Preprocessing

-->

Training

Training --> YOLOv8

Training --> YOLO11

Training --> RTDETR

YOLOv8 --> Evaluation

YOLO11 --> Evaluation

RTDETR --> Evaluation

Evaluation --> Benchmark

Evaluation --> ErrorAnalysis

Evaluation --> Deployment
```

## Experimental Setup
GPU: RTX 3060 Laptop

Epochs: 50

Image Size: 640

Batch: 8

Optimizer: AdamW

Framework: Ultralytics


## Repository Structure

```text
.
├── docs
├── notebooks
├── results
├── benchmark.py
├── computational.py
├── quantitative.py
├── qualitative_results.py
├── requirements.txt
└── README.md
```


## Models Evaluated

| Model   | Architecture      |
| ------- | ----------------- |
| YOLOv8  | CNN-based         |
| YOLO26  | CNN-based         |
| RT-DETR | Transformer-based |


## Evaluation Criteria

- mAP
- Precision
- Recall
- F1-score
- IoU
- Computational Cost
- Inference Speed
- Model Size
- Qualitative Prediction
- Failure Case Analysis


## Dataset

The dataset is **not included** in this repository because of its large size.

Please download the PPE dataset in roboflow (https://universe.roboflow.com/ppe-ihvqu/ppe-8k2vo/dataset/3#) separately and organize it according to the required directory structure before training.


## Running Experiments

```bash
pip install -r requirements.txt
```

Run training notebooks:

- YOLOv8_PPE_Final_Training_Notebook.ipynb
- YOLO26_PPE_Final_Training_Notebook.ipynb
- RTDETR_PPE_Final_Training_Notebook.ipynb

Workflow:

Dataset

↓

Data Validation

↓

Training

↓

Evaluation

↓

Benchmark

↓

Failure Analysis

↓

Comparison

↓

Conclusion

## Results

The repository includes:

- Benchmark comparison

<p align="center">
  <img src="image.png" width="900"/>
</p>

- Qualitative Predictions
<p align="center">
  <img src="https://github.com/user-attachments/assets/13638ae6-dac9-4dac-be5c-58f09f9209fc" width="900"/>
</p>

- Failure Case Visualization
<p align="center">
  <img src="Failure Case.png" width="900"/>
</p>

- Computational Analysis
<p align="center">
  <img src="https://github.com/user-attachments/assets/80379bd6-4327-4466-9fee-18b91d444901"
       width="700"/>
</p>

## Future Improvements

- D-FINE comparison
- Grounding DINO
- OWLv2
- Edge deployment benchmarking
- TensorRT optimization
- ONNX inference


## Author

- Pham Tan Minh 
- Doan Duy Long
- Nguyen Tran Hai Phong

Computer Vision • AI Engineering • Backend AI Systems

- Linkedln: Minh Pham Tan 
- Email: phamtanminh2004@gmail.com
