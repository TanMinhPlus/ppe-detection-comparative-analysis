![Python](https://img.shields.io/badge/Python-3.11-blue)

![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)

![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO-purple)

![RT-DETR](https://img.shields.io/badge/RT--DETR-Transformer-green)

![License](https://img.shields.io/badge/License-MIT-yellow)

# 🦺 PPE Detection Comparative Analysis

> Comparative Performance Analysis of CNN-based and Transformer-based Architectures for Real-Time PPE Detection in IoT Systems.

---

## 📌 Overview

This project presents a comprehensive comparative study of modern object detection architectures for Personal Protective Equipment (PPE) detection.

Instead of focusing on a single model, this work evaluates multiple state-of-the-art CNN-based and Transformer-based detectors under identical experimental conditions to analyze their effectiveness for real-time IoT deployment.

The comparison includes:

* YOLOv8
* YOLO11
* RT-DETR

The study evaluates detection accuracy, computational efficiency, inference speed, qualitative performance, and deployment suitability.

---

## ✨ Key Features

* Comparative analysis of multiple object detection architectures
* Benchmarking under identical datasets
* Quantitative evaluation
* Qualitative evaluation
* Failure case analysis
* Computational performance analysis
* COCO dataset validation utilities
* Ready for real-time IoT deployment experiments

---

## 🏗 Project Architecture

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

---

## 📂 Repository Structure

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

---

## 📊 Models Evaluated

| Model   | Architecture      |
| ------- | ----------------- |
| YOLOv8  | CNN-based         |
| YOLO11  | CNN-based         |
| RT-DETR | Transformer-based |

---

## 📈 Evaluation Criteria

* mAP
* Precision
* Recall
* F1-score
* IoU
* Computational Cost
* Inference Speed
* Model Size
* Qualitative Prediction
* Failure Case Analysis

---

## 📁 Dataset

The dataset is **not included** in this repository because of its large size.

Please download the PPE dataset in roboflow separately and organize it according to the required directory structure before training.

---

## 🚀 Running Experiments

```bash
pip install -r requirements.txt
```

Run training notebooks:

* YOLOv8_PPE_Final_Training_Notebook.ipynb
* YOLO26_PPE_Final_Training_Notebook.ipynb
* RTDETR_PPE_Final_Training_Notebook.ipynb

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

---

## 📸 Results

The repository includes:

* Benchmark comparison
* Qualitative predictions
* Failure case visualization
* Computational analysis

---

## ❌ Failure case image

![alt text](<Failure Case.png>)

---

## Result Table

![alt text](image.png)

---

## 🎯 Future Improvements

* YOLO12 or D-FINE comparison
* Grounding DINO
* OWLv2
* Edge deployment benchmarking
* TensorRT optimization
* ONNX inference

---

## 👨‍💻 Author

Pham Tan Minh 
Doan Duy Long
Nguyen Tran Hai Phong

Computer Vision • AI Engineering • Backend AI Systems

Linkedln: Minh Pham Tan 
Email: phamtanminh2004@gmail.com
