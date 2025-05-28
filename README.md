# Oral Cancer Detection
This project focuses on detecting histopathologic oral cancer using Convolutional Neural Networks (CNNs).

## Features
- Trained a deep learning model using histopathology images
- Achieved high accuracy in classifying cancerous vs. non-cancerous images
- Visualized performance with loss and accuracy plots

## Dataset
The dataset consists of 4946 histopathology images of oral cancer. It includes both positive (cancerous) and negative (non-cancerous) samples.

## Model
- Architecture: Custom CNN with multiple convolutional and pooling layers
- Optimizer: AdamW
- Loss Function: Categorical Crossentropy
- Metrics: Accuracy

## How to Run
1. Clone this repository
2. Install dependencies:
    ```bash
    pip install tensorflow matplotlib numpy seaborn cv2 
    ```
3. Run the notebook:
    ```bash
    jupyter notebook kan oraldataset.ipynb
    ```

## Results
- Accuracy: 92.45%
- Model format: `.keras`

## Authors
- Yash Karnani
