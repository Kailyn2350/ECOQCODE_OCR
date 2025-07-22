# ECOQCODE OCR Project

This project focuses on developing an Optical Character Recognition (OCR) system specifically designed to detect the presence of 'ECOQCODE' text within images. It utilizes a Convolutional Recurrent Neural Network (CRNN) for binary classification, determining whether an image contains the specified text or not.

## Features

-   **Synthetic Data Generation**: Generates a synthetic dataset of images with and without 'ECOQCODE' text, using various backgrounds and font sizes.
-   **CRNN Model**: Utilizes a Convolutional Recurrent Neural Network (CRNN) architecture designed to capture both spatial and sequential features for detecting the presence of specific text.
-   **Training and Validation**: Provides a complete pipeline for loading data, applying transformations, and training the CRNN model with binary labels (presence or absence of 'ECOQCODE').
-   **Model Evaluation**: Evaluates the model's performance using binary classification metrics such as accuracy, precision, recall, and F1-score.
-   **ONNX Export**: Supports exporting the trained CRNN model to ONNX format for deployment and inference using ONNX Runtime.


## Project Structure

```
ECOQCODE_OCR/
├── CRNN.ipynb               # Notebook for data generation, model training, and ONNX export (editable in VSCode or any .ipynb-compatible editor)
├── eco_dataset/            # Directory for generated images and labels
│   ├── images/             # Generated image files
│   ├── labels.txt          # All generated image paths and their corresponding labels
│   ├── train_labels.txt    # Training set labels
│   └── val_labels.txt      # Validation set labels
├── backgrounds/            # Directory for background images used in data generation
└── real_backgrounds/       # Placeholder for real-world background images (optional)
```

## Setup

### Prerequisites

-   Python 3.x
-   Visual Studio Code (with Python and Jupyter extensions enabled) or any compatible `.ipynb` editor
-   `arial.ttf` font file (or specify another font path in `CRNN.ipynb`)


### Installation

1.  **Clone the repository (if applicable):**
    ```bash
    git clone <repository_url>
    cd ECOQCODE_OCR
    ```

2.  **Install required Python packages:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Prepare background images:**
    Place various background images (e.g., `.jpg`, `.png`) into the `backgrounds/` directory. These will be used for synthetic data generation in `CRNN.ipynb`.



## Usage

All steps are contained within the `CRNN.ipynb` notebook file.

1.  **Open the notebook in VSCode:**
    Open `CRNN.ipynb` using Visual Studio Code with the Python and Jupyter extensions enabled.

2.  **Run cells sequentially:**
    Execute the cells from top to bottom in order.

    -   **Data Generation**: Generates synthetic images and their labels, saving them to `eco_dataset/images/` and `eco_dataset/labels.txt`.
    -   **Dataset and Model Definition**: Defines the custom dataset class, image transformations, and the CRNN model architecture.
    -   **Data Splitting**: Splits the generated labels into training and validation sets (`train_labels.txt` and `val_labels.txt`).
    -   **Training**: Trains the CRNN model using the labeled synthetic dataset.
    -   **Evaluation**: Evaluates model performance on the validation set and prints out binary classification metrics.
    -   **ONNX Export**: Exports the trained model to ONNX format and saves it as `ecoq_classifier.onnx` in the project root directory.


## ONNX Model Export

The trained `ECOQClassifier` CRNN model is exported to ONNX format (`ecoq_classifier.onnx`). This allows for deployment and inference across various platforms and runtimes that support ONNX, such as ONNX Runtime.

The export process includes:
-   **`dummy_input`**: A dummy tensor is created to trace the model's computation graph. It assumes an input image size of `(1, 3, 224, 224)` (batch size 1, 3 color channels, 224x224 pixels).
-   **`opset_version=11`**: Specifies the ONNX operator set version for compatibility.
-   **`dynamic_axes`**: Enables dynamic batch sizing so the ONNX model can accept variable batch sizes at inference time.

## Future Improvements

-   Integrate real-world background images from `real_backgrounds/` to enhance generalization.
-   Explore more advanced CRNN architectures or use pre-trained backbone encoders to improve accuracy.
-   Implement a dedicated Python inference script using the exported ONNX model.
-   Apply more diverse and robust data augmentation techniques during training.
