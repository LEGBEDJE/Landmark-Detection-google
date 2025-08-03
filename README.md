# Landmark Image Detection with Deep Learning 🏛️

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.5%2B-FF6F00.svg)
![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)
![Keras](https://img.shields.io/badge/Keras-2.5%2B-red.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

This project is a deep learning application that identifies famous landmarks from images. It leverages **transfer learning** with a pre-trained **MobileNetV2** model to achieve high accuracy without needing to train a massive network from scratch. This repository is designed as a portfolio piece to showcase skills in computer vision, deep learning, and model deployment.

## 📖 Table of Contents
- [Project Overview](#project-overview)
- [Methodology: Transfer Learning](#methodology-transfer-learning)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [🚀 How to Run This Project](#-how-to-run-this-project)
- [📈 Evaluation and Performance](#-evaluation-and-performance)
- [🛠️ Technologies Used](#️-technologies-used)
- [📄 License](#-license)

## Project Overview

The goal is to build a model capable of classifying images of famous landmarks from around the world. This is a challenging computer vision task that requires a sophisticated model to discern subtle differences between various architectural styles and natural formations.

## Methodology: Transfer Learning

Training a deep convolutional neural network (CNN) from scratch requires a massive amount of data and computational power. Instead, we use **transfer learning**. This technique involves:

1.  **Using a Pre-trained Model:** We start with **MobileNetV2**, a state-of-the-art model pre-trained on the ImageNet dataset (which contains millions of images across 1,000 categories). This model has already learned a rich hierarchy of features (edges, textures, shapes) from a wide variety of images.

2.  **Freezing the Base:** We "freeze" the weights of the early layers of MobileNetV2. This preserves the general-purpose feature extraction capabilities it has already learned.

3.  **Adding a Custom Head:** We remove the original top classification layer of MobileNetV2 and add our own custom layers. This new "head" will be trained specifically to classify our landmark categories.

4.  **Fine-Tuning (Optional but recommended):** After an initial training phase, we can unfreeze some of the later layers of the base model and continue training with a very low learning rate. This allows the model to fine-tune its existing feature extractors for the specifics of our landmark images.

This approach is highly efficient and leads to excellent performance even with a smaller dataset.

## Dataset

This project is designed to work with a curated subset of the Google Landmarks Dataset. A good starting point is the dataset from the Kaggle competition "Landmark Recognition 2020", which is more manageable than the full Google Landmarks dataset.

**You will need to download the data and place it in the `data/` directory.** A recommended dataset can be found here: [Kaggle Landmark Recognition 2020](https://www.kaggle.com/c/landmark-recognition-2020/data)

*Note: You may need to perform some preprocessing to structure the data into class-specific folders if it isn't already.* 

## Project Structure

```
Landmark-Detection/
├── data/
│   └── (Image data, organized by class)
├── notebooks/
│   └── Landmark_Detection.ipynb
├── saved_model/
│   └── (The final trained model will be saved here)
├── src/
│   └── (For helper scripts, e.g., predict.py)
├── .gitignore
├── README.md
└── requirements.txt
```

## 🚀 How to Run This Project

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/LEGBEDJE/Landmark-Detection-google.git
    cd Landmark-Detection
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download and Prepare the Dataset:**
    - Download the image dataset from the link provided above (or use your own).
    - Organize the images into subdirectories within the `data/` folder. Each subdirectory should be named after a landmark class (e.g., `data/eiffel_tower`, `data/golden_gate_bridge`).

5.  **Launch Jupyter and run the notebook:**
    ```bash
    jupyter lab
    ```
    - Open `notebooks/Landmark_Detection.ipynb` and execute the cells to train the model.

## 📈 Evaluation and Performance

The model's performance is evaluated using standard classification metrics:
- **Accuracy:** The percentage of correctly classified images.
- **Confusion Matrix:** To visualize which classes are being confused with each other.
- **Classification Report:** Providing precision, recall, and F1-score for each landmark class.

The notebook will contain visualizations of the training process (accuracy and loss curves) to diagnose overfitting and ensure the model is learning effectively.

## 🛠️ Technologies Used

- **TensorFlow & Keras**: For building and training the deep learning model.
- **Python**: The core programming language.
- **Pandas & NumPy**: For data manipulation.
- **Matplotlib & Seaborn**: For data visualization.
- **Pillow**: For image processing.
- **Jupyter Notebook**: For interactive development.

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
