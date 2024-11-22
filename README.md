# Binary Image Classification: Cats vs Dogs Using CNNs

## 📌 Project Overview
This project focuses on building a **Convolutional Neural Network (CNN)** to classify images of cats and dogs. Using the Kaggle "Dogs vs Cats" dataset, the goal is to develop a binary classification model capable of distinguishing between the two classes. The project involves preprocessing data, building a deep learning model, training and evaluation, and testing with real-world inference.

---

## 📂 Dataset
The dataset is sourced from Kaggle and consists of labeled images of cats and dogs.

- **Training Set**: Contains images of cats and dogs.
- **Validation Set**: Split from the training set (20% of the total data) for performance monitoring.
- **Test Set**: Unseen data used to evaluate the model's generalization.

### Dataset Directory Structure
The dataset is organized into the following directory structure:

```
/dogs_vs_cats
    /train
        /cats
        /dogs
    /test
        /cats
        /dogs
```

---

## 🛠️ Project Workflow

### 1. Data Preprocessing
- **Resizing**: All images are resized to \(150 \times 150\) pixels.
- **Normalization**: Pixel values are rescaled between \(0\) and \(1\) for efficient training.
- **Data Splitting**: The training set is split into training (80%) and validation (20%) subsets using `ImageDataGenerator`.

---

### 2. Model Architecture
The model is built using the **Keras Sequential API**, consisting of:

- **Convolutional Layers**:
  - Extract spatial features with filters.
  - Use `ReLU` activation for non-linearity.
- **Pooling Layers**:
  - Downsample feature maps using MaxPooling.
- **Fully Connected Layers**:
  - Flatten the feature maps.
  - Dense layers with `ReLU` activation.
  - Sigmoid activation at the output layer for binary classification.

**Model Summary:**

![image](https://github.com/user-attachments/assets/ccb53927-f2f7-4f00-9f20-ae1f33f10d6a)

### 3. Model Training
- **Loss Function**: Binary Cross-Entropy
- **Optimizer**: Adam
- **Metrics**: Accuracy

The model is trained for **5 epochs** using `train_generator` and validated using `validation_generator`.

---

### 4. Evaluation
- Performance metrics like accuracy and loss are visualized using plots.
- Confusion matrix and classification reports provide insights into prediction accuracy for each class.

---

### 5. Testing and Inference
The model is tested on an unseen dataset to evaluate its generalization capability. A utility function enables single-image prediction, displaying the image and its predicted class.

---

## 🔍 Results
- **Training Accuracy**: ~98%
- **Validation Accuracy**: ~95%
- **Test Accuracy**: ~94%
- Confusion matrix and classification report are generated to analyze the predictions.

### Performance Plots:
![image](https://github.com/user-attachments/assets/ff32953a-d2c1-44da-84e3-6332a0c1c000)

![Accuracy Plot](link_to_accuracy_plot.png)
![Loss Plot](link_to_loss_plot.png)

---

## 🧑‍💻 Prerequisites
### Libraries:
- Python 3.x
- TensorFlow
- Keras
- OpenCV
- NumPy
- Matplotlib
- Seaborn

### Installation:
Install the required libraries using:
```bash
pip install tensorflow numpy matplotlib keras opencv-python seaborn
```

---

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/dogs-vs-cats-classification.git
   cd dogs-vs-cats-classification
   ```

2. Download the dataset from [Kaggle](https://www.kaggle.com/salader/dogs-vs-cats) and place it in the project directory.

3. Run the training script:
   ```bash
   python train.py
   ```

4. For single image prediction:
   ```bash
   python predict.py --image_path /path/to/image.jpg
   ```

---

## 📊 Confusion Matrix
![Confusion Matrix](link_to_confusion_matrix.png)

---

## 🔗 Future Improvements
- Implement transfer learning using pre-trained models (e.g., VGG16, ResNet).
- Add advanced data augmentation techniques for better generalization.
- Tune hyperparameters using tools like Keras Tuner.
- Experiment with multi-class classification by adding more categories.

---

## 👤 Author
- **Name**: [Your Name]
- **Contact**: [Your Email]
- **LinkedIn**: [Your LinkedIn Profile](https://linkedin.com/in/your-profile)

Feel free to reach out for any questions or collaboration opportunities! 😊

---

Let me know if you'd like to customize this further or if you'd like me to generate specific visuals like confusion matrix and plots for you!