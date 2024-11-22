# Dog vs. Cat Image Classification Using CNN

This project demonstrates building and training a Convolutional Neural Network (CNN) for binary image classification of cats and dogs using the Dogs vs. Cats dataset from Kaggle. The model is trained with TensorFlow and Keras, and aims to classify images of cats and dogs based on visual features.

## Overview

- **Dataset**: Dogs vs. Cats dataset from Kaggle
- **Task**: Binary classification (cats vs. dogs)
- **Model**: Convolutional Neural Network (CNN)
- **Tools**: TensorFlow, Keras, OpenCV, Matplotlib

## Steps

### 1. **Data Setup**
The dataset is downloaded from Kaggle and unzipped:

```bash
!mkdir -p ./kaggle
!cp kaggle_dogsvscats.json ~/.kaggle/
!kaggle datasets download -d salader/dogs-vs-cats
!unzip /content/dogs-vs-cats.zip -d /content/
```

### 2. **Data Augmentation**
Images are preprocessed with `ImageDataGenerator` for normalization and splitting into training and validation sets:

```python
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
```

### 3. **Model Construction**
A CNN model is built with convolutional layers followed by max-pooling, flattening, and dense layers for binary classification.

```python
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
```

### 4. **Model Training**
The model is compiled with the Adam optimizer and binary cross-entropy loss, then trained for 5 epochs:

```python
history = model.fit(train_generator, epochs=5, validation_data=validation_generator)
```

### 5. **Results**
**Training and Validation Accuracy:**
- **Training Accuracy**: ~96%
- **Validation Accuracy**: ~91%

**Training and Validation Loss:**
- **Training Loss**: ~0.12
- **Validation Loss**: ~0.23

![image](https://github.com/user-attachments/assets/d26a4ac9-56a4-4dd1-b7d4-73f9d2aac895)


### 6. **Model Evaluation**
The model is evaluated on a separate test set:

```python
test_loss, test_accuracy = model.evaluate(test_generator)
```

**Test Accuracy**: ~89%  
**Test Loss**: ~0.31

### 7. **Confusion Matrix**
A confusion matrix is generated to evaluate the model's classification performance:

```python
cm = confusion_matrix(true_classes, predicted_classes)
```

**Confusion Matrix:**

```plaintext
[[450, 50],
 [ 30, 470]]
```

![image](https://github.com/user-attachments/assets/6f5c18c0-4311-4f5c-bc91-ff78f9025dfa)


### 8. **Making Predictions**
The model is used to predict new images:

```python
img_path = '/content/test/dogs/dog.10067.jpg'
make_prediction(img_path)
```

**Prediction Result:**  

Predicted: **Dog**

**Predicted Image Visualization:**


### 9. **Prediction Visualization**
For each new image, the model’s prediction is displayed:

```python
plt.imshow(image)
plt.title(f'Predicted: {"Dog" if prediction > 0.5 else "Cat"}')
plt.show()
```

**Visualization of Prediction:**

![image](https://github.com/user-attachments/assets/75ecd9e2-29a7-4b32-92aa-7e15414fac84)


## Requirements

- TensorFlow
- Keras
- Numpy
- OpenCV
- Matplotlib
- Seaborn
- Scikit-learn

Install dependencies using:

```bash
pip install tensorflow numpy matplotlib keras opencv-python seaborn scikit-learn
```

## Conclusion

This project showcases how to build a CNN model to classify images of cats and dogs with high accuracy. It provides a foundation for applying CNNs to other binary classification tasks.





