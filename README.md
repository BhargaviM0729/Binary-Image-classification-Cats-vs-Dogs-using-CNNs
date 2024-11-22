# Dog vs. Cat Image Classification Using CNN

This project demonstrates how to build and train a **Convolutional Neural Network (CNN)** for **binary image classification** of cats and dogs using TensorFlow and Keras. The dataset used is the popular **Dogs vs. Cats** dataset from Kaggle, and the goal is to create a model capable of distinguishing between images of dogs and cats.

## Project Overview

In this project, we perform the following steps:

- **Data preprocessing**: Load and preprocess the dataset.
- **Model construction**: Build a CNN for binary classification.
- **Model training**: Train the model with augmented data.
- **Model evaluation**: Evaluate the model on test data.
- **Prediction and visualization**: Make predictions on new images and visualize the results.

## Steps in the Project

### 1. **Environment Setup and Data Preparation**

Before starting with model development, we download the dataset from Kaggle and unzip it for use in training:


!mkdir -p ./kaggle  # Create directory for Kaggle API credentials
!cp kaggle_dogsvscats.json ~/.kaggle/  # Copy Kaggle API credentials to the correct directory
!kaggle datasets download -d salader/dogs-vs-cats  # Download the Dogs vs Cats dataset from Kaggle
!unzip /content/dogs-vs-cats.zip -d /content/  # Unzip the dataset
```

- **Dataset Path**: Defines paths for images of cats and dogs to easily access them during training.

### 2. **Data Exploration and Preprocessing**

We explore the dataset and load images for further processing:

```python
cat_directory_path = '/content/dogs_vs_cats/train/cats'
dogs_directory_path = '/content/dogs_vs_cats/train/dogs'
```

- This step ensures that the images are properly categorized for binary classification (cats vs. dogs).

### 3. **Data Augmentation and Generator Setup**

Data augmentation is applied using the `ImageDataGenerator` to improve model generalization:

```python
train_datagen = ImageDataGenerator(rescale=1./225, validation_split=0.2)
```

- **`rescale`**: Normalizes pixel values to range `[0, 1]` by dividing by 255.
- **`validation_split=0.2`**: Uses 20% of the dataset for validation.

We create the data generators for training and validation:

```python
train_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary',
    subset='training'
)
```

- **Data generators** ensure that the data is loaded in batches and augmented in real time during training.

### 4. **Building the CNN Model**

The model consists of several convolutional layers for feature extraction, followed by max-pooling layers, flattening, and dense layers for classification:

```python
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
```

- **Conv2D**: Convolutional layers to extract features from the images.
- **MaxPooling2D**: Reduces the spatial dimensions of the image.
- **Dense Layers**: Fully connected layers that perform the classification.

### 5. **Compiling the Model**

We compile the model using the **Adam optimizer** and **binary crossentropy loss** for binary classification:

```python
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```

- **Binary Cross-Entropy**: Used for binary classification tasks.
- **Accuracy**: The model's performance is evaluated based on classification accuracy.

### 6. **Model Training**

We train the model on the augmented data:

```python
history = model.fit(train_generator, epochs=5, validation_data=validation_generator)
```

- **Epochs**: The number of times the model will see the entire dataset.
- **Validation Data**: Used to evaluate the model's performance during training.

### 7. **Visualization of Training Progress**

Training and validation accuracy, along with loss curves, are plotted to monitor the model’s learning process:

```python
plt.figure(figsize=(20,5))
plt.subplot(1,2,1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1,2,2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
```

### 8. **Model Evaluation and Testing**

The trained model is evaluated on a test dataset:

```python
test_loss, test_accuracy = model.evaluate(test_generator, steps=len(test_generator))
```

- **Evaluate**: Calculates the loss and accuracy of the model on the test set.

### 9. **Confusion Matrix**

The confusion matrix is generated to analyze the model’s classification performance:

```python
cm = confusion_matrix(true_classes, predicted_classes)
```

- The confusion matrix shows the true vs. predicted class counts.

### 10. **Inference and Prediction**

We use the trained model to make predictions on individual images:

```python
img_path = '/content/test/dogs/dog.10067.jpg'
make_prediction(img_path)
```

- **Prediction**: For each image, the model outputs whether it’s more likely to be a dog or a cat.

The result is visualized by displaying the image and the predicted label.

## Requirements

- TensorFlow
- Keras
- Numpy
- OpenCV
- Matplotlib
- Seaborn
- Scikit-learn

Install the required libraries using:

```bash
pip install tensorflow numpy matplotlib keras opencv-python seaborn scikit-learn
```

## Conclusion

This project demonstrates the end-to-end process of building and training a **Convolutional Neural Network (CNN)** for **binary image classification**. The model is able to classify images of dogs and cats with high accuracy using data augmentation, model evaluation, and visualization techniques. The code provides a good foundation for building other image classification models and experimenting with more complex datasets.




