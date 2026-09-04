# ✍️ Handwritten Character Recognition using CNN

A Deep Learning project for recognizing handwritten digits using **Convolutional Neural Networks (CNNs)** and the **MNIST dataset**.

The model learns visual patterns from handwritten images and classifies them into one of **10 digit classes (0–9)**.

---

## 🚀 Project Overview

Handwritten Character Recognition is a computer vision problem where a machine learning model identifies handwritten symbols from images.

In this project, a **Convolutional Neural Network (CNN)** is trained on the MNIST handwritten digit dataset.

The complete workflow includes:

* Dataset loading
* Exploratory Data Analysis
* Image preprocessing
* Pixel normalization
* CNN architecture design
* Model training
* Model evaluation
* Classification report
* Confusion matrix
* Individual image prediction
* Model saving and loading

---

## 🎯 Objective

The main objective of this project is to build a deep learning model capable of recognizing handwritten digits from grayscale images.

### Input

A handwritten digit image of size:

```text
28 × 28 pixels
```

### Output

One of the following classes:

```text
0 1 2 3 4 5 6 7 8 9
```

---

## 📊 Dataset

This project uses the **MNIST Handwritten Digit Dataset**.

MNIST contains:

* **60,000 training images**
* **10,000 testing images**
* Image size: **28 × 28 pixels**
* Image type: **grayscale**
* Number of classes: **10**

Each image represents a handwritten digit from 0 to 9.

The dataset is loaded directly using TensorFlow/Keras:

```python
from tensorflow.keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()
```

---

## 🧠 Model Architecture

A Convolutional Neural Network is used for image classification.

### Architecture

```text
Input Image
    │
    ▼
28 × 28 × 1
    │
    ▼
Conv2D (32 filters, 3×3)
    │
    ▼
ReLU Activation
    │
    ▼
MaxPooling2D
    │
    ▼
Conv2D (64 filters, 3×3)
    │
    ▼
ReLU Activation
    │
    ▼
MaxPooling2D
    │
    ▼
Flatten
    │
    ▼
Dense Layer (128 neurons)
    │
    ▼
ReLU
    │
    ▼
Dense Layer (10 neurons)
    │
    ▼
Softmax
    │
    ▼
Predicted Digit
```

---

## 🔧 Technologies Used

| Technology                      | Purpose                        |
| ------------------------------- | ------------------------------ |
| Python                          | Programming Language           |
| TensorFlow                      | Deep Learning Framework        |
| Keras                           | Neural Network API             |
| NumPy                           | Numerical Computation          |
| Matplotlib                      | Data Visualization             |
| Seaborn                         | Confusion Matrix Visualization |
| Scikit-learn                    | Model Evaluation               |
| Jupyter Notebook / Google Colab | Development Environment        |

---

## ⚙️ Image Preprocessing

Before training, the images are normalized.

Original pixel values:

```text
0 – 255
```

are converted to:

```text
0 – 1
```

using:

```python
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0
```

The image dimensions are also reshaped to include the grayscale channel:

```python
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)
```

---

## 🏋️ Model Training

The CNN is compiled using the Adam optimizer and sparse categorical cross-entropy loss.

```python
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
```

The model is trained for multiple epochs:

```python
history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.1
)
```

---

## 📈 Model Evaluation

The trained model is evaluated on the unseen MNIST test dataset.

Evaluation metrics include:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

Example evaluation:

```python
test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("Test Accuracy:", test_accuracy)
```

> **Note:** Exact performance may vary slightly depending on training configuration and environment.

---

## 📋 Classification Report

The project generates a detailed classification report using Scikit-learn:

```python
from sklearn.metrics import classification_report

print(
    classification_report(
        y_test,
        y_pred
    )
)
```

This provides:

* Precision
* Recall
* F1-score
* Support

for every digit class.

---

## 🔥 Confusion Matrix

A confusion matrix is used to understand how accurately the model recognizes each digit.

```python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
```

It helps identify which digits are most commonly confused with each other.

---

## 🖼️ Prediction Example

The trained model can predict an individual handwritten digit:

```python
prediction = model.predict(
    image.reshape(1, 28, 28, 1)
)

predicted_digit = np.argmax(prediction)

print("Predicted Digit:", predicted_digit)
```

The predicted digit can then be compared with the actual label.

---

## 💾 Model Saving

The trained model is saved using Keras:

```python
model.save("handwritten_character_cnn.h5")
```

The saved model can later be loaded without retraining:

```python
from tensorflow.keras.models import load_model

model = load_model(
    "handwritten_character_cnn.h5"
)
```

---

## 📁 Project Structure

```text
handwritten-character-recognition/
│
├── handwritten_character_recognition.ipynb
│
├── handwritten_character_cnn.h5
│
├── requirements.txt
│
├── .gitignore
│
└── README.md
```

---

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/AbhishekNimaje435/handwritten-character-recognition.git
```

### 2. Navigate to the project

```bash
cd handwritten-character-recognition
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Open the notebook

```bash
jupyter notebook
```

Or open the notebook directly in **Google Colab**.

### 5. Run all cells

The MNIST dataset will automatically be downloaded through TensorFlow/Keras.

---

## 🔮 Future Improvements

This project can be extended beyond handwritten digit recognition.

### 🔤 EMNIST Character Recognition

The model can be trained on the **EMNIST dataset** to recognize handwritten alphabets and characters.

```text
A B C D E ... Z
a b c d e ... z
```

depending on the selected EMNIST split.

### 📝 Word Recognition

Multiple characters can be combined to recognize complete words.

```text
H → E → L → L → O
```

### 📄 Sentence Recognition

The project can further evolve into handwritten sentence recognition.

### 🧠 CRNN

A **Convolutional Recurrent Neural Network (CRNN)** with sequence modeling can be used for recognizing sequences of handwritten characters.

Possible architecture:

```text
Input Image
     ↓
CNN
     ↓
Feature Extraction
     ↓
RNN / LSTM / BiLSTM
     ↓
CTC
     ↓
Text Output
```

---

## 🌟 Key Learning Outcomes

Through this project, I learned:

* Image preprocessing
* Computer vision fundamentals
* CNN architecture
* Convolution and pooling
* Image normalization
* Deep learning model training
* Model evaluation
* Classification metrics
* Confusion matrix analysis
* Model saving and loading
* Handwritten digit classification

---

## 💡 Applications

Handwritten character recognition can be used in:

* 🏦 Bank cheque processing
* 📄 Digitizing handwritten documents
* 📚 Automated form processing
* 📝 OCR systems
* 📬 Postal address recognition
* 🏫 Educational applications
* 📱 Handwriting-based interfaces
* 🔢 Digit recognition systems

---

## 🛠️ Possible Production Extension

A future version can include a web interface where users draw a digit and receive a real-time prediction.

```text
User Draws Digit
       ↓
Canvas / Image Upload
       ↓
Image Preprocessing
       ↓
CNN Model
       ↓
Prediction
       ↓
Recognized Character
```

This can be deployed using **Streamlit**, **FastAPI**, or another web framework.

---

## 📌 Project Status

**Status:** ✅ Completed

Current version supports:

* MNIST handwritten digit recognition
* CNN-based classification
* Model evaluation
* Individual digit prediction
* Saved trained model

Future versions can support:

* EMNIST alphabet recognition
* Multi-character recognition
* Word recognition
* Sentence recognition
* CRNN-based sequence recognition
* Web deployment

---

## 👨‍💻 Author

**Abhishek Nimaje**

B.Tech – Artificial Intelligence & Data Science

Interested in:

* Data Science
* Machine Learning
* Deep Learning
* Generative AI
* Computer Vision

---

## ⭐ If You Like This Project

If you found this project useful, consider giving the repository a ⭐ on GitHub.

---

## 📜 License

This project is intended for educational and learning purposes.
