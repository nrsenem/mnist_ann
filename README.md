MNIST ANN Classification with Keras
This project demonstrates how to build, train, and evaluate a simple Artificial Neural Network (ANN) on the MNIST handwritten digits dataset using Keras and TensorFlow.

📌 Project Overview
Dataset: MNIST(digits 0–9)
Model: Fully connected neural network (ANN)
Goal: Classify 28x28 grayscale digit images into 10 classes (0–9)

⚙️ Model Architecture
Input Layer: 784 neurons (28×28 flattened image)
Hidden Layer 1: 512 neurons, ReLU activation
Hidden Layer 2: 256 neurons, Tanh activation
Output Layer: 10 neurons, Softmax activation

🚀 Training Setup
Optimizer: Adam
Loss Function: Categorical Crossentropy
Metrics: Accuracy
Epochs: 10
Batch Size: 60
Validation Split: 20

Callbacks:
EarlyStopping: Stops training when validation loss does not improve for 3 epochs.
ModelCheckpoint: Saves the best model based on validation loss.

📊 Results
Test Accuracy: ~97–98%
Test Loss: ~0.07


💾 Model Saving & Loading
The trained model is saved in HDF5 format:
  model.save("final_mnist_ann_model.h5")

It can be reloaded for future use:
  from keras.models import load_model
  loaded_model = load_model("final_mnist_ann_model.h5")

📂 Project Structure
.
├── ann.py                    # Main ANN script
├── final_mnist_ann_model.h5  # Final saved model
├── ann_best_model.h5         # Best model (via checkpoint)
└── README.md                 # Documentation

🛠️ Requirements
Install dependencies with:
pip install tensorflow keras matplotlib

👩‍💻 Author
Developed by Nursenem Zirek
📧 nrsenemzrk1121@gmail.com



📌 Proje Hakkında
Bu proje, Keras ve TensorFlow kullanarak MNIST el yazısı rakam veri seti üzerinde basit bir Yapay Sinir Ağı (ANN) kurmayı, eğitmeyi ve değerlendirmeyi göstermektedir.

Veri Seti: MNIST (0–9 rakamları)
Model: Tam bağlı yapay sinir ağı (ANN)
Amaç: 28x28 gri tonlamalı rakam görsellerini 10 sınıfa (0–9) ayırmak

⚙️ Model Mimarisi
Girdi Katmanı: 784 nöron (28×28 piksel, düzleştirilmiş)
Gizli Katman 1: 512 nöron, ReLU aktivasyon
Gizli Katman 2: 256 nöron, Tanh aktivasyon
Çıkış Katmanı: 10 nöron, Softmax aktivasyon

🚀 Eğitim Ayarları
Optimizer: Adam
Loss Fonksiyonu: Categorical Crossentropy
Metrik: Accuracy (doğruluk)
Epochs: 10
Batch Size: 60
Validation Split: %20

Callbacks:
EarlyStopping: Val_loss iyileşmediğinde eğitimi durdurur
ModelCheckpoint: En iyi modeli kaydeder

📊 Sonuçlar
Test Doğruluğu (Accuracy): ~%97–98
Test Kaybı (Loss): ~0.07

💾 Model Kaydetme & Yükleme
model.save("final_mnist_ann_model.h5")

from keras.models import load_model
loaded_model = load_model("final_mnist_ann_model.h5")

📂 Proje Yapısı
.
├── ann.py                    # ANN model kodu
├── final_mnist_ann_model.h5  # Eğitilmiş model
├── ann_best_model.h5         # En iyi model (checkpoint ile)
└── README.md                 # Açıklama

🛠️ Gereksinimler
pip install tensorflow keras matplotlib

👩‍💻 Yazar
Geliştiren: Nursenem Zirek
📧 nrsenemzrk1121@gmail.com
