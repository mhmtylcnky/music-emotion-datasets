import os
import numpy as np
import pickle
from featureExtraction import extract_features

# Veri klasörleri
train_dir = "Audio_files/train"
test_dir = "Audio_files/test"

def load_and_extract(directory):
    features, labels = [], []
    for label in os.listdir(directory):
        class_dir = os.path.join(directory, label)
        for file in os.listdir(class_dir):
            file_path = os.path.join(class_dir, file)
            feature = extract_features(file_path)
            if feature is not None:
                features.append(feature)
                labels.append(label)
    return np.array(features), np.array(labels)

# Eğitim ve test verileri için özellikleri çıkart
X_train, y_train = load_and_extract(train_dir)
X_test, y_test = load_and_extract(test_dir)

# Kaydet
with open("features.pkl", "wb") as f:
    pickle.dump((X_train, y_train, X_test, y_test), f)

print("Özellik çıkarımı tamamlandı ve features.pkl dosyasına kaydedildi.")
