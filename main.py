import os
import numpy as np
import pickle
from sklearn.metrics import classification_report
from featureExtraction import extract_features

# Dosya yolları
test_dir = "Audio_files/test"

def load_data(directory):
    features, labels = [], []
    for label in os.listdir(directory):
        class_dir = os.path.join(directory, label)
        for file_name in os.listdir(class_dir):
            file_path = os.path.join(class_dir, file_name)
            feature = extract_features(file_path)
            if feature is not None:
                features.append(feature)
                labels.append(label)
    return np.array(features), np.array(labels)

# Test verisini yükle
X_test, y_test = load_data(test_dir)

# Model dosyaları
model_files = [
    "random_forest.pkl",
    "logistic_regression.pkl",
    "naive_bayes.pkl",
    "svm.pkl",
    "adaboost.pkl",
    "ann.pkl",
    "decision_tree.pkl",
    "knn.pkl",
    "xgboost.pkl"
]

# Her modeli test et
for model_file in model_files:
    with open(model_file, "rb") as f:
        model, encoder = pickle.load(f)

    # Test etiketlerini encode et
    y_test_enc = encoder.transform(y_test)
    y_pred = model.predict(X_test)

    print(f"\n--- {model_file.replace('.pkl', '').replace('_', ' ').title()} ---")
    print("Accuracy:", np.mean(y_pred == y_test_enc))
    print(classification_report(y_test_enc, y_pred, target_names=encoder.classes_))
