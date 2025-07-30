import os
import numpy as np
import pickle
from sklearn.metrics import classification_report
from featureDefini import extract_features

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
    "models/random_forest.pkl",
    "models/logistic_regression.pkl",
    "models/naive_bayes.pkl",
    "models/svm.pkl",
    "models/adaboost.pkl",
    "models/ann.pkl",
    "models/decision_tree.pkl",
    "models/knn.pkl",
    "models/xgboost.pkl"
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
