import os
import numpy as np
import pickle
from sklearn.metrics import classification_report
from featureDefini import extract_features

# Dosya yolları
test_dir = "Audio_files/test"

# Sonuçların kaydedileceği klasör ve dosya
results_dir = "models"
os.makedirs(results_dir, exist_ok=True)
results_file = os.path.join(results_dir, "all_models_results.txt")

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

with open(results_file, "w") as f:
    for model_file in model_files:
        with open(model_file, "rb") as model_f:
            model, encoder = pickle.load(model_f)

        y_test_enc = encoder.transform(y_test)
        y_pred = model.predict(X_test)

        accuracy = np.mean(y_pred == y_test_enc)
        report = classification_report(y_test_enc, y_pred, target_names=encoder.classes_)

        model_name = os.path.splitext(os.path.basename(model_file))[0]

        header = f"\n--- {model_name.replace('_', ' ').title()} ---\n"
        summary = f"Accuracy: {accuracy:.4f}\n"

        print(header)
        print(summary)
        print(report)

        # Dosyaya yaz
        f.write(header)
        f.write(summary)
        f.write(report)
        f.write("\n" + "="*60 + "\n")

print(f"\nTüm modellerin sonuçları '{results_file}' dosyasına kaydedildi.")
