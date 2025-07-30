import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, RocCurveDisplay
from sklearn.preprocessing import label_binarize
from featureDefini import extract_features

# Test verisini yükleme
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

# Test verisi
test_dir = "Audio_files/test"
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

# ROC eğrisi çizimi
plt.figure(figsize=(12, 8))

# AUC değerlerini tutmak için
auc_scores = {}

for model_file in model_files:
    with open(model_file, "rb") as f:
        model, encoder = pickle.load(f)

    y_test_enc = encoder.transform(y_test)
    y_test_bin = label_binarize(y_test_enc, classes=[0, 1])  # binary
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)
    else:
        continue  # ROC için skor veremeyen model

    fpr, tpr, _ = roc_curve(y_test_bin, y_score)
    roc_auc = auc(fpr, tpr)
    auc_scores[model_file] = roc_auc
    plt.plot(fpr, tpr, label=f"{model_file.replace('.pkl', '').title()} (AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Eğrileri ve AUC Değerleri")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()

plt_path = "roc_auc_graph.png"
plt.savefig(plt_path)
plt_path
