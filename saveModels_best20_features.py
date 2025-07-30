import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Klasör oluştur
os.makedirs("models_best20", exist_ok=True)

# Feature isimleri (tam listeyi buraya ekle)
feature_names = [
    'MFCC1', 'MFCC2', 'MFCC3', 'MFCC4', 'MFCC5', 'MFCC6', 'MFCC7', 'MFCC8', 'MFCC9', 'MFCC10', 'MFCC11', 'MFCC12', 'MFCC13',  # MFCC'ler
    'Chroma1', 'Chroma2', 'Chroma3', 'Chroma4', 'Chroma5', 'Chroma6', 'Chroma7', 'Chroma8', 'Chroma9', 'Chroma10', 'Chroma11', 'Chroma12',  # Chroma
    'SpectralContrast1', 'SpectralContrast2', 'SpectralContrast3', 'SpectralContrast4', 'SpectralContrast5', 'SpectralContrast6', 'SpectralContrast7',  # Spectral Contrast
    'ZCR',  # Zero-Crossing Rate
    # 'MelSpectrogram1', 'MelSpectrogram2', 'MelSpectrogram3', 'MelSpectrogram4', 'MelSpectrogram5', 'MelSpectrogram6', 'MelSpectrogram7', 'MelSpectrogram8', 'MelSpectrogram9', 'MelSpectrogram10', 'MelSpectrogram11', 'MelSpectrogram12', 'MelSpectrogram13', 'MelSpectrogram14', 'MelSpectrogram15', 'MelSpectrogram16', 'MelSpectrogram17', 'MelSpectrogram18', 'MelSpectrogram19', 'MelSpectrogram20', 'MelSpectrogram21', 'MelSpectrogram22', 'MelSpectrogram23', 'MelSpectrogram24', 'MelSpectrogram25', 'MelSpectrogram26', 'MelSpectrogram27', 'MelSpectrogram28', 'MelSpectrogram29', 'MelSpectrogram30', 'MelSpectrogram31', 'MelSpectrogram32', 'MelSpectrogram33', 'MelSpectrogram34', 'MelSpectrogram35', 'MelSpectrogram36', 'MelSpectrogram37', 'MelSpectrogram38', 'MelSpectrogram39', 'MelSpectrogram40', 'MelSpectrogram41', 'MelSpectrogram42', 'MelSpectrogram43', 'MelSpectrogram44', 'MelSpectrogram45', 'MelSpectrogram46', 'MelSpectrogram47', 'MelSpectrogram48', 'MelSpectrogram49', 'MelSpectrogram50', 'MelSpectrogram51', 'MelSpectrogram52', 'MelSpectrogram53', 'MelSpectrogram54', 'MelSpectrogram55', 'MelSpectrogram56', 'MelSpectrogram57', 'MelSpectrogram58', 'MelSpectrogram59', 'MelSpectrogram60', 'MelSpectrogram61', 'MelSpectrogram62', 'MelSpectrogram63', 'MelSpectrogram64', 'MelSpectrogram65', 'MelSpectrogram66', 'MelSpectrogram67', 'MelSpectrogram68', 'MelSpectrogram69', 'MelSpectrogram70', 'MelSpectrogram71', 'MelSpectrogram72', 'MelSpectrogram73', 'MelSpectrogram74', 'MelSpectrogram75', 'MelSpectrogram76', 'MelSpectrogram77', 'MelSpectrogram78', 'MelSpectrogram79', 'MelSpectrogram80', 'MelSpectrogram81', 'MelSpectrogram82', 'MelSpectrogram83', 'MelSpectrogram84', 'MelSpectrogram85', 'MelSpectrogram86', 'MelSpectrogram87', 'MelSpectrogram88', 'MelSpectrogram89', 'MelSpectrogram90', 'MelSpectrogram91', 'MelSpectrogram92', 'MelSpectrogram93', 'MelSpectrogram94', 'MelSpectrogram95', 'MelSpectrogram96', 'MelSpectrogram97', 'MelSpectrogram98', 'MelSpectrogram99', 'MelSpectrogram100', 'MelSpectrogram101', 'MelSpectrogram102', 'MelSpectrogram103', 'MelSpectrogram104', 'MelSpectrogram105', 'MelSpectrogram106', 'MelSpectrogram107', 'MelSpectrogram108', 'MelSpectrogram109', 'MelSpectrogram110', 'MelSpectrogram111', 'MelSpectrogram112', 'MelSpectrogram113', 'MelSpectrogram114', 'MelSpectrogram115', 'MelSpectrogram116', 'MelSpectrogram117', 'MelSpectrogram118', 'MelSpectrogram119', 'MelSpectrogram120', 'MelSpectrogram121', 'MelSpectrogram122', 'MelSpectrogram123', 'MelSpectrogram124', 'MelSpectrogram125', 'MelSpectrogram126', 'MelSpectrogram127', 'MelSpectrogram128',  # Mel Spectrogram
    'RMSE',  # RMSE
    'SpectralRollOff'  # Spectral Roll-off
]


# Özellikleri yükle
with open("features/features.pkl", "rb") as f:
    X_train, y_train, X_test, y_test = pickle.load(f)

# Etiketleri encode et
encoder = LabelEncoder()
y_train_enc = encoder.fit_transform(y_train)
y_test_enc = encoder.transform(y_test)  # test etiketleri de encode edildi

# Modeller ve dosya isimleri
model_files = {
    "random_forest.pkl": RandomForestClassifier(n_estimators=100, random_state=42),
    "logistic_regression.pkl": LogisticRegression(max_iter=1000),
    "naive_bayes.pkl": GaussianNB(),
    "svm.pkl": SVC(probability=True),
    "adaboost.pkl": AdaBoostClassifier(n_estimators=100, random_state=42),
    "ann.pkl": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42),
    "decision_tree.pkl": DecisionTreeClassifier(random_state=42),
    "knn.pkl": KNeighborsClassifier(n_neighbors=5),
    "xgboost.pkl": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
}

# Test doğruluklarını tutacak liste
accuracy_results = []

# Her model için
for filename, model in model_files.items():
    print(f"\n{filename} için best 20 özellik alınıyor ve model eğitiliyor...")

    fi_path = f"feature_importance_results/{filename.replace('.pkl', '')}_feature_importance.csv"
    if not os.path.exists(fi_path):
        print(f"{fi_path} bulunamadı, atlanıyor.")
        continue

    importance_df = pd.read_csv(fi_path)
    best_20_features = importance_df["Feature"].head(20).tolist()

    try:
        indices = [feature_names.index(f) for f in best_20_features]
    except ValueError as e:
        print(f"Özellik bulunamadı hatası: {e}")
        continue

    X_train_selected = X_train[:, indices]
    X_test_selected = X_test[:, indices]

    model.fit(X_train_selected, y_train_enc)

    # Tahmin ve doğruluk
    y_pred = model.predict(X_test_selected)
    acc = accuracy_score(y_test_enc, y_pred)
    print(f"{filename} test doğruluğu: {acc:.4f}")

    accuracy_results.append({
        "Model": filename.replace(".pkl", ""),
        "Test Accuracy": acc
    })

    # Model kaydet
    save_path = f"models_best20/{filename}"
    with open(save_path, "wb") as f:
        pickle.dump((model, encoder), f)
    print(f"{filename} başarıyla kaydedildi.")

# Sonuçları CSV’ye kaydet
results_df = pd.DataFrame(accuracy_results)
results_df.to_csv("models_best20/accuracy_report.csv", index=False)
print("\nTest doğrulukları 'models_best20/accuracy_report.csv' dosyasına kaydedildi.")
