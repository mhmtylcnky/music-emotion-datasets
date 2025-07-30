import os
import numpy as np
import pickle
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt
import seaborn as sns

# Klasörü oluştur
results_dir = "feature_importance_results"
os.makedirs(results_dir, exist_ok=True)

# Özellik isimleri
feature_names = [
    'MFCC1', 'MFCC2', 'MFCC3', 'MFCC4', 'MFCC5', 'MFCC6', 'MFCC7', 'MFCC8', 'MFCC9', 'MFCC10', 'MFCC11', 'MFCC12', 'MFCC13',  # MFCC'ler
    'Chroma1', 'Chroma2', 'Chroma3', 'Chroma4', 'Chroma5', 'Chroma6', 'Chroma7', 'Chroma8', 'Chroma9', 'Chroma10', 'Chroma11', 'Chroma12',  # Chroma
    'SpectralContrast1', 'SpectralContrast2', 'SpectralContrast3', 'SpectralContrast4', 'SpectralContrast5', 'SpectralContrast6', 'SpectralContrast7',  # Spectral Contrast
    'ZCR',  # Zero-Crossing Rate
    'MelSpectrogram1', 'MelSpectrogram2', 'MelSpectrogram3', 'MelSpectrogram4', 'MelSpectrogram5', 'MelSpectrogram6', 'MelSpectrogram7', 'MelSpectrogram8', 'MelSpectrogram9', 'MelSpectrogram10', 'MelSpectrogram11', 'MelSpectrogram12', 'MelSpectrogram13', 'MelSpectrogram14', 'MelSpectrogram15', 'MelSpectrogram16', 'MelSpectrogram17', 'MelSpectrogram18', 'MelSpectrogram19', 'MelSpectrogram20', 'MelSpectrogram21', 'MelSpectrogram22', 'MelSpectrogram23', 'MelSpectrogram24', 'MelSpectrogram25', 'MelSpectrogram26', 'MelSpectrogram27', 'MelSpectrogram28', 'MelSpectrogram29', 'MelSpectrogram30', 'MelSpectrogram31', 'MelSpectrogram32', 'MelSpectrogram33', 'MelSpectrogram34', 'MelSpectrogram35', 'MelSpectrogram36', 'MelSpectrogram37', 'MelSpectrogram38', 'MelSpectrogram39', 'MelSpectrogram40', 'MelSpectrogram41', 'MelSpectrogram42', 'MelSpectrogram43', 'MelSpectrogram44', 'MelSpectrogram45', 'MelSpectrogram46', 'MelSpectrogram47', 'MelSpectrogram48', 'MelSpectrogram49', 'MelSpectrogram50', 'MelSpectrogram51', 'MelSpectrogram52', 'MelSpectrogram53', 'MelSpectrogram54', 'MelSpectrogram55', 'MelSpectrogram56', 'MelSpectrogram57', 'MelSpectrogram58', 'MelSpectrogram59', 'MelSpectrogram60', 'MelSpectrogram61', 'MelSpectrogram62', 'MelSpectrogram63', 'MelSpectrogram64', 'MelSpectrogram65', 'MelSpectrogram66', 'MelSpectrogram67', 'MelSpectrogram68', 'MelSpectrogram69', 'MelSpectrogram70', 'MelSpectrogram71', 'MelSpectrogram72', 'MelSpectrogram73', 'MelSpectrogram74', 'MelSpectrogram75', 'MelSpectrogram76', 'MelSpectrogram77', 'MelSpectrogram78', 'MelSpectrogram79', 'MelSpectrogram80', 'MelSpectrogram81', 'MelSpectrogram82', 'MelSpectrogram83', 'MelSpectrogram84', 'MelSpectrogram85', 'MelSpectrogram86', 'MelSpectrogram87', 'MelSpectrogram88', 'MelSpectrogram89', 'MelSpectrogram90', 'MelSpectrogram91', 'MelSpectrogram92', 'MelSpectrogram93', 'MelSpectrogram94', 'MelSpectrogram95', 'MelSpectrogram96', 'MelSpectrogram97', 'MelSpectrogram98', 'MelSpectrogram99', 'MelSpectrogram100', 'MelSpectrogram101', 'MelSpectrogram102', 'MelSpectrogram103', 'MelSpectrogram104', 'MelSpectrogram105', 'MelSpectrogram106', 'MelSpectrogram107', 'MelSpectrogram108', 'MelSpectrogram109', 'MelSpectrogram110', 'MelSpectrogram111', 'MelSpectrogram112', 'MelSpectrogram113', 'MelSpectrogram114', 'MelSpectrogram115', 'MelSpectrogram116', 'MelSpectrogram117', 'MelSpectrogram118', 'MelSpectrogram119', 'MelSpectrogram120', 'MelSpectrogram121', 'MelSpectrogram122', 'MelSpectrogram123', 'MelSpectrogram124', 'MelSpectrogram125', 'MelSpectrogram126', 'MelSpectrogram127', 'MelSpectrogram128',  # Mel Spectrogram
    'RMSE',  # RMSE
    'SpectralRollOff'  # Spectral Roll-off
]

# Veri yükle
with open("features/features.pkl", "rb") as f:
    X_train, y_train, X_test, y_test = pickle.load(f)

# Model dosyaları
model_files = {
    "RandomForest": "models/random_forest.pkl",
    "LogisticRegression": "models/logistic_regression.pkl",
    "NaiveBayes": "models/naive_bayes.pkl",
    "SVM": "models/svm.pkl",
    "AdaBoost": "models/adaboost.pkl",
    "ANN": "models/ann.pkl",
    "DecisionTree": "models/decision_tree.pkl",
    "KNN": "models/knn.pkl",
    "XGBoost": "models/xgboost.pkl"
}

# Özellik önemi hesaplama
def calculate_importance(model, model_name):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        method = "Built-in"
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_).flatten()
        method = "Coefficient"
    else:
        print(f"{model_name}: mutual_info_classif yöntemi kullanılıyor...")
        importances = mutual_info_classif(X_train, y_train, discrete_features=False)
        method = "Mutual Info"

    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    return importance_df, method

# Grafik çizme
def plot_top_features(importance_df, model_name, method_used, top_n=20):
    top_features = importance_df.head(top_n)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=top_features, palette='viridis')
    plt.title(f"{model_name} – Top {top_n} Features ({method_used})")
    plt.xlabel("Önem Skoru")
    plt.ylabel("Özellik")
    plt.tight_layout()
    
    # Grafik kaydı
    plt.savefig(os.path.join(results_dir, f"{model_name}_top{top_n}_features.png"))
    plt.close()

# Ana döngü: tüm modeller için
for model_name, file_name in model_files.items():
    try:
        with open(file_name, 'rb') as f:
            model, _ = pickle.load(f)

        importance_df, method_used = calculate_importance(model, model_name)

        # CSV çıktısı
        csv_path = os.path.join(results_dir, f"{model_name}_feature_importance.csv")
        importance_df.to_csv(csv_path, index=False)

        # Grafik çıktısı
        plot_top_features(importance_df, model_name, method_used, top_n=20)

        print(f"{model_name} için sonuçlar '{results_dir}/' klasörüne kaydedildi.")

    except Exception as e:
        print(f"{model_name} için hata oluştu: {e}")
