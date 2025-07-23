feature_names = [
    'MFCC1', 'MFCC2', 'MFCC3', 'MFCC4', 'MFCC5', 'MFCC6', 'MFCC7', 'MFCC8', 'MFCC9', 'MFCC10', 'MFCC11', 'MFCC12', 'MFCC13',  # MFCC'ler
    'Chroma1', 'Chroma2', 'Chroma3', 'Chroma4', 'Chroma5', 'Chroma6', 'Chroma7', 'Chroma8', 'Chroma9', 'Chroma10', 'Chroma11', 'Chroma12',  # Chroma
    'SpectralContrast1', 'SpectralContrast2', 'SpectralContrast3', 'SpectralContrast4', 'SpectralContrast5', 'SpectralContrast6', 'SpectralContrast7',  # Spectral Contrast
    'ZCR',  # Zero-Crossing Rate
    'MelSpectrogram1', 'MelSpectrogram2', 'MelSpectrogram3', 'MelSpectrogram4', 'MelSpectrogram5', 'MelSpectrogram6', 'MelSpectrogram7', 'MelSpectrogram8', 'MelSpectrogram9', 'MelSpectrogram10', 'MelSpectrogram11', 'MelSpectrogram12', 'MelSpectrogram13', 'MelSpectrogram14', 'MelSpectrogram15', 'MelSpectrogram16', 'MelSpectrogram17', 'MelSpectrogram18', 'MelSpectrogram19', 'MelSpectrogram20', 'MelSpectrogram21', 'MelSpectrogram22', 'MelSpectrogram23', 'MelSpectrogram24', 'MelSpectrogram25', 'MelSpectrogram26', 'MelSpectrogram27', 'MelSpectrogram28', 'MelSpectrogram29', 'MelSpectrogram30', 'MelSpectrogram31', 'MelSpectrogram32', 'MelSpectrogram33', 'MelSpectrogram34', 'MelSpectrogram35', 'MelSpectrogram36', 'MelSpectrogram37', 'MelSpectrogram38', 'MelSpectrogram39', 'MelSpectrogram40', 'MelSpectrogram41', 'MelSpectrogram42', 'MelSpectrogram43', 'MelSpectrogram44', 'MelSpectrogram45', 'MelSpectrogram46', 'MelSpectrogram47', 'MelSpectrogram48', 'MelSpectrogram49', 'MelSpectrogram50', 'MelSpectrogram51', 'MelSpectrogram52', 'MelSpectrogram53', 'MelSpectrogram54', 'MelSpectrogram55', 'MelSpectrogram56', 'MelSpectrogram57', 'MelSpectrogram58', 'MelSpectrogram59', 'MelSpectrogram60', 'MelSpectrogram61', 'MelSpectrogram62', 'MelSpectrogram63', 'MelSpectrogram64', 'MelSpectrogram65', 'MelSpectrogram66', 'MelSpectrogram67', 'MelSpectrogram68', 'MelSpectrogram69', 'MelSpectrogram70', 'MelSpectrogram71', 'MelSpectrogram72', 'MelSpectrogram73', 'MelSpectrogram74', 'MelSpectrogram75', 'MelSpectrogram76', 'MelSpectrogram77', 'MelSpectrogram78', 'MelSpectrogram79', 'MelSpectrogram80', 'MelSpectrogram81', 'MelSpectrogram82', 'MelSpectrogram83', 'MelSpectrogram84', 'MelSpectrogram85', 'MelSpectrogram86', 'MelSpectrogram87', 'MelSpectrogram88', 'MelSpectrogram89', 'MelSpectrogram90', 'MelSpectrogram91', 'MelSpectrogram92', 'MelSpectrogram93', 'MelSpectrogram94', 'MelSpectrogram95', 'MelSpectrogram96', 'MelSpectrogram97', 'MelSpectrogram98', 'MelSpectrogram99', 'MelSpectrogram100', 'MelSpectrogram101', 'MelSpectrogram102', 'MelSpectrogram103', 'MelSpectrogram104', 'MelSpectrogram105', 'MelSpectrogram106', 'MelSpectrogram107', 'MelSpectrogram108', 'MelSpectrogram109', 'MelSpectrogram110', 'MelSpectrogram111', 'MelSpectrogram112', 'MelSpectrogram113', 'MelSpectrogram114', 'MelSpectrogram115', 'MelSpectrogram116', 'MelSpectrogram117', 'MelSpectrogram118', 'MelSpectrogram119', 'MelSpectrogram120', 'MelSpectrogram121', 'MelSpectrogram122', 'MelSpectrogram123', 'MelSpectrogram124', 'MelSpectrogram125', 'MelSpectrogram126', 'MelSpectrogram127', 'MelSpectrogram128',  # Mel Spectrogram
    'RMSE',  # RMSE
    'SpectralRollOff'  # Spectral Roll-off
]

import pickle
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import mutual_info_classif


with open("features.pkl", "rb") as f:
    X_train, y_train, X_test, y_test = pickle.load(f)

# Model dosyaları
model_files = {
    "RandomForest": "random_forest.pkl",
    "LogisticRegression": "logistic_regression.pkl",
    "NaiveBayes": "naive_bayes.pkl",
    "SVM": "svm.pkl",
    "AdaBoost": "adaboost.pkl",
    "ANN": "ann.pkl",
    "DecisionTree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "XGBoost": "xgboost.pkl"
}

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

# İşlem
for model_name, file_name in model_files.items():
    try:
        with open(file_name, 'rb') as f:
            model, _ = pickle.load(f)

        print(f"\n--- {model_name} ---")
        importance_df, method_used = calculate_importance(model, model_name)
        print(f"Yöntem: {method_used}")
        print(importance_df.head(10))

        importance_df.to_csv(f"{model_name}_feature_importance.csv", index=False)

    except Exception as e:
        print(f"{model_name} için hata oluştu: {e}")