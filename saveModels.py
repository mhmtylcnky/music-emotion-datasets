import os
import pickle
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier  # scikit-learn'in ANN modeli
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

# verileri yükle
with open("features/features.pkl", "rb") as f:
    X_train, y_train, X_test, y_test = pickle.load(f)

#Etiketleri sayısal değerlere dönüştür
encoder = LabelEncoder()
y_train_enc = encoder.fit_transform(y_train)

# "models/" klasörü yoksa oluştur
os.makedirs("models", exist_ok=True)

# Sınıflandırıcılar
models = {
    "random_forest.pkl": RandomForestClassifier(n_estimators=100, random_state=42),
    "logistic_regression.pkl": LogisticRegression(max_iter=1000),
    "naive_bayes.pkl": GaussianNB(),
    "svm.pkl": SVC(probability=True),
    "adaboost.pkl": AdaBoostClassifier(n_estimators=100, random_state=42),
    "ann.pkl": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42),  # ANN modeli
    "decision_tree.pkl": DecisionTreeClassifier(random_state=42),
    "knn.pkl": KNeighborsClassifier(n_neighbors=5),
    "xgboost.pkl": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
}

# Her bir modeli eğitip kaydet
for filename, model in models.items():
    print(f"{filename} eğitiliyor...")
    model.fit(X_train, y_train_enc)

    model_path = os.path.join("models", filename)
    with open(model_path, "wb") as f:
        pickle.dump((model, encoder), f)

    print(f"{filename} başarıyla 'models/' klasörüne kaydedildi.")
