import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.utils import shuffle
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Veri Kümesini Yükle
df = pd.read_csv('C:\\Users\\Hunkar\\Desktop\\BigData\\ProjectIoTBigData\\ML-EdgeIIoT-dataset.csv', low_memory=False)

# 2. Gereksiz veya çok eksik olan sütunları çıkar
df.drop(columns=["frame.time", "ip.src_host", "ip.dst_host", "arp.src.proto_ipv4", "arp.dst.proto_ipv4",
                 "http.file_data", "http.request.full_uri", "icmp.transmit_timestamp",
                 "http.request.uri.query", "tcp.options", "tcp.payload", "tcp.srcport", "tcp.dstport", 
                 "udp.port", "mqtt.msg"], inplace=True, errors='ignore')

# 3. Eksik verileri temizle
df.dropna(axis=0, how='any', inplace=True)

# 4. Tekrarlı verileri kaldır
df.drop_duplicates(inplace=True)

# 5. Veri karıştır
df = shuffle(df, random_state=42)

# 6. Etiket kolonunu ayır ve etiketleri encode et
y = df["Attack_type"]
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# atak tiplerini yazdır
print("\nAtak tipleri:")
print(label_encoder.classes_)
print("\n\n")

# 7. Özellikleri ayır
X = df.drop(columns=["Attack_type", "Attack_label"], errors='ignore')

# 8. Kategorik verileri one-hot encode et
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
X = pd.get_dummies(X, columns=categorical_cols)

# 9. Ölçekleme
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 10. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.3, stratify=y_encoded, random_state=42)

# 11. Modelleri Tanımla
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'KNN': KNeighborsClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
}

# 12. Model Eğitimi ve Değerlendirmesi
results = []

for name, model in tqdm(models.items(), desc='Modeller Eğitiliyor'):
    print(f"\nTraining: {name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'Recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'F1 Score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
    })

# 13. Sonuçları Göster
results_df = pd.DataFrame(results)
print("\n🔍 Model Karşılaştırma Sonuçları:\n")
print(results_df.sort_values(by='F1 Score', ascending=False).reset_index(drop=True))

# 14. Grafikle Göster
plt.figure(figsize=(12,6))
sns.barplot(x='Model', y='F1 Score', data=results_df.sort_values(by='F1 Score', ascending=False))
plt.title("Model Karşılaştırması (F1 Score)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
