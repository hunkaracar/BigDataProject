import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Veri setini oku
df = pd.read_csv("C:\\Users\\Hunkar\\Desktop\\BigData\\ProjectIoTBigData\\RT_IOT2022.csv")

# Kategorik verileri sayısal yap
df_encoded = df.copy()
for col in df_encoded.columns:
    if df_encoded[col].dtype == 'object':
        df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col].astype(str))

# Hedef değişken
target_col = 'Attack_type'

# Özellik önemine göre seçilecek sütunlar (önceki görsele göre)
important_features = [
    'fwd_pkts_payload.avg', 'id.resp_p', 'fwd_pkts_payload.min', 'service',
    'fwd_subflow_bytes', 'fwd_pkts_payload.tot', 'fwd_pkts_payload.max',
    'flow_iat.min', 'Unnamed: 0', 'flow_iat.avg', 'flow_pkts_payload.avg',
    'flow_iat.max', 'active.min', 'fwd_iat.min', 'fwd_PSH_flag_count',
    'flow_pkts_payload.max', 'fwd_pkts_tot', 'active.max',
    'flow_pkts_payload.tot', 'fwd_URG_flag_count'
]

X = df_encoded[important_features]
y = df_encoded[target_col]

# Eğitim/test ayrımı
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Modeli oluştur
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Tahmin ve sonuç
y_pred = clf.predict(X_test)

# Performans ölçütleri
print("Sınıflandırma Raporu:")
print(classification_report(y_test, y_pred))

print("Karmaşıklık Matrisi:")
print(confusion_matrix(y_test, y_pred))
