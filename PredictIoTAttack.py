import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
from collections import Counter



df = pd.read_csv('ML-EdgeIIoT-dataset.csv')
print(df.head(10))

print("DataFrame Information:")
print("-" * 30)
print('DF Shape: ' ,df.shape)
print('number of Columns: ' ,len(df.columns))
print('number of Observations: ' ,len(df))
print('Number of values in df: ' , df.count().sum())
print('Total Number of Missing values in df: ' , df.isna().sum().sum())
print('percentage of Missing values : ' ,  "{:.2f}".format(df.isna().sum().sum()/df.count().sum() *100),'%')
print('Total Number of Duplicated records in df : ' , df.duplicated().sum().sum())
print('percentage of Duplicated values : ' ,  "{:.2f}".format(df.duplicated().sum().sum()/df.count().sum() *100),'%')


# Original attack type names before encoding
original_attack_types = df['Attack_type'].unique()
print("\nOrijinal saldırı türleri:")
print(original_attack_types)

pd.set_option('display.max_columns', None)
print(df.info())

print(df.nunique())
print("\n")
print(df.describe())
print("\n")
print(df.describe(include='O'))

df.hist(bins = 25, color='steelblue', edgecolor='black', grid=False , figsize = (10, 10) )
# Add titles and labels
plt.suptitle('Histograms of Data', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust spacing between subplots
plt.xlabel('Value', fontsize=12)
plt.ylabel('Frequency', fontsize=12)

# Modify tick font size
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# Remove outer spines
for spine in plt.gca().spines.values():
    spine.set_visible(False)

plt.show()

fig = px.pie(df, names='Attack_type', title='Distribution of Attack Type')
fig.show()


fig = px.pie(df, names='Attack_type', title='Distribution of Attack Type')
fig.show()


for i in df.columns:
    print("Column Name:" ,i   ," " ,df[i].nunique() , " " ,df[i].dtype)

numerical_col = []
categorical_col = []

for i in df.columns:
    if df[i].nunique() == 1: 
        df.drop(i , axis =1 , inplace = True)
        print("dropped column : ", i)
    
    elif i != 'Attack_label' or i != 'Attack_type':
        if df[i].dtype == object or df[i].nunique() <= 20:
            categorical_col.append(i)
        else:
            numerical_col.append(i)

traget_col = ['Attack_label', 'Attack_type']
print(traget_col) 
print(categorical_col)
print(numerical_col)


# High 0 values count can make model biased
zero_percent_columns = []

for col in df.columns:
    try:
        zero_ratio = (df[col] == 0).sum() / len(df)
        if zero_ratio >= 0.50:
            zero_percent_columns.append((col, zero_ratio))
            df.drop(col , axis=1 , inplace=True)
    except:
        # Skip non-numeric or problematic columns
        continue

# Display the results
for col, ratio in zero_percent_columns:
    print(f"{col}: {ratio:.2%} zeros")

print("number of dropped columns: " ,len(zero_percent_columns))
print("New DF Shape: " ,df.shape)

df.drop_duplicates()

drop_columns = [
    "frame.time",            # Date and time (high cardinality, not useful for modeling directly)
    "ip.src_host",           # Source Host (character string, high cardinality, likely unique or noisy)
    "ip.dst_host",           # Destination Host (same reason as above)
    "arp.dst.proto_ipv4",    # Target IP address (IPv4, high cardinality)
    "arp.src.proto_ipv4",    # Sender IP address (IPv4, high cardinality)
    "http.file_data",        # File Data (raw content, unstructured string/bytes)
    "http.request.method",   # HTTP method (likely has little variation; use only if modeling request types)
    "http.request.full_uri", # Full request URI (very high cardinality string, likely unique per request)
    "http.request.version",  # HTTP version (may have little variance and minimal predictive value)
    "tcp.dstport",           # Destination Port (label type, likely high cardinality or redundant with `tcp.srcport`)
    "tcp.options",           # TCP Options (sequence of bytes, not directly useful without custom parsing)
    "tcp.payload",           # TCP Payload (raw bytes, high dimensionality and unstructured)
    "tcp.srcport"            # Source Port (unsigned int, possibly high cardinality, often randomized)
]

df.drop(drop_columns, axis=1, inplace=True)
print(df.shape)

"""
# Then, Here We will pass the categorical columns to the LabelEncoder and the numerical columns to the StandardScaler.
# indeed , the LabelEncoder will convert the categorical columns into numerical values, and the StandardScaler will standardize the numerical columns.
"""

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import xgboost as xgb
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore")

le = LabelEncoder()
df["Attack_type"] = le.fit_transform(df["Attack_type"])
print(df)
print("\n")

y_label = df.pop("Attack_label")
y_type = df.pop('Attack_type')

# sınıf sayısı
print("Number of classes in y_label: ", len(set(y_label)))
print("Number of classes in y_type: ", len(set(y_type)))
print("Number of classes in y_label: ", y_label.nunique())
# y_type isimleri DDOS, DoS, PortScan, etc. 
print("y_type isimleri: ", y_type.unique())


sc = StandardScaler()
df = sc.fit_transform(df)

X_train, X_test, y_train_label, y_test_label = train_test_split(df, y_label, test_size=0.2, random_state=42)
print("X_train shape: " ,X_train.shape)

# 1. Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train_label)
rf_pred = rf.predict(X_test)
print(f"Random Forest Accuracy: {accuracy_score(y_test_label, rf_pred):.4f}")

# 2. Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train_label)
lr_pred = lr.predict(X_test)
print(f"Logistic Regression Accuracy: {accuracy_score(y_test_label, lr_pred):.4f}")

# 3. Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train_label)
dt_pred = dt.predict(X_test)
print(f"Decision Tree Accuracy: {accuracy_score(y_test_label, dt_pred):.4f}")

# 4. XGBoost
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train_label)
xgb_pred = xgb_model.predict(X_test)
print(f"XGBoost Accuracy: {accuracy_score(y_test_label, xgb_pred):.4f}")

# 5. LightGBM
lgb_model = lgb.LGBMClassifier(verbosity=-1)
lgb_model.fit(X_train, y_train_label)
lgb_pred = lgb_model.predict(X_test)
print(f"LightGBM Accuracy: {accuracy_score(y_test_label, lgb_pred):.4f}")

# 6. Gradient Boosting (from sklearn)
gb = GradientBoostingClassifier(random_state=42)
gb.fit(X_train, y_train_label)
gb_pred = gb.predict(X_test)
print(f"Gradient Boosting (sklearn) Accuracy: {accuracy_score(y_test_label, gb_pred):.4f}")

# Store model names and predictions
models = {
    'Random Forest': rf_pred,
    'Logistic Regression': lr_pred,
    'Decision Tree': dt_pred,
    'XGBoost': xgb_pred,
    'LightGBM': lgb_pred,
    'Gradient Boosting': gb_pred
}

# Initialize list to hold results
results = []

# Calculate metrics
for model_name, y_pred in models.items():
    acc = accuracy_score(y_test_label, y_pred)
    f1 = f1_score(y_test_label, y_pred)
    precision = precision_score(y_test_label, y_pred)
    recall = recall_score(y_test_label, y_pred)
    
    results.append({
        'Model': model_name,
        'Accuracy': acc,
        'F1 Score': f1,
        'Precision': precision,
        'Recall': recall
    })

# Convert to DataFrame
results_df = pd.DataFrame(results)

# Print the table
print(results_df.sort_values(by='Accuracy', ascending=False).reset_index(drop=True))

# Create confusion matrix
cm = confusion_matrix(y_test_label, lgb_pred)

# Plot confusion matrix as a heatmap
plt.figure(figsize=(4, 2))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", cbar=False)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()

# Predicting Attack Types
X_train, X_test, y_train_type, y_test_type = train_test_split(df, y_type, test_size=0.2, random_state=42)
print(y_train_type.value_counts())

# 1. Random Forest
rf = RandomForestClassifier(criterion = 'entropy', max_depth = None, min_samples_leaf = 2, min_samples_split = 10)
rf.fit(X_train, y_train_type)
rf_pred = rf.predict(X_test)
print(f"Random Forest Accuracy: {accuracy_score(y_test_type, rf_pred):.4f}")


# 3. Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train_type)
dt_pred = dt.predict(X_test)
print(f"Decision Tree Accuracy: {accuracy_score(y_test_type, dt_pred):.4f}")

# 4. XGBoost
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train_type)
xgb_pred = xgb_model.predict(X_test)
print(f"XGBoost Accuracy: {accuracy_score(y_test_type, xgb_pred):.4f}")

# 5. LightGBM
lgb_model = lgb.LGBMClassifier(objective='multiclass',num_class=15, verbosity=-1)
lgb_model.fit(X_train, y_train_type)
lgb_pred = lgb_model.predict(X_test)
print(f"LightGBM Accuracy: {accuracy_score(y_test_type, lgb_pred):.4f}")

# Store model names and predictions
models = {
    'Random Forest': rf_pred,
    'Decision Tree': dt_pred,
    'XGBoost': xgb_pred,
    'LightGBM': lgb_pred,
}

# Initialize list to hold results
results = []

# Calculate metrics
for model_name, y_pred in models.items():
    acc = accuracy_score(y_test_type, y_pred)
    f1 = f1_score(y_test_type, y_pred , average='micro')
    precision = precision_score(y_test_type, y_pred ,average='micro')
    recall = recall_score(y_test_type, y_pred, average='micro')

    
    results.append({
        'Model': model_name,
        'Accuracy': acc,
        'F1 Score': f1,
        'Precision': precision,
        'Recall': recall
    })

# Convert to DataFrame
results_df = pd.DataFrame(results)

# Print the table
print(results_df.sort_values(by='Accuracy', ascending=False).reset_index(drop=True))

lgb_model1 = lgb.LGBMClassifier(verbosity=-1)
lgb_model1.fit(X_train, y_train_label)
lgb_pred = lgb_model1.predict(X_test)
print(f"LightGBM Accuracy: {accuracy_score(y_test_label, lgb_pred):.4f}")

lgb_model2 = lgb.LGBMClassifier(objective='multiclass',num_class=15, verbosity=-1)
lgb_model2.fit(X_train, y_train_type)
lgb_pred = lgb_model2.predict(X_test)
print(f"LightGBM Accuracy: {accuracy_score(y_test_type, lgb_pred):.4f}")



import pickle

# Save model
with open('AttackCLF.pkl', 'wb') as f:
    pickle.dump(lgb_model1, f)


with open('AttackTypeCLF.pkl', 'wb') as f:
    pickle.dump(lgb_model2, f)

# Save label encoder
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

# Save scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(sc, f)


# Load model 1
with open('AttackCLF.pkl', 'rb') as f:
    lgb_model1 = pickle.load(f)

# Load model 2
with open('AttackTypeCLF.pkl', 'rb') as f:
    lgb_model2 = pickle.load(f)

# Load label encoder
with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

# Load scaler
with open('scaler.pkl', 'rb') as f:
    sc = pickle.load(f)