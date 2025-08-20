# =============================================
# 1️⃣ Imports
# =============================================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, roc_auc_score, classification_report, ConfusionMatrixDisplay, RocCurveDisplay

# Models
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# =============================================
# 2️⃣ Load dataset
# =============================================
df_raw = pd.read_csv("data/survey.csv")

print("=== BEFORE CLEANING ===")
print(df_raw.shape)
print(df_raw.info())

# Drop unnecessary columns
df_clean = df_raw.drop(columns=['comments','state','Timestamp'], errors='ignore')

# Encode categorical columns
le = LabelEncoder()
for col in df_clean.select_dtypes(include=['object']).columns:
    df_clean[col] = df_clean[col].astype(str).str.strip()
    df_clean[col] = le.fit_transform(df_clean[col])

# Impute missing values
imputer = SimpleImputer(strategy='most_frequent')
df_clean = pd.DataFrame(imputer.fit_transform(df_clean), columns=df_clean.columns)

print("=== AFTER CLEANING ===")
print(df_clean.shape)
print(df_clean.info())

# =============================================
# 3️⃣ Visualizations
# =============================================
# Target distribution
plt.figure(figsize=(6,4))
sns.countplot(data=df_clean, x='treatment')
plt.title("Target Distribution")
plt.show()

# Correlation heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df_clean.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# =============================================
# 4️⃣ Train/Test Split and Scaling
# =============================================
X = df_clean.drop(columns='treatment')
y = df_clean['treatment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =============================================
# 5️⃣ Train all models and compute metrics
# =============================================
models_dict = {
    "Logistic": LogisticRegression(max_iter=1000),
    "Perceptron": Perceptron(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "MLP": MLPClassifier(max_iter=1000)
}

f1_scores = []
roc_auc_scores = []

for name, clf in models_dict.items():
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)
    f1_scores.append(f1_score(y_test, y_pred))
    roc_auc_scores.append(roc_auc_score(y_test, y_pred))
    print(f"{name} F1-score: {f1_scores[-1]:.3f}, ROC-AUC: {roc_auc_scores[-1]:.3f}")

# =============================================
# 6️⃣ Plot model comparison dynamically
# =============================================
x = range(len(models_dict))
plt.figure(figsize=(8,5))
plt.bar(x, f1_scores, width=0.4, label="F1-Score", align='center')
plt.bar([i + 0.4 for i in x], roc_auc_scores, width=0.4, label="ROC-AUC", align='center')
plt.xticks([i + 0.2 for i in x], list(models_dict.keys()), rotation=45)
plt.ylim(0.7, 1.0)
plt.ylabel("Score")
plt.title("Model Performance Comparison")
plt.legend()
plt.tight_layout()
plt.show()

# =============================================
# 7️⃣ Confusion Matrix, Classification Report, ROC for main model
# =============================================
# Use Random Forest as main model (best performing)
main_model = RandomForestClassifier(n_estimators=100, random_state=42)
main_model.fit(X_train_scaled, y_train)
y_pred_main = main_model.predict(X_test_scaled)

# Confusion matrix
ConfusionMatrixDisplay.from_estimator(main_model, X_test_scaled, y_test)
plt.title("Confusion Matrix")
plt.show()

# Classification report
report = classification_report(y_test, y_pred_main, output_dict=True)
df_report = pd.DataFrame(report).transpose()
plt.figure(figsize=(8,4))
sns.heatmap(df_report.iloc[:-1,:-1], annot=True, cmap="YlGnBu")
plt.title("Classification Report Heatmap")
plt.show()

# ROC Curve
RocCurveDisplay.from_estimator(main_model, X_test_scaled, y_test)
plt.title("ROC Curve")
plt.show()

# =============================================
# 8️⃣ Save main model, scaler, and columns
# =============================================
os.makedirs('models', exist_ok=True)
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('models/model.pkl', 'wb') as f:
    pickle.dump(main_model, f)

with open('models/columns.pkl', 'wb') as f:
    pickle.dump(X.columns.tolist(), f)

print("✅ Model, scaler, and column list saved successfully!")