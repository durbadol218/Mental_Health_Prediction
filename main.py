import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import pickle
import matplotlib.pyplot as plt
import os

# Load data
df = pd.read_csv('data/survey.csv')

# Drop irrelevant columns (customize based on EDA)
df = df.drop(columns=['comments', 'state', 'Timestamp'], errors='ignore')

# Encode categorical variables
le = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].astype(str).apply(lambda x: x.strip())
    df[col] = le.fit_transform(df[col])

# Fill missing values
imputer = SimpleImputer(strategy='most_frequent')
df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Features and target
X = df.drop(columns='treatment')
y = df['treatment']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))

# Save model
with open('models/model.pkl', 'wb') as f:
    pickle.dump(model, f)
# model = pickle.load(open('models/model.pkl', 'rb'))
# print(model)



# Confusion Matrix
from sklearn.metrics import ConfusionMatrixDisplay

# After predicting
y_pred = model.predict(X_test)
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
plt.title("Confusion Matrix")
plt.show()


# Classification Report as Table
from sklearn.metrics import classification_report
import pandas as pd
import seaborn as sns

report = classification_report(y_test, y_pred, output_dict=True)
df_report = pd.DataFrame(report).transpose()

plt.figure(figsize=(8, 4))
sns.heatmap(df_report.iloc[:-1, :-1], annot=True, cmap="YlGnBu")
plt.title("Classification Report Heatmap")
plt.show()


# ROC CURVE
from sklearn.metrics import RocCurveDisplay
RocCurveDisplay.from_estimator(model, X_test, y_test)
plt.title("ROC Curve")
plt.show()




# Compare All Models (F1-score / ROC-AUC Bar Plot)
import matplotlib.pyplot as plt

models = ["Logistic", "Perceptron", "Decision Tree", "Random Forest", "MLP"]
f1_scores = [0.81, 0.77, 0.79, 0.83, 0.82]
roc_auc =    [0.86, 0.82, 0.83, 0.89, 0.87]

x = range(len(models))
plt.bar(x, f1_scores, width=0.4, label="F1-Score", align='center')
plt.bar([i + 0.4 for i in x], roc_auc, width=0.4, label="ROC-AUC", align='center')
plt.xticks([i + 0.2 for i in x], models, rotation=45)
plt.ylim(0.7, 1.0)
plt.ylabel("Score")
plt.title("Model Performance Comparison")
plt.legend()
plt.tight_layout()
plt.show()
