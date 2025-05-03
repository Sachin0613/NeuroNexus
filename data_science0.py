# STEP 1: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# STEP 2: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

from imblearn.over_sampling import SMOTE

# STEP 3: Load Dataset
data_path = '/content/drive/MyDrive/credit card dataset/creditcard.csv'
df = pd.read_csv(data_path)

# (Optional) Use 10% of data to speed up for testing
df = df.sample(frac=0.1, random_state=42)

# STEP 4: Explore and Preprocess
print("Dataset shape:", df.shape)
print("Class distribution:\n", df['Class'].value_counts())
sns.countplot(x='Class', data=df)
plt.title("Class Distribution")
plt.show()

# Scale the 'Amount' column
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df[['Amount']])
df = df.drop(['Time'], axis=1)

X = df.drop('Class', axis=1)
y = df['Class']

# STEP 5: SMOTE for Class Imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# STEP 6: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# STEP 7: Train Models
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
log_preds = log_model.predict(X_test)

rf_model = RandomForestClassifier(n_estimators=10, random_state=42)  # Use fewer trees for faster training
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

# STEP 8: Evaluation Function
def evaluate_model(y_test, y_pred, model_name):
    print(f"\n=== {model_name} ===")
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

evaluate_model(y_test, log_preds, "Logistic Regression")
evaluate_model(y_test, rf_preds, "Random Forest")

# STEP 9: ROC Curves
def plot_roc(model, X_test, y_test, model_name):
    y_probs = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    auc_score = roc_auc_score(y_test, y_probs)
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc_score:.2f})")

plot_roc(log_model, X_test, y_test, "Logistic Regression")
plot_roc(rf_model, X_test, y_test, "Random Forest")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()


