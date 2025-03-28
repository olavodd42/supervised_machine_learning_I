# Income Classification using Logistic Regression

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Column names for the dataset
col_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 
             'marital-status', 'occupation', 'relationship', 'race', 'sex',
             'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']

# Load the dataset
df = pd.read_csv('adult.data', header=None, names=col_names)

# Clean columns by stripping extra whitespace for columns of type "object"
for c in df.select_dtypes(include=['object']).columns:
    df[c] = df[c].str.strip()

# 1. Check Class Imbalance
print("Class Distribution:")
print(df['income'].value_counts(normalize=True))

# 2. Create feature dataframe X with feature columns and dummy variables for categorical features
feature_cols = ['age', 'capital-gain', 'capital-loss', 'hours-per-week', 'sex', 'race', 'education']
X = pd.get_dummies(df[feature_cols], drop_first=True)

# 3. Create a heatmap of X data to see feature correlation
plt.figure(figsize=(10, 8))
sns.heatmap(X.corr(), annot=True, cmap='coolwarm', cbar=True)
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.show()
plt.close()

# 4. Scale the features and create binary output variable
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Create binary output variable
y = [1 if income == '>50K' else 0 for income in df['income']]

# 5. Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=1)

# Fit Logistic Regression model
log_reg = LogisticRegression(C=0.05, penalty='l1', solver='liblinear')
log_reg.fit(X_train, y_train)

# Make predictions
y_pred = log_reg.predict(X_test)

# 6. Print model parameters
print("\nModel Parameters:")
print("Intercept:", log_reg.intercept_)
print("\nCoefficients:")
for feature, coef in zip(X.columns, log_reg.coef_[0]):
    print(f"{feature}: {coef}")

# 7. Evaluate model predictions
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nAccuracy Score:")
print(accuracy_score(y_test, y_pred))

# 8. Create DataFrame of model coefficients
coefs = pd.DataFrame({
    'feature': X.columns,
    'coefficient': log_reg.coef_[0]
})
# Remove zero coefficients and sort
coefs = coefs[coefs['coefficient'] != 0].sort_values("coefficient")
print("\nNon-zero Coefficients:")
print(coefs)

# 9. Barplot of coefficients
plt.figure(figsize=(10, 6))
sns.barplot(x='coefficient', y='feature', data=coefs)
plt.title('Feature Impact on Income')
plt.xlabel('Coefficient')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()
plt.close()

# 10. Plot ROC curve
from sklearn.metrics import roc_curve, roc_auc_score

y_pred_prob = log_reg.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = roc_auc_score(y_test, y_pred_prob)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
plt.close()

print(f"\nROC AUC Score: {roc_auc:.4f}")