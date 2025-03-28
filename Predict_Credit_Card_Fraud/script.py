# Credit Card Fraud Detection Project

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

def load_and_explore_data(filepath):
    """
    Load the dataset and perform initial exploration.
    
    Args:
        filepath (str): Path to the CSV file
    
    Returns:
        pandas.DataFrame: Loaded transactions dataframe
    """
    # Load the dataset
    transactions = pd.read_csv(filepath)
    
    # Print basic information about the dataset
    print("Dataset Overview:")
    print(transactions.head())
    print("\nDataset Information:")
    print(transactions.info())
    
    # Analyze fraudulent transactions
    fraud_counts = transactions['isFraud'].value_counts()
    fraud_percentage = fraud_counts / len(transactions) * 100
    print("\nFraudulent Transactions:")
    print(fraud_counts)
    print(f"Percentage of Fraudulent Transactions: {fraud_percentage[1]:.2f}%")
    
    return transactions

def preprocess_data(transactions):
    """
    Preprocess the transaction data by creating new features.
    
    Args:
        transactions (pandas.DataFrame): Input dataframe
    
    Returns:
        tuple: Features and label arrays
    """
    # Create new feature columns
    transactions['isPayment'] = transactions['type'].apply(
        lambda type_val: 1 if type_val in ['PAYMENT', 'DEBIT'] else 0
    )
    
    transactions['isMovement'] = transactions['type'].apply(
        lambda type_val: 1 if type_val in ['CASH_OUT', 'TRANSFER'] else 0
    )
    
    transactions['accountDiff'] = abs(
        transactions['oldbalanceOrg'] - transactions['oldbalanceDest']
    )
    
    # Select features and label
    features = transactions[['amount', 'isPayment', 'isMovement', 'accountDiff']]
    label = transactions['isFraud']
    
    return features, label

def train_and_evaluate_model(features, label):
    """
    Split data, train logistic regression model, and evaluate performance.
    
    Args:
        features (pandas.DataFrame): Input features
        label (pandas.Series): Target variable
    
    Returns:
        tuple: Trained model, scaler, and performance metrics
    """
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        features, label, test_size=0.3, random_state=42
    )
    
    # Normalize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the logistic regression model
    model = LogisticRegression(random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate model performance
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    
    print("\nModel Performance:")
    print(f"Training Accuracy: {train_score:.4f}")
    print(f"Test Accuracy: {test_score:.4f}")
    
    # Print feature importance
    feature_importance = pd.DataFrame({
        'feature': features.columns,
        'importance': np.abs(model.coef_[0])
    }).sort_values('importance', ascending=False)
    print("\nFeature Importance:")
    print(feature_importance)
    
    # Detailed classification report
    y_pred = model.predict(X_test_scaled)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return model, scaler

def predict_new_transactions(model, scaler):
    """
    Predict fraud for new sample transactions.
    
    Args:
        model (LogisticRegression): Trained logistic regression model
        scaler (StandardScaler): Fitted standard scaler
    """
    # New transaction data
    transaction1 = np.array([123456.78, 0.0, 1.0, 54670.1])
    transaction2 = np.array([98765.43, 1.0, 0.0, 8524.75])
    transaction3 = np.array([543678.31, 1.0, 0.0, 510025.5])
    
    # Your custom transaction
    your_transaction = np.array([1543000.77, 0., 1., 20522.9])
    
    # Combine transactions
    sample_transactions = np.vstack([
        transaction1, transaction2, transaction3, your_transaction
    ])
    
    # Normalize new transactions
    sample_transactions_scaled = scaler.transform(sample_transactions)
    
    # Predict fraud
    predictions = model.predict(sample_transactions_scaled)
    probabilities = model.predict_proba(sample_transactions_scaled)
    
    print("\nNew Transaction Predictions:")
    for i, (pred, prob) in enumerate(zip(predictions, probabilities), 1):
        print(f"Transaction {i}:")
        print(f"  Fraud Prediction: {'Fraudulent' if pred else 'Not Fraudulent'}")
        print(f"  Probabilities (Not Fraud, Fraud): {prob}")

def visualize_feature_distributions(transactions):
    """
    Create visualizations to understand feature distributions.
    
    Args:
        transactions (pandas.DataFrame): Input dataframe
    """
    plt.figure(figsize=(12, 6))
    
    # Histogram of transaction amounts
    plt.subplot(1, 2, 1)
    transactions[transactions['isFraud'] == 0]['amount'].hist(
        alpha=0.5, bins=50, label='Non-Fraudulent'
    )
    transactions[transactions['isFraud'] == 1]['amount'].hist(
        alpha=0.5, bins=50, label='Fraudulent'
    )
    plt.title('Transaction Amount Distribution')
    plt.xlabel('Amount')
    plt.ylabel('Frequency')
    plt.legend()
    
    # Boxplot of account differences
    plt.subplot(1, 2, 2)
    sns.boxplot(x='isFraud', y='accountDiff', data=transactions)
    plt.title('Account Difference by Fraud Status')
    
    plt.tight_layout()
    plt.show()

def main():
    """
    Main function to orchestrate the fraud detection workflow.
    """
    # Load and explore the dataset
    transactions = load_and_explore_data('transactions_modified.csv')
    #transactions = load_and_explore_data('transactions.csv')
    
    # Visualize feature distributions
    visualize_feature_distributions(transactions)
    
    # Preprocess data
    features, label = preprocess_data(transactions)
    
    # Train and evaluate the model
    model, scaler = train_and_evaluate_model(features, label)
    
    # Predict on new transactions
    predict_new_transactions(model, scaler)

if __name__ == '__main__':
    main()