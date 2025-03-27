import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

def load_and_explore_data(filepath):
    """
    Load the tennis statistics dataset and perform initial exploratory data analysis.
    
    Args:
        filepath (str): Path to the CSV file
    
    Returns:
        pandas.DataFrame: Loaded dataset
    """
    try:
        # Load the dataset
        df = pd.read_csv(filepath)
        
        # Basic dataset information
        print("Dataset Overview:")
        print("=" * 50)
        print(f"Total number of records: {len(df)}")
        print("\nColumn Information:")
        print(df.info())
        
        print("\nBasic Statistical Summary:")
        print(df.describe())
        
        return df
    
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None

def perform_exploratory_analysis(df):
    """
    Conduct comprehensive exploratory data analysis with visualizations.
    
    Args:
        df (pandas.DataFrame): Tennis statistics dataset
    """
    # Set up the plotting style
    plt.style.use('seaborn')
    
    # Correlation heatmap
    plt.figure(figsize=(16, 12))
    correlation_matrix = df.select_dtypes(include=[np.number]).corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt='.2f', square=True)
    plt.title('Correlation Heatmap of Tennis Performance Features')
    plt.tight_layout()
    plt.show()
    plt.close()
    
    # Pairplot for key performance features
    key_features = ['Aces', 'FirstServePointsWon', 'BreakPointsOpportunities', 
                    'ReturnGamesWon', 'Winnings', 'Wins']
    sns.pairplot(df[key_features], diag_kind='kde')
    plt.suptitle('Pairwise Relationships Between Key Tennis Performance Features', y=1.02)
    plt.show()
    plt.close()

def single_feature_regression(df, feature, target='Winnings'):
    """
    Perform single feature linear regression and evaluate performance.
    
    Args:
        df (pandas.DataFrame): Dataset
        feature (str): Feature column name
        target (str, optional): Target column name. Defaults to 'Winnings'.
    
    Returns:
        dict: Regression model performance metrics
    """
    # Prepare the data
    X = df[[feature]]
    y = df[target]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    # Predictions and evaluation
    y_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    # Visualization
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test, y_test, color='blue', alpha=0.7, label='Actual Data')
    plt.plot(X_test, y_pred, color='red', label='Regression Line')
    plt.xlabel(feature)
    plt.ylabel(target)
    plt.title(f'{feature} vs {target}: Linear Regression')
    plt.legend()
    plt.show()
    plt.close()
    
    return {
        'feature': feature,
        'r2_score': r2,
        'mean_squared_error': mse
    }

def multi_feature_regression(df, features, target='Winnings'):
    """
    Perform multi-feature linear regression and evaluate performance.
    
    Args:
        df (pandas.DataFrame): Dataset
        features (list): List of feature column names
        target (str, optional): Target column name. Defaults to 'Winnings'.
    
    Returns:
        dict: Regression model performance metrics
    """
    # Prepare the data
    X = df[features]
    y = df[target]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    # Predictions and evaluation
    y_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    return {
        'features': features,
        'r2_score': r2,
        'mean_squared_error': mse
    }

def main():
    # Load the dataset
    filepath = 'tennis_stats.csv'
    df = load_and_explore_data(filepath)
    
    if df is None:
        return
    
    # Perform exploratory analysis
    perform_exploratory_analysis(df)
    
    # Single feature regression tests
    single_feature_results = []
    single_features = [
        'Aces', 'FirstServePointsWon', 'BreakPointsOpportunities', 
        'ReturnGamesWon', 'ServiceGamesWon'
    ]
    
    for feature in single_features:
        result = single_feature_regression(df, feature)
        single_feature_results.append(result)
    
    # Print single feature regression results
    print("\nSingle Feature Regression Results:")
    for result in sorted(single_feature_results, key=lambda x: x['r2_score'], reverse=True):
        print(f"{result['feature']}: R² = {result['r2_score']:.4f}, MSE = {result['mean_squared_error']:.2f}")
    
    # Multi-feature regression tests
    multi_feature_tests = [
        ['BreakPointsOpportunities', 'FirstServePointsWon'],
        ['Aces', 'ServiceGamesWon'],
        ['FirstServePointsWon', 'ReturnGamesWon'],
        ['BreakPointsOpportunities', 'Aces', 'FirstServeReturnPointsWon'],
        ['ServiceGamesWon', 'ReturnGamesWon', 'TotalPointsWon'],
        ['FirstServe','FirstServePointsWon','FirstServeReturnPointsWon',
        'SecondServePointsWon','SecondServeReturnPointsWon','Aces',
        'BreakPointsConverted','BreakPointsFaced','BreakPointsOpportunities',
        'BreakPointsSaved','DoubleFaults','ReturnGamesPlayed','ReturnGamesWon',
        'ReturnPointsWon','ServiceGamesPlayed','ServiceGamesWon','TotalPointsWon',
        'TotalServicePointsWon']
    ]
    
    print("\nMulti-Feature Regression Results:")
    for features in multi_feature_tests:
        result = multi_feature_regression(df, features)
        print(f"{', '.join(features)}: R² = {result['r2_score']:.4f}, MSE = {result['mean_squared_error']:.2f}")

if __name__ == '__main__':
    main()