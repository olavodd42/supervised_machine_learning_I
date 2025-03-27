import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats

class HoneyProductionAnalysis:
    def __init__(self, csv_path):
        """
        Initialize the analysis with data loading and preprocessing
        
        Args:
            csv_path (str): Path to the CSV file containing honey production data
        """
        try:
            # Load the dataset with error handling
            self.df = pd.read_csv(csv_path)
            
            # Basic data validation
            required_columns = ['state', 'year', 'totalprod', 'numcol', 'yieldpercol']
            if not all(col in self.df.columns for col in required_columns):
                raise ValueError("Missing required columns in the dataset")
            
            # Preprocess the data
            self.preprocess_data()
        
        except FileNotFoundError:
            print(f"Error: File not found at {csv_path}")
            raise
        except pd.errors.EmptyDataError:
            print("Error: The CSV file is empty")
            raise
        except Exception as e:
            print(f"Unexpected error loading the dataset: {e}")
            raise

    def preprocess_data(self):
        """
        Preprocess the honey production data
        """
        # Remove any rows with missing values
        self.df.dropna(subset=['year', 'totalprod'], inplace=True)
        
        # Group by year and calculate mean total production
        self.prod_per_year = self.df.groupby('year')['totalprod'].mean()

    def prepare_regression_data(self):
        """
        Prepare data for linear regression
        
        Returns:
            tuple: X (years), y (total production)
        """
        # Prepare X (years)
        X = self.prod_per_year.index.values.reshape(-1, 1)
        
        # Prepare y (total production)
        y = self.prod_per_year.values
        
        return X, y

    def perform_linear_regression(self):
        """
        Perform linear regression and calculate key metrics
        
        Returns:
            dict: Regression results and metrics
        """
        # Prepare data
        X, y = self.prepare_regression_data()
        
        # Create and fit the model
        regr = LinearRegression()
        regr.fit(X, y)
        
        # Make predictions
        y_pred = regr.predict(X)
        
        # Calculate metrics
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        # Perform statistical significance test
        slope, intercept, r_value, p_value, std_err = stats.linregress(X.flatten(), y)
        
        return {
            'model': regr,
            'slope': regr.coef_[0],
            'intercept': regr.intercept_,
            'mse': mse,
            'r2': r2,
            'p_value': p_value,
            'std_err': std_err
        }

    def predict_future_production(self, regression_results, start_year=2013, end_year=2050):
        """
        Predict future honey production
        
        Args:
            regression_results (dict): Results from linear regression
            start_year (int): Starting year for prediction
            end_year (int): Ending year for prediction
        
        Returns:
            tuple: Future years and their predicted production
        """
        # Create future years array
        X_future = np.array(range(start_year, end_year + 1)).reshape(-1, 1)
        
        # Predict future production
        future_predict = regression_results['model'].predict(X_future)
        
        return X_future, future_predict

    def visualize_results(self, X, y, y_pred, X_future, future_predict, regression_results):
        """
        Create comprehensive visualizations of the analysis
        
        Args:
            X (array): Years of existing data
            y (array): Total production of existing data
            y_pred (array): Predicted values for existing data
            X_future (array): Future years
            future_predict (array): Predicted future production
            regression_results (dict): Regression analysis results
        """
        # Set up a clean, professional plot style
        plt.style.use('seaborn')
        
        # Create a figure with multiple subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # First subplot: Historical Data and Regression Line
        ax1.scatter(X, y, color='blue', label='Actual Production', alpha=0.7)
        ax1.plot(X, y_pred, color='red', label='Regression Line')
        ax1.set_title('Honey Production Over Time')
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Total Production')
        
        # Add R-squared and p-value to the plot
        ax1.text(0.05, 0.95, 
                 f'R² = {regression_results["r2"]:.4f}\n'
                 f'p-value = {regression_results["p_value"]:.4f}', 
                 transform=ax1.transAxes, 
                 verticalalignment='top')
        
        ax1.legend()
        
        # Second subplot: Future Production Projection
        ax2.plot(X_future, future_predict, color='green', label='Projected Production')
        ax2.axhline(y=future_predict[-1], color='red', linestyle='--', 
                    label=f'2050 Projected Production: {future_predict[-1]:.0f}')
        ax2.set_title('Projected Honey Production to 2050')
        ax2.set_xlabel('Year')
        ax2.set_ylabel('Projected Total Production')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()

    def comprehensive_analysis(self):
        """
        Perform a comprehensive analysis of honey production data
        """
        # Prepare data and perform regression
        regression_results = self.perform_linear_regression()
        
        # Prepare existing data predictions
        X, y = self.prepare_regression_data()
        y_pred = regression_results['model'].predict(X)
        
        # Predict future production
        X_future, future_predict = self.predict_future_production(regression_results)
        
        # Visualize results
        self.visualize_results(X, y, y_pred, X_future, future_predict, regression_results)
        
        # Print detailed analysis
        print("\n--- Honey Production Analysis Results ---")
        print(f"Slope (Annual Change): {regression_results['slope']:.2f}")
        print(f"Intercept: {regression_results['intercept']:.2f}")
        print(f"Mean Squared Error: {regression_results['mse']:.2f}")
        print(f"R-squared: {regression_results['r2']:.4f}")
        print(f"P-value: {regression_results['p_value']:.4f}")
        print(f"Projected 2050 Production: {future_predict[-1]:.0f}")

# Main execution
if __name__ == "__main__":
    try:
        # Replace with the actual path to your CSV file
        csv_path = "https://content.codecademy.com/programs/data-science-path/linear_regression/honeyproduction.csv"
        
        # Create analysis instance
        analysis = HoneyProductionAnalysis(csv_path)
        
        # Run comprehensive analysis
        analysis.comprehensive_analysis()
    
    except Exception as e:
        print(f"An error occurred during analysis: {e}")