import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import missingno as msno
from matplotlib.dates import DateFormatter
import matplotlib.ticker as mtick

# Plotting parameters
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

class DataExplorer:
    def __init__(self, price_data_monthly, price_data_annually, factor_data):
        self.price_data_monthly = price_data_monthly
        self.price_data_annually = price_data_annually
        self.factor_data = factor_data
    
    def summary_statistics(self):
        """
        Computes summary statistics for each dataset.
        """
        print("Price Data Monthly Summary:")
        print(self.price_data_monthly.describe())
        
        print("\nPrice Data Annually Summary:")
        print(self.price_data_annually.describe())
        
        print("\nFactor Data Summary:")
        print(self.factor_data.describe())
        
    def missing_values(self):
        """
        Check for missing values in the data
        """
        print("Missing values in price data monthly:")
        print(self.price_data_monthly.isnull().sum())
        print("\nMissing values in price data annually:")
        print(self.price_data_annually.isnull().sum())
        print("\nMissing values in factor data:")
        print(self.factor_data.isnull().sum())
        
    def explore_distributions(self, save_path=None):
        """
        Create comprehensive distribution plots for numerical variables
        """
        def create_distribution_plots(data, dataset_name, num_cols):
            """
            Create distribution plots for a single dataset
            """
            if len(num_cols) == 0:
                print(f"No numerical columns found in {dataset_name}")
                return
            
            num_total_plots = len(num_cols) * 3  # histogram, Q-Q, boxplot
            
            # Calculate rows and columns
            n_cols = 3
            n_rows = (num_total_plots + n_cols - 1) // n_cols
            
            # Create figure with calculated size
            plt.figure(figsize=(15, 5 * n_rows))
            plt.suptitle(f'Distribution Plots for {dataset_name}', fontsize=16)
            
            # Iterate through columns
            plot_index = 1
            for col in num_cols:
                # Prepare data, removing NaNs
                clean_data = data[col].dropna()
                
                # Histogram subplot
                plt.subplot(n_rows, n_cols, plot_index)
                plot_index += 1
                try:
                    sns.histplot(clean_data, kde=True)
                except Exception as e:
                    plt.text(0.5, 0.5, f'Error in histogram\n{str(e)}', 
                             horizontalalignment='center', 
                             verticalalignment='center')
                plt.title(f'{col} Distribution')
                
                # Q-Q plot subplot
                plt.subplot(n_rows, n_cols, plot_index)
                plot_index += 1
                try:
                    stats.probplot(clean_data, plot=plt,rvalue=True)
                except Exception as e:
                    plt.text(0.5, 0.5, f'Error in Q-Q plot\n{str(e)}', 
                             horizontalalignment='center', 
                             verticalalignment='center')
                plt.title(f'Q-Q Plot for {col}')
                
                # Box plot subplot
                plt.subplot(n_rows, n_cols, plot_index)
                plot_index += 1
                try:
                    sns.boxplot(x=clean_data)
                except Exception as e:
                    plt.text(0.5, 0.5, f'Error in Box plot\n{str(e)}', 
                             horizontalalignment='center', 
                             verticalalignment='center')
                plt.title(f'Box Plot for {col}')
            
            # Adjust layout
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            # Save or show
            if save_path:
                plt.savefig(f'{save_path}/{dataset_name}_distributions.png')
                plt.close()
            else:
                plt.show()
        
        # Identify numerical columns
        monthly_num_cols = ['RET', 'RET11', 'ME', 'b', 'h', 's', 'ivol']
        annually_num_cols = ['fyear', 'BEME', 'OP', 'INV']
        factor_num_cols = ['MktRF', 'SMB', 'HML', 'RF', 'WML']
        
        # Print out columns for debugging
        print("Monthly Numerical Columns:", monthly_num_cols)
        print("Annual Numerical Columns:", annually_num_cols)
        print("Factor Numerical Columns:", factor_num_cols)
        
        # Create plots for each dataset
        create_distribution_plots(self.price_data_monthly, 'Monthly Price Data', monthly_num_cols)
        create_distribution_plots(self.price_data_annually, 'Annual Price Data', annually_num_cols)
        create_distribution_plots(self.factor_data, 'Factor Data', factor_num_cols)
        
    def correlation_heatmap(self, save_path=None):
        """
        Create correlation heatmaps for numerical variables in each dataset
        """
        # Identify numerical columns
        monthly_num_cols = ['RET', 'RET11', 'ME', 'b', 'h', 's', 'ivol']
        annually_num_cols = ['fyear', 'BEME', 'OP', 'INV']
        factor_num_cols = ['MktRF', 'SMB', 'HML', 'RF', 'WML']
        
        # Correlation heatmaps
        plt.figure(figsize=(15, 5))
        
        # Monthly data correlation
        plt.subplot(1, 3, 1)
        corr_monthly = self.price_data_monthly[monthly_num_cols].corr()
        sns.heatmap(corr_monthly, annot=True, cmap='coolwarm', center=0, vmin=-1, vmax=1, 
                    square=True, cbar=False)
        plt.title('Monthly Price Data Correlation')
        
        # Annual data correlation
        plt.subplot(1, 3, 2)
        corr_annually = self.price_data_annually[annually_num_cols].corr()
        sns.heatmap(corr_annually, annot=True, cmap='coolwarm', center=0, vmin=-1, vmax=1, 
                    square=True, cbar=False)
        plt.title('Annual Price Data Correlation')
        
        # Factor data correlation
        plt.subplot(1, 3, 3)
        corr_factor = self.factor_data[factor_num_cols].corr()
        sns.heatmap(corr_factor, annot=True, cmap='coolwarm', center=0, vmin=-1, vmax=1, 
                    square=True, cbar=False)
        plt.title('Factor Data Correlation')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f'{save_path}/correlation_heatmaps.png')
            plt.close()
        else:
            plt.show()
            
        