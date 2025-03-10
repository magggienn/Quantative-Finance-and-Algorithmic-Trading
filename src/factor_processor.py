import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns

class FactorProcessor:
    """
    Handles factor calculation, standardization, and preprocessing
    """
    def __init__(self, data):
        self.data = data
        
    def winsorize(self, column, lower_percentile=1, 
                  upper_percentile= 99):
        """
        Winsorize a given column to reduce impact of outliers
        """
        data = self.data[column].values
    
        # winsorization limits
        lower_limit = lower_percentile / 100
        upper_limit = (100 - upper_percentile) / 100
    
        winsorized = winsorize(data, limits=[lower_limit, upper_limit])
        
        return pd.Series(winsorized, name=column, index=self.data.index)
    
    def calculate_z_scores(self, column):
        """
        Calculate cross-sectional z-scores for a given factor.
        Groups BEME stocks per month and standardizes each group
        """
        def safe_zscore(x):
            # If we have enough non-NaN values, calculate z-score
            non_null_count = x.count()
            
            if non_null_count > 1:
                # Calculate z-score on non-null values
                z_scores = (x - x.mean()) / x.std()
                return z_scores
            else:
                # If insufficient data, return NaN for the entire group
                return pd.Series(np.nan, index=x.index)
    
        # Group by month and calculate z-scores
        z_scores = self.data.groupby('mdate')[column].transform(safe_zscore)
        
        return z_scores
        
    def preprocess_factors(self):
        """
        Preprocess and standardize all factors
        """
        factors = {
            'value': 'BEME',
            'momentum': 'RET11',
            'profitability': 'OP',
            'investment': 'INV'
        }
        
        processed_factors = {}
        for factor_name, column in factors.items():
            # Winsorize first
            winsorized = self.winsorize(column)
            
            # Calculate z-scores
            z_scores = self.calculate_z_scores(column)
            
            # Adjust sign for investment factor (lower is better)
            if factor_name == 'investment':
                z_scores = -z_scores
            
            processed_factors[factor_name] = z_scores
    
        return processed_factors