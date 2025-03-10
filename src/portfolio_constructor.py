import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns

class PortfolioConstructor:
    """
    Constructs portfolios based on factor scores
    """
    def __init__(self, factor_scores, data, weights=None):
        """
        Initialize PortfolioConstructor
        """
        self.factor_scores = factor_scores
        self.data = data
        self.default_weights  = weights or {
            'value': 0.25,
            'momentum': 0.25,
            'profitability': 0.25,
            'investment': 0.25
        }
        
        self.weights = weights or self.default_weights
        
        # factor scores to NumPy arrays
        self.factor_arrays = np.column_stack([
            factor_scores[factor].values 
            for factor in self.weights.keys()
        ])
        
        # weights to NumPy array
        self.weight_array = np.array([
            self.weights.get(factor, 0.25) 
            for factor in self.weights.keys()
        ])
        
        self.index = next(iter(factor_scores.values())).index
    
    def compute_composite_score(self):
        """
        Compute composite score by combining factor z-scores
        """
        
        # computationally efficient way to compute composite score
        composite_score_array = np.dot(self.factor_arrays, self.weight_array)
        
        # for easy readibility convert to PD series
        composite_score = pd.Series(
            composite_score_array, 
            index=self.index,
            name='composite_score'
        )
        
        return composite_score
    
    def construct_portfolio(self, num_portfolios=5, min_unique_scores=10):
        """
        Construct portfolios based on composite score
        """
        composite_score = self.compute_composite_score()
        unique_periods = self.data['mdate'].unique()
        
        # Dictionary to store portfolios
        portfolios = {}
        skipped_periods = []
        for period in unique_periods:
            # Filter data for current period
            period_data = self.data[self.data['mdate'] == period]
            period_scores = composite_score[self.data['mdate'] == period]
            
            # NaN values
            valid_mask = ~period_scores.isna()
            period_data_filtered = period_data[valid_mask]
            period_scores_filtered = period_scores[valid_mask]
            
            # unique scores
            unique_scores = period_scores_filtered.unique()
            if len(unique_scores) < min_unique_scores:
                skipped_periods.append({
                    'period': period, 
                    'total_stocks': len(period_data),
                    'valid_stocks': len(period_data_filtered),
                    'unique_scores': len(unique_scores)
                })
                continue
            
            # Create portfolio assignments
            try:
                portfolio_assignments = pd.qcut(
                    period_scores_filtered, 
                    q=num_portfolios, 
                    labels=False,
                    duplicates='drop' 
                )
            except ValueError:
                # Fallback to using quantiles if qcut fails
                portfolio_assignments = pd.qcut(
                    period_scores_filtered, 
                    q=num_portfolios, 
                    labels=False,
                    duplicates='keep'
                )
                
            period_portfolios = {}
            for portfolio in range(num_portfolios):
                # Get ISINs for stocks in this portfolio
                portfolio_stocks = period_data_filtered[portfolio_assignments == portfolio]['ISIN'].tolist()
                period_portfolios[f'Portfolio_{portfolio+1}'] = portfolio_stocks
                
            portfolios[period] = period_portfolios
            
        if skipped_periods:
            print("\nSkipped Periods Details:")
            for skip_info in skipped_periods:
                print(f"Period: {skip_info['period']}")
                print(f"  Total Stocks: {skip_info['total_stocks']}")
                print(f"  Valid Stocks: {skip_info['valid_stocks']}")
                print(f"  Unique Scores: {skip_info['unique_scores']}")
    
        return portfolios