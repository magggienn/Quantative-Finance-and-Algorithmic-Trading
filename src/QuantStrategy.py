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


class Backtester:
    """
    Performs backtesting of the multi-factor strategy
    """
    def __init__(self, data, portfolios,factor_data):
        self.data = data
        self.portfolios = portfolios
        self.factor_data = factor_data
    
    def calculate_portfolio_returns(self):
        """
        Calculate portfolio returns with Fama-French factor integration
        """
        portfolio_returns = {}
    
        for period, period_portfolios in self.portfolios.items():
            # Convert period to match factor data format
            factor_period = int(str(period)[:6])
            
            # factor data for the period
            try:
                period_factors = self.factor_data[self.factor_data['mdate'] == factor_period]
                
                # If no matching factor data, skip this period
                if len(period_factors) == 0:
                    continue
                
                # extract factor returns
                rf = period_factors['RF'].values[0]  
                mktrf = period_factors['MktRF'].values[0]  
                smb = period_factors['SMB'].values[0] 
                hml = period_factors['HML'].values[0]  
                period_returns = {}
                
                for portfolio_name, stocks in period_portfolios.items():
                    # Get returns for stocks in this portfolio
                    portfolio_stock_returns = self.data[
                        (self.data['mdate'] == period) & 
                        (self.data['ISIN'].isin(stocks))
                    ]['RET']
                    
                    # Calculate equal-weighted portfolio return
                    portfolio_return = portfolio_stock_returns.mean()
                    
                    # Calculate excess return
                    excess_return = portfolio_return - rf
                    
                    # Store detailed return information
                    period_returns[portfolio_name] = {
                        'portfolio_return': portfolio_return,
                        'excess_return': excess_return,
                        'risk_free_rate': rf,
                        'market_excess_return': mktrf,
                        'smb': smb,
                        'hml': hml,
                        'num_stocks': len(stocks)
                    }
                
                portfolio_returns[period] = period_returns
            
            except Exception as e:
                print(f"Error processing period {period}: {e}")
    
        return portfolio_returns
    
    def compute_performance_metrics(self):
        """
        Compute key performance metrics for all portfolios.
        """
    
        performance_summary = {}
        portfolio_returns = self.calculate_portfolio_returns()
        
        # Get unique portfolio names
        portfolio_names = []
        for period, period_portfolios in portfolio_returns.items():
            portfolio_names.extend(list(period_portfolios.keys()))
        portfolio_names = list(set(portfolio_names))

        # Calculate metrics for each portfolio
        for portfolio_name in portfolio_names:
            # Extract time series of returns and excess returns
            returns_series = self._extract_portfolio_return_series(portfolio_returns, portfolio_name)
            
            if not returns_series:
                continue  
                            
            # Calculate individual metrics
            basic_stats = self._calculate_basic_stats(returns_series)
            risk_metrics = self._calculate_risk_metrics(returns_series)
            risk_adjusted_metrics = self._calculate_risk_adjusted_metrics(returns_series)
            
            # Combine all metrics
            performance_summary[portfolio_name] = {**basic_stats, **risk_metrics, **risk_adjusted_metrics}
    
        return performance_summary

    def _extract_portfolio_return_series(self, portfolio_returns, portfolio_name):
        """
        Extract time series of returns and related data for a specific portfolio.
        """
        returns = {
            'portfolio_returns': [],
            'excess_returns': [],
            'market_excess_returns': [],
            'dates': []
        }
        
        for period, period_returns in portfolio_returns.items():
            if portfolio_name in period_returns:
                returns['portfolio_returns'].append(period_returns[portfolio_name]['portfolio_return'])
                returns['excess_returns'].append(period_returns[portfolio_name]['excess_return'])
                returns['market_excess_returns'].append(period_returns[portfolio_name]['market_excess_return'])
                returns['dates'].append(period)
        
        return returns

    def _calculate_basic_stats(self, returns_series):
        """
        Calculate basic return statistics.
        """
        TRADING_PERIODS = 12 
        
        portfolio_returns = np.array(returns_series['portfolio_returns'])
        
        mean_return = np.mean(portfolio_returns)
        annualized_return = (1 + mean_return) ** TRADING_PERIODS - 1
        volatility = np.std(portfolio_returns)
        annualized_volatility = volatility * np.sqrt(TRADING_PERIODS)
        
        return {
            'Mean Return': mean_return,
            'Annualized Return': annualized_return,
            'Return Volatility': volatility,
            'Annualized Volatility': annualized_volatility,
            'Number of Periods': len(portfolio_returns),
            'Positive Periods Ratio': np.mean(portfolio_returns > 0)
        }

    def _calculate_risk_metrics(self, returns_series):
        """
        Calculate risk-related metrics including drawdown.
        """
        portfolio_returns = np.array(returns_series['portfolio_returns'])
        
        # Maximum Drawdown calculation
        cumulative_returns = np.cumprod(1 + portfolio_returns) - 1
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / (1 + running_max)
        max_drawdown = np.min(drawdown)
        
        return {
            'Maximum Drawdown': max_drawdown
        }

    def _calculate_risk_adjusted_metrics(self, returns_series):
        """
        Calculate risk-adjusted performance metrics.
        """
        TRADING_PERIODS = 12  # 12 months in a year
        
        excess_returns = np.array(returns_series['excess_returns'])
        market_excess_returns = np.array(returns_series['market_excess_returns'])
        
        # Sharpe Ratio
        sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(TRADING_PERIODS)
        
        # Information Ratio
        tracking_error = np.std(excess_returns - market_excess_returns)
        information_ratio = (np.mean(excess_returns) - np.mean(market_excess_returns)) / tracking_error if tracking_error != 0 else 0
        
        return {
            'Sharpe Ratio': sharpe_ratio,
            'Information Ratio': information_ratio,
            'Tracking Error': tracking_error
        }

