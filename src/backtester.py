import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns

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
