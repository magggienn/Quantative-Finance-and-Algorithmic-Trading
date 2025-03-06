from QuantStrategy import *


def main():
    """
    Main execution function for the multi-factor strategy
    """
    # Load data
    path = "C:\\Users\\magggien\\Documents\\Masters\\QFAT\\group_project\\data\\merge_annual_monthly_data.csv"
    data = pd.read_csv(path)
    factor_data = pd.read_csv("C:\\Users\\magggien\\Documents\\Masters\\QFAT\\group_project\\data\\Europe_FF_Factors.csv")
    
    # Process factors
    factor_processor = FactorProcessor(data)
    processed_factors = factor_processor.preprocess_factors()

    # Construct portfolios
    portfolio_constructor = PortfolioConstructor(
        factor_scores=processed_factors, 
        data=data
    )
    composite_score = portfolio_constructor.compute_composite_score()
    portfolios = portfolio_constructor.construct_portfolio(min_unique_scores=10)
  
    # Backtest
    backtester = Backtester(data, portfolios,factor_data=factor_data)
    portfolio_returns = backtester.calculate_portfolio_returns()
    performance_metrics = backtester.compute_performance_metrics()
    backtester_results = pd.DataFrame(performance_metrics).T
    # backtester_results.to_csv("C:\\Users\\magggien\\Documents\\Masters\\QFAT\\group_project\\data\\backtester_results.csv")
    print(backtester_results)

    
    # Visualize and report results
    def plot_portfolio_metrics(results_df):
        """
        Simple visualization for portfolio performance metrics.
        """
        # Set style
        sns.set(style="whitegrid")
    
        # 1. Create single row of key metric bar charts
        key_metrics = ['Annualized Return', 'Sharpe Ratio', 'Information Ratio', 'Maximum Drawdown']
        
        fig, axes = plt.subplots(1, len(key_metrics), figsize=(16, 5))
        
        # Sort portfolios by annualized return
        sorted_df = results_df.sort_values('Annualized Return', ascending=False)
        
        for i, metric in enumerate(key_metrics):
            ax = axes[i]
            sorted_df[metric].plot(kind='bar', ax=ax, color='skyblue')
            ax.set_title(metric)
            ax.set_ylabel('Value')
            
            # Add value labels
            for j, v in enumerate(sorted_df[metric]):
                ax.text(j, v, f"{v:.2f}", ha='center', 
                    va='bottom' if v >= 0 else 'top', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('C:\\Users\\magggien\\Documents\\Masters\\QFAT\\group_project\\data\\portfolio_metrics.png', dpi=300)
        
        # 2. Risk-Return scatter plot with added Information Ratio information
        plt.figure(figsize=(8, 6))
        
        # Use Information Ratio to determine point size
        sizes = sorted_df['Information Ratio'].apply(lambda x: max(abs(x) * 80, 30))
        
        plt.scatter(sorted_df['Annualized Volatility'], sorted_df['Annualized Return'], 
                s=sizes, alpha=0.7)
        
        for idx, row in sorted_df.iterrows():
            plt.annotate(f"{idx} (IR: {row['Information Ratio']:.2f})", 
                        (row['Annualized Volatility'], row['Annualized Return']),
                        xytext=(5, 5), textcoords='offset points')
        
        plt.xlabel('Annualized Volatility')
        plt.ylabel('Annualized Return')
        plt.title('Risk-Return Profile with Information Ratio')
        plt.grid(True, alpha=0.3)
        
        plt.savefig('C:\\Users\\magggien\\Documents\\Masters\\QFAT\\group_project\\data\\risk_return.png', dpi=300)
        plt.show()
        
    plot_portfolio_metrics(backtester_results)

        
if __name__ == "__main__":
    main()