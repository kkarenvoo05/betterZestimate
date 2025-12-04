"""
Error Analysis Script for Real Estate Portfolio RL Models
Implements three key analyses:
1. Maximum Drawdown
2. Win Rate on Sales
3. Temporal Performance Breakdown
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import sys
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from env_setup import RealEstatePortfolioEnv

class TransactionTracker:
    """Tracks buy/sell transactions for analysis"""
    def __init__(self):
        self.buys = []  # List of (date, zpid, price)
        self.sells = []  # List of (date, zpid, price)
        self.owned_properties = {}  # zpid -> list of purchase prices (FIFO)
        
    def record_buy(self, date, zpid, price):
        """Record a property purchase"""
        self.buys.append({'date': date, 'zpid': zpid, 'price': price})
        if zpid not in self.owned_properties:
            self.owned_properties[zpid] = []
        self.owned_properties[zpid].append({'date': date, 'price': price})
    
    def record_sell(self, date, zpid, price):
        """Record a property sale and return profit/loss"""
        if zpid not in self.owned_properties or len(self.owned_properties[zpid]) == 0:
            return None  # Can't sell what we don't own
        
        # FIFO: sell the oldest purchase
        purchase = self.owned_properties[zpid].pop(0)
        profit = price - purchase['price']
        
        self.sells.append({
            'date': date,
            'zpid': zpid,
            'sell_price': price,
            'buy_price': purchase['price'],
            'buy_date': purchase['date'],
            'profit': profit,
            'return_pct': (profit / purchase['price']) * 100
        })
        
        if len(self.owned_properties[zpid]) == 0:
            del self.owned_properties[zpid]
        
        return profit
    
    def get_sell_df(self):
        """Get DataFrame of all sales with profit/loss"""
        if not self.sells:
            return pd.DataFrame()
        return pd.DataFrame(self.sells)


def evaluate_strategy(env, model_or_strategy, strategy_name, deterministic=True):
    """
    Evaluate a strategy (model or function) and return detailed results
    
    Args:
        env: RealEstatePortfolioEnv instance
        model_or_strategy: Either a PPO model or a function that takes obs and returns action
        strategy_name: Name of the strategy
        deterministic: Whether to use deterministic predictions (for models)
    
    Returns:
        dict with portfolio_values, dates, actions, transactions
    """
    obs, info = env.reset()
    done = False
    
    portfolio_values = [info.get('portfolio_value', env.initial_cash)]
    dates = [env.dates[0] if len(env.dates) > 0 else None]
    actions_taken = []
    tracker = TransactionTracker()
    
    step_idx = 0
    
    while not done:
        # Get action
        if hasattr(model_or_strategy, 'predict'):
            # It's a model
            action, _ = model_or_strategy.predict(obs, deterministic=deterministic)
        else:
            # It's a function
            action = model_or_strategy(obs, env)
        
        actions_taken.append(action.copy())
        
        # Get current period and properties before step
        current_period = env.dates[env.current_idx] if env.current_idx < len(env.dates) else None
        top_props, _ = env._get_current_data()
        
        # Record transactions before step
        for i, act in enumerate(action):
            if i >= len(top_props):
                break
            p = top_props.iloc[i]
            zpid = p['zpid']
            price = p['price']
            
            if act == 1:  # buy
                tracker.record_buy(current_period, zpid, price)
            elif act == 2:  # sell
                tracker.record_sell(current_period, zpid, price)
        
        # Step environment
        obs, reward, done, truncated, info = env.step(action)
        
        # Record portfolio value and date
        portfolio_value = info.get('portfolio_value', env.cash)
        portfolio_values.append(portfolio_value)
        
        if not done and env.current_idx < len(env.dates):
            dates.append(env.dates[env.current_idx])
        else:
            dates.append(current_period)
        
        step_idx += 1
    
    # Convert to DataFrames
    results_df = pd.DataFrame({
        'date': dates[:len(portfolio_values)],
        'portfolio_value': portfolio_values
    })
    
    return {
        'strategy_name': strategy_name,
        'portfolio_values': portfolio_values,
        'dates': dates[:len(portfolio_values)],
        'results_df': results_df,
        'actions': actions_taken,
        'transactions': tracker,
        'final_value': portfolio_values[-1],
        'initial_value': portfolio_values[0]
    }


def calculate_max_drawdown(portfolio_values):
    """
    Calculate maximum drawdown
    
    Returns:
        max_drawdown (float): Maximum peak-to-trough decline as a percentage
        drawdown_series (pd.Series): Drawdown at each time point
    """
    portfolio_series = pd.Series(portfolio_values)
    peak = portfolio_series.expanding(min_periods=1).max()
    drawdown = (portfolio_series - peak) / peak
    max_drawdown = drawdown.min()
    
    return max_drawdown, drawdown


def calculate_win_rate(transactions):
    """
    Calculate win rate on sales
    
    Returns:
        win_rate (float): Percentage of profitable sales
        avg_profit_per_sale (float): Average profit per sale
        total_sales (int): Total number of sales
    """
    sell_df = transactions.get_sell_df()
    
    if sell_df.empty:
        return 0.0, 0.0, 0
    
    wins = (sell_df['profit'] > 0).sum()
    total_sales = len(sell_df)
    win_rate = (wins / total_sales) * 100 if total_sales > 0 else 0.0
    avg_profit = sell_df['profit'].mean()
    
    return win_rate, avg_profit, total_sales


def temporal_performance_breakdown(results_dict, periods):
    """
    Calculate performance for different time periods
    
    Args:
        results_dict: Results from evaluate_strategy
        periods: List of tuples (start_year, end_year)
    
    Returns:
        pd.DataFrame with period returns
    """
    df = results_dict['results_df'].copy()
    # Handle Period objects or string dates
    if df['date'].dtype == 'object':
        # Convert Period to string then to datetime
        df['date'] = pd.to_datetime([str(d) if d is not None else '2014-01' for d in df['date']])
    df['year'] = df['date'].dt.year
    
    period_results = []
    
    for start_year, end_year in periods:
        period_df = df[(df['year'] >= start_year) & (df['year'] <= end_year)]
        
        if len(period_df) == 0:
            continue
        
        initial_val = period_df['portfolio_value'].iloc[0]
        final_val = period_df['portfolio_value'].iloc[-1]
        years = end_year - start_year + 1
        
        total_return = (final_val - initial_val) / initial_val
        annualized_return = ((final_val / initial_val) ** (1 / years) - 1) if years > 0 else 0
        
        period_results.append({
            'period': f"{start_year}-{end_year}",
            'start_year': start_year,
            'end_year': end_year,
            'years': years,
            'initial_value': initial_val,
            'final_value': final_val,
            'total_return': total_return * 100,
            'annualized_return': annualized_return * 100
        })
    
    return pd.DataFrame(period_results)


def run_all_analyses(model_paths, baseline_strategies=None):
    """
    Run all three analyses on multiple strategies
    
    Args:
        model_paths: Dict of {strategy_name: model_path}
        baseline_strategies: Dict of {strategy_name: strategy_function}
    """
    # Setup environment
    base_dir = Path(__file__).parent.parent
    property_data_path = base_dir / "data" / "processed_data.csv"
    market_data_path = base_dir / "data" / "market_data.csv"
    
    # Use absolute paths
    property_data_path = str(property_data_path.absolute())
    market_data_path = str(market_data_path.absolute())
    
    env = RealEstatePortfolioEnv(
        property_data_path=property_data_path,
        market_data_path=market_data_path,
        initial_cash=100_000_000,
        num_properties=5
    )
    
    all_results = {}
    
    # Evaluate models
    if model_paths:
        for strategy_name, model_path in model_paths.items():
            print(f"Evaluating {strategy_name}...")
            try:
                model = PPO.load(model_path)
                results = evaluate_strategy(env, model, strategy_name)
                all_results[strategy_name] = results
            except Exception as e:
                print(f"Error loading {strategy_name}: {e}")
    
    # Evaluate baseline strategies
    if baseline_strategies:
        for strategy_name, strategy_func in baseline_strategies.items():
            print(f"Evaluating {strategy_name}...")
            results = evaluate_strategy(env, strategy_func, strategy_name)
            all_results[strategy_name] = results
    
    # Run analyses
    print("\n" + "="*80)
    print("ANALYSIS RESULTS")
    print("="*80)
    
    # 1. Maximum Drawdown
    print("\n1. MAXIMUM DRAWDOWN ANALYSIS")
    print("-" * 80)
    drawdown_results = []
    
    for strategy_name, results in all_results.items():
        max_dd, dd_series = calculate_max_drawdown(results['portfolio_values'])
        drawdown_results.append({
            'Strategy': strategy_name,
            'Max Drawdown (%)': f"{max_dd * 100:.2f}%",
            'Final Value': f"${results['final_value']:,.0f}"
        })
        print(f"{strategy_name}:")
        print(f"  Maximum Drawdown: {max_dd * 100:.2f}%")
        print(f"  Final Portfolio Value: ${results['final_value']:,.0f}")
    
    drawdown_df = pd.DataFrame(drawdown_results)
    print("\n" + drawdown_df.to_string(index=False))
    
    # 2. Win Rate on Sales
    print("\n\n2. WIN RATE ON SALES ANALYSIS")
    print("-" * 80)
    winrate_results = []
    
    for strategy_name, results in all_results.items():
        win_rate, avg_profit, total_sales = calculate_win_rate(results['transactions'])
        winrate_results.append({
            'Strategy': strategy_name,
            'Win Rate (%)': f"{win_rate:.2f}%",
            'Avg Profit/Sale': f"${avg_profit:,.0f}",
            'Total Sales': total_sales
        })
        print(f"{strategy_name}:")
        print(f"  Win Rate: {win_rate:.2f}%")
        print(f"  Average Profit per Sale: ${avg_profit:,.0f}")
        print(f"  Total Sales: {total_sales}")
        
        # Show distribution if we have sales
        sell_df = results['transactions'].get_sell_df()
        if not sell_df.empty:
            print(f"  Profitable Sales: {(sell_df['profit'] > 0).sum()}")
            print(f"  Loss Sales: {(sell_df['profit'] <= 0).sum()}")
            print(f"  Best Sale: ${sell_df['profit'].max():,.0f}")
            print(f"  Worst Sale: ${sell_df['profit'].min():,.0f}")
    
    winrate_df = pd.DataFrame(winrate_results)
    print("\n" + winrate_df.to_string(index=False))
    
    # 3. Temporal Performance Breakdown
    print("\n\n3. TEMPORAL PERFORMANCE BREAKDOWN")
    print("-" * 80)
    
    # Define periods (adjust based on your data range)
    periods = [(2014, 2017), (2018, 2020), (2021, 2024)]
    
    temporal_results = []
    
    for strategy_name, results in all_results.items():
        period_df = temporal_performance_breakdown(results, periods)
        if not period_df.empty:
            print(f"\n{strategy_name}:")
            for _, row in period_df.iterrows():
                print(f"  {row['period']}: {row['annualized_return']:.2f}% annualized return")
                temporal_results.append({
                    'Strategy': strategy_name,
                    'Period': row['period'],
                    'Annualized Return (%)': f"{row['annualized_return']:.2f}%",
                    'Total Return (%)': f"{row['total_return']:.2f}%"
                })
    
    temporal_df = pd.DataFrame(temporal_results)
    if not temporal_df.empty:
        print("\n" + temporal_df.to_string(index=False))
    
    # Generate visualizations
    print("\n\nGenerating visualizations...")
    generate_visualizations(all_results, drawdown_df, winrate_df, temporal_df)
    
    return all_results, drawdown_df, winrate_df, temporal_df


def generate_visualizations(all_results, drawdown_df, winrate_df, temporal_df):
    """Generate visualization plots"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Error Analysis: Real Estate Portfolio RL Models', fontsize=16, fontweight='bold')
    
    # 1. Portfolio Value Over Time
    ax1 = axes[0, 0]
    for strategy_name, results in all_results.items():
        dates = [str(d) for d in results['dates']]
        ax1.plot(dates[::max(1, len(dates)//50)], 
                results['portfolio_values'][::max(1, len(results['portfolio_values'])//50)],
                label=strategy_name, linewidth=2)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.set_title('Portfolio Value Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Maximum Drawdown
    ax2 = axes[0, 1]
    strategies = drawdown_df['Strategy'].values
    max_dds = [float(x.replace('%', '')) for x in drawdown_df['Max Drawdown (%)'].values]
    ax2.barh(strategies, max_dds, color='coral')
    ax2.set_xlabel('Maximum Drawdown (%)')
    ax2.set_title('Maximum Drawdown by Strategy')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # 3. Win Rate
    ax3 = axes[1, 0]
    strategies = winrate_df['Strategy'].values
    win_rates = [float(x.replace('%', '')) for x in winrate_df['Win Rate (%)'].values]
    ax3.barh(strategies, win_rates, color='lightgreen')
    ax3.set_xlabel('Win Rate (%)')
    ax3.set_title('Win Rate on Sales by Strategy')
    ax3.grid(True, alpha=0.3, axis='x')
    
    # 4. Temporal Performance
    ax4 = axes[1, 1]
    if not temporal_df.empty:
        pivot_data = temporal_df.pivot(index='Strategy', columns='Period', values='Annualized Return (%)')
        pivot_data_clean = pivot_data.map(lambda x: float(x.replace('%', '')) if isinstance(x, str) else x)
        pivot_data_clean.plot(kind='bar', ax=ax4, width=0.8)
        ax4.set_ylabel('Annualized Return (%)')
        ax4.set_title('Temporal Performance Breakdown')
        ax4.legend(title='Period')
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(__file__).parent.parent / "error_analysis_results.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    
    # Also create drawdown visualization
    fig2, ax = plt.subplots(figsize=(12, 6))
    for strategy_name, results in all_results.items():
        _, dd_series = calculate_max_drawdown(results['portfolio_values'])
        dates = [str(d) for d in results['dates']]
        ax.plot(dates[::max(1, len(dates)//50)], 
               dd_series.values[::max(1, len(dd_series)//50)] * 100,
               label=strategy_name, linewidth=2)
    ax.set_xlabel('Date')
    ax.set_ylabel('Drawdown (%)')
    ax.set_title('Drawdown Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    
    output_path2 = Path(__file__).parent.parent / "drawdown_over_time.png"
    plt.savefig(output_path2, dpi=300, bbox_inches='tight')
    print(f"Drawdown visualization saved to: {output_path2}")


# Baseline strategy functions
def hold_strategy(obs, env):
    """Always hold (do nothing)"""
    return [0] * env.num_properties

def random_strategy(obs, env):
    """Random actions"""
    import random
    return [random.randint(0, 2) for _ in range(env.num_properties)]


if __name__ == "__main__":
    # Define model paths (adjust based on your actual model files)
    base_dir = Path(__file__).parent.parent
    
    model_paths = {}
    
    # Try to find model files (both directories and zip files)
    possible_models = [
        "re_portfolio_ppo_new_0.001_lr_long",
        "re_portfolio_ppo_new_0.001_lr",
        "re_portfolio_ppo_0.0001_lr",
        "re_portfolio_ppo_new_0.001_lr_long.zip",
        "re_portfolio_ppo_new_0.001_lr.zip",
        "re_portfolio_ppo_0.0001_lr.zip"
    ]
    
    for model_name in possible_models:
        model_path = base_dir / model_name
        if model_path.exists():
            # For zip files, PPO.load can handle them directly
            model_paths[model_name.replace('.zip', '')] = str(model_path.absolute())
            print(f"Found model: {model_name}")
    
    # Add baseline strategies
    baseline_strategies = {
        'Hold (Baseline)': hold_strategy,
        'Random (Baseline)': random_strategy
    }
    
    if not model_paths:
        print("Warning: No model files found. Running with baseline strategies only.")
        print("To analyze models, ensure model files exist in the project root.")
    
    # Run analyses
    all_results, drawdown_df, winrate_df, temporal_df = run_all_analyses(
        model_paths=model_paths,
        baseline_strategies=baseline_strategies
    )
    
    # Save results to CSV
    output_dir = base_dir
    drawdown_df.to_csv(output_dir / "max_drawdown_results.csv", index=False)
    winrate_df.to_csv(output_dir / "win_rate_results.csv", index=False)
    temporal_df.to_csv(output_dir / "temporal_performance_results.csv", index=False)
    
    print("\n" + "="*80)
    print("Analysis complete! Results saved to CSV files.")
    print("="*80)

