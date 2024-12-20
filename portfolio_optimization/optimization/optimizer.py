import os
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from scipy.optimize import minimize
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

class PortfolioOptimizer:
    """Class to handle portfolio optimization using mean-variance approach."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize optimizer with configuration."""
        # Get the project root directory (two levels up from this file)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        config_file = os.path.join(project_root, config_path)
        
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.risk_free_rate = self.config['optimization']['risk_free_rate']
        self.constraints = self.config['constraints']
        
        # Adjust max_position if it's too restrictive
        n_stocks = self.config['universe']['n_stocks']
        min_max_position = 1.0 / n_stocks
        self.constraints['max_position'] = max(min_max_position, self.constraints['max_position'])
    
    def calculate_portfolio_metrics(
        self, 
        weights: np.ndarray, 
        returns: np.ndarray, 
        cov_matrix: np.ndarray
    ) -> Tuple[float, float, float]:
        """Calculate portfolio return, volatility, and Sharpe ratio."""
        weights = np.array(weights)
        
        # Ensure the inputs are properly scaled
        returns = np.array(returns)
        cov_matrix = np.array(cov_matrix)
        
        # Calculate annualized metrics
        portfolio_return = np.sum(returns.mean() * weights) * 252
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
        
        # Handle numerical instability
        if portfolio_vol < 1e-8:
            portfolio_vol = 1e-8
            
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_vol
        return portfolio_return, portfolio_vol, sharpe_ratio
    
    def objective_function(
        self, 
        weights: np.ndarray, 
        returns: np.ndarray, 
        cov_matrix: np.ndarray, 
        objective: str = 'sharpe'
    ) -> float:
        """Define the optimization objective function."""
        if objective not in ['sharpe', 'volatility', 'return']:
            raise ValueError(f"Unsupported objective: {objective}")
            
        try:
            portfolio_return, portfolio_vol, sharpe_ratio = self.calculate_portfolio_metrics(
                weights, returns, cov_matrix
            )
            
            if objective == 'sharpe':
                return -sharpe_ratio  # Minimize negative Sharpe ratio (maximize Sharpe ratio)
            elif objective == 'volatility':
                return portfolio_vol
            elif objective == 'return':
                return -portfolio_return
        except Exception as e:
            # Return a large number if calculation fails
            return 1e6
    
    def optimize_portfolio(
        self, 
        returns: pd.DataFrame, 
        objective: str = 'sharpe',
        initial_weights: Optional[np.ndarray] = None,
        verbose: bool = True
    ) -> Dict:
        """Optimize portfolio weights given returns data."""
        if objective not in ['sharpe', 'volatility', 'return']:
            raise ValueError(f"Unsupported objective: {objective}")
            
        if verbose:
            print(f"\nOptimizing portfolio for objective: {objective}")
            
        # Calculate inputs
        mean_returns = returns.mean()
        cov_matrix = returns.cov()
        
        if verbose:
            print("\nInput Statistics:")
            print("Mean Returns (annualized):")
            print((mean_returns * 252).to_string())
            print("\nVolatilities (annualized):")
            print((returns.std() * np.sqrt(252)).to_string())
        
        n_assets = len(returns.columns)
        
        # Adjust max_position if needed
        min_max_position = 1.0 / n_assets
        max_position = max(min_max_position, self.constraints['max_position'])
        
        if verbose:
            print(f"\nConstraints:")
            print(f"Number of assets: {n_assets}")
            print(f"Min position: {self.constraints['min_position']:.2%}")
            print(f"Max position: {max_position:.2%}")
        
        # Initial weights if not provided
        if initial_weights is None:
            initial_weights = np.array([1/n_assets] * n_assets)
        
        # Ensure initial weights are valid
        initial_weights = np.clip(
            initial_weights,
            self.constraints['min_position'],
            max_position
        )
        initial_weights = initial_weights / np.sum(initial_weights)
        
        if verbose:
            print("\nInitial weights:")
            for asset, weight in zip(returns.columns, initial_weights):
                print(f"{asset}: {weight:.2%}")
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Weights sum to 1
        ]
        
        # Bounds for individual weights
        bounds = tuple(
            (max(0.0, self.constraints['min_position']), min(1.0, max_position))
            for _ in range(n_assets)
        )
        
        best_result = None
        best_metric = float('inf')
        
        # Try multiple starting points with different methods
        methods = ['SLSQP']  # Remove trust-constr as it requires different parameters
        for method in methods:
            for attempt in range(3):  # Increase attempts per method
                try:
                    if verbose:
                        print(f"\nTrying optimization with method {method}, attempt {attempt + 1}")
                    
                    result = minimize(
                        fun=self.objective_function,
                        x0=initial_weights,
                        args=(returns.values, cov_matrix.values, objective),
                        method=method,
                        bounds=bounds,
                        constraints=constraints,
                        options={
                            'ftol': 1e-8,
                            'maxiter': 1000,
                            'disp': verbose
                        }
                    )
                    
                    if result.success:
                        metric = self.objective_function(
                            result.x, returns.values, cov_matrix.values, objective
                        )
                        if metric < best_metric:
                            if verbose:
                                print(f"Found better solution with metric: {metric:.6f}")
                            best_result = result
                            best_metric = metric
                    else:
                        if verbose:
                            print(f"Optimization failed: {result.message}")
                    
                    # Try new random weights for next iteration
                    initial_weights = np.random.dirichlet(np.ones(n_assets))
                    initial_weights = np.clip(
                        initial_weights,
                        self.constraints['min_position'],
                        max_position
                    )
                    initial_weights = initial_weights / np.sum(initial_weights)
                    
                except Exception as e:
                    if verbose:
                        print(f"Optimization attempt failed with error: {str(e)}")
                    continue
        
        if best_result is None:
            raise Exception("Failed to find optimal portfolio after multiple attempts")
        
        # Calculate metrics for optimal portfolio
        optimal_weights = best_result.x
        port_return, port_vol, sharpe = self.calculate_portfolio_metrics(
            optimal_weights, returns.values, cov_matrix.values
        )
        
        if verbose:
            print("\nOptimal portfolio found:")
            print(f"Return (annualized): {port_return:.2%}")
            print(f"Volatility (annualized): {port_vol:.2%}")
            print(f"Sharpe Ratio: {sharpe:.2f}")
            print("\nOptimal weights:")
            for asset, weight in zip(returns.columns, optimal_weights):
                print(f"{asset}: {weight:.2%}")
        
        return {
            'weights': dict(zip(returns.columns, optimal_weights)),
            'return': port_return,
            'volatility': port_vol,
            'sharpe_ratio': sharpe
        }
    
    def analyze_ef_uniqueness(self, ef: pd.DataFrame) -> pd.DataFrame:
        """Analyze the uniqueness of efficient frontier points."""
        if len(ef) == 0:
            return ef
            
        # Round to fewer decimal places for more lenient comparison
        ef_rounded = ef.round(4)  # Changed from 6 to 4 decimal places
        
        # Check for duplicates
        duplicates = ef_rounded.duplicated(subset=['return', 'volatility'], keep='first')  # Changed from False to 'first'
        
        unique_ef = ef[~duplicates].copy()  # Create explicit copy
        
        if len(unique_ef) == 0:
            # If all points were considered duplicates, keep at least one point
            unique_ef = ef.iloc[[0]].copy()  # Create explicit copy
        
        if duplicates.any():
            print(f"\nFound {duplicates.sum()} similar points in the efficient frontier")
            print(f"Keeping {len(unique_ef)} unique points")
            
        # Calculate point-to-point differences for remaining points
        unique_ef['return_diff'] = unique_ef['return'].diff()
        unique_ef['volatility_diff'] = unique_ef['volatility'].diff()
        
        return unique_ef

    def plot_efficient_frontier(self, ef_df, show_sharpe=True, show_assets=True, returns=None, show_plot=True):
        """Plot the efficient frontier with Monte Carlo simulation points."""
        # Clear any existing plots
        plt.clf()
        
        # Create new figure
        fig = plt.figure(figsize=(14, 10), dpi=300)
        plt.rcParams.update({'font.size': 12})
        
        # Generate Monte Carlo simulation points first (so they appear in the background)
        if returns is not None:
            n_assets = len(returns.columns)
            n_portfolios = 1000  # Number of random portfolios to generate
            
            # Generate random weights that satisfy constraints
            weights = np.zeros((n_portfolios, n_assets))
            for i in range(n_portfolios):
                # Generate weights that sum to 1 and satisfy position constraints
                w = np.random.random(n_assets)
                w = w / w.sum()  # Normalize to sum to 1
                # Apply position constraints
                w = np.clip(w, self.constraints['min_position'], self.constraints['max_position'])
                # Iteratively adjust weights to satisfy both sum and position constraints
                while not np.isclose(np.sum(w), 1.0, atol=1e-3) or np.any(w > self.constraints['max_position']):
                    w = w / w.sum()
                    w = np.clip(w, self.constraints['min_position'], self.constraints['max_position'])
                weights[i] = w
            
            # Calculate portfolio metrics using actual returns and covariance
            annual_returns = returns.mean() * 252
            cov_matrix = returns.cov() * 252
            
            port_returns = np.zeros(n_portfolios)
            port_volatilities = np.zeros(n_portfolios)
            port_sharpe_ratios = np.zeros(n_portfolios)
            
            for i in range(n_portfolios):
                w = weights[i]
                port_returns[i] = np.sum(w * annual_returns)
                port_volatilities[i] = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
                port_sharpe_ratios[i] = (port_returns[i] - self.risk_free_rate) / port_volatilities[i]
            
            # Plot Monte Carlo points with color based on Sharpe ratio
            scatter = plt.scatter(port_volatilities, port_returns, 
                                c=port_sharpe_ratios, cmap='viridis', 
                                alpha=0.3, s=15,
                                label='Random Portfolios')
            cbar = plt.colorbar(scatter, label='Sharpe Ratio')
            cbar.ax.tick_params(labelsize=12)
        
        # Plot individual assets if requested
        if show_assets and returns is not None:
            asset_returns = returns.mean() * 252  # Annualized returns
            asset_volatility = returns.std() * np.sqrt(252)  # Annualized volatility
            plt.scatter(asset_volatility, asset_returns, 
                       marker='o', s=150, c='red', label='Individual Assets')
            
            # Add asset labels with offset
            for i, (asset, vol, ret) in enumerate(zip(returns.columns, asset_volatility, asset_returns)):
                plt.annotate(asset, (vol, ret), 
                            xytext=(5, 5), textcoords='offset points',
                            fontsize=10, alpha=0.8)
        
        # Plot efficient frontier
        if not ef_df.empty:
            plt.plot(ef_df['volatility'], ef_df['return'], 
                    'b-', linewidth=3, label='Efficient Frontier')
            
            # Highlight maximum Sharpe ratio point if requested
            if show_sharpe:
                max_sharpe_idx = ef_df['sharpe_ratio'].idxmax()
                if not pd.isna(max_sharpe_idx):
                    max_sharpe_point = ef_df.loc[max_sharpe_idx]
                    plt.scatter(max_sharpe_point['volatility'], max_sharpe_point['return'], 
                              color='gold', marker='*', s=300, 
                              label='Maximum Sharpe Ratio')
                    # Add annotation for max Sharpe ratio point
                    plt.annotate(f'Sharpe: {max_sharpe_point["sharpe_ratio"]:.2f}',
                               (max_sharpe_point['volatility'], max_sharpe_point['return']),
                               xytext=(10, 10), textcoords='offset points',
                               fontsize=10,
                               bbox=dict(facecolor='white', alpha=0.7))
        
        plt.xlabel('Expected Volatility (Annualized)', fontsize=14)
        plt.ylabel('Expected Return (Annualized)', fontsize=14)
        plt.title('Portfolio Optimization: Efficient Frontier with Monte Carlo Simulation', fontsize=14)
        plt.legend(bbox_to_anchor=(0.95, 0.05), loc='lower right', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tick_params(axis='both', which='major', labelsize=12)
        
        # Add text box with summary statistics
        if not ef_df.empty:
            stats_text = (
                f'Risk-free Rate: {self.risk_free_rate:.2%}\n'
                f'Min Return: {ef_df["return"].min():.2%}\n'
                f'Max Return: {ef_df["return"].max():.2%}\n'
                f'Min Volatility: {ef_df["volatility"].min():.2%}\n'
                f'Max Volatility: {ef_df["volatility"].max():.2%}'
            )
            if returns is not None:
                stats_text = f'Number of Assets: {len(returns.columns)}\n' + stats_text
            
            plt.text(1.20, 0.5, stats_text, transform=plt.gca().transAxes,
                    bbox=dict(facecolor='white', alpha=0.7),
                    verticalalignment='center',
                    fontsize=12)
        
        plt.tight_layout()
        if show_plot:
            return fig
        else:
            plt.close(fig)
            plt.clf()
            return None

    def plot_portfolio_weights(self, ef: pd.DataFrame, n_points: int = 5, show_plot: bool = True) -> None:
        """Plot portfolio weights for selected points along the efficient frontier."""
        # Clear any existing plots
        plt.clf()
        
        # Ensure we don't request more points than we have
        n_points = min(n_points, len(ef))
        
        if n_points == 0:
            print("No points available to plot weights")
            return None
            
        # Select evenly spaced points
        if n_points == 1:
            indices = [0]
        else:
            indices = np.linspace(0, len(ef)-1, n_points, dtype=int)
        
        selected_portfolios = ef.iloc[indices]
        
        try:
            # Create weight matrix
            weights_list = []
            for _, row in selected_portfolios.iterrows():
                weights_dict = row['weights']
                weights_list.append(weights_dict)
            
            weights_df = pd.DataFrame(weights_list)
            
            # Create figure and axis objects
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot stacked bar chart
            weights_df.plot(kind='bar', stacked=True, ax=ax)
            plt.title(f'Portfolio Weights Along the Efficient Frontier ({n_points} points)')
            plt.xlabel('Portfolio Index (Low to High Risk)')
            plt.ylabel('Weight')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
                        
            # Print the actual returns and volatilities for these points
            print("\nSelected portfolios metrics:")
            for i, (_, row) in enumerate(selected_portfolios.iterrows()):
                print(f"\nPortfolio {i+1}:")
                print(f"Return: {row['return']:.2%}")
                print(f"Volatility: {row['volatility']:.2%}")
                print(f"Sharpe Ratio: {row['sharpe_ratio']:.2f}")
                print("\nWeights:")
                weights = row['weights']
                for asset, weight in weights.items():
                    print(f"{asset}: {weight:.2%}")
            if show_plot:
                return fig
            else:
                plt.close(fig)
                plt.clf()
                return None
        
        except Exception as e:
            print(f"Error plotting weights: {str(e)}")
            print("Raw data for debugging:")
            print(f"Number of portfolios: {len(ef)}")
            print(f"Selected indices: {indices}")
            print(f"Selected portfolios shape: {selected_portfolios.shape}")
            if len(selected_portfolios) > 0:
                print("Sample weights structure:", dict(selected_portfolios.iloc[0]['weights']))
            return None

    def generate_efficient_frontier(
        self, 
        returns: pd.DataFrame, 
        n_points: int = 50,
        verbose: bool = True,
        show_plots: bool = True
    ) -> pd.DataFrame:
        """Generate efficient frontier points."""
        try:
            if verbose:
                print("\nStarting efficient frontier generation...")
                print(f"Number of assets: {len(returns.columns)}")
                print(f"Target number of points: {n_points}")
            
            # First, get the boundary portfolios
            if verbose:
                print("\nFinding minimum volatility portfolio...")
            min_vol_port = self.optimize_portfolio(returns, objective='volatility', verbose=verbose)
            
            if verbose:
                print("\nFinding maximum return portfolio...")
            max_ret_port = self.optimize_portfolio(returns, objective='return', verbose=verbose)
            
            # Add some margin to the return range to ensure diversity
            min_ret = min_vol_port['return'] * 0.95  # Go 5% below min return
            max_ret = max_ret_port['return'] * 1.05  # Go 5% above max return
            
            if verbose:
                print("\nBoundary portfolios found:")
                print(f"Min return: {min_ret:.2%}")
                print(f"Max return: {max_ret:.2%}")
                print(f"Return range: {max_ret - min_ret:.2%}")
            
            # Generate target returns with some randomness
            base_returns = np.linspace(min_ret, max_ret, n_points)
            noise = np.random.normal(0, (max_ret - min_ret) * 0.02, n_points)  # Add 2% noise
            target_returns = np.clip(base_returns + noise, min_ret, max_ret)
            target_returns.sort()
            
            if verbose:
                print(f"\nGenerated {len(target_returns)} target return points")
                print(f"Min target return: {target_returns[0]:.2%}")
                print(f"Max target return: {target_returns[-1]:.2%}")
            
            efficient_portfolios = []
            n_assets = len(returns.columns)
            successful_points = 0
            failed_points = 0
            
            if verbose:
                print("\nGenerating efficient frontier points...")
            
            for i, target_return in enumerate(tqdm(target_returns, disable=not verbose)):
                if verbose:
                    print(f"\nOptimizing for target return {target_return:.2%} ({i+1}/{n_points})")
                
                try:
                    # Initial weights - try different starting points
                    initial_weights_list = [
                        np.array([1/n_assets] * n_assets),  # Equal weights
                        np.array(list(min_vol_port['weights'].values())),  # Min vol weights
                        np.array(list(max_ret_port['weights'].values()))   # Max return weights
                    ]
                    
                    best_portfolio = None
                    min_volatility = float('inf')
                    attempts_for_point = 0
                    
                    for j, initial_weights in enumerate(initial_weights_list):
                        try:
                            if verbose:
                                print(f"\nTrying initial weights set {j+1}")
                                print("Initial weights sum:", np.sum(initial_weights))
                            
                            # Constraints including target return
                            constraints = [
                                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Weights sum to 1
                                {'type': 'eq', 'fun': lambda x: np.sum(returns.mean() * x) * 252 - target_return}  # Target return
                            ]
                            
                            # Bounds for individual weights
                            bounds = tuple(
                                (self.constraints['min_position'], self.constraints['max_position'])
                                for _ in range(n_assets)
                            )
                            
                            # Optimize for minimum volatility at this return level
                            result = minimize(
                                fun=lambda x: np.sqrt(np.dot(x.T, np.dot(returns.cov(), x))) * np.sqrt(252),
                                x0=initial_weights,
                                method='SLSQP',
                                bounds=bounds,
                                constraints=constraints,
                                options={
                                    'ftol': 1e-9,
                                    'maxiter': 1000,
                                    'disp': verbose
                                }
                            )
                            
                            if result.success:
                                attempts_for_point += 1
                                volatility = result.fun
                                if volatility < min_volatility:
                                    if verbose:
                                        print(f"Found better solution with volatility: {volatility:.2%}")
                                    min_volatility = volatility
                                    weights = result.x
                                    portfolio_return = np.sum(returns.mean() * weights) * 252
                                    sharpe = (portfolio_return - self.risk_free_rate) / volatility
                                    
                                    best_portfolio = {
                                        'return': portfolio_return,
                                        'volatility': volatility,
                                        'sharpe_ratio': sharpe,
                                        'weights': dict(zip(returns.columns, weights))
                                    }
                            else:
                                if verbose:
                                    print(f"Optimization failed: {result.message}")
                        
                        except Exception as e:
                            if verbose:
                                print(f"Failed with initial weights set {j+1}: {str(e)}")
                            continue
                    
                    if best_portfolio is not None:
                        efficient_portfolios.append(best_portfolio)
                        successful_points += 1
                        if verbose:
                            print(f"\nSuccessfully found portfolio for target return {target_return:.2%}")
                            print(f"Attempts needed: {attempts_for_point}")
                    else:
                        failed_points += 1
                        if verbose:
                            print(f"\nFailed to find portfolio for target return {target_return:.2%}")
                
                except Exception as e:
                    failed_points += 1
                    if verbose:
                        print(f"\nFailed to optimize for target return {target_return:.2%}: {str(e)}")
                    continue
            
            if verbose:
                print("\nEfficient Frontier Generation Summary:")
                print(f"Successful points: {successful_points}")
                print(f"Failed points: {failed_points}")
                print(f"Success rate: {successful_points/(successful_points+failed_points)*100:.1f}%")
            
            if not efficient_portfolios:
                raise Exception("Failed to generate any efficient frontier points")
            
            # Create DataFrame and sort by return
            ef = pd.DataFrame(efficient_portfolios)
            
            if verbose:
                print("\nRaw efficient frontier points:", len(ef))
            
            ef = ef.sort_values('return')
            
            # Analyze uniqueness and remove duplicates
            ef = self.analyze_ef_uniqueness(ef)
            
            if verbose:
                print("\nEfficient frontier generated successfully:")
                print(f"Number of unique points: {len(ef)}")
                if len(ef) > 0:
                    print(f"Return range: {ef['return'].min():.2%} to {ef['return'].max():.2%}")
                    print(f"Volatility range: {ef['volatility'].min():.2%} to {ef['volatility'].max():.2%}")
                
                # Plot the frontier
                if len(ef) > 0 and verbose:
                    self.plot_efficient_frontier(ef, show_sharpe=True, show_assets=True, returns=returns, show_plot=show_plots)
                    self.plot_portfolio_weights(ef, show_plot=show_plots)
            
            return ef
            
        except Exception as e:
            print("\nDetailed error information:")
            print(f"Returns shape: {returns.shape}")
            print("\nReturns summary:")
            print(returns.describe())
            print("\nConstraints:")
            print(self.constraints)
            raise Exception(f"Failed to generate efficient frontier: {str(e)}")
    
    def generate_efficient_frontier_relaxed(self, returns: pd.DataFrame, n_points: int = 50) -> pd.DataFrame:
        """Generate efficient frontier points with relaxed constraints."""
        # Temporarily relax constraints
        original_max_position = self.constraints['max_position']
        original_min_position = self.constraints['min_position']
        
        try:
            # Relax position constraints
            self.constraints['max_position'] = min(0.5, 2.0 / len(returns.columns))
            self.constraints['min_position'] = 0.0
            
            # Try to generate frontier with relaxed constraints
            ef = self.generate_efficient_frontier(returns, n_points)
            
            return ef
        finally:
            # Restore original constraints
            self.constraints['max_position'] = original_max_position
            self.constraints['min_position'] = original_min_position

if __name__ == "__main__":
    # Example usage
    import pandas as pd
    
    # Load returns data
    returns = pd.read_csv('data/returns_daily.csv', index_col=0)
    
    # Initialize optimizer
    optimizer = PortfolioOptimizer()
    
    # Optimize portfolio
    result = optimizer.optimize_portfolio(returns)
    print("\nOptimal Portfolio:")
    print(f"Sharpe Ratio: {result['sharpe_ratio']:.2f}")
    print(f"Annual Return: {result['return']:.2%}")
    print(f"Annual Volatility: {result['volatility']:.2%}")
    
    # Generate efficient frontier
    ef = optimizer.generate_efficient_frontier(returns)
    print("\nEfficient Frontier Generated") 