import os
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from scipy.optimize import minimize
import yaml
from tqdm import tqdm

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
        initial_weights: Optional[np.ndarray] = None
    ) -> Dict:
        """Optimize portfolio weights given returns data."""
        if objective not in ['sharpe', 'volatility', 'return']:
            raise ValueError(f"Unsupported objective: {objective}")
            
        # Calculate inputs
        mean_returns = returns.mean()
        cov_matrix = returns.cov()
        
        # Scale the covariance matrix to improve numerical stability
        scale = np.median(np.diag(cov_matrix))
        if scale > 0:
            cov_matrix = cov_matrix / scale
        
        n_assets = len(returns.columns)
        
        # Adjust max_position if needed
        min_max_position = 1.0 / n_assets
        max_position = max(min_max_position, self.constraints['max_position'])
        
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
        methods = ['SLSQP', 'trust-constr']
        for method in methods:
            for _ in range(2):  # Try each method twice
                try:
                    result = minimize(
                        fun=self.objective_function,
                        x0=initial_weights,
                        args=(returns.values, cov_matrix.values, objective),
                        method=method,
                        bounds=bounds,
                        constraints=constraints,
                        options={
                            'ftol': 1e-8,
                            'maxiter': 2000,
                            'disp': False
                        }
                    )
                    
                    if result.success:
                        metric = self.objective_function(
                            result.x, returns.values, cov_matrix.values, objective
                        )
                        if metric < best_metric:
                            best_result = result
                            best_metric = metric
                    
                    # Try new random weights for next iteration
                    initial_weights = np.random.dirichlet(np.ones(n_assets))
                    initial_weights = np.clip(
                        initial_weights,
                        self.constraints['min_position'],
                        max_position
                    )
                    initial_weights = initial_weights / np.sum(initial_weights)
                    
                except Exception:
                    continue
        
        if best_result is None:
            raise Exception("Failed to find optimal portfolio after multiple attempts")
        
        # Calculate metrics for optimal portfolio
        optimal_weights = best_result.x
        port_return, port_vol, sharpe = self.calculate_portfolio_metrics(
            optimal_weights, returns.values, cov_matrix.values
        )
        
        return {
            'weights': dict(zip(returns.columns, optimal_weights)),
            'return': port_return,
            'volatility': port_vol,
            'sharpe_ratio': sharpe
        }
    
    def generate_efficient_frontier(
        self, 
        returns: pd.DataFrame, 
        n_points: int = 50
    ) -> pd.DataFrame:
        """Generate efficient frontier points."""
        try:
            # Get minimum volatility and maximum return portfolios
            min_vol_port = self.optimize_portfolio(returns, objective='volatility')
            max_ret_port = self.optimize_portfolio(returns, objective='return')
            
            min_ret = min_vol_port['return']
            max_ret = max_ret_port['return']
            
            # Ensure min_ret is less than max_ret
            if min_ret > max_ret:
                min_ret, max_ret = max_ret, min_ret
            
            # Add a small buffer to avoid numerical issues
            ret_range = max_ret - min_ret
            min_ret = min_ret - 0.05 * ret_range  # Increased buffer
            max_ret = max_ret + 0.05 * ret_range  # Increased buffer
            
            # Generate return targets
            target_returns = np.linspace(min_ret, max_ret, n_points)
            
            efficient_portfolios = []
            n_assets = len(returns.columns)
            
            # Scale returns and covariance matrix
            returns_scaled = returns.copy()
            returns_scaled = (returns_scaled - returns_scaled.mean()) / returns_scaled.std()
            cov_matrix = returns_scaled.cov()
            
            # Try different initial weights for each target return
            for target_return in tqdm(target_returns, desc="Generating Efficient Frontier"):
                success = False
                
                # Try different initial weights
                initial_weights_list = [
                    np.array([1/n_assets] * n_assets),  # Equal weights
                    np.random.dirichlet(np.ones(n_assets)),  # Random weights
                    np.array(list(min_vol_port['weights'].values())),  # Min vol weights
                    np.array(list(max_ret_port['weights'].values()))  # Max return weights
                ]
                
                for initial_weights in initial_weights_list:
                    if success:
                        break
                        
                    try:
                        # Ensure initial weights are valid
                        initial_weights = np.clip(
                            initial_weights,
                            self.constraints['min_position'],
                            self.constraints['max_position']
                        )
                        initial_weights = initial_weights / np.sum(initial_weights)
                        
                        # Relaxed return constraint with wider tolerance
                        return_tolerance = 0.001  # Increased tolerance
                        constraints = [
                            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Weights sum to 1
                            {'type': 'ineq', 'fun': lambda x:  # Return within tolerance
                                self.calculate_portfolio_metrics(x, returns_scaled.values, cov_matrix.values)[0] - target_return + return_tolerance
                            },
                            {'type': 'ineq', 'fun': lambda x:
                                target_return + return_tolerance - self.calculate_portfolio_metrics(x, returns_scaled.values, cov_matrix.values)[0]
                            }
                        ]
                        
                        bounds = tuple(
                            (max(0.0, self.constraints['min_position']), min(1.0, self.constraints['max_position']))
                            for _ in range(n_assets)
                        )
                        
                        # Try both optimization methods with increased iterations
                        for method in ['SLSQP', 'trust-constr']:
                            try:
                                result = minimize(
                                    fun=lambda x: self.calculate_portfolio_metrics(x, returns_scaled.values, cov_matrix.values)[1],
                                    x0=initial_weights,
                                    method=method,
                                    bounds=bounds,
                                    constraints=constraints,
                                    options={
                                        'ftol': 1e-6,  # Relaxed tolerance
                                        'maxiter': 5000,  # Increased iterations
                                        'disp': False
                                    }
                                )
                                
                                if result.success:
                                    weights = result.x
                                    port_return, port_vol, sharpe = self.calculate_portfolio_metrics(
                                        weights, returns.values, returns.cov().values
                                    )
                                    
                                    efficient_portfolios.append({
                                        'return': port_return,
                                        'volatility': port_vol,
                                        'sharpe_ratio': sharpe,
                                        'weights': dict(zip(returns.columns, weights))
                                    })
                                    success = True
                                    break
                            except Exception:
                                continue
                        
                    except Exception:
                        continue
            
            if not efficient_portfolios:
                raise Exception("Failed to generate any efficient frontier points")
                
            # Create DataFrame and sort by return
            ef = pd.DataFrame(efficient_portfolios)
            ef = ef.sort_values('return')
            
            return ef
            
        except Exception as e:
            raise Exception(f"Failed to generate efficient frontier: {str(e)}")

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