import os
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from scipy.optimize import minimize
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from logging.handlers import RotatingFileHandler

# Configure logging
def setup_logger(
    name: str = __name__,
    console_level: Optional[int] = None,  # None means no console output
    file_level: int = logging.INFO
) -> logging.Logger:
    """Configure and return a logger instance.
    
    Args:
        name: Logger name
        console_level: Logging level for console output. None to disable console logging.
        file_level: Logging level for file output
        
    Returns:
        logging.Logger: Configured logger instance
        
    Notes:
        - Always writes to a rotating log file
        - Console output is optional and can be disabled by setting console_level to None
        - File handler rotates at 10MB with 5 backup files
    """
    logger = logging.getLogger(name)
    logger.setLevel(min(file_level, console_level or logging.INFO))
    
    # Remove any existing handlers
    logger.handlers = []
    
    # Create logs directory if it doesn't exist
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # File handler with rotation (always enabled)
    file_handler = RotatingFileHandler(
        os.path.join(log_dir, "portfolio_optimizer.log"),
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(file_level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler (optional)
    if console_level is not None:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger

# Initialize logger with default settings
# In scripts: Enable console output at INFO level
# In notebooks: Disable console output by not specifying console_level
logger = setup_logger()

class PortfolioOptimizer:
    """A comprehensive portfolio optimization class using mean-variance approach.
    
    This class implements modern portfolio theory (MPT) to find optimal asset allocations
    that maximize expected return for a given level of risk, or minimize risk for a 
    given level of expected return. It supports various optimization objectives including
    maximizing Sharpe ratio, minimizing volatility, and maximizing returns.
    
    Features:
        - Mean-variance optimization
        - Efficient frontier generation
        - Portfolio constraints handling
        - Monte Carlo simulation
        - Visualization tools
        
    Attributes:
        config (dict): Configuration parameters loaded from YAML file
        risk_free_rate (float): Annual risk-free rate used in Sharpe ratio calculations
        constraints (dict): Portfolio constraints including position limits
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the portfolio optimizer with configuration settings.
        
        Args:
            config_path (str): Path to the YAML configuration file relative to project root.
                Defaults to "config/config.yaml".
                
        Raises:
            FileNotFoundError: If the configuration file cannot be found
            yaml.YAMLError: If the configuration file is invalid
        """
        logger.info("Initializing PortfolioOptimizer")
        
        # Get the project root directory (two levels up from this file)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        config_file = os.path.join(project_root, config_path)
        
        logger.debug(f"Loading configuration from: {config_file}")
        try:
            with open(config_file, 'r') as f:
                self.config = yaml.safe_load(f)
            
            self.risk_free_rate = self.config['optimization']['risk_free_rate']
            self.constraints = self.config['constraints']
            
            # Adjust max_position if it's too restrictive
            n_stocks = self.config['universe']['n_stocks']
            min_max_position = 1.0 / n_stocks
            self.constraints['max_position'] = max(min_max_position, self.constraints['max_position'])
            
            logger.info(f"Successfully initialized with {n_stocks} stocks")
            logger.debug(f"Risk-free rate: {self.risk_free_rate:.2%}")
            logger.debug(f"Position constraints: {self.constraints}")
            
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_file}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Invalid YAML configuration: {str(e)}")
            raise
        except KeyError as e:
            logger.error(f"Missing required configuration key: {str(e)}")
            raise
    
    def calculate_portfolio_metrics(
        self, 
        weights: np.ndarray, 
        returns: np.ndarray, 
        cov_matrix: np.ndarray
    ) -> Tuple[float, float, float]:
        """Calculate key portfolio performance metrics.
        
        This method computes the annualized return, volatility, and Sharpe ratio
        for a given portfolio allocation. All metrics are annualized assuming
        252 trading days per year.
        
        Args:
            weights (np.ndarray): Portfolio weights for each asset
            returns (np.ndarray): Historical returns data
            cov_matrix (np.ndarray): Covariance matrix of returns
            
        Returns:
            Tuple[float, float, float]: A tuple containing:
                - portfolio_return (float): Annualized portfolio return
                - portfolio_vol (float): Annualized portfolio volatility
                - sharpe_ratio (float): Portfolio Sharpe ratio
                
        Notes:
            - Returns and volatility are annualized assuming 252 trading days
            - Uses the risk-free rate specified in the configuration
            - Handles numerical instability in volatility calculation
        """
        logger.debug("Calculating portfolio metrics")
        weights = np.array(weights)
        
        # Ensure the inputs are properly scaled
        returns = np.array(returns)
        cov_matrix = np.array(cov_matrix)
        
        # Calculate annualized metrics
        portfolio_return = np.sum(returns.mean() * weights) * 252
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
        
        # Handle numerical instability
        if portfolio_vol < 1e-8:
            logger.warning("Portfolio volatility near zero, adjusting to prevent division by zero")
            portfolio_vol = 1e-8
            
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_vol
        
        logger.debug(f"Portfolio metrics calculated - Return: {portfolio_return:.2%}, "
                    f"Volatility: {portfolio_vol:.2%}, Sharpe: {sharpe_ratio:.2f}")
        
        return portfolio_return, portfolio_vol, sharpe_ratio
    
    def objective_function(
        self, 
        weights: np.ndarray, 
        returns: np.ndarray, 
        cov_matrix: np.ndarray, 
        objective: str = 'sharpe'
    ) -> float:
        """Calculate the objective function value for portfolio optimization.
        
        This method computes the objective function value based on the specified
        optimization goal. It supports three objectives:
        - 'sharpe': Maximize Sharpe ratio (minimize negative Sharpe)
        - 'volatility': Minimize portfolio volatility
        - 'return': Maximize portfolio return (minimize negative return)
        
        Args:
            weights (np.ndarray): Portfolio weights for each asset
            returns (np.ndarray): Historical returns data
            cov_matrix (np.ndarray): Covariance matrix of returns
            objective (str, optional): Optimization objective.
                Must be one of ['sharpe', 'volatility', 'return'].
                Defaults to 'sharpe'.
                
        Returns:
            float: The objective function value to be minimized
            
        Raises:
            ValueError: If an unsupported objective is specified
            
        Notes:
            - Returns a large number (1e6) if calculation fails to ensure
              the optimizer avoids invalid solutions
            - For 'sharpe' and 'return' objectives, returns negative values
              since we're minimizing
        """
        if objective not in ['sharpe', 'volatility', 'return']:
            logger.error(f"Invalid objective function specified: {objective}")
            raise ValueError(f"Unsupported objective: {objective}")
            
        try:
            logger.debug(f"Calculating {objective} objective for weights sum: {np.sum(weights):.4f}")
            
            portfolio_return, portfolio_vol, sharpe_ratio = self.calculate_portfolio_metrics(
                weights, returns, cov_matrix
            )
            
            if objective == 'sharpe':
                result = -sharpe_ratio  # Minimize negative Sharpe ratio
            elif objective == 'volatility':
                result = portfolio_vol
            else:  # objective == 'return'
                result = -portfolio_return  # Minimize negative return
                
            logger.debug(f"Objective '{objective}' value: {result:.6f}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to calculate objective function: {str(e)}")
            # Return a large number if calculation fails
            return 1e6
        
    def optimize_portfolio(
        self,
        returns: pd.DataFrame,
        objective: str = 'sharpe',
        target_return: Optional[float] = None,
        initial_weights: Optional[np.ndarray] = None,
        verbose: bool = True
    ) -> Dict:
        """Optimize portfolio weights to achieve the specified objective.
        
        This method performs portfolio optimization using the scipy.optimize
        minimize function with the SLSQP method. It supports various objectives
        and constraints, including target return constraints.
        
        Args:
            returns (pd.DataFrame): Historical returns data with assets as columns
            objective (str, optional): Optimization objective.
                Must be one of ['sharpe', 'volatility', 'return'].
                Defaults to 'sharpe'.
            target_return (float, optional): Target portfolio return.
                Required if objective is 'volatility'.
                Defaults to None.
            initial_weights (np.ndarray, optional): Starting point for optimization.
                If None, uses equal weights.
                Defaults to None.
            verbose (bool, optional): Whether to print optimization progress.
                Defaults to True.
                
        Returns:
            Dict: A dictionary containing:
                - weights (Dict[str, float]): Optimal portfolio weights
                - return (float): Expected portfolio return
                - volatility (float): Expected portfolio volatility
                - sharpe_ratio (float): Portfolio Sharpe ratio
                
        Raises:
            ValueError: If input validation fails
            Exception: If optimization fails
            
        Notes:
            - Uses multiple initial weight attempts if optimization fails
            - Implements position constraints from configuration
            - Validates inputs for numerical stability
        """
        logger.info("Starting portfolio optimization")
        logger.debug(f"Objective: {objective}, Target return: {target_return}")
        
        # Input validation
        if returns.empty:
            logger.error("Empty returns data provided")
            raise ValueError("Empty returns data")
        if not returns.select_dtypes(include=[np.number]).equals(returns):
            logger.error("Non-numeric data in returns")
            raise ValueError("Non-numeric data in returns")
        if returns.isna().any().any():
            logger.error("NaN values in returns")
            raise ValueError("NaN values in returns")
        if np.isinf(returns).any().any():
            logger.error("Infinite values in returns")
            raise ValueError("Infinite values in returns")

        # Calculate expected returns and covariance matrix
        expected_returns = returns.mean()
        cov_matrix = returns.cov()
        n_assets = len(returns.columns)
        
        logger.debug(f"Number of assets: {n_assets}")
        logger.debug(f"Expected returns range: {expected_returns.min():.2%} to {expected_returns.max():.2%}")

        # Set default target return if not provided
        if target_return is None and objective == 'return':
            target_return = np.mean(expected_returns)
            logger.info(f"Using default target return: {target_return:.2%}")

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}  # Weights sum to 1
        ]
        
        if target_return is not None:
            constraints.append({
                'type': 'eq',
                'fun': lambda x: np.dot(x, expected_returns) - target_return
            })
            logger.debug(f"Added target return constraint: {target_return:.2%}")

        # Bounds
        bounds = [(self.constraints['min_position'], self.constraints['max_position'])] * n_assets
        logger.debug(f"Position bounds: [{self.constraints['min_position']:.2%}, {self.constraints['max_position']:.2%}]")

        # Try multiple initial weights if the first attempt fails
        if initial_weights is None:
            attempts = [
                np.array([1.0 / n_assets] * n_assets),  # Equal weights
                np.random.dirichlet(np.ones(n_assets)),  # Random weights
                np.array([self.constraints['min_position']] * n_assets)  # Minimum weights
            ]
            logger.info("Using multiple initial weight attempts")
        else:
            attempts = [initial_weights]
            logger.info("Using provided initial weights")

        best_result = None
        min_objective = float('inf')

        for i, weights in enumerate(attempts):
            try:
                logger.debug(f"Optimization attempt {i+1} with initial weights sum: {np.sum(weights):.4f}")
                
                result = minimize(
                    fun=self.objective_function,
                    x0=weights,
                    args=(returns.values, cov_matrix.values, objective),
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={
                        'disp': verbose,
                        'maxiter': 1000,
                        'ftol': 1e-8,
                        'eps': 1e-8
                    }
                )

                if result.success and result.fun < min_objective:
                    logger.debug(f"Found better solution with objective value: {result.fun:.6f}")
                    best_result = result
                    min_objective = result.fun

            except Exception as e:
                logger.warning(f"Optimization attempt {i+1} failed: {str(e)}")
                if verbose:
                    print(f"Optimization attempt failed: {str(e)}")
                continue

        if best_result is None:
            logger.error("All optimization attempts failed")
            raise Exception("All optimization attempts failed")

        # Optimal weights
        optimal_weights = best_result.x
        portfolio_return, portfolio_vol, sharpe_ratio = self.calculate_portfolio_metrics(
            optimal_weights, returns.values, cov_matrix.values
        )
        
        result_dict = {
            'weights': dict(zip(returns.columns, optimal_weights)),
            'return': portfolio_return,
            'volatility': portfolio_vol,
            'sharpe_ratio': sharpe_ratio
        }
        
        logger.info("Portfolio optimization completed successfully")
        logger.info(f"Final metrics - Return: {portfolio_return:.2%}, "
                   f"Volatility: {portfolio_vol:.2%}, Sharpe: {sharpe_ratio:.2f}")
        
        return result_dict

    def analyze_ef_uniqueness(self, ef: pd.DataFrame) -> pd.DataFrame:
        """Analyze and filter unique points on the efficient frontier.
        
        This method processes the efficient frontier points to remove near-duplicate
        points and ensure a clean, well-distributed frontier. Points are considered
        duplicates if they have the same return and volatility values when rounded
        to 4 decimal places.
        
        Args:
            ef (pd.DataFrame): Efficient frontier DataFrame containing columns:
                - return: Portfolio returns
                - volatility: Portfolio volatilities
                - weights: Portfolio weights
                - sharpe_ratio: Portfolio Sharpe ratios
                
        Returns:
            pd.DataFrame: Filtered efficient frontier with unique points and
                additional columns for point-to-point differences
                
        Notes:
            - Rounds values to 4 decimal places for comparison
            - Keeps at least one point even if all points are similar
            - Adds columns for return_diff and volatility_diff between points
        """
        logger.info("Analyzing efficient frontier uniqueness")
        
        if len(ef) == 0:
            logger.warning("Empty efficient frontier provided")
            return ef
            
        # Round to fewer decimal places for more lenient comparison
        ef_rounded = ef.round(4)
        
        # Check for duplicates
        duplicates = ef_rounded.duplicated(subset=['return', 'volatility'], keep='first')
        
        unique_ef = ef[~duplicates].copy()
        
        if len(unique_ef) == 0:
            logger.warning("All points were considered duplicates, keeping one point")
            unique_ef = ef.iloc[[0]].copy()
        
        if duplicates.any():
            logger.info(f"Found {duplicates.sum()} similar points in the efficient frontier")
            logger.info(f"Keeping {len(unique_ef)} unique points")
            
        # Calculate point-to-point differences for remaining points
        unique_ef['return_diff'] = unique_ef['return'].diff()
        unique_ef['volatility_diff'] = unique_ef['volatility'].diff()
        
        logger.debug("Point-to-point differences statistics:")
        logger.debug(f"Return differences - Mean: {unique_ef['return_diff'].mean():.4%}, "
                    f"Std: {unique_ef['return_diff'].std():.4%}")
        logger.debug(f"Volatility differences - Mean: {unique_ef['volatility_diff'].mean():.4%}, "
                    f"Std: {unique_ef['volatility_diff'].std():.4%}")
        
        return unique_ef

    def plot_efficient_frontier(
        self, 
        ef_df: pd.DataFrame, 
        show_sharpe: bool = True, 
        show_assets: bool = True, 
        returns: Optional[pd.DataFrame] = None, 
        show_plot: bool = True
    ) -> Optional[plt.Figure]:
        """Create a comprehensive visualization of the efficient frontier.
        
        This method generates a detailed plot of the efficient frontier, optionally
        including Monte Carlo simulation points, individual assets, and the maximum
        Sharpe ratio point. The plot includes a color-coded representation of
        Sharpe ratios and detailed statistics.
        
        Args:
            ef_df (pd.DataFrame): Efficient frontier DataFrame with portfolio metrics
            show_sharpe (bool, optional): Whether to highlight max Sharpe ratio point.
                Defaults to True.
            show_assets (bool, optional): Whether to plot individual assets.
                Defaults to True.
            returns (pd.DataFrame, optional): Historical returns data for Monte Carlo
                simulation. If None, Monte Carlo points are not plotted.
                Defaults to None.
            show_plot (bool, optional): Whether to display the plot.
                If False, returns the figure object without displaying.
                Defaults to True.
                
        Returns:
            Optional[plt.Figure]: Matplotlib figure object if show_plot is True,
                None otherwise
                
        Notes:
            - Generates 1000 random portfolios for Monte Carlo simulation
            - Uses high DPI (300) for publication-quality plots
            - Includes a detailed legend and statistics box
            - Automatically adjusts layout for readability
        """
        logger.info("Creating efficient frontier plot")
        
        # Clear any existing plots
        plt.clf()
        
        # Create new figure
        fig = plt.figure(figsize=(14, 10), dpi=300)
        plt.rcParams.update({'font.size': 12})
        
        logger.debug("Generating Monte Carlo simulation points")
        # Generate Monte Carlo simulation points first (so they appear in the background)
        if returns is not None:
            n_assets = len(returns.columns)
            n_portfolios = 1000
            
            logger.debug(f"Generating {n_portfolios} random portfolios for {n_assets} assets")
            
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
            
            logger.debug("Calculating portfolio metrics for Monte Carlo points")
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
            
            logger.debug("Plotting Monte Carlo simulation points")
            # Plot Monte Carlo points with color based on Sharpe ratio
            scatter = plt.scatter(port_volatilities, port_returns, 
                                c=port_sharpe_ratios, cmap='viridis', 
                                alpha=0.3, s=15,
                                label='Random Portfolios')
            cbar = plt.colorbar(scatter, label='Sharpe Ratio')
            cbar.ax.tick_params(labelsize=12)
        
        logger.debug("Adding individual assets to plot")
        # Plot individual assets if requested
        if show_assets and returns is not None:
            asset_returns = returns.mean() * 252
            asset_volatility = returns.std() * np.sqrt(252)
            plt.scatter(asset_volatility, asset_returns, 
                       marker='o', s=150, c='red', label='Individual Assets')
            
            # Add asset labels with offset
            for i, (asset, vol, ret) in enumerate(zip(returns.columns, asset_volatility, asset_returns)):
                plt.annotate(asset, (vol, ret), 
                            xytext=(5, 5), textcoords='offset points',
                            fontsize=10, alpha=0.8)
        
        logger.debug("Plotting efficient frontier line")
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
                    logger.info(f"Maximum Sharpe ratio: {max_sharpe_point['sharpe_ratio']:.2f}")
        
        plt.xlabel('Expected Volatility (Annualized)', fontsize=14)
        plt.ylabel('Expected Return (Annualized)', fontsize=14)
        plt.title('Portfolio Optimization: Efficient Frontier with Monte Carlo Simulation', fontsize=14)
        plt.legend(bbox_to_anchor=(0.95, 0.05), loc='lower right', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tick_params(axis='both', which='major', labelsize=12)
        
        logger.debug("Adding statistics box to plot")
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
        
        logger.info("Efficient frontier plot created successfully")
        if show_plot:
            return fig
        else:
            plt.close(fig)
            plt.clf()
            return None

    def plot_portfolio_weights(
        self, 
        ef: pd.DataFrame, 
        n_points: int = 5, 
        show_plot: bool = True
    ) -> Optional[plt.Figure]:
        """Create a visualization of portfolio weights along the efficient frontier.
        
        This method generates a stacked bar chart showing the weight distribution
        of assets for selected points along the efficient frontier. It also prints
        detailed metrics for each selected portfolio.
        
        Args:
            ef (pd.DataFrame): Efficient frontier DataFrame containing portfolio
                weights and metrics
            n_points (int, optional): Number of portfolios to display.
                Points are selected evenly along the frontier.
                Defaults to 5.
            show_plot (bool, optional): Whether to display the plot.
                If False, returns the figure object without displaying.
                Defaults to True.
                
        Returns:
            Optional[plt.Figure]: Matplotlib figure object if show_plot is True,
                None otherwise
                
        Notes:
            - Selects evenly spaced points along the frontier
            - Prints detailed metrics for each selected portfolio
            - Uses a stacked bar chart for weight visualization
            - Automatically adjusts layout for readability
        """
        logger.info(f"Creating portfolio weights plot for {n_points} points")
        
        # Clear any existing plots
        plt.clf()
        
        # Ensure we don't request more points than we have
        n_points = min(n_points, len(ef))
        
        if n_points == 0:
            logger.warning("No points available to plot weights")
            return None
            
        # Select evenly spaced points
        if n_points == 1:
            indices = [0]
        else:
            indices = np.linspace(0, len(ef)-1, n_points, dtype=int)
            
        # Select portfolios at the chosen indices
        selected_portfolios = ef.iloc[indices]
        
        logger.debug(f"Selected {len(indices)} points from efficient frontier")
        
        try:
            # Create weight matrix
            weights_list = []
            for _, row in selected_portfolios.iterrows():
                weights_dict = row['weights']
                weights_list.append(weights_dict)
            
            weights_df = pd.DataFrame(weights_list)
            
            logger.debug(f"Created weights matrix with shape {weights_df.shape}")
            
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
            logger.info("\nSelected portfolios metrics:")
            for i, (_, row) in enumerate(selected_portfolios.iterrows()):
                logger.info(f"\nPortfolio {i+1}:")
                logger.info(f"Return: {row['return']:.2%}")
                logger.info(f"Volatility: {row['volatility']:.2%}")
                logger.info(f"Sharpe Ratio: {row['sharpe_ratio']:.2f}")
                logger.debug("\nWeights:")
                weights = row['weights']
                for asset, weight in weights.items():
                    logger.debug(f"{asset}: {weight:.2%}")
                    
            logger.info("Portfolio weights plot created successfully")
            
            if show_plot:
                return fig
            else:
                plt.close(fig)
                plt.clf()
                return None
        
        except Exception as e:
            logger.error(f"Error plotting weights: {str(e)}")
            logger.debug("Raw data for debugging:")
            logger.debug(f"Number of portfolios: {len(ef)}")
            logger.debug(f"Selected indices: {indices}")
            logger.debug(f"Selected portfolios shape: {selected_portfolios.shape}")
            if len(selected_portfolios) > 0:
                logger.debug("Sample weights structure:", dict(selected_portfolios.iloc[0]['weights']))
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
    
    def generate_efficient_frontier_relaxed(
        self, 
        returns: pd.DataFrame, 
        n_points: int = 50
    ) -> pd.DataFrame:
        """Generate efficient frontier points with relaxed position constraints.
        
        This method temporarily relaxes the position constraints to allow for
        more flexibility in portfolio construction. This can be useful when
        the original constraints are too restrictive to find feasible solutions.
        
        Args:
            returns (pd.DataFrame): Historical returns data with assets as columns
            n_points (int, optional): Number of points to generate on the frontier.
                Defaults to 50.
                
        Returns:
            pd.DataFrame: Efficient frontier points with relaxed constraints
            
        Notes:
            - Temporarily sets maximum position to min(0.5, 2/n_assets)
            - Sets minimum position to 0.0
            - Restores original constraints after generation
            - Uses the same generation process as the standard method
        """
        logger.info("Generating efficient frontier with relaxed constraints")
        
        # Temporarily relax constraints
        original_max_position = self.constraints['max_position']
        original_min_position = self.constraints['min_position']
        
        try:
            # Relax position constraints
            self.constraints['max_position'] = min(0.5, 2.0 / len(returns.columns))
            self.constraints['min_position'] = 0.0
            
            logger.debug("Relaxed constraints:")
            logger.debug(f"Max position: {self.constraints['max_position']:.2%}")
            logger.debug(f"Min position: {self.constraints['min_position']:.2%}")
            
            # Try to generate frontier with relaxed constraints
            ef = self.generate_efficient_frontier(returns, n_points)
            
            logger.info("Successfully generated efficient frontier with relaxed constraints")
            return ef
            
        except Exception as e:
            logger.error(f"Failed to generate efficient frontier with relaxed constraints: {str(e)}")
            raise
            
        finally:
            # Restore original constraints
            logger.debug("Restoring original constraints")
            self.constraints['max_position'] = original_max_position
            self.constraints['min_position'] = original_min_position

if __name__ == "__main__":
    # Configure logging for the main script
    main_logger = setup_logger("portfolio_optimization.main")
    main_logger.info("Starting portfolio optimization example")
    
    try:
        # Load returns data
        main_logger.info("Loading returns data")
        returns = pd.read_csv('data/returns_daily.csv', index_col=0)
        main_logger.debug(f"Loaded returns data with shape: {returns.shape}")
        
        # Initialize optimizer
        main_logger.info("Initializing portfolio optimizer")
        optimizer = PortfolioOptimizer()
        
        # Optimize portfolio
        main_logger.info("Optimizing portfolio")
        result = optimizer.optimize_portfolio(returns)
        main_logger.info("\nOptimal Portfolio:")
        main_logger.info(f"Sharpe Ratio: {result['sharpe_ratio']:.2f}")
        main_logger.info(f"Annual Return: {result['return']:.2%}")
        main_logger.info(f"Annual Volatility: {result['volatility']:.2%}")
        
        # Generate efficient frontier
        main_logger.info("\nGenerating efficient frontier")
        ef = optimizer.generate_efficient_frontier(returns)
        main_logger.info("Efficient Frontier Generated")
        
    except Exception as e:
        main_logger.error(f"Portfolio optimization failed: {str(e)}")
        raise 
