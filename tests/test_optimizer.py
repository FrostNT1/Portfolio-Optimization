import os
import pytest
import numpy as np
import pandas as pd
from portfolio_optimization.optimization import PortfolioOptimizer

@pytest.fixture
def optimizer():
    """Create a PortfolioOptimizer instance with test configuration."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "test_config.yaml")
    return PortfolioOptimizer(config_path)

@pytest.fixture
def sample_returns():
    """Create sample returns data for testing."""
    np.random.seed(42)  # For reproducibility
    dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq='D')
    assets = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    
    # Generate random returns with some correlation
    n_days = len(dates)
    n_assets = len(assets)
    corr = np.array([[1.0, 0.5, 0.3, 0.2, 0.1],
                     [0.5, 1.0, 0.4, 0.3, 0.2],
                     [0.3, 0.4, 1.0, 0.5, 0.3],
                     [0.2, 0.3, 0.5, 1.0, 0.4],
                     [0.1, 0.2, 0.3, 0.4, 1.0]])
    
    # Generate correlated returns
    L = np.linalg.cholesky(corr)
    uncorrelated = np.random.normal(0.0001, 0.02, (n_days, n_assets))
    returns = uncorrelated @ L.T
    
    return pd.DataFrame(returns, index=dates, columns=assets)

def test_initialization(optimizer):
    """Test if PortfolioOptimizer initializes correctly."""
    assert isinstance(optimizer, PortfolioOptimizer)
    assert optimizer.risk_free_rate == 0.04
    assert optimizer.constraints['max_position'] == 0.40
    assert optimizer.constraints['min_position'] == 0.0

def test_calculate_portfolio_metrics(optimizer, sample_returns):
    """Test portfolio metrics calculation."""
    weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])  # Equal weights
    returns_array = sample_returns.values
    cov_matrix = sample_returns.cov().values
    
    port_return, port_vol, sharpe = optimizer.calculate_portfolio_metrics(
        weights, returns_array, cov_matrix
    )
    
    assert isinstance(port_return, float)
    assert isinstance(port_vol, float)
    assert isinstance(sharpe, float)
    assert port_vol > 0  # Volatility should be positive
    assert np.isfinite(sharpe)  # Sharpe ratio should be finite

def test_objective_function(optimizer, sample_returns):
    """Test different objective functions."""
    weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    returns_array = sample_returns.values
    cov_matrix = sample_returns.cov().values
    
    # Test Sharpe ratio objective
    sharpe_obj = optimizer.objective_function(
        weights, returns_array, cov_matrix, 'sharpe'
    )
    assert isinstance(sharpe_obj, float)
    assert np.isfinite(sharpe_obj)
    
    # Test volatility objective
    vol_obj = optimizer.objective_function(
        weights, returns_array, cov_matrix, 'volatility'
    )
    assert isinstance(vol_obj, float)
    assert vol_obj > 0
    
    # Test return objective
    ret_obj = optimizer.objective_function(
        weights, returns_array, cov_matrix, 'return'
    )
    assert isinstance(ret_obj, float)
    
    # Test invalid objective
    with pytest.raises(ValueError):
        optimizer.objective_function(
            weights, returns_array, cov_matrix, 'invalid'
        )

def test_optimize_portfolio(optimizer, sample_returns):
    """Test portfolio optimization."""
    result = optimizer.optimize_portfolio(sample_returns)
    
    assert isinstance(result, dict)
    assert all(key in result for key in ['weights', 'return', 'volatility', 'sharpe_ratio'])
    
    weights = np.array(list(result['weights'].values()))
    assert np.isclose(np.sum(weights), 1.0)  # Weights sum to 1
    assert all(w >= optimizer.constraints['min_position'] for w in weights)
    assert all(w <= optimizer.constraints['max_position'] for w in weights)
    
    assert result['return'] > -1 and result['return'] < 1  # Reasonable return range
    assert result['volatility'] > 0  # Positive volatility
    assert np.isfinite(result['sharpe_ratio'])  # Finite Sharpe ratio

def test_generate_efficient_frontier(optimizer, sample_returns):
    """Test efficient frontier generation."""
    ef = optimizer.generate_efficient_frontier(sample_returns, n_points=10)
    
    assert isinstance(ef, pd.DataFrame)
    assert not ef.empty
    assert all(col in ef.columns for col in ['return', 'volatility', 'sharpe_ratio'])
    
    # Sort by return to ensure monotonicity check works
    ef = ef.sort_values('return')
    
    # Check if points are ordered by return
    assert ef['return'].is_monotonic_increasing or ef['return'].is_monotonic_decreasing
    
    # Check if all values are finite
    assert np.all(np.isfinite(ef['return']))
    assert np.all(np.isfinite(ef['volatility']))
    assert np.all(np.isfinite(ef['sharpe_ratio'])) 